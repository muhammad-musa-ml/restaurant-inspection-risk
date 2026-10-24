"""The measurements: everything in results/ is produced by the functions here, run by scripts/run_pipeline.py.

Randomness and how it is handled: seeds 0-4 drive the boosting (row and column sampling), the random
tie breaks among equal scores, and the random splits; every figure that depends on them is reported as the
median and the range over the five seeds, with the five values kept in the results file.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from . import config, drift, features, metrics, modeling
from .study import Study

SPLITS = ("train", "valid", "test", "post")


# --------------------------------------------------------------------------------------------- helpers
def stat(values) -> dict:
    """Median, range and the individual values of something measured once per seed."""
    arr = np.asarray(list(values), float)
    return {"median": float(np.median(arr)), "min": float(arr.min()), "max": float(arr.max()),
            "values": [float(v) for v in arr]}


def masks_of(rows: pd.DataFrame) -> dict[str, np.ndarray]:
    return {name: (rows["split"] == name).to_numpy() for name in SPLITS}


def tiebreaks(n: int, seeds=config.SEEDS) -> dict[int, np.ndarray]:
    """A random number per row and seed, used only to order rows whose scores are equal."""
    return {seed: np.random.default_rng(1000 + seed).random(n) for seed in seeds}


def previous_result_score(feats: pd.DataFrame, k: int) -> np.ndarray:
    """1.0 when the previous initial inspection (else the previous inspection of any type) cited at least
    k critical violations, 0.0 otherwise or when there is no earlier record."""
    prev = feats["prev_ci_n_crit"]
    chosen = prev.where(prev.notna(), feats["last_n_crit"])
    return (chosen >= k).astype(float).to_numpy()


def evaluate(rows: pd.DataFrame, mask: np.ndarray, score: np.ndarray, tiebreak: np.ndarray, capacity: int) -> dict:
    sub = rows[mask]
    y = sub["y"].to_numpy()
    s = np.asarray(score, float)[mask]
    cap = metrics.capacity_metrics(sub["week"], s, y, tiebreak[mask], capacity)
    return {"auc": float(roc_auc_score(y, s)), "average_precision": float(average_precision_score(y, s)),
            **{k: cap[k] for k in ("precision", "recall", "lift", "hits_per_week", "visits_per_week", "base_rate")}}


def stat_of_runs(runs: list[dict]) -> dict:
    return {key: stat(r[key] for r in runs) for key in runs[0]}


# ---------------------------------------------------------------------------------------------- models
def fit_gbm(frame, y, masks, seeds=config.SEEDS, grid=modeling.GRID):
    """Choose the grid point with the lowest validation log loss (seed 0), then fit every seed with it."""
    xtr, ytr = frame[masks["train"]], y[masks["train"]]
    xva, yva = frame[masks["valid"]], y[masks["valid"]]
    selection = []
    for params in grid:
        _, best_iteration, loss = modeling.fit_lgbm(xtr, ytr, xva, yva, seed=seeds[0], params=params)
        selection.append({"params": params, "best_iteration": best_iteration, "valid_log_loss": loss})
    best = min(selection, key=lambda r: r["valid_log_loss"])
    return best, selection, fit_seeds(frame, y, masks, best["params"], seeds)


def fit_seeds(frame, y, masks, params, seeds=config.SEEDS):
    xtr, ytr = frame[masks["train"]], y[masks["train"]]
    xva, yva = frame[masks["valid"]], y[masks["valid"]]
    return [modeling.fit_lgbm(xtr, ytr, xva, yva, seed=seed, params=params)[0] for seed in seeds]


def predict_matrix(boosters, frame) -> np.ndarray:
    return np.column_stack([modeling.predict_lgbm(b, frame) for b in boosters])


def choose_calibrator(raw_valid, y_valid, weeks_valid):
    """Fit on the first half of the validation weeks, score the second half; pick the lower Brier score."""
    order = np.sort(pd.unique(np.asarray(weeks_valid)))
    first = pd.Series(np.asarray(weeks_valid)).isin(set(order[: len(order) // 2])).to_numpy()
    scores = {"uncalibrated": metrics.brier(y_valid[~first], raw_valid[~first])}
    for cls in (modeling.Platt, modeling.Isotonic):
        calibrator = cls().fit(raw_valid[first], y_valid[first])
        scores[cls.name] = metrics.brier(y_valid[~first], calibrator.predict(raw_valid[~first]))
    chosen = min(("platt", "isotonic"), key=scores.get)
    return chosen, scores


CALIBRATORS = {"platt": modeling.Platt, "isotonic": modeling.Isotonic}


def calibration_numbers(y, p) -> dict:
    return {"ece": metrics.ece(y, p), "brier": metrics.brier(y, p),
            "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))}


def operating_point(p_valid, weeks_valid, p_test, weeks_test, y_test, capacity) -> dict:
    """The probability cut-off that flags `capacity` a week on average over the validation weeks, applied
    unchanged to the test weeks."""
    threshold, fraction = metrics.threshold_for_capacity(p_valid, weeks_valid, capacity)
    result = metrics.flagged_metrics(weeks_test, y_test, p_test >= threshold)
    result.update(threshold=threshold, validation_flag_fraction=fraction)
    return result


# ------------------------------------------------------------------------ random split vs time split
def random_assignment(rows: pd.DataFrame, idx: np.ndarray, sizes: tuple[int, int, int], seed: int, by: str):
    """Row positions for a random train / validation / test split with (about) the time split's sizes.
    by='inspection' shuffles inspections; by='restaurant' keeps each restaurant on one side."""
    rng = np.random.default_rng(2000 + seed)
    n_train, n_valid, _ = sizes
    if by == "inspection":
        perm = rng.permutation(idx)
        return perm[:n_train], perm[n_train:n_train + n_valid], perm[n_train + n_valid:]
    camis = rows["camis"].to_numpy()
    counts = pd.Series(camis[idx]).value_counts()
    order = counts.index.to_numpy().copy()
    rng.shuffle(order)
    cumulative = counts.loc[order].cumsum().to_numpy()
    bucket = (cumulative > n_train).astype(int) + (cumulative > n_train + n_valid).astype(int)
    side = dict(zip(order, bucket))
    assigned = np.array([side[c] for c in camis[idx]])
    return idx[assigned == 0], idx[assigned == 1], idx[assigned == 2]


def split_comparison(st: Study, frame: pd.DataFrame, params: dict, test_raw: np.ndarray, fraction: float,
                     seeds=config.SEEDS) -> dict:
    """Same model, same rows, three ways of cutting them; capacity is stated as a share of each week's pool
    because a random split's test set holds only a share of every week."""
    rows, masks = st.rows, masks_of(st.rows)
    y = rows["y"].to_numpy()
    labelled = np.flatnonzero(~masks["post"])
    sizes = (int(masks["train"].sum()), int(masks["valid"].sum()), int(masks["test"].sum()))
    ties = tiebreaks(len(rows), seeds)
    week = rows["week"].to_numpy()

    def score(test_idx, predictions, seed):
        sub_y = y[test_idx]
        out = metrics.fraction_metrics(week[test_idx], predictions, sub_y, ties[seed][test_idx], fraction)
        return {"auc": float(roc_auc_score(sub_y, predictions)),
                "average_precision": float(average_precision_score(sub_y, predictions)),
                "precision": out["precision"], "recall": out["recall"], "lift": out["lift"],
                "base_rate": out["base_rate"]}

    test_idx = np.flatnonzero(masks["test"])
    result = {"capacity_fraction_of_weekly_pool": fraction,
              "time_split_test_window": stat_of_runs([score(test_idx, test_raw[masks["test"], k], s)
                                                      for k, s in enumerate(seeds)])}
    for by in ("inspection", "restaurant"):
        runs = []
        for seed in seeds:
            tr, va, te = random_assignment(rows, labelled, sizes, seed, by)
            booster = modeling.fit_lgbm(frame.iloc[tr], y[tr], frame.iloc[va], y[va], seed=seed, params=params)[0]
            runs.append(score(te, modeling.predict_lgbm(booster, frame.iloc[te]), seed))
        result[f"random_by_{by}"] = stat_of_runs(runs)
    return result


# ------------------------------------------------------------------------------------------ variants
def variant_runs(st: Study, columns: list[str], params: dict, capacity: int, seeds=config.SEEDS) -> dict:
    """Retrain with the chosen parameters on a variant of the features (5 seeds) and score the test window."""
    rows, masks = st.rows, masks_of(st.rows)
    y = rows["y"].to_numpy()
    frame = modeling.model_frame(st.feats, columns, st.cuisine_levels)
    boosters = fit_seeds(frame, y, masks, params, seeds)
    raw = predict_matrix(boosters, frame)
    ties = tiebreaks(len(rows), seeds)
    return stat_of_runs([evaluate(rows, masks["test"], raw[:, k], ties[s], capacity) for k, s in enumerate(seeds)])


# ------------------------------------------------------------------------------------------- drift
def monthly_drift(st: Study, ensemble_raw: np.ndarray, calibrated: np.ndarray, min_rows: int = 500):
    """PSI per feature and for the prediction, month by month, against fixed reference windows."""
    rows, masks = st.rows, masks_of(st.rows)
    columns = st.numeric + features.CATEGORICAL
    profile = drift.reference_profile(st.feats[masks["train"]], columns)
    prediction_profile = drift.reference_profile(pd.DataFrame({"prediction": ensemble_raw[masks["valid"]]}),
                                                 ["prediction"])
    month = rows["date"].dt.strftime("%Y-%m")
    monitored = rows["split"].isin(["valid", "test", "post"])
    psi_records, calibration_records, skipped = [], [], []
    for name in sorted(month[monitored].unique()):
        selected = monitored & (month == name)
        n = int(selected.sum())
        if n < min_rows:
            skipped.append({"month": name, "rows": n})
            continue
        split = rows.loc[selected, "split"].mode().iloc[0]
        values = drift.psi_against(profile, st.feats[selected])
        if split != "valid":  # the validation window is the prediction reference, so it is not scored against itself
            values["__prediction__"] = drift.psi_against(
                prediction_profile, pd.DataFrame({"prediction": ensemble_raw[selected.to_numpy()]}))["prediction"]
        for feature, value in values.items():
            psi_records.append({"month": name, "split": split, "rows": n, "feature": feature, "psi": value,
                                "status": drift.status(value)})
        y = rows.loc[selected, "y"].to_numpy()
        calibration_records.append({
            "month": name, "split": split, "rows": n, "observed_rate": float(y.mean()),
            "mean_calibrated_probability": float(calibrated[selected.to_numpy()].mean()),
            "mean_raw_probability": float(ensemble_raw[selected.to_numpy()].mean())})
    median_rows = int(np.median([r["rows"] for r in calibration_records]))
    floor = drift.noise_floor(st.feats[masks["train"]], columns, size=median_rows)
    return (pd.DataFrame(psi_records), pd.DataFrame(calibration_records), floor, profile, prediction_profile,
            skipped, median_rows)


def importance_table(boosters, names: list[str]) -> pd.DataFrame:
    """Share of total split gain per feature, per seed; sorted by the median over seeds."""
    shares = []
    for booster in boosters:
        gain = np.asarray(booster.feature_importance(importance_type="gain"), float)
        shares.append(gain / gain.sum())
    arr = np.vstack(shares)
    table = pd.DataFrame({"feature": names, "gain_share_median": np.median(arr, axis=0),
                          "gain_share_min": arr.min(axis=0), "gain_share_max": arr.max(axis=0)})
    return table.sort_values("gain_share_median", ascending=False).reset_index(drop=True)
