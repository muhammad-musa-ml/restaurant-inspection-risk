"""Run every measurement and write results/ and models/.

    python scripts/run_pipeline.py

Reads data/inspections.csv.gz only. Takes a few minutes: it fits the boosted model for 5 seeds on the main
setup, on the secondary target, on the ablations and on the random splits.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inspection_risk import config, drift, features, metrics, modeling, pipeline, runinfo  # noqa: E402
from inspection_risk import study as study_module  # noqa: E402

T0 = time.time()


def log(message: str) -> None:
    print(f"[{time.time() - T0:6.0f}s] {message}", flush=True)


def sha256_of(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/inspections.csv.gz")
    parser.add_argument("--results", default="results")
    parser.add_argument("--models", default="models")
    args = parser.parse_args()
    results, models_dir = Path(args.results), Path(args.models)
    results.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    command = "python scripts/run_pipeline.py"
    seeds, capacity = config.SEEDS, config.CAPACITY_PER_WEEK

    st = study_module.prepare(args.data)
    rows, masks = st.rows, pipeline.masks_of(st.rows)
    y, n, weeks = rows["y"].to_numpy(), len(rows), rows["week"].to_numpy()
    vm, tm, pm = masks["valid"], masks["test"], masks["post"]
    ties = pipeline.tiebreaks(n)
    frame = modeling.model_frame(st.feats, st.numeric, st.cuisine_levels)
    log(f"table ready: {n} inspections, {frame.shape[1]} features")

    # ------------------------------------------------------------------ models
    best, grid, boosters = pipeline.fit_gbm(frame, y, masks)
    raw = pipeline.predict_matrix(boosters, frame)
    ensemble_raw = raw.mean(axis=1)
    log(f"gbm fitted: best params {best['params']}, iterations {[b.best_iteration for b in boosters]}")
    logistic = modeling.fit_logistic(frame[masks["train"]], y[masks["train"]])
    p_logistic = modeling.predict_logistic(logistic, frame)
    rule = pipeline.previous_result_score(st.feats, config.TARGET_MIN_CRITICAL)
    zeros = np.zeros(n)

    def models_at(mask):
        return {
            "base_rate": pipeline.stat_of_runs([pipeline.evaluate(rows, mask, zeros, ties[s], capacity) for s in seeds]),
            "previous_result_rule": pipeline.stat_of_runs(
                [pipeline.evaluate(rows, mask, rule, ties[s], capacity) for s in seeds]),
            "logistic": pipeline.stat_of_runs(
                [pipeline.evaluate(rows, mask, p_logistic, ties[s], capacity) for s in seeds]),
            "gbm": pipeline.stat_of_runs(
                [pipeline.evaluate(rows, mask, raw[:, k], ties[s], capacity) for k, s in enumerate(seeds)]),
            "gbm_5_seed_average": pipeline.evaluate(rows, mask, ensemble_raw, ties[0], capacity),
        }

    ranking = {"test": models_at(tm), "post": models_at(pm)}
    log("headline ranking done")

    # ------------------------------------------------------------ calibration
    chosen, second_half_brier = pipeline.choose_calibrator(ensemble_raw[vm], y[vm], weeks[vm])
    classes = pipeline.CALIBRATORS
    ensemble_calibrators = {name: cls().fit(ensemble_raw[vm], y[vm]) for name, cls in classes.items()}
    per_seed_calibrated = {name: np.column_stack([cls().fit(raw[vm, k], y[vm]).predict(raw[:, k])
                                                  for k in range(len(seeds))]) for name, cls in classes.items()}
    ensemble_cal = {name: cal.predict(ensemble_raw) for name, cal in ensemble_calibrators.items()}
    calibration = {"chosen_method": chosen,
                   "choice_rule": "fit on the first half of the validation weeks, lower Brier score on the second half",
                   "second_half_of_validation_brier": second_half_brier, "windows": {}}
    for window, mask in (("test", tm), ("post", pm)):
        yw = y[mask]
        block = {"base_rate": float(yw.mean()),
                 "gbm_5_seed_average": {"before": pipeline.calibration_numbers(yw, ensemble_raw[mask])},
                 "per_seed": {"before": pipeline.stat_of_runs(
                     [pipeline.calibration_numbers(yw, raw[mask, k]) for k in range(len(seeds))])}}
        for name in classes:
            block["gbm_5_seed_average"][name] = pipeline.calibration_numbers(yw, ensemble_cal[name][mask])
            block["per_seed"][name] = pipeline.stat_of_runs(
                [pipeline.calibration_numbers(yw, per_seed_calibrated[name][mask, k]) for k in range(len(seeds))])
        calibration["windows"][window] = block
    calibration["reliability_test_5_seed_average"] = {
        "before": metrics.reliability_table(y[tm], ensemble_raw[tm]).to_dict("records"),
        "after": metrics.reliability_table(y[tm], ensemble_cal[chosen][tm]).to_dict("records")}
    log(f"calibration done (chosen: {chosen})")

    # ---------------------------------------------- operating point from capacity
    def operating(p_all):
        return pipeline.operating_point(p_all[vm], weeks[vm], p_all[tm], weeks[tm], y[tm], capacity)

    seed_points = [operating(per_seed_calibrated[chosen][:, k]) for k in range(len(seeds))]
    ensemble_point = operating(ensemble_cal[chosen])
    threshold = {
        "capacity_per_week": capacity, "calibration": chosen,
        "rule": "the calibrated-probability cut-off that flags the stated capacity per week on average over the "
                "validation weeks, applied unchanged to the test weeks",
        "gbm_5_seed_average": ensemble_point,
        "per_seed": {key: pipeline.stat(p[key] for p in seed_points)
                     for key in ("threshold", "precision", "recall", "lift", "visits_per_week_median",
                                 "visits_per_week_min", "visits_per_week_max")}}
    log("operating point done")

    # ----------------------------------------------------- curves and weekly series
    curve_rows = []
    oracle = y.astype(float)
    test_curves = {"base_rate": [(s, zeros, ties[s]) for s in seeds],
                   "previous_result_rule": [(s, rule, ties[s]) for s in seeds],
                   "logistic": [(s, p_logistic, ties[s]) for s in seeds],
                   "gbm": [(s, raw[:, k], ties[s]) for k, s in enumerate(seeds)],
                   "gbm_5_seed_average": [(0, ensemble_raw, ties[0])],
                   "perfect_ranking": [(0, oracle, ties[0])]}
    for name, variants in test_curves.items():
        for seed, score, tie in variants:
            curve = metrics.capacity_curve(rows.loc[tm, "week"], score[tm], y[tm], tie[tm])
            curve.insert(0, "seed", seed)
            curve.insert(0, "model", name)
            curve_rows.append(curve)
    pd.concat(curve_rows).to_csv(results / "capacity_curve_test.csv", index=False, lineterminator="\n")

    ensemble_frame = metrics.ranked(weeks[tm], ensemble_raw[tm], y[tm], ties[0][tm])
    rule_frame = metrics.ranked(weeks[tm], rule[tm], y[tm], ties[0][tm])
    logistic_frame = metrics.ranked(weeks[tm], p_logistic[tm], y[tm], ties[0][tm])
    ensemble_counts = metrics.per_week_counts(ensemble_frame, capacity)
    rule_counts = metrics.per_week_counts(rule_frame, capacity)
    logistic_counts = metrics.per_week_counts(logistic_frame, capacity)
    weekly = pd.DataFrame({
        "week": [d.strftime("%Y-%m-%d") for d in ensemble_counts.index], "pool": ensemble_counts["pool"].to_numpy(),
        "positives": ensemble_counts["positives"].to_numpy(),
        "visits": ensemble_counts["visits"].to_numpy(), "hits_gbm": ensemble_counts["hits"].to_numpy(),
        "hits_previous_result_rule": rule_counts["hits"].to_numpy(),
        "hits_logistic": logistic_counts["hits"].to_numpy()})
    weekly.to_csv(results / "weekly_test.csv", index=False, lineterminator="\n")
    bootstrap = {
        "what": "resampling the test weeks with replacement (2000 draws, seed 0); the 5 seeds do not capture which "
                "weeks happened to be in the test window",
        "capacity_per_week": capacity,
        "gbm_5_seed_average": metrics.bootstrap_weeks(ensemble_counts),
        "gbm_minus_previous_result_rule": metrics.bootstrap_paired(ensemble_counts, rule_counts),
        "gbm_minus_logistic": metrics.bootstrap_paired(ensemble_counts, logistic_counts)}
    log("curves and bootstrap done")

    # ------------------------------------------------------ random split vs time
    mean_pool_test = float(tm.sum() / rows.loc[tm, "week"].nunique())
    fraction = capacity / mean_pool_test
    comparison = pipeline.split_comparison(st, frame, best["params"], raw, fraction)
    log("random vs time split done")

    # --------------------------------------------------------------- variants
    history = features.history_columns(st.families)
    no_history = [c for c in st.numeric if c not in history]
    ablations = {
        "full_model": ranking["test"]["gbm"],
        "no_history_features": pipeline.variant_runs(st, no_history, best["params"], capacity),
        "history_not_floored": pipeline.variant_runs(
            study_module.variant(st, history_floor=pd.Timestamp("2007-01-01")), st.numeric, best["params"], capacity),
        "dropped_columns_in_no_history": [c for c in st.numeric if c in history]}
    embargo = {"0": ranking["test"]["gbm"]}
    for days in (30, 60, 90):
        embargo[str(days)] = pipeline.variant_runs(study_module.variant(st, embargo_days=days), st.numeric,
                                                   best["params"], capacity)
    log("ablations and embargo done")

    # ---------------------------------------------------------- secondary target
    st1 = study_module.variant(st, min_critical=config.SECONDARY_MIN_CRITICAL)
    y1 = st1.rows["y"].to_numpy()
    frame1 = modeling.model_frame(st1.feats, st1.numeric, st1.cuisine_levels)
    best1, _, boosters1 = pipeline.fit_gbm(frame1, y1, masks)
    raw1 = pipeline.predict_matrix(boosters1, frame1)
    rule1 = pipeline.previous_result_score(st1.feats, config.SECONDARY_MIN_CRITICAL)
    secondary = {
        "target": f"at least {config.SECONDARY_MIN_CRITICAL} critical violation (the first-posed question)",
        "base_rate_test": float(y1[tm].mean()), "lift_ceiling_at_capacity": float(1 / y1[tm].mean()),
        "best_params": best1["params"],
        "gbm": pipeline.stat_of_runs([pipeline.evaluate(st1.rows, tm, raw1[:, k], ties[s], capacity)
                                      for k, s in enumerate(seeds)]),
        "previous_result_rule": pipeline.stat_of_runs([pipeline.evaluate(st1.rows, tm, rule1, ties[s], capacity)
                                                       for s in seeds]),
        "base_rate": pipeline.stat_of_runs([pipeline.evaluate(st1.rows, tm, zeros, ties[s], capacity)
                                            for s in seeds])}
    log("secondary target done")

    # ------------------------------------------------------------------- drift
    psi_table, month_table, floor, profile, prediction_profile, skipped, median_rows = pipeline.monthly_drift(
        st, ensemble_raw, ensemble_cal[chosen])
    psi_table.to_csv(results / "drift_monthly.csv", index=False, lineterminator="\n")
    month_table.to_csv(results / "monthly_calibration.csv", index=False, lineterminator="\n")
    columns = [c for c in floor.columns if c != "seed"]
    drift_summary = {
        "reference": {"features": "training window", "prediction": "validation window (the 5-seed average, raw)"},
        "months_scored": month_table["month"].tolist(), "months_skipped_too_few_rows": skipped,
        "noise_floor": {"sample_size": median_rows, "seeds": list(seeds),
                        "median_over_features_and_seeds": float(np.median(floor[columns].to_numpy())),
                        "max_over_features_and_seeds": float(floor[columns].to_numpy().max())},
        "cutoffs": {"moderate": config.PSI_MODERATE, "major": config.PSI_MAJOR},
        "feature_months_by_status": psi_table.groupby("status").size().to_dict(),
        "features_with_a_major_month": sorted(psi_table.loc[psi_table["status"] == "major", "feature"].unique()),
        "max_psi_by_feature": {f: float(v) for f, v in psi_table.groupby("feature")["psi"].max().sort_values(
            ascending=False).items()}}
    log("drift done")

    # --------------------------------------------------- importance, predictions
    importance = pipeline.importance_table(boosters, list(frame.columns))
    importance.to_csv(results / "feature_importance.csv", index=False, lineterminator="\n")
    shown = rows["split"].isin(["valid", "test", "post"]).to_numpy()
    out = pd.DataFrame({
        "camis": rows["camis"], "date": rows["date"].dt.strftime("%Y-%m-%d"),
        "week": rows["week"].dt.strftime("%Y-%m-%d"), "split": rows["split"], "n_critical": rows["n_crit"],
        "y": y, "previous_result_rule": rule, "logistic": p_logistic.round(6),
        "gbm_raw": ensemble_raw.round(6), "gbm_calibrated": ensemble_cal[chosen].round(6)})
    for k, s in enumerate(seeds):
        out[f"gbm_raw_seed{s}"] = raw[:, k].round(6)
    runinfo.write_gzip_text(results / "predictions.csv.gz", out[shown].to_csv(index=False, lineterminator="\n"))

    for k, s in enumerate(seeds):
        runinfo.write_gzip_text(models_dir / f"gbm_seed{s}.txt.gz",
                                boosters[k].model_to_string(num_iteration=boosters[k].best_iteration))
    card = {
        "created_utc": runinfo.run_meta(command)["timestamp_utc"],
        "code_git_head": runinfo.git_head(ROOT),
        "data_sha256_of_gzip": sha256_of(Path(args.data)),
        "target": {"min_critical_violations": config.TARGET_MIN_CRITICAL},
        "history_floor": str(st.floor.date()), "history_embargo_days": st.embargo_days,
        "families": st.families, "cuisine_levels": st.cuisine_levels, "boro_levels": features.BORO_LEVELS,
        "numeric_columns": st.numeric, "categorical_columns": features.CATEGORICAL,
        "lightgbm": {"params": best["params"], "seeds": list(seeds),
                     "best_iterations": [int(b.best_iteration) for b in boosters],
                     "files": [f"gbm_seed{s}.txt.gz" for s in seeds]},
        "calibration": {"method": chosen, "fitted_on": f"validation window {config.VALID_START} to {config.TEST_START}",
                        "calibrator": ensemble_calibrators[chosen].to_dict()},
        "capacity": {"per_week": capacity, "probability_threshold": ensemble_point["threshold"],
                     "share_of_validation_inspections_flagged": ensemble_point["validation_flag_fraction"]},
        "drift_reference": {"features": profile, "prediction": prediction_profile["prediction"],
                            "noise_floor_median": drift_summary["noise_floor"]["median_over_features_and_seeds"]}}
    runinfo.write_json(models_dir / "model_card.json", card)

    # ---------------------------------------------------------------- summary
    summary = {
        "meta": runinfo.run_meta(command, ROOT),
        "setup": {
            "target": f"at least {config.TARGET_MIN_CRITICAL} critical violations at an initial cycle inspection",
            "capacity_per_week": capacity, "seeds": list(seeds), "history_floor": str(st.floor.date()),
            "mean_pool_per_week_test": mean_pool_test, "capacity_share_of_test_pool": fraction,
            "windows": {name: {"rows": int(masks[name].sum()), "weeks": int(rows.loc[masks[name], "week"].nunique()),
                               "first_date": str(rows.loc[masks[name], "date"].min().date()),
                               "last_date": str(rows.loc[masks[name], "date"].max().date()),
                               "positive_share": float(y[masks[name]].mean())}
                        for name in pipeline.SPLITS}},
        "model_selection": {"grid_on_validation_log_loss": grid, "chosen": best,
                            "best_iterations_by_seed": [int(b.best_iteration) for b in boosters]},
        "ranking_at_capacity": ranking, "calibration": calibration, "operating_threshold": threshold,
        "bootstrap_over_test_weeks": bootstrap, "random_vs_time_split": comparison,
        "ablations": ablations, "history_embargo_days": embargo, "secondary_target": secondary,
        "drift": drift_summary}
    runinfo.write_json(results / "summary.json", summary)
    log("wrote results/summary.json, models/model_card.json and the tables")
    return 0


if __name__ == "__main__":
    sys.exit(main())
