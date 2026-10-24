"""Metrics: calibration (ECE, Brier), ranking (AUC) and what a fixed weekly capacity buys."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from . import config


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def reliability_table(y: np.ndarray, p: np.ndarray, bins: int = config.ECE_BINS) -> pd.DataFrame:
    """Equal-width bins on [0, 1]: count, mean predicted probability and observed rate per bin."""
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    index = np.minimum((p * bins).astype(int), bins - 1)
    rows = []
    for b in range(bins):
        mask = index == b
        n = int(mask.sum())
        rows.append({"bin": b, "lo": b / bins, "hi": (b + 1) / bins, "n": n,
                     "mean_predicted": float(p[mask].mean()) if n else None,
                     "observed_rate": float(y[mask].mean()) if n else None})
    return pd.DataFrame(rows)


def ece(y: np.ndarray, p: np.ndarray, bins: int = config.ECE_BINS) -> float:
    """Expected calibration error: bin-size-weighted mean of |observed rate - mean predicted| over
    `bins` equal-width probability bins."""
    table = reliability_table(y, p, bins)
    filled = table[table["n"] > 0]
    weights = filled["n"] / filled["n"].sum()
    return float((weights * (filled["observed_rate"] - filled["mean_predicted"]).abs()).sum())


def ranking_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y = np.asarray(y, int)
    return {"auc": float(roc_auc_score(y, p)), "average_precision": float(average_precision_score(y, p)),
            "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))), "base_rate": float(y.mean())}


def week_start(dates: pd.Series) -> pd.Series:
    """The Monday of each date's week."""
    dates = pd.to_datetime(dates)
    return (dates - pd.to_timedelta(dates.dt.day_of_week, unit="D")).astype("datetime64[ns]")


def ranked(week: pd.Series, score, y, tiebreak) -> pd.DataFrame:
    """Rows sorted within each week by descending score, ties broken by `tiebreak` (a seeded random
    number, so a tie never favours a row by its position in the file). Adds rank and the running count
    of positives, so every capacity can be read from one sort."""
    frame = pd.DataFrame({"week": np.asarray(week), "score": np.asarray(score, float), "y": np.asarray(y, int),
                          "tb": np.asarray(tiebreak, float)})
    frame = frame.sort_values(["week", "score", "tb"], ascending=[True, False, True], kind="stable")
    grouped = frame.groupby("week", sort=True)
    frame["rank"] = grouped.cumcount() + 1
    frame["cum_y"] = grouped["y"].cumsum()
    return frame


def per_week_counts(frame: pd.DataFrame, capacity: int) -> pd.DataFrame:
    """For every week: pool size, positives in the pool, visits made (min(capacity, pool)) and hits."""
    weeks = frame.groupby("week", sort=True)
    pool = weeks.size().rename("pool")
    positives = weeks["y"].sum().rename("positives")
    top = frame[frame["rank"] <= capacity].groupby("week", sort=True)
    visits = top.size().reindex(pool.index).fillna(0).rename("visits")
    hits = top["y"].sum().reindex(pool.index).fillna(0).rename("hits")
    return pd.concat([pool, positives, visits, hits], axis=1).astype(int)


def summarise_counts(counts: pd.DataFrame) -> dict:
    """Pooled over weeks: precision = hits/visits, recall = hits/positives, lift = precision/base rate."""
    visits = int(counts["visits"].sum())
    hits = int(counts["hits"].sum())
    positives = int(counts["positives"].sum())
    pool = int(counts["pool"].sum())
    base = positives / pool
    precision = hits / visits if visits else float("nan")
    return {"weeks": int(len(counts)), "pool": pool, "positives": positives, "base_rate": base,
            "visits": visits, "hits": hits, "precision": precision, "recall": hits / positives,
            "lift": precision / base, "hits_per_week": hits / len(counts),
            "visits_per_week": visits / len(counts)}


def capacity_metrics(week, score, y, tiebreak, capacity: int) -> dict:
    frame = ranked(week, score, y, tiebreak)
    return summarise_counts(per_week_counts(frame, capacity))


def capacity_curve(week, score, y, tiebreak, capacities=config.CAPACITY_SWEEP) -> pd.DataFrame:
    frame = ranked(week, score, y, tiebreak)
    rows = []
    for capacity in capacities:
        rows.append({"capacity": capacity, **summarise_counts(per_week_counts(frame, capacity))})
    return pd.DataFrame(rows)


def threshold_for_capacity(valid_scores: np.ndarray, valid_weeks: pd.Series, capacity: int) -> tuple[float, float]:
    """The score cut-off that, on the validation weeks, flags `capacity` inspections per week on average.
    Returns (threshold, flagged fraction)."""
    weeks = pd.Series(np.asarray(valid_weeks)).nunique()
    fraction = min(1.0, capacity * weeks / len(valid_scores))
    threshold = float(np.quantile(valid_scores, 1 - fraction, method="higher"))
    return threshold, fraction


def flagged_metrics(week, y, flagged) -> dict:
    """Precision, recall and lift of a fixed flag, plus how many inspections it flags per week."""
    frame = pd.DataFrame({"week": np.asarray(week), "y": np.asarray(y, int), "flag": np.asarray(flagged, bool)})
    weekly = frame.groupby("week", sort=True).agg(pool=("y", "size"), positives=("y", "sum"),
                                                  visits=("flag", "sum"))
    weekly["hits"] = frame[frame["flag"]].groupby("week", sort=True)["y"].sum().reindex(weekly.index).fillna(0)
    out = summarise_counts(weekly.astype(int))
    out["visits_per_week_median"] = float(weekly["visits"].median())
    out["visits_per_week_min"] = int(weekly["visits"].min())
    out["visits_per_week_max"] = int(weekly["visits"].max())
    return out


def bootstrap_weeks(counts: pd.DataFrame, draws: int = 2000, seed: int = 0) -> dict:
    """Resample whole weeks with replacement and recompute precision, recall and lift: how much the
    result moves with which weeks happened to be in the test window (the seeds do not capture that)."""
    rng = np.random.default_rng(seed)
    arrays = {c: counts[c].to_numpy() for c in ("pool", "positives", "visits", "hits")}
    n = len(counts)
    precision, recall, lift = [], [], []
    for _ in range(draws):
        pick = rng.integers(0, n, n)
        hits = arrays["hits"][pick].sum()
        visits = arrays["visits"][pick].sum()
        positives = arrays["positives"][pick].sum()
        pool = arrays["pool"][pick].sum()
        precision.append(hits / visits)
        recall.append(hits / positives)
        lift.append((hits / visits) / (positives / pool))

    def interval(values):
        low, mid, high = np.percentile(values, [2.5, 50, 97.5])
        return {"p2.5": float(low), "median": float(mid), "p97.5": float(high)}

    return {"draws": draws, "seed": seed, "precision": interval(precision), "recall": interval(recall),
            "lift": interval(lift)}
