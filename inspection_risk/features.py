"""Leakage-safe features: for an inspection on date d, only records dated before d are used.

Rules every feature obeys
  1. Only records dated strictly before d (the day itself is excluded, whatever its type).
  2. Only records dated on or after `floor`, the same date for every restaurant. The publisher keeps a
     restaurant's records only up to three years before its latest inspection, so how far back a
     restaurant's visible history reaches depends on how recently it was last inspected, which is
     information from after d. A common floor removes that dependence.
  3. History is expressed as rates and "nearest previous" values, not as counts that grow with the
     length of the visible window.
  4. Neighbourhood and cuisine rates use only earlier inspections of other (and the same) restaurants.
tests/test_features_leakage.py checks 1-4 by truncating the data and comparing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

CATEGORICAL = ["boro", "cuisine_group"]
BORO_LEVELS = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island", "Unknown"]


def numeric_columns(families: list[str]) -> list[str]:
    base = [
        "prev_ci_n_crit", "prev_ci_n_viol", "prev_ci_score", "prev_ci_closed", "days_since_prev_ci",
        "last_n_crit", "last_score", "days_since_last_inspection", "n_inspection_days_365", "ever_closed_prior",
        "zip_prior_rate", "cuisine_prior_rate", "month", "weekday",
    ]
    prev = [f"prev_ci_fam_{f}" for f in families]
    share = [f"fam_{f}_prior_share" for f in families]
    return base + prev + share


def history_columns(families: list[str]) -> list[str]:
    """The columns that need the restaurant's own earlier records (the ablation drops these)."""
    return [c for c in numeric_columns(families) if c.startswith(("prev_ci", "last_", "days_since", "n_inspection",
                                                                     "ever_closed", "fam_"))]


def cuisine_levels_from(targets: pd.DataFrame) -> list[str]:
    """The most common cuisines among the given (training) rows; everything else becomes 'Other'."""
    counts = targets["cuisine"].value_counts()
    return list(counts.index[: config.CUISINE_TOP])


def _asof(left: pd.DataFrame, right: pd.DataFrame, by: str | None, exact: bool = False) -> pd.DataFrame:
    """For each left row, the last right row of the same `by` group dated before (or, if `exact`, on or
    before) the left row's date. Both sides are sorted by date, as merge_asof requires."""
    return pd.merge_asof(left.sort_values("date", kind="stable"), right.sort_values("date", kind="stable"),
                         by=by, on="date", direction="backward", allow_exact_matches=exact)


def build_features(events: pd.DataFrame, targets: pd.DataFrame, *, floor, families: list[str],
                   cuisine_levels: list[str], min_critical: int = config.TARGET_MIN_CRITICAL) -> pd.DataFrame:
    """Features for each row of `targets` (columns camis, date, boro, zipcode, cuisine).

    Nothing from `targets` other than the identifiers and static attributes is read: a target's own
    outcome columns are never consulted."""
    floor = pd.Timestamp(floor)
    ev = events[events["date"] >= floor]
    fam_cols = [f"fam_{f}" for f in families]

    zip_key = targets["zipcode"].fillna("UNKNOWN").to_numpy()
    cuisine_key = targets["cuisine"].fillna("UNKNOWN").to_numpy()
    left = pd.DataFrame({"_row": np.arange(len(targets)), "camis": targets["camis"].to_numpy(),
                         "date": targets["date"].to_numpy(), "zip": zip_key, "cuisine": cuisine_key})

    # ---- the restaurant's own earlier inspection days (any type) -----------------------------------
    agg = {"n_crit": ("n_crit", "sum"), "score": ("score", "max"), "closed": ("closed", "max")}
    agg.update({c: (c, "max") for c in fam_cols})
    day = ev.groupby(["camis", "date"], sort=True).agg(**agg).reset_index()
    by_camis = day.groupby("camis")
    day["cum_days"] = by_camis.cumcount() + 1
    day["cum_closed"] = by_camis["closed"].cumsum()
    for c in fam_cols:
        day[f"cum_{c}"] = by_camis[c].cumsum()
    day["last_date"] = day["date"]
    right = day[["camis", "date", "last_date", "n_crit", "score", "cum_days", "cum_closed"]
                + [f"cum_{c}" for c in fam_cols]].rename(columns={"n_crit": "last_n_crit", "score": "last_score"})
    prior = _asof(left[["_row", "camis", "date"]], right, "camis")

    year_ago = left[["_row", "camis", "date"]].copy()
    year_ago["date"] = year_ago["date"] - pd.Timedelta(days=365)
    older = _asof(year_ago, day[["camis", "date", "cum_days"]], "camis", exact=True)

    # ---- the restaurant's previous initial cycle inspection --------------------------------------
    ci_events = ev[ev["inspection_type"] == config.CYCLE_INITIAL]
    ci = ci_events[["camis", "date", "n_crit", "n_viol", "score", "closed"] + fam_cols].copy()
    ci = ci.rename(columns={"n_crit": "prev_ci_n_crit", "n_viol": "prev_ci_n_viol", "score": "prev_ci_score",
                            "closed": "prev_ci_closed", **{c: f"prev_ci_{c}" for c in fam_cols}})
    ci["prev_ci_date"] = ci["date"]
    prev_ci = _asof(left[["_row", "camis", "date"]], ci, "camis")

    # ---- neighbourhood and cuisine rates from earlier initial cycle inspections ---------------------
    labelled = pd.DataFrame({
        "date": ci_events["date"].to_numpy(), "zip": ci_events["zipcode"].fillna("UNKNOWN").to_numpy(),
        "cuisine": ci_events["cuisine"].fillna("UNKNOWN").to_numpy(),
        "label": (ci_events["n_crit"] >= min_critical).astype(int).to_numpy()})

    def cumulative(key: str | None) -> pd.DataFrame:
        keys = ["date"] if key is None else [key, "date"]
        daily = labelled.groupby(keys, sort=True).agg(n=("label", "size"), pos=("label", "sum")).reset_index()
        grouped = daily if key is None else daily.groupby(key)
        daily["cum_n"] = grouped["n"].cumsum()
        daily["cum_pos"] = grouped["pos"].cumsum()
        return daily[keys + ["cum_n", "cum_pos"]]

    overall = _asof(left[["_row", "date"]], cumulative(None), None)
    global_rate = overall["cum_pos"] / overall["cum_n"]
    rates = {}
    for key in ("zip", "cuisine"):
        got = _asof(left[["_row", "date", key]], cumulative(key), key)
        n = got["cum_n"].fillna(0)
        pos = got["cum_pos"].fillna(0)
        rates[key] = ((pos + config.SMOOTHING * global_rate.to_numpy()) / (n + config.SMOOTHING)).to_numpy()
        rates[key + "_row"] = got["_row"].to_numpy()

    # ---- assemble in the target order ----------------------------------------------------------------
    out = pd.DataFrame({"_row": np.arange(len(targets))}).set_index("_row")

    def put(name, frame, column, transform=None):
        series = frame.set_index("_row")[column]
        out[name] = series.reindex(out.index) if transform is None else transform(series.reindex(out.index))

    put("prev_ci_n_crit", prev_ci, "prev_ci_n_crit")
    put("prev_ci_n_viol", prev_ci, "prev_ci_n_viol")
    put("prev_ci_score", prev_ci, "prev_ci_score")
    put("prev_ci_closed", prev_ci, "prev_ci_closed", lambda s: s.astype("float"))
    days = (left.set_index("_row")["date"] - prev_ci.set_index("_row")["prev_ci_date"].reindex(out.index)).dt.days
    out["days_since_prev_ci"] = days.reindex(out.index).astype("float")
    for f in families:
        put(f"prev_ci_fam_{f}", prev_ci, f"prev_ci_fam_{f}", lambda s: s.astype("float"))

    put("last_n_crit", prior, "last_n_crit")
    put("last_score", prior, "last_score")
    since = (left.set_index("_row")["date"] - prior.set_index("_row")["last_date"].reindex(out.index)).dt.days
    out["days_since_last_inspection"] = since.reindex(out.index).astype("float")
    cum_days = prior.set_index("_row")["cum_days"].reindex(out.index)
    cum_old = older.set_index("_row")["cum_days"].reindex(out.index).fillna(0)
    out["n_inspection_days_365"] = (cum_days.fillna(0) - cum_old).astype("float")
    cum_closed = prior.set_index("_row")["cum_closed"].reindex(out.index)
    out["ever_closed_prior"] = np.where(cum_days.isna(), np.nan, (cum_closed.fillna(0) > 0).astype(float))
    for f in families:
        cum_f = prior.set_index("_row")[f"cum_fam_{f}"].reindex(out.index)
        out[f"fam_{f}_prior_share"] = (cum_f / cum_days).astype("float")

    out["zip_prior_rate"] = pd.Series(rates["zip"], index=rates["zip_row"]).reindex(out.index)
    out["cuisine_prior_rate"] = pd.Series(rates["cuisine"], index=rates["cuisine_row"]).reindex(out.index)
    dates = pd.DatetimeIndex(targets["date"].to_numpy())
    out["month"] = dates.month.astype("float")
    out["weekday"] = dates.day_of_week.astype("float")
    out["boro"] = targets["boro"].fillna("Unknown").to_numpy()
    cuisine = targets["cuisine"].fillna("Unknown")
    out["cuisine_group"] = np.where(cuisine.isin(cuisine_levels), cuisine, np.where(cuisine == "Unknown", "Unknown", "Other"))

    out.index = targets.index
    return out[numeric_columns(families) + CATEGORICAL]
