"""Load the frozen snapshot and collapse citation rows into one row per inspection."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

PLACEHOLDER_DATE = "1900-01-01"  # the publisher's marker for "applied for a permit, never inspected"
EVENT_KEY = ["camis", "date", "inspection_type"]


def load_raw(path: str | Path) -> pd.DataFrame:
    """One row per violation citation (or per inspection that cited nothing), typed.

    IDs stay strings (leading zeros are content). Dates are calendar dates: no time zone is attached.
    The 1900-01-01 placeholder rows are dropped; they are restaurants that were never inspected."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df = df[df["inspection_date"] != PLACEHOLDER_DATE].copy()
    df["date"] = pd.to_datetime(df["inspection_date"], format="%Y-%m-%d").astype("datetime64[ns]")
    df["is_critical"] = df["critical_flag"].eq("Critical")
    df["has_violation"] = df["violation_code"].ne("")
    df["closed"] = df["action"].str.contains(r"closed by dohmh|re-closed", case=False, regex=True)
    df["score_num"] = pd.to_numeric(df["score"], errors="coerce")
    df["family"] = df["violation_code"].str[:2]
    for column in ("boro", "zipcode", "cuisine_description"):
        df[column] = df[column].replace({"": np.nan, "0": np.nan})
    return df


def visibility_floor(raw_or_events: pd.DataFrame) -> pd.Timestamp:
    """First date from which every active restaurant's records are complete in this snapshot."""
    return raw_or_events["date"].max() - pd.Timedelta(days=config.VISIBILITY_DAYS)


def max_span_days(raw: pd.DataFrame) -> int:
    """Longest distance between a restaurant's earliest and latest visible inspection date."""
    span = raw.groupby("camis")["date"].agg(["min", "max"])
    return int((span["max"] - span["min"]).dt.days.max())


def build_events(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per (restaurant, date, inspection type) with the counts the features need.

    n_crit counts critical citations; fam_XX is 1 when any citation's code starts with XX."""
    grouped = raw.groupby(EVENT_KEY, sort=True)
    events = grouped.agg(
        n_crit=("is_critical", "sum"),
        n_viol=("has_violation", "sum"),
        score=("score_num", "max"),
        closed=("closed", "max"),
        boro=("boro", "first"),
        zipcode=("zipcode", "first"),
        cuisine=("cuisine_description", "first"),
    ).reset_index()
    citations = raw.loc[raw["has_violation"], EVENT_KEY + ["family"]].drop_duplicates()
    flags = pd.crosstab([citations["camis"], citations["date"], citations["inspection_type"]],
                        citations["family"]).clip(upper=1)
    flags.columns = [f"fam_{name}" for name in flags.columns]
    events = events.merge(flags.reset_index(), on=EVENT_KEY, how="left")
    for column in flags.columns:
        events[column] = events[column].fillna(0).astype("int8")
    events["n_crit"] = events["n_crit"].astype("int32")
    events["n_viol"] = events["n_viol"].astype("int32")
    events["closed"] = events["closed"].astype(bool)
    return events


def select_families(raw: pd.DataFrame, before: dt.date) -> list[str]:
    """Violation-code families (first two characters) cited often enough to be features, counted over
    citations at initial cycle inspections dated before `before` (the end of the training window).
    Frequency only, never labels; families that only occur in other programs (smoke-free air, calorie
    posting) would be constant columns, so they are not counted."""
    window = raw[raw["has_violation"] & (raw["date"] < pd.Timestamp(before))
                 & (raw["inspection_type"] == config.CYCLE_INITIAL)]
    counts = window["family"].value_counts()
    return sorted(counts[counts >= config.FAMILY_MIN_CITATIONS].index)


def cycle_initial(events: pd.DataFrame) -> pd.DataFrame:
    """The inspections that are ranked: initial cycle inspections. A restaurant has at most one per day."""
    rows = events[events["inspection_type"] == config.CYCLE_INITIAL].copy()
    if rows.duplicated(["camis", "date"]).any():
        raise ValueError("a restaurant has two initial cycle inspections on one date")
    return rows.reset_index(drop=True)


def assign_split(dates: pd.Series) -> pd.Series:
    """'train' | 'valid' | 'test' | 'post' by date; NaN outside [TRAIN_START, POST_END)."""
    edges = [config.TRAIN_START, config.VALID_START, config.TEST_START, config.POST_START, config.POST_END]
    bins = [pd.Timestamp(d) for d in edges]
    labels = ["train", "valid", "test", "post"]
    cut = pd.cut(dates, bins=bins, labels=labels, right=False)
    return cut.astype("object")
