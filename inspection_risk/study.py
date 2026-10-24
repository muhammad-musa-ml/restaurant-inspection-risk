"""Assemble the modelling table once: events, floor, rows to rank with their split, features, labels."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import config, data, features, metrics


@dataclass
class Study:
    raw: pd.DataFrame
    events: pd.DataFrame
    floor: pd.Timestamp
    families: list[str]
    cuisine_levels: list[str]
    numeric: list[str]
    rows: pd.DataFrame       # one row per ranked inspection: camis, date, week, split, y (target), y1, n_crit ...
    feats: pd.DataFrame      # features aligned to `rows`
    min_critical: int = config.TARGET_MIN_CRITICAL
    embargo_days: int = 0


def prepare(path: str | Path, min_critical: int = config.TARGET_MIN_CRITICAL, embargo_days: int = 0,
            history_floor: pd.Timestamp | None = None) -> Study:
    """Load the snapshot and build the table. `history_floor` replaces the derived floor in the features
    only (used to show what ignoring the publisher's three-year window would do); the checks on the study
    dates still use the derived one."""
    raw = data.load_raw(path)
    events = data.build_events(raw)
    return _build(raw, events, min_critical, embargo_days, history_floor)


def variant(base: Study, *, min_critical: int | None = None, embargo_days: int | None = None,
            history_floor: pd.Timestamp | None = None) -> Study:
    """The same snapshot with the target, embargo or floor changed; nothing is read from disk again."""
    return _build(base.raw, base.events,
                  base.min_critical if min_critical is None else min_critical,
                  base.embargo_days if embargo_days is None else embargo_days, history_floor)


def _build(raw, events, min_critical, embargo_days, history_floor) -> Study:
    floor = data.visibility_floor(events)
    first_ok = floor + pd.Timedelta(days=config.MIN_WINDOW_DAYS)
    if pd.Timestamp(config.TRAIN_START) < first_ok:
        raise ValueError(f"TRAIN_START {config.TRAIN_START} is earlier than the first date with "
                         f"{config.MIN_WINDOW_DAYS} days of visible history ({first_ok.date()})")
    ranked = data.cycle_initial(events)
    ranked["split"] = data.assign_split(ranked["date"])
    rows = ranked[ranked["split"].notna()].reset_index(drop=True)
    rows["week"] = metrics.week_start(rows["date"])
    rows["y"] = (rows["n_crit"] >= min_critical).astype(int)
    rows["y1"] = (rows["n_crit"] >= config.SECONDARY_MIN_CRITICAL).astype(int)
    families = data.select_families(raw, config.VALID_START)
    train = rows[rows["split"] == "train"]
    cuisine_levels = features.cuisine_levels_from(train)
    feats = features.build_features(events, rows, floor=floor if history_floor is None else history_floor,
                                    families=families, cuisine_levels=cuisine_levels,
                                    min_critical=min_critical, embargo_days=embargo_days)
    return Study(raw=raw, events=events, floor=floor, families=families, cuisine_levels=cuisine_levels,
                 numeric=features.numeric_columns(families), rows=rows, feats=feats,
                 min_critical=min_critical, embargo_days=embargo_days)

