"""Batch scoring: rank a month of initial inspections with the saved models and report the month's drift."""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config, data, drift, features, metrics, modeling, runinfo


class ScoringError(ValueError):
    """The request cannot be scored honestly (no rows, dates the model was not built for, bad input)."""


def load_artifacts(models_dir: str | Path) -> dict:
    models_dir = Path(models_dir)
    card = json.loads((models_dir / "model_card.json").read_text(encoding="utf-8"))
    boosters = [lgb.Booster(model_str=runinfo.read_gzip_text(models_dir / name)) for name in card["lightgbm"]["files"]]
    return {"card": card, "boosters": boosters,
            "calibrator": modeling.calibrator_from_dict(card["calibration"]["calibrator"])}


def month_targets(events: pd.DataFrame, month: str) -> pd.DataFrame:
    """Backtest mode: the initial cycle inspections that happened in `month`. Only identifiers and static
    attributes are kept; the outcome columns are set aside by the caller."""
    ranked = data.cycle_initial(events)
    chosen = ranked[ranked["date"].dt.strftime("%Y-%m") == month]
    if chosen.empty:
        raise ScoringError(f"no initial cycle inspections in {month} (the snapshot ends "
                           f"{events['date'].max().date()})")
    return chosen.reset_index(drop=True)


def due_targets(events: pd.DataFrame, due: pd.DataFrame) -> pd.DataFrame:
    """Forward mode: restaurants with a planned inspection date. Static attributes come from the
    restaurant's most recent record; a restaurant the snapshot does not know gets missing attributes."""
    missing = {"camis", "planned_date"} - set(due.columns)
    if missing:
        raise ScoringError(f"the due list needs the columns camis and planned_date (missing {sorted(missing)})")
    due = due.copy()
    due["camis"] = due["camis"].astype(str)
    due["date"] = pd.to_datetime(due["planned_date"], format="%Y-%m-%d", errors="coerce").astype("datetime64[ns]")
    if due["date"].isna().any():
        raise ScoringError("planned_date must be YYYY-MM-DD")
    if due.duplicated(["camis", "date"]).any():
        raise ScoringError("the due list repeats a restaurant on the same date")
    latest = (events.sort_values("date").groupby("camis")[["boro", "zipcode", "cuisine"]].last().reset_index())
    merged = due[["camis", "date"]].merge(latest, on="camis", how="left")
    return merged


def score_targets(events: pd.DataFrame, targets: pd.DataFrame, artifacts: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Features and scores for `targets` (columns camis, date, boro, zipcode, cuisine).

    Returns (scored table, feature table). Nothing but identifiers and static attributes is read from
    `targets`, so a month's own outcomes cannot reach its scores."""
    card = artifacts["card"]
    floor = pd.Timestamp(card["history_floor"])
    earliest = floor + pd.Timedelta(days=config.MIN_WINDOW_DAYS)
    if targets["date"].min() < earliest:
        raise ScoringError(f"the model needs {config.MIN_WINDOW_DAYS} days of visible history; dates before "
                           f"{earliest.date()} cannot be scored")
    inputs = targets[["camis", "date", "boro", "zipcode", "cuisine"]].reset_index(drop=True)
    feats = features.build_features(
        events, inputs, floor=floor, families=card["families"], cuisine_levels=card["cuisine_levels"],
        min_critical=card["target"]["min_critical_violations"], embargo_days=card["history_embargo_days"])
    frame = modeling.model_frame(feats, card["numeric_columns"], card["cuisine_levels"])
    raw = np.column_stack([modeling.predict_lgbm(b, frame) for b in artifacts["boosters"]]).mean(axis=1)
    calibrated = artifacts["calibrator"].predict(raw)
    capacity = card["capacity"]["per_week"]
    week = metrics.week_start(inputs["date"])
    order = pd.DataFrame({"camis": inputs["camis"], "week": week, "raw": raw}).sort_values(
        ["week", "raw", "camis"], ascending=[True, False, True], kind="stable")
    order["rank_in_week"] = order.groupby("week").cumcount() + 1
    ranks = order["rank_in_week"].reindex(inputs.index)
    scored = pd.DataFrame({
        "camis": inputs["camis"], "inspection_date": inputs["date"].dt.strftime("%Y-%m-%d"),
        "week": week.dt.strftime("%Y-%m-%d"), "probability": calibrated, "raw_score": raw,
        "rank_in_week": ranks.astype(int), "pool_in_week": week.map(week.value_counts()).astype(int)})
    scored["top_capacity"] = scored["rank_in_week"] <= capacity
    scored["above_threshold"] = calibrated >= card["capacity"]["probability_threshold"]
    return scored.sort_values(["week", "rank_in_week"]).reset_index(drop=True), feats


def drift_report(feats: pd.DataFrame, raw_scores: np.ndarray, card: dict, month: str) -> dict:
    """PSI of this month's features and predictions against the stored reference windows."""
    profile = card["drift_reference"]["features"]
    columns = [c for c in profile if c in feats.columns]
    psi = drift.psi_against({c: profile[c] for c in columns}, feats[columns])
    psi["__prediction__"] = drift.psi_against(
        {"prediction": card["drift_reference"]["prediction"]}, pd.DataFrame({"prediction": raw_scores}))["prediction"]
    floor = card["drift_reference"]["noise_floor_median"]
    rows = [{"feature": name, "psi": value, "status": drift.status(value)} for name, value in psi.items()]
    rows.sort(key=lambda r: -r["psi"])
    return {"month": month, "inspections": int(len(feats)),
            "reference": "training window for features, validation window for predictions",
            "cutoffs": {"moderate": config.PSI_MODERATE, "major": config.PSI_MAJOR},
            "sampling_noise_floor_median": floor,
            "counts": {s: sum(r["status"] == s for r in rows) for s in ("stable", "moderate", "major")},
            "features": rows}
