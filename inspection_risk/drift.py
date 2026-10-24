"""Population stability index (PSI) per feature and for the prediction distribution.

PSI = sum over bins of (actual share - reference share) * ln(actual share / reference share).
Bins are fixed from the reference window: quantile edges for continuous values, the observed values
for columns with few distinct values, the observed categories for strings; a missing value is its own
bin, and a value the reference never had falls in an "other" bin.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

FEW_VALUES = 12
FLOOR = 1e-4  # shares are floored so an empty bin cannot produce an infinite term


def psi_from_shares(reference: np.ndarray, actual: np.ndarray) -> float:
    reference = np.clip(np.asarray(reference, float), FLOOR, None)
    actual = np.clip(np.asarray(actual, float), FLOOR, None)
    return float(np.sum((actual - reference) * np.log(actual / reference)))


def _is_categorical(series: pd.Series) -> bool:
    if not pd.api.types.is_numeric_dtype(series):
        return True
    return series.nunique(dropna=True) <= FEW_VALUES


def _levels_for(series: pd.Series) -> list:
    if pd.api.types.is_numeric_dtype(series):
        return sorted(float(v) for v in series.dropna().unique())
    return sorted(str(v) for v in series.dropna().unique())


def _assign(series: pd.Series, spec: dict) -> np.ndarray:
    """Bin index per value for a reference profile entry."""
    if spec["kind"] == "levels":
        levels = spec["levels"]
        lookup = {level: i for i, level in enumerate(levels)}
        other = len(levels)
        missing = len(levels) + 1
        if pd.api.types.is_numeric_dtype(series):
            values = series.astype(float)
            codes = [missing if pd.isna(v) else lookup.get(float(v), other) for v in values]
        else:
            codes = [missing if pd.isna(v) else lookup.get(str(v), other) for v in series]
        return np.asarray(codes)
    edges = np.asarray(spec["edges"], float)
    values = series.astype(float).to_numpy()
    codes = np.searchsorted(edges, values, side="right") - 1
    codes = np.clip(codes, 0, len(edges) - 2)
    return np.where(np.isnan(values), len(edges) - 1, codes)


def _shares(codes: np.ndarray, n_bins: int) -> np.ndarray:
    counts = np.bincount(codes, minlength=n_bins).astype(float)
    return counts / counts.sum()


def reference_profile(frame: pd.DataFrame, columns: list[str], bins: int = config.PSI_BINS) -> dict:
    """Bin definitions and reference shares for each column of the reference window."""
    profile = {}
    for column in columns:
        series = frame[column]
        if _is_categorical(series):
            spec = {"kind": "levels", "levels": _levels_for(series)}
            n_bins = len(spec["levels"]) + 2
        else:
            values = series.dropna().astype(float).to_numpy()
            inner = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
            spec = {"kind": "edges", "edges": inner.tolist()}
            n_bins = len(inner)  # len(inner)-1 value bins + 1 missing bin
        spec["shares"] = _shares(_assign(series, spec), n_bins).tolist()
        profile[column] = spec
    return profile


def psi_against(profile: dict, frame: pd.DataFrame) -> dict[str, float]:
    result = {}
    for column, spec in profile.items():
        n_bins = len(spec["shares"])
        result[column] = psi_from_shares(spec["shares"], _shares(_assign(frame[column], spec), n_bins))
    return result


def status(value: float) -> str:
    if value >= config.PSI_MAJOR:
        return "major"
    if value >= config.PSI_MODERATE:
        return "moderate"
    return "stable"


def noise_floor(frame: pd.DataFrame, columns: list[str], size: int, seeds=config.SEEDS) -> pd.DataFrame:
    """PSI between the reference window and random same-size samples of itself: what sampling noise alone
    produces at this month size. One row per seed; columns are the features."""
    profile = reference_profile(frame, columns)
    rows = []
    for seed in seeds:
        sample = frame.sample(n=size, random_state=seed)
        rows.append({"seed": seed, **psi_against(profile, sample)})
    return pd.DataFrame(rows)
