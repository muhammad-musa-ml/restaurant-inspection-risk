"""Do the leakage checks bite? Compile deliberately leaky copies of the feature builder in memory and
require the checks to catch each one. The repository files are only read."""
import types
from pathlib import Path

import pandas as pd
import pytest

SOURCE = (Path(__file__).resolve().parents[1] / "inspection_risk" / "features.py").read_text(encoding="utf-8")

MUTANTS = {
    # records dated the same day count as history
    "same_day_counts": ("cut", [
        ('def _asof(left: pd.DataFrame, right: pd.DataFrame, by: str | None, exact: bool = False)',
         'def _asof(left: pd.DataFrame, right: pd.DataFrame, by: str | None, exact: bool = True)'),
        ('direction="backward", allow_exact_matches=exact)', 'direction="backward", allow_exact_matches=True)')]),
    # records before the common floor are used
    "no_floor": ("floor", [('ev = events[events["date"] >= floor]', 'ev = events')]),
    # a feature copies the inspection's own outcome
    "uses_own_outcome": ("cut", [('    put("prev_ci_n_crit", prev_ci, "prev_ci_n_crit")',
                                  '    out["prev_ci_n_crit"] = targets["n_crit"].to_numpy()')]),
    # the neighbourhood and cuisine rates include the following days
    "future_prior_rate": ("cut", [
        ('daily["cum_n"] = grouped["n"].cumsum()', 'daily["cum_n"] = grouped["n"].cumsum().shift(-1).fillna(0)'),
        ('daily["cum_pos"] = grouped["pos"].cumsum()', 'daily["cum_pos"] = grouped["pos"].cumsum().shift(-1).fillna(0)')]),
}


def compile_variant(name, replacements):
    source = SOURCE
    for old, new in replacements:
        assert source.count(old) == 1, f"{name}: {old!r} must appear exactly once in features.py"
        source = source.replace(old, new)
    module = types.ModuleType(f"inspection_risk.features_{name}")
    module.__package__ = "inspection_risk"
    exec(compile(source, f"features_{name}", "exec"), module.__dict__)
    return module


def tamper(events, mask):
    events = events.copy()
    events.loc[mask, "n_crit"] = 99
    events.loc[mask, "n_viol"] = 99
    events.loc[mask, "score"] = 999.0
    events.loc[mask, "closed"] = True
    events.loc[mask, "zipcode"] = "99999"
    events.loc[mask, "cuisine"] = "Tampered"
    for column in events.columns[events.columns.str.startswith("fam_")]:
        events.loc[mask, column] = 1
    return events


def unchanged(a, b):
    try:
        pd.testing.assert_frame_equal(a, b)
        return True
    except AssertionError:
        return False


def detectors(module, study):
    """(passes the cut/own-outcome check, passes the floor check) for a builder module."""
    kwargs = dict(floor=study.floor, families=study.families, cuisine_levels=study.cuisine_levels)
    cut = pd.Timestamp("2026-03-23")
    rows = study.rows[study.rows["date"] <= cut]
    garbled = rows.copy()
    for column in ("n_crit", "n_viol", "score", "y", "y1"):
        garbled[column] = 99
    baseline = module.build_features(study.events, rows, **kwargs)
    cut_ok = unchanged(baseline, module.build_features(tamper(study.events, study.events["date"] >= cut), garbled, **kwargs))
    floor_ok = unchanged(module.build_features(study.events, study.rows, **kwargs),
                         module.build_features(tamper(study.events, study.events["date"] < study.floor), study.rows, **kwargs))
    return cut_ok, floor_ok


def test_the_real_builder_passes_both_checks(study):
    assert detectors(compile_variant("original", []), study) == (True, True)


@pytest.mark.parametrize("name", sorted(MUTANTS))
def test_each_leaky_variant_is_caught(study, name):
    caught_by, replacements = MUTANTS[name]
    cut_ok, floor_ok = detectors(compile_variant(name, replacements), study)
    if caught_by == "cut":
        assert not cut_ok, f"{name} slipped past the cut/own-outcome check"
    else:
        assert not floor_ok, f"{name} slipped past the floor check"
