"""Leakage checks on the real data: change what must not matter, and the features must not move."""
import pandas as pd
import pytest

from inspection_risk import features

CUTS = ["2024-12-09", "2025-03-10", "2025-11-17", "2026-03-23", "2026-07-06"]


def build(study, events, rows):
    return features.build_features(events, rows, floor=study.floor, families=study.families,
                                   cuisine_levels=study.cuisine_levels)


def tamper(study, mask):
    """Replace every outcome and attribute of the selected events with values that would shift any feature."""
    events = study.events.copy()
    events.loc[mask, "n_crit"] = 99
    events.loc[mask, "n_viol"] = 99
    events.loc[mask, "score"] = 999.0
    events.loc[mask, "closed"] = True
    events.loc[mask, "zipcode"] = "99999"
    events.loc[mask, "cuisine"] = "Tampered"
    for column in events.columns[events.columns.str.startswith("fam_")]:
        events.loc[mask, column] = 1
    return events


@pytest.mark.parametrize("cut", CUTS)
def test_records_on_or_after_the_cut_do_not_change_earlier_inspections(study, cut):
    cut = pd.Timestamp(cut)
    rows = study.rows[study.rows["date"] <= cut]
    garbled = rows.copy()
    for column in ("n_crit", "n_viol", "score", "y", "y1"):  # an inspection's own outcome must never be read
        garbled[column] = 99
    baseline = build(study, study.events, rows)
    tampered = tamper(study, study.events["date"] >= cut)
    pd.testing.assert_frame_equal(baseline, build(study, tampered, garbled))
    assert len(rows) > 500, "the check must cover a real number of inspections"


@pytest.mark.parametrize("cut", CUTS)
def test_removing_the_future_changes_nothing(study, cut):
    cut = pd.Timestamp(cut)
    rows = study.rows[study.rows["date"] <= cut]
    truncated = study.events[study.events["date"] <= cut]
    pd.testing.assert_frame_equal(build(study, study.events, rows), build(study, truncated, rows))


def test_records_before_the_floor_are_ignored(study):
    old = tamper(study, study.events["date"] < study.floor)
    assert (study.events["date"] < study.floor).sum() > 0, "the snapshot has records older than the floor"
    pd.testing.assert_frame_equal(build(study, study.events, study.rows), build(study, old, study.rows))


def test_one_restaurants_later_records_do_not_change_its_earlier_rows(study):
    """The publisher drops records by each restaurant's own latest inspection, so check restaurant by restaurant."""
    rows = study.rows
    sample = rows["camis"].drop_duplicates().sample(400, random_state=0)
    cut = pd.Timestamp("2026-01-05")
    mine = rows[rows["camis"].isin(sample) & (rows["date"] < cut)]
    kept = study.events[~(study.events["camis"].isin(sample) & (study.events["date"] >= cut))]
    pd.testing.assert_frame_equal(build(study, study.events, mine), build(study, kept, mine))
    assert len(mine) > 200


def test_the_check_has_teeth(study, monkeypatch):
    """Plant a leak (let a record dated on the same day count as history) and the check must see it."""
    cut = pd.Timestamp("2026-03-23")
    rows = study.rows[study.rows["date"] <= cut]
    original = features._asof

    def leaky(left, right, by, exact=False):
        return original(left, right, by, exact=True)

    monkeypatch.setattr(features, "_asof", leaky)
    baseline = build(study, study.events, rows)
    changed = build(study, tamper(study, study.events["date"] >= cut), rows)
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(baseline, changed)


def test_features_have_no_outcome_columns(study):
    forbidden = {"n_crit", "n_viol", "score", "closed", "grade", "action", "critical_flag", "y", "y1"}
    assert not forbidden & set(study.feats.columns)


def test_feature_rows_align_with_ranked_rows(study):
    assert len(study.feats) == len(study.rows)
    assert (study.feats.index == study.rows.index).all()
