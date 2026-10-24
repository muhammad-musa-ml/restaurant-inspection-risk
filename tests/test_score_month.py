import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from inspection_risk import config, scoring

MONTH = "2026-04"


@pytest.fixture(scope="module")
def artifacts(root):
    return scoring.load_artifacts(root / "models")


def blank_outcomes(events, mask):
    """Replace the outcomes of the selected events; identifiers and static attributes stay."""
    events = events.copy()
    events.loc[mask, "n_crit"] = 99
    events.loc[mask, "n_viol"] = 99
    events.loc[mask, "score"] = 999.0
    events.loc[mask, "closed"] = True
    for column in events.columns[events.columns.str.startswith("fam_")]:
        events.loc[mask, column] = 1
    return events


def test_scores_do_not_depend_on_the_months_own_outcomes(study, artifacts):
    targets = scoring.month_targets(study.events, MONTH)
    base, _ = scoring.score_targets(study.events, targets, artifacts)
    start = pd.Timestamp(f"{MONTH}-01")
    changed = blank_outcomes(study.events, study.events["date"] >= start)
    again, _ = scoring.score_targets(changed, scoring.month_targets(changed, MONTH), artifacts)
    pd.testing.assert_frame_equal(base, again)
    assert len(base) > 500


def test_batch_scores_reproduce_the_pipelines_predictions(study, artifacts, root):
    targets = scoring.month_targets(study.events, MONTH)
    scored, _ = scoring.score_targets(study.events, targets, artifacts)
    saved = pd.read_csv(root / "results" / "predictions.csv.gz", dtype={"camis": str})
    saved = saved.rename(columns={"date": "inspection_date"})
    merged = scored.merge(saved, on=["camis", "inspection_date"], how="left")
    assert merged["gbm_raw"].notna().all(), "every scored inspection must be in results/predictions.csv.gz"
    np.testing.assert_allclose(merged["raw_score"], merged["gbm_raw"], atol=2e-6)
    np.testing.assert_allclose(merged["probability"], merged["gbm_calibrated"], atol=2e-6)


def test_ranking_table_is_well_formed(study, artifacts):
    targets = scoring.month_targets(study.events, MONTH)
    scored, _ = scoring.score_targets(study.events, targets, artifacts)
    capacity = artifacts["card"]["capacity"]["per_week"]
    assert len(scored) == len(targets)
    assert scored["probability"].between(0, 1).all()
    for _, week in scored.groupby("week"):
        assert sorted(week["rank_in_week"]) == list(range(1, len(week) + 1))
        assert week["top_capacity"].sum() == min(capacity, len(week))
        ordered = week.sort_values("rank_in_week")["raw_score"].to_numpy()
        assert (np.diff(ordered) <= 1e-12).all(), "rank 1 must be the highest score"
        assert (week["pool_in_week"] == len(week)).all()


def test_dates_the_model_was_not_built_for_are_refused(study, artifacts):
    early = pd.DataFrame({"camis": [study.rows["camis"].iloc[0]], "planned_date": ["2024-01-02"]})
    with pytest.raises(scoring.ScoringError):
        scoring.score_targets(study.events, scoring.due_targets(study.events, early), artifacts)
    with pytest.raises(scoring.ScoringError):
        scoring.month_targets(study.events, "2031-01")
    with pytest.raises(scoring.ScoringError):
        scoring.due_targets(study.events, pd.DataFrame({"id": ["1"]}))


def test_forward_mode_scores_a_planned_list_and_handles_an_unknown_restaurant(study, artifacts):
    known = study.rows.loc[study.rows["split"] == "test", "camis"].drop_duplicates().head(6).tolist()
    due = pd.DataFrame({"camis": known + ["99999999"], "planned_date": ["2026-09-15"] * 7})
    scored, feats = scoring.score_targets(study.events, scoring.due_targets(study.events, due), artifacts)
    assert len(scored) == 7 and scored["probability"].between(0, 1).all()
    unknown = scored["camis"] == "99999999"
    assert feats.loc[unknown.to_numpy(), "prev_ci_n_crit"].isna().all()


def test_drift_report_covers_every_feature_and_the_prediction(study, artifacts):
    targets = scoring.month_targets(study.events, MONTH)
    scored, feats = scoring.score_targets(study.events, targets, artifacts)
    report = scoring.drift_report(feats, scored.sort_index()["raw_score"].to_numpy()[: len(feats)], artifacts["card"], MONTH)
    names = {row["feature"] for row in report["features"]}
    assert "__prediction__" in names and set(artifacts["card"]["drift_reference"]["features"]) <= names
    assert sum(report["counts"].values()) == len(report["features"])


def test_command_line_writes_scores_and_drift(root, tmp_path):
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / "score_month.py"), "--month", config.NEW_MONTH,
         "--data", str(root / "data" / "inspections.csv.gz"), "--models", str(root / "models"),
         "--out-dir", str(tmp_path)], capture_output=True, text=True, cwd=root)
    assert result.returncode == 0, result.stderr
    scores = pd.read_csv(tmp_path / f"scores_{config.NEW_MONTH}.csv", dtype={"camis": str})
    assert len(scores) > 1000 and scores["observed_hit"].notna().all()
    report = json.loads((tmp_path / f"drift_{config.NEW_MONTH}.json").read_text(encoding="utf-8"))
    assert report["month"] == config.NEW_MONTH
    refused = subprocess.run([sys.executable, str(root / "scripts" / "score_month.py"), "--month", "2031-01",
                              "--out-dir", str(tmp_path)], capture_output=True, text=True, cwd=root)
    assert refused.returncode == 2 and "cannot score" in refused.stderr
