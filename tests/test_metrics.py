import numpy as np
import pandas as pd
import pytest

from inspection_risk import metrics


def test_brier_known_value():
    assert metrics.brier([1, 0, 1, 0], [1.0, 0.0, 0.5, 0.5]) == pytest.approx((0 + 0 + 0.25 + 0.25) / 4)


def test_ece_is_zero_for_a_calibrated_forecast():
    p = np.repeat([0.1, 0.5, 0.9], 1000)
    y = np.concatenate([[1] * 100 + [0] * 900, [1] * 500 + [0] * 500, [1] * 900 + [0] * 100])
    assert metrics.ece(y, p) == pytest.approx(0.0, abs=1e-12)


def test_ece_known_overconfident_forecast():
    # 1000 rows say 0.9 but only 60% are positive; 1000 say 0.1 and 40% are positive: gap 0.3 in both bins
    p = np.repeat([0.9, 0.1], 1000)
    y = np.concatenate([[1] * 600 + [0] * 400, [1] * 400 + [0] * 600])
    assert metrics.ece(y, p) == pytest.approx(0.3)


def test_ece_puts_probability_one_in_the_last_bin():
    table = metrics.reliability_table(np.array([1, 1]), np.array([1.0, 0.95]))
    assert table.loc[9, "n"] == 2 and table["n"].sum() == 2


def test_week_start_is_the_monday():
    dates = pd.Series(pd.to_datetime(["2026-02-02", "2026-02-08", "2026-02-09"]))
    assert metrics.week_start(dates).dt.strftime("%Y-%m-%d").tolist() == ["2026-02-02", "2026-02-02", "2026-02-09"]


def toy():
    # two weeks; scores rank the positives first in week 1 and last in week 2
    week = np.array(["w1"] * 4 + ["w2"] * 4)
    score = np.array([0.9, 0.8, 0.3, 0.1, 0.9, 0.8, 0.3, 0.1])
    y = np.array([1, 1, 0, 0, 0, 0, 1, 1])
    return week, score, y, np.zeros(8)


def test_capacity_metrics_known_toy_case():
    week, score, y, tb = toy()
    got = metrics.capacity_metrics(week, score, y, tb, capacity=2)
    # top 2 of each week: week 1 hits 2, week 2 hits 0
    assert got["visits"] == 4 and got["hits"] == 2 and got["positives"] == 4
    assert got["precision"] == pytest.approx(0.5) and got["recall"] == pytest.approx(0.5)
    assert got["base_rate"] == pytest.approx(0.5) and got["lift"] == pytest.approx(1.0)


def test_capacity_larger_than_the_pool_takes_everything():
    week, score, y, tb = toy()
    got = metrics.capacity_metrics(week, score, y, tb, capacity=100)
    assert got["visits"] == 8 and got["recall"] == pytest.approx(1.0) and got["precision"] == pytest.approx(0.5)


def test_ties_are_broken_by_the_tiebreak_not_by_row_order():
    week = np.array(["w"] * 4)
    score = np.zeros(4)
    y = np.array([1, 0, 0, 0])
    first = metrics.capacity_metrics(week, score, y, np.array([0.1, 0.2, 0.3, 0.4]), capacity=1)
    last = metrics.capacity_metrics(week, score, y, np.array([0.4, 0.3, 0.2, 0.1]), capacity=1)
    assert first["hits"] == 1 and last["hits"] == 0


def test_capacity_curve_matches_single_capacity_calls():
    week, score, y, tb = toy()
    curve = metrics.capacity_curve(week, score, y, tb, capacities=(1, 2, 3))
    for _, row in curve.iterrows():
        single = metrics.capacity_metrics(week, score, y, tb, int(row["capacity"]))
        assert row["precision"] == pytest.approx(single["precision"]) and row["hits"] == single["hits"]


def test_threshold_flags_about_the_capacity_on_validation_weeks():
    rng = np.random.default_rng(0)
    scores = rng.random(1700)
    weeks = np.repeat(np.arange(17), 100)
    threshold, fraction = metrics.threshold_for_capacity(scores, weeks, capacity=30)
    assert fraction == pytest.approx(0.3)
    assert (scores >= threshold).sum() / 17 == pytest.approx(30, abs=1.0)


def test_flagged_metrics_known_case():
    week = np.array(["w1"] * 4 + ["w2"] * 4)
    y = np.array([1, 1, 0, 0, 0, 0, 1, 1])
    flag = np.array([1, 0, 0, 0, 1, 1, 0, 0], bool)
    got = metrics.flagged_metrics(week, y, flag)
    assert got["visits"] == 3 and got["hits"] == 1 and got["positives"] == 4
    assert got["visits_per_week_min"] == 1 and got["visits_per_week_max"] == 2


def test_bootstrap_is_reproducible_and_brackets_the_estimate():
    week, score, y, tb = toy()
    frame = metrics.ranked(week, score, y, tb)
    counts = metrics.per_week_counts(frame, 2)
    a = metrics.bootstrap_weeks(counts, draws=300, seed=1)
    b = metrics.bootstrap_weeks(counts, draws=300, seed=1)
    assert a == b
    assert a["precision"]["p2.5"] <= 0.5 <= a["precision"]["p97.5"]
