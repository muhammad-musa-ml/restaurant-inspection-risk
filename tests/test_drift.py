import numpy as np
import pandas as pd
import pytest

from inspection_risk import drift


def frame(n=20000, seed=0, shift=0.0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "x": rng.normal(shift, 1.0, n),
        "flag": (rng.random(n) < 0.3).astype(float),
        "kind": rng.choice(["a", "b", "c"], n, p=[0.5, 0.3, 0.2]),
    })


def test_psi_of_a_distribution_with_itself_is_zero():
    ref = frame()
    profile = drift.reference_profile(ref, ["x", "flag", "kind"])
    assert max(drift.psi_against(profile, ref).values()) == pytest.approx(0.0, abs=1e-12)


def test_psi_known_value_for_shares():
    ref = np.array([0.5, 0.5])
    act = np.array([0.7, 0.3])
    expected = (0.7 - 0.5) * np.log(0.7 / 0.5) + (0.3 - 0.5) * np.log(0.3 / 0.5)
    assert drift.psi_from_shares(ref, act) == pytest.approx(expected)


def test_a_shifted_feature_is_major_and_an_unshifted_one_is_stable():
    ref = frame(seed=0)
    cur = frame(seed=1, shift=1.0)
    profile = drift.reference_profile(ref, ["x", "flag", "kind"])
    got = drift.psi_against(profile, cur)
    assert got["x"] > 0.25 and drift.status(got["x"]) == "major"
    assert drift.status(got["flag"]) == "stable" and drift.status(got["kind"]) == "stable"


def test_missing_values_and_unseen_categories_get_their_own_bins():
    ref = pd.DataFrame({"x": np.arange(1000, dtype=float), "kind": ["a"] * 500 + ["b"] * 500})
    cur = pd.DataFrame({"x": [np.nan] * 500 + list(np.arange(500, dtype=float)), "kind": ["a"] * 250 + ["z"] * 250 + ["b"] * 500})
    profile = drift.reference_profile(ref, ["x", "kind"])
    got = drift.psi_against(profile, cur)
    assert got["x"] > 0.25 and got["kind"] > 0.25


def test_values_outside_the_reference_range_fall_in_the_end_bins():
    ref = pd.DataFrame({"x": np.linspace(0, 1, 5000)})
    cur = pd.DataFrame({"x": [-5.0] * 100 + [5.0] * 100})
    profile = drift.reference_profile(ref, ["x"])
    assert drift.psi_against(profile, cur)["x"] > 0.25


def test_noise_floor_is_small_at_month_size_and_deterministic():
    ref = frame(n=11000)
    a = drift.noise_floor(ref, ["x", "flag", "kind"], size=1300)
    b = drift.noise_floor(ref, ["x", "flag", "kind"], size=1300)
    pd.testing.assert_frame_equal(a, b)
    assert a[["x", "flag", "kind"]].to_numpy().max() < 0.05
    assert list(a["seed"]) == [0, 1, 2, 3, 4]
