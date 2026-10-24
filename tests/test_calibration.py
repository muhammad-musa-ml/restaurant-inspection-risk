"""Calibrators on synthetic scores (made up here; nothing from the inspection data)."""
import json

import numpy as np
import pytest

from inspection_risk import metrics, modeling


def overconfident(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    true_p = rng.uniform(0.2, 0.8, n)
    y = (rng.random(n) < true_p).astype(int)
    stretched = 1 / (1 + np.exp(-3.0 * np.log(true_p / (1 - true_p))))  # pushed towards 0 and 1
    return stretched, y


@pytest.mark.parametrize("cls", [modeling.Platt, modeling.Isotonic])
def test_calibrators_reduce_ece_and_brier_on_held_out_data(cls):
    p_fit, y_fit = overconfident(seed=0)
    p_new, y_new = overconfident(seed=1)
    calibrator = cls().fit(p_fit, y_fit)
    after = calibrator.predict(p_new)
    assert metrics.ece(y_new, after) < metrics.ece(y_new, p_new) / 3
    assert metrics.brier(y_new, after) < metrics.brier(y_new, p_new)


@pytest.mark.parametrize("cls", [modeling.Platt, modeling.Isotonic])
def test_calibrators_round_trip_through_json(cls):
    p, y = overconfident()
    calibrator = cls().fit(p, y)
    spec = json.loads(json.dumps(calibrator.to_dict()))
    restored = modeling.calibrator_from_dict(spec)
    probe = np.linspace(0, 1, 101)
    np.testing.assert_allclose(calibrator.predict(probe), restored.predict(probe))


def test_isotonic_output_is_monotone_and_inside_zero_one():
    p, y = overconfident()
    out = modeling.Isotonic().fit(p, y).predict(np.linspace(-0.1, 1.1, 200))
    assert (np.diff(out) >= -1e-12).all() and out.min() >= 0 and out.max() <= 1


def test_unknown_calibrator_kind_is_refused():
    with pytest.raises(ValueError):
        modeling.calibrator_from_dict({"kind": "none"})
