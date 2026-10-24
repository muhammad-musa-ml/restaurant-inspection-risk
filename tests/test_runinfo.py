import json

import numpy as np
import pytest

from inspection_risk import runinfo


def test_write_json_turns_non_finite_numbers_into_null_and_stays_strict(tmp_path):
    path = tmp_path / "x.json"
    runinfo.write_json(path, {"a": float("nan"), "b": [np.float64("inf"), np.int64(3), np.float32(0.5)],
                              "c": np.array([1.0, float("nan")])})
    text = path.read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text) == {"a": None, "b": [None, 3, 0.5], "c": [1.0, None]}


def test_write_json_uses_unix_line_endings(tmp_path):
    path = tmp_path / "y.json"
    runinfo.write_json(path, {"k": 1})
    assert b"\r" not in path.read_bytes() and path.read_bytes().endswith(b"\n")


def test_write_json_refuses_unknown_types(tmp_path):
    with pytest.raises(TypeError):
        runinfo.write_json(tmp_path / "z.json", {"k": object()})
