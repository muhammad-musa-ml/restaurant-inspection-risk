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


def test_gzip_text_round_trips_with_lf_endings(tmp_path):
    path = tmp_path / "m.txt.gz"
    runinfo.write_gzip_text(path, "tree\r\nversion=v4\n")
    assert runinfo.read_gzip_text(path) == "tree\nversion=v4\n"


def test_committed_compressed_files_carry_no_time_stamp_or_file_name(root):
    """A header with the write time or a name would change the file on every run, whatever the content."""
    paths = [path for folder in ("data", "results", "models") for path in sorted((root / folder).glob("*.gz"))]
    assert len(paths) >= 7  # the snapshot, the predictions and one model per seed
    for path in paths:
        head = path.read_bytes()[:10]
        assert head[:2] == b"\x1f\x8b" and head[3] == 0 and head[4:8] == b"\x00\x00\x00\x00", path.name


def test_gzip_text_bytes_depend_only_on_the_text(tmp_path):
    first, second = tmp_path / "a" / "m.txt.gz", tmp_path / "b" / "other_name.txt.gz"
    runinfo.write_gzip_text(first, "same text\n" * 50)
    runinfo.write_gzip_text(second, "same text\n" * 50)
    raw = first.read_bytes()
    assert raw == second.read_bytes()
    assert raw[4:8] == b"\x00\x00\x00\x00"  # no time stamp
    assert raw[3] == 0                       # no flags: no stored file name
    runinfo.write_gzip_text(tmp_path / "c.txt.gz", "same text\n" * 50, level=1)
    assert (tmp_path / "c.txt.gz").read_bytes() != raw
