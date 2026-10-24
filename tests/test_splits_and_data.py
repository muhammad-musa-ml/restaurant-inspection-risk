import gzip
import hashlib
import json

import pandas as pd

from inspection_risk import config, data


def test_every_split_boundary_is_a_monday():
    for day in (config.TRAIN_START, config.VALID_START, config.TEST_START, config.POST_START, config.POST_END):
        assert day.weekday() == 0, day


def test_split_boundaries_are_in_order():
    edges = [config.TRAIN_START, config.VALID_START, config.TEST_START, config.POST_START, config.POST_END]
    assert edges == sorted(edges) and len(set(edges)) == len(edges)


def test_splits_are_disjoint_in_time_and_cover_the_window(study):
    rows = study.rows
    spans = {name: (g["date"].min(), g["date"].max()) for name, g in rows.groupby("split")}
    assert spans["train"][1] < spans["valid"][0] <= spans["valid"][1] < spans["test"][0]
    assert spans["test"][1] < spans["post"][0]
    assert rows["split"].isin(["train", "valid", "test", "post"]).all()
    assert rows["date"].min() >= pd.Timestamp(config.TRAIN_START)
    assert rows["date"].max() < pd.Timestamp(config.POST_END)


def test_no_week_straddles_two_splits(study):
    assert (study.rows.groupby("week")["split"].nunique() == 1).all()


def test_modelled_rows_have_a_full_window_of_visible_history(study):
    first_ok = study.floor + pd.Timedelta(days=config.MIN_WINDOW_DAYS)
    assert study.rows["date"].min() >= first_ok


def test_floor_is_the_latest_date_minus_the_visibility_window(study):
    assert study.floor == study.raw["date"].max() - pd.Timedelta(days=config.VISIBILITY_DAYS)
    assert str(study.floor.date()) == "2023-10-22"


def test_the_publishers_three_year_rule_holds_in_the_snapshot(study):
    assert data.max_span_days(study.raw) <= config.VISIBILITY_DAYS


def test_one_ranked_row_per_restaurant_and_date(study):
    assert not study.rows.duplicated(["camis", "date"]).any()
    assert (study.rows["inspection_type"] == config.CYCLE_INITIAL).all()


def test_target_definition(study):
    rows = study.rows
    assert ((rows["n_crit"] >= config.TARGET_MIN_CRITICAL).astype(int) == rows["y"]).all()
    assert ((rows["n_crit"] >= config.SECONDARY_MIN_CRITICAL).astype(int) == rows["y1"]).all()
    assert (rows["y"] <= rows["y1"]).all()


def test_snapshot_matches_its_manifest(root):
    manifest = json.loads((root / "data" / "MANIFEST.json").read_text(encoding="utf-8"))
    gz_path = root / "data" / "inspections.csv.gz"
    assert hashlib.sha256(gz_path.read_bytes()).hexdigest() == manifest["files"]["inspections.csv.gz"]["sha256_of_gzip"]
    with gzip.open(gz_path, "rb") as handle:
        csv_bytes = handle.read()
    assert hashlib.sha256(csv_bytes).hexdigest() == manifest["files"]["inspections.csv.gz"]["sha256_of_csv"]
    assert csv_bytes.count(b"\n") - 1 == manifest["files"]["inspections.csv.gz"]["rows"]
    codes = (root / "data" / "violation_codes.csv").read_bytes()
    assert hashlib.sha256(codes).hexdigest() == manifest["files"]["violation_codes.csv"]["sha256"]


def test_snapshot_has_no_names_addresses_or_phones(root):
    header = gzip.open(root / "data" / "inspections.csv.gz", "rt", encoding="utf-8").readline().strip().split(",")
    assert not {"dba", "building", "street", "phone", "latitude", "longitude"} & set(header)
