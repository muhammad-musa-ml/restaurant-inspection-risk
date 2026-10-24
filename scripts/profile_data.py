"""Describe the frozen snapshot: the facts the design in docs/design.md rests on.

    python scripts/profile_data.py            writes results/data_profile.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inspection_risk import config, data, runinfo, study  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/inspections.csv.gz")
    parser.add_argument("--out", default="results/data_profile.json")
    args = parser.parse_args()

    stamps = pd.read_csv(args.data, dtype=str, keep_default_na=False, usecols=["camis", "inspection_date", "record_date"])
    st = study.prepare(args.data)
    raw, events, rows = st.raw, st.events, st.rows

    ci_all = data.cycle_initial(events)
    since_floor = ci_all[ci_all["date"] >= st.floor]
    months = since_floor.assign(month=since_floor["date"].dt.strftime("%Y-%m")).groupby("month").agg(
        initial_cycle_inspections=("camis", "size"),
        share_ge1_critical=("n_crit", lambda s: float((s >= 1).mean())),
        share_ge2_critical=("n_crit", lambda s: float((s >= 2).mean())))

    train = rows[rows["split"] == "train"]
    target_choice = {"window": f"{config.TRAIN_START} to {config.VALID_START} (exclusive)", "rows": int(len(train))}
    for k in range(1, 6):
        target_choice[f"share_at_least_{k}_critical"] = float((train["n_crit"] >= k).mean())

    by_split = {}
    for name in ("train", "valid", "test", "post"):
        sub = rows[rows["split"] == name]
        weeks = sub["week"].nunique()
        mask = (rows["split"] == name).to_numpy()
        by_split[name] = {
            "rows": int(len(sub)), "weeks": int(weeks), "mean_rows_per_week": float(len(sub) / weeks),
            "first_date": str(sub["date"].min().date()), "last_date": str(sub["date"].max().date()),
            "share_ge1_critical": float(sub["y1"].mean()), "share_ge2_critical": float(sub["y"].mean()),
            "share_with_previous_initial_inspection_visible": float(st.feats.loc[mask, "prev_ci_n_crit"].notna().mean()),
            "share_with_any_earlier_inspection_visible": float(st.feats.loc[mask, "last_n_crit"].notna().mean()),
        }

    ordered = ci_all[ci_all["date"] >= st.floor].sort_values(["camis", "date"])
    gaps = ordered.groupby("camis")["date"].diff().dt.days.dropna()
    quantiles = {f"p{int(q * 100)}": float(gaps.quantile(q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)}

    visible = events[events["date"] >= st.floor]
    multi = visible.groupby("camis").filter(lambda g: len(g) >= 2).groupby("camis")
    stability = {
        "restaurants_with_two_or_more_visible_events": int(multi.ngroups),
        "share_with_more_than_one_zipcode": float((multi["zipcode"].nunique() > 1).mean()),
        "share_with_more_than_one_cuisine": float((multi["cuisine"].nunique() > 1).mean()),
        "share_with_more_than_one_boro": float((multi["boro"].nunique() > 1).mean()),
    }

    selection = raw[raw["has_violation"] & (raw["date"] < pd.Timestamp(config.VALID_START))
                    & (raw["inspection_type"] == config.CYCLE_INITIAL)]
    family_counts = selection["family"].value_counts()
    profile = {
        "meta": runinfo.run_meta("python scripts/profile_data.py", ROOT),
        "snapshot": {
            "citation_rows": int(len(stamps)),
            "placeholder_rows_never_inspected": int((stamps["inspection_date"] == data.PLACEHOLDER_DATE).sum()),
            "record_dates": sorted(stamps["record_date"].unique().tolist()),
            "restaurants_with_a_real_inspection": int(raw["camis"].nunique()),
            "inspection_events": int(len(events)),
            "earliest_inspection_date": str(raw["date"].min().date()),
            "latest_inspection_date": str(raw["date"].max().date()),
        },
        "visibility": {
            "max_span_days_between_a_restaurants_earliest_and_latest_record": data.max_span_days(raw),
            "rule_in_publisher_description": "records up to three years before each restaurant's most recent inspection",
            "history_floor_date": str(st.floor.date()),
            "min_window_days": config.MIN_WINDOW_DAYS,
            "first_date_with_min_window": str((st.floor + pd.Timedelta(days=config.MIN_WINDOW_DAYS)).date()),
        },
        "initial_cycle_inspections_since_floor": {
            "inspections": int(len(since_floor)), "restaurants": int(since_floor["camis"].nunique()),
            "gap_days_between_consecutive_initial_inspections": quantiles,
            "by_month": {m: {k: (float(v) if isinstance(v, (float, np.floating)) else int(v)) for k, v in r.items()}
                         for m, r in months.iterrows()},
        },
        "target_choice_from_training_window_only": target_choice,
        "splits": by_split,
        "restaurant_attribute_stability_within_visible_history": stability,
        "violation_families": {
            "selection_rule": f"first two characters of the code; at least {config.FAMILY_MIN_CITATIONS} citations "
                              f"at initial cycle inspections before {config.VALID_START}",
            "selected": st.families,
            "citations_by_family_in_selection_window": {k: int(v) for k, v in family_counts.items()},
        },
        "cuisine_levels_kept": st.cuisine_levels,
        "week_boundaries_are_mondays": all(d.weekday() == 0 for d in (
            config.TRAIN_START, config.VALID_START, config.TEST_START, config.POST_START, config.POST_END)),
    }
    runinfo.write_json(args.out, profile)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
