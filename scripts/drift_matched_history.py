"""Is the feature drift only the publisher's growing history?

    python scripts/drift_matched_history.py            # writes results/drift_matched_history.json

The monthly PSI in results/drift_monthly.csv compares each month with the training window. The publisher
keeps three years of history per restaurant, so a row dated later in the study sees more of the past than a
row dated earlier, and features that describe the past move for that reason alone. This script repeats the
comparison on rows whose visible history is the same in every window: the inspections whose previous initial
inspection took place within the last 365 days. From the first modelled date on, every restaurant's last 365
days are visible in the file, so that rule means the same thing in the training window and in any later one.

Only about one inspection in ten qualifies, so a month holds roughly 140 matching rows, too few for a stable
PSI. The comparison is therefore made per window (validation, test, after the test) against the matching
training rows, and the sampling-noise floor is measured at each window's matching size. The same windows are
also scored on all rows (the setup of the monthly table) so the two columns of numbers are comparable.

What this does not remove: features that accumulate over the whole visible history (the share of earlier
days that cited a violation family, ever closed before) still see a longer past in later windows. They are
reported as their own group so the reader can see which alerts could still be the visible-history effect.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inspection_risk import config, drift, features, pipeline, runinfo  # noqa: E402
from inspection_risk import study as study_module  # noqa: E402

MATCH_DAYS = 365
WINDOWS = ("valid", "test", "post")
GROUP_ORDER = ["previous initial inspection", "earlier history", "neighbourhood and cuisine rates", "calendar and static"]


def group_of(feature: str) -> str:
    if feature.startswith("prev_ci") or feature == "days_since_prev_ci":
        return GROUP_ORDER[0]
    if feature.startswith(("last_", "days_since_last", "n_inspection", "ever_closed", "fam_")):
        return GROUP_ORDER[1]
    if feature in ("zip_prior_rate", "cuisine_prior_rate"):
        return GROUP_ORDER[2]
    return GROUP_ORDER[3]


def counts(frame: pd.DataFrame) -> dict:
    """Feature-windows by status overall and by feature group."""
    frame = frame.copy()
    frame["group"] = frame["feature"].map(group_of)
    statuses = ("stable", "moderate", "major")
    out = {"feature_windows": int(len(frame)),
           "all": {s: int((frame["status"] == s).sum()) for s in statuses},
           "without_calendar_month": {s: int(((frame["status"] == s) & (frame["feature"] != "month")).sum())
                                      for s in statuses},
           "by_group": {}}
    for group in GROUP_ORDER:
        part = frame[frame["group"] == group]
        out["by_group"][group] = {"feature_windows": int(len(part)), **{s: int((part["status"] == s).sum()) for s in statuses}}
    return out


def psi_table(profile: dict, feats: pd.DataFrame, masks: dict, extra: np.ndarray | None) -> pd.DataFrame:
    records = []
    for window in WINDOWS:
        selected = masks[window] if extra is None else masks[window] & extra
        for feature, value in drift.psi_against(profile, feats[selected]).items():
            records.append({"window": window, "rows": int(selected.sum()), "feature": feature, "psi": value,
                            "status": drift.status(value)})
    return pd.DataFrame(records)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/inspections.csv.gz")
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    results = Path(args.results)

    st = study_module.prepare(args.data)
    rows, feats = st.rows, st.feats
    masks = {k: np.asarray(v, bool) for k, v in pipeline.masks_of(rows).items()}
    columns = st.numeric + features.CATEGORICAL
    matched = (feats["days_since_prev_ci"] <= MATCH_DAYS).to_numpy()

    all_rows = psi_table(drift.reference_profile(feats[masks["train"]], columns), feats, masks, None)
    matched_rows = psi_table(drift.reference_profile(feats[masks["train"] & matched], columns), feats, masks, matched)

    floors = {}
    for window in WINDOWS:
        size = int((masks[window] & matched).sum())
        floor = drift.noise_floor(feats[masks["train"] & matched], columns, size=size)
        values = floor[[c for c in floor.columns if c != "seed"]].to_numpy()
        floors[window] = {"sample_size": size, "median": float(np.median(values)), "max": float(values.max())}

    per_feature = {}
    for feature in columns:
        entry = {"group": group_of(feature)}
        for label, table in (("all_rows", all_rows), ("matched", matched_rows)):
            part = table[table["feature"] == feature].set_index("window")["psi"]
            entry[label] = {w: float(part[w]) for w in WINDOWS}
        per_feature[feature] = entry

    windows = {}
    for name in ("train",) + WINDOWS:
        in_window = masks[name]
        windows[name] = {"rows": int(in_window.sum()), "rows_matched": int((in_window & matched).sum()),
                         "matched_share": float((in_window & matched).sum() / in_window.sum()),
                         "positive_share_matched": float(rows.loc[in_window & matched, "y"].mean())}

    summary = {
        "meta": runinfo.run_meta("python scripts/drift_matched_history.py", ROOT),
        "rule": f"rows whose previous initial inspection took place at most {MATCH_DAYS} days earlier",
        "why": "the same rule means the same visible history in every window, so these rows are comparable",
        "reference": "the training window (matching rows only for the matched columns)",
        "windows": windows, "sampling_noise_floor_matched": floors,
        "cutoffs": {"moderate": config.PSI_MODERATE, "major": config.PSI_MAJOR},
        "all_rows": counts(all_rows), "matched": counts(matched_rows), "per_feature": per_feature,
    }
    runinfo.write_json(results / "drift_matched_history.json", summary)
    pd.concat([all_rows.assign(rows_used="all"), matched_rows.assign(rows_used="matched")]).to_csv(
        results / "drift_matched_history.csv", index=False, lineterminator="\n")
    print("wrote results/drift_matched_history.json and .csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
