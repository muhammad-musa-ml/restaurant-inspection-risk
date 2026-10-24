"""Score a month of initial inspections and report that month's drift.

Backtest (the month is in the snapshot): scores the initial cycle inspections that happened that month from
records dated before each one, then adds what was observed afterwards so the ranking can be checked.

    python scripts/score_month.py --month 2026-09

Forward (a list of planned inspections): scores restaurants for planned dates; history is whatever the
snapshot holds before each date.

    python scripts/score_month.py --due planned.csv --month 2026-11      # columns: camis,planned_date

Writes <out-dir>/scores_<month>.csv (one row per inspection, ranked within its week; top_capacity marks the
first 100 of the week, above_threshold the calibrated-probability cut-off) and <out-dir>/drift_<month>.json.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inspection_risk import config, data, runinfo, scoring  # noqa: E402


def run(args) -> dict:
    raw = data.load_raw(args.data)
    events = data.build_events(raw)
    artifacts = scoring.load_artifacts(args.models)
    card = artifacts["card"]
    if args.due:
        due = pd.read_csv(args.due, dtype=str, keep_default_na=False)
        targets = scoring.due_targets(events, due)
        label = args.month or "due"
    else:
        if not args.month:
            raise scoring.ScoringError("give --month YYYY-MM or --due FILE")
        targets = scoring.month_targets(events, args.month)
        label = args.month
    scored, feats = scoring.score_targets(events, targets, artifacts)
    if not args.due:
        observed = targets[["camis", "date", "n_crit"]].copy()
        observed["inspection_date"] = observed["date"].dt.strftime("%Y-%m-%d")
        scored = scored.merge(observed[["camis", "inspection_date", "n_crit"]], on=["camis", "inspection_date"],
                              how="left").rename(columns={"n_crit": "observed_critical_violations"})
        scored["observed_hit"] = scored["observed_critical_violations"] >= card["target"]["min_critical_violations"]
    # the drift report needs the raw scores in the feature table's order
    ordered = scored.set_index(["camis", "inspection_date"])["raw_score"]
    keys = list(zip(targets["camis"], targets["date"].dt.strftime("%Y-%m-%d")))
    raw_scores = ordered.reindex(keys).to_numpy()
    report = scoring.drift_report(feats, raw_scores, card, label)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = out_dir / f"scores_{label}.csv"
    scored.to_csv(scores_path, index=False, lineterminator="\n")
    runinfo.write_json(out_dir / f"drift_{label}.json", report)
    return {"label": label, "inspections": len(scored), "weeks": int(scored["week"].nunique()),
            "flagged_top_capacity": int(scored["top_capacity"].sum()),
            "flagged_above_threshold": int(scored["above_threshold"].sum()),
            "drift_counts": report["counts"], "scores": str(scores_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--month", help="YYYY-MM")
    parser.add_argument("--due", help="CSV with columns camis,planned_date")
    parser.add_argument("--data", default="data/inspections.csv.gz")
    parser.add_argument("--models", default="models")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()
    try:
        summary = run(args)
    except scoring.ScoringError as error:
        print(f"cannot score: {error}", file=sys.stderr)
        return 2
    print(f"scored {summary['inspections']} inspections in {summary['weeks']} weeks for {summary['label']}: "
          f"{summary['flagged_top_capacity']} in the first {config.CAPACITY_PER_WEEK} of their week, "
          f"{summary['flagged_above_threshold']} above the probability threshold; drift {summary['drift_counts']}; "
          f"wrote {summary['scores']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
