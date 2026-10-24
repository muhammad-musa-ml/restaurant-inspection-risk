"""Time the batch scoring of one month, 5 repetitions each of two measurements.

    python scripts/time_batch.py --month 2026-09

1. end to end: a fresh `python scripts/score_month.py` process per repetition (interpreter start,
   imports, reading the snapshot, features, scoring, writing the two files);
2. by stage inside one process: reading the snapshot, building the inspection table, loading the models,
   building the month's features and scoring them (the feature and scoring split is taken inside
   score_targets, so they are timed together).

Wall clock (time.perf_counter). Run it when nothing else heavy is running on the machine; the file
records the number of inspections scored and the machine, not the load.
"""
from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inspection_risk import data, runinfo, scoring  # noqa: E402

REPETITIONS = 5


def spread(values):
    return {"median": statistics.median(values), "min": min(values), "max": max(values),
            "values": [round(v, 4) for v in values]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", default="2026-09")
    parser.add_argument("--out", default="results/timing_batch_scoring.json")
    args = parser.parse_args()
    snapshot = ROOT / "data" / "inspections.csv.gz"

    end_to_end, inspections = [], None
    with tempfile.TemporaryDirectory() as tmp:
        for _ in range(REPETITIONS):
            start = time.perf_counter()
            done = subprocess.run([sys.executable, str(ROOT / "scripts" / "score_month.py"), "--month", args.month,
                                   "--out-dir", tmp], cwd=ROOT, capture_output=True, text=True)
            end_to_end.append(time.perf_counter() - start)
            if done.returncode != 0:
                print(done.stderr)
                return 1
            inspections = int(done.stdout.split()[1])

    stages = {"read_snapshot": [], "build_inspection_table": [], "load_models": [], "features_and_scoring": []}
    for _ in range(REPETITIONS):
        t = time.perf_counter()
        raw = data.load_raw(snapshot)
        stages["read_snapshot"].append(time.perf_counter() - t)
        t = time.perf_counter()
        events = data.build_events(raw)
        stages["build_inspection_table"].append(time.perf_counter() - t)
        t = time.perf_counter()
        artifacts = scoring.load_artifacts(ROOT / "models")
        stages["load_models"].append(time.perf_counter() - t)
        targets = scoring.month_targets(events, args.month)
        t = time.perf_counter()
        scoring.score_targets(events, targets, artifacts)
        stages["features_and_scoring"].append(time.perf_counter() - t)

    result = {
        "meta": runinfo.run_meta(f"python scripts/time_batch.py --month {args.month}", ROOT),
        "month": args.month, "inspections_scored": inspections, "repetitions": REPETITIONS,
        "seconds_end_to_end_fresh_process": spread(end_to_end),
        "seconds_by_stage_in_one_process": {name: spread(values) for name, values in stages.items()},
        "note": "wall clock; the machine was shared with other containers, see timing_docker_ps.txt beside this file",
    }
    runinfo.write_json(args.out, result)
    print(f"wrote {args.out}: end to end median {result['seconds_end_to_end_fresh_process']['median']:.2f} s "
          f"for {inspections} inspections")
    return 0


if __name__ == "__main__":
    sys.exit(main())
