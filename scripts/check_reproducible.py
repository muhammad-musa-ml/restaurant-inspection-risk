"""Re-run the pipeline into a temporary folder and compare it with the committed results and models.

    python scripts/check_reproducible.py                    # exact comparison
    python scripts/check_reproducible.py --tolerance 1e-6   # allow floating-point differences between machines

Ignored on purpose: the run metadata (time, code revision) and the model card's creation time.
Exit code 0 when every compared file matches.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inspection_risk import runinfo  # noqa: E402

IGNORED_KEYS = {"meta", "created_utc", "code_git_head"}


def differences(a, b, tolerance: float, path: str = "") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for key in sorted(set(a) | set(b)):
            if key in IGNORED_KEYS:
                continue
            if key not in a or key not in b:
                out.append(f"{path}/{key}: present in only one")
            else:
                out += differences(a[key], b[key], tolerance, f"{path}/{key}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} vs {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in differences(x, y, tolerance, f"{path}[{i}]")]
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if a == b or (tolerance and math.isclose(a, b, rel_tol=tolerance, abs_tol=tolerance)):
            return []
        return [f"{path}: {a} vs {b}"]
    return [] if a == b else [f"{path}: {a!r} vs {b!r}"]


def compare_json(committed: Path, fresh: Path, tolerance: float) -> list[str]:
    return differences(json.loads(committed.read_text(encoding="utf-8")),
                       json.loads(fresh.read_text(encoding="utf-8")), tolerance)


def same_text(committed: Path, fresh: Path) -> bool:
    """Model files are gzip-compressed text: compare the text, with LF line endings."""
    return runinfo.read_gzip_text(committed).replace("\r\n", "\n") == runinfo.read_gzip_text(fresh).replace("\r\n", "\n")


def compare_csv(committed: Path, fresh: Path, tolerance: float) -> list[str]:
    a, b = pd.read_csv(committed, dtype=str), pd.read_csv(fresh, dtype=str)
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return [f"shape or columns differ ({a.shape} vs {b.shape})"]
    out = []
    for column in a.columns:
        x, y = a[column], b[column]
        if x.equals(y):
            continue
        try:
            fx, fy = pd.to_numeric(x), pd.to_numeric(y)
            worst = float((fx - fy).abs().max())
            if tolerance and worst <= tolerance:
                continue
            out.append(f"{column}: largest difference {worst}")
        except (ValueError, TypeError):
            out.append(f"{column}: text differs")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--tolerance", type=float, default=0.0)
    args = parser.parse_args()
    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        results, models = Path(tmp) / "results", Path(tmp) / "models"
        done = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_pipeline.py"), "--results", str(results),
                               "--models", str(models)], cwd=ROOT, capture_output=True, text=True)
        if done.returncode != 0:
            print(done.stderr[-2000:])
            return 1
        compared = 0
        for fresh in sorted(results.iterdir()):
            committed = ROOT / "results" / fresh.name
            if not committed.exists():
                print(f"MISSING in repo: results/{fresh.name}")
                failures += 1
                continue
            if fresh.suffix == ".json":
                problems = compare_json(committed, fresh, args.tolerance)
            else:
                problems = compare_csv(committed, fresh, args.tolerance)
            compared += 1
            print(("OK   " if not problems else "DIFF ") + f"results/{fresh.name}")
            for line in problems[:5]:
                print("     ", line)
            failures += bool(problems)
        for fresh in sorted(models.iterdir()):
            committed = ROOT / "models" / fresh.name
            problems = (compare_json(committed, fresh, args.tolerance) if fresh.suffix == ".json"
                        else ([] if same_text(committed, fresh) else ["text differs"]))
            compared += 1
            print(("OK   " if not problems else "DIFF ") + f"models/{fresh.name}")
            for line in problems[:5]:
                print("     ", line)
            failures += bool(problems)
    print(f"compared {compared} files, {failures} with differences (tolerance {args.tolerance})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
