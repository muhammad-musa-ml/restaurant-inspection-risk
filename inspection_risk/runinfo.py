"""Facts recorded beside every results file: when, on what, with which versions."""
from __future__ import annotations

import datetime as dt
import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import numpy as np

PACKAGES = ("pandas", "numpy", "scikit-learn", "lightgbm", "matplotlib", "scipy")


def _plain(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def _finite_or_none(obj):
    """Strict JSON has no NaN or Infinity (JSON.parse in a browser rejects them): an empty reliability bin,
    for example, has no mean, and is written as null."""
    if isinstance(obj, dict):
        return {key: _finite_or_none(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite_or_none(value) for value in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, np.ndarray):
        return _finite_or_none(obj.tolist())
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def write_json(path: Path | str, obj) -> None:
    """Strict JSON (no NaN) with LF line endings and a final newline, whatever the platform, so committed
    files do not change with the machine that wrote them. numpy numbers and arrays become plain ones."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_finite_or_none(obj), indent=2, default=_plain, allow_nan=False) + "\n"
    Path(path).write_text(text, encoding="utf-8", newline="\n")


def git_head(repo: Path) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True,
                             check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_meta(command: str, repo: Path | None = None) -> dict:
    """`command` is the command line as the reader would type it (relative paths only)."""
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    meta = {
        "command": command,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "machine": f"{platform.system()} {platform.release()} ({platform.version()}), {platform.machine()}",
        "python": sys.version.split()[0],
        "packages": versions,
    }
    if repo is not None:
        meta["code_git_head"] = git_head(repo)
    return meta
