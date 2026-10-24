"""Facts recorded beside every results file: when, on what, with which versions."""
from __future__ import annotations

import datetime as dt
import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

PACKAGES = ("pandas", "numpy", "scikit-learn", "lightgbm", "matplotlib", "scipy")


def write_json(path: Path | str, obj) -> None:
    """JSON with LF line endings and a final newline, whatever the platform (so committed files do not
    change with the machine that wrote them)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8", newline="\n")


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
