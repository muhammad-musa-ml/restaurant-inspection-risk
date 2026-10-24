from pathlib import Path

import pytest

from inspection_risk import study as study_module

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "inspections.csv.gz"


@pytest.fixture(scope="session")
def root():
    return ROOT


@pytest.fixture(scope="session")
def study():
    """The real frozen snapshot, prepared once for the whole test session."""
    return study_module.prepare(DATA)
