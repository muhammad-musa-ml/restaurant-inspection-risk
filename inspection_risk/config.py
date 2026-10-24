"""Constants that define the study. Changing any of them changes the results; re-run the pipeline after."""
from __future__ import annotations

import datetime as dt

CYCLE_INITIAL = "Cycle Inspection / Initial Inspection"

# The publisher keeps, for each restaurant, only records dated within three years before that
# restaurant's most recent inspection (checked in results/data_profile.json: no restaurant spans more
# than 1,095 days). Every active restaurant's records are therefore complete only from
# (latest inspection date in the file) - 1,095 days on; scripts and tests derive that date from the data.
VISIBILITY_DAYS = 1095
# Modelled inspections need at least this many days of fully visible history before them.
MIN_WINDOW_DAYS = 365

# Target: the inspection cites at least this many critical violations. 87.5% of initial cycle inspections
# cite at least one, so "at least one" cannot be prioritised on; the cut-off is chosen from the training
# window only (docs/design.md). Results for the first-posed question (>= 1) are reported as well.
TARGET_MIN_CRITICAL = 2
SECONDARY_MIN_CRITICAL = 1

# Time split. Half-open intervals [start, end); every boundary is a Monday so a week never straddles two.
TRAIN_START = dt.date(2024, 10, 21)  # the first Monday with a full MIN_WINDOW_DAYS of visible history
VALID_START = dt.date(2025, 10, 6)
TEST_START = dt.date(2026, 2, 2)
POST_START = dt.date(2026, 8, 3)   # months after the test window: never used to fit, tune or choose anything
POST_END = dt.date(2026, 10, 5)    # the last full week of data ends Sunday 2026-10-04
NEW_MONTH = "2026-09"              # the month the batch scoring script is demonstrated on

# Stated assumption: the team can visit this many inspections per week ahead of the rest of the week's pool.
CAPACITY_PER_WEEK = 100
CAPACITY_SWEEP = tuple(range(10, 301, 10))  # the dashboard's capacity slider reads this grid

SEEDS = (0, 1, 2, 3, 4)
SMOOTHING = 20            # pseudo-count for the neighbourhood and cuisine prior rates
CUISINE_TOP = 25          # cuisines outside the most common 25 (by training rows) become "Other"
FAMILY_MIN_CITATIONS = 2000  # violation-code families (first two characters) with at least this many citations

# PSI cut-offs are the usual rule of thumb, not a derived quantity.
PSI_MODERATE = 0.10
PSI_MAJOR = 0.25
PSI_BINS = 10
ECE_BINS = 10
