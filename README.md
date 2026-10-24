# restaurant-inspection-risk

A backtest of one question: when a health department can only put some of its routine restaurant
inspections first each week, which ones should they be? It takes New York City's public restaurant
inspection results, ranks each week's routine inspections by how likely each is to cite
at least two critical violations, and measures what the top of the list buys compared with simpler rules, on weeks the
model never saw. Around that sit the parts I wanted to practise: features that use only records from
before the inspection (the tests try to break that), probabilities that are checked instead of trusted,
drift monitoring, a script that scores a new month, and a small static dashboard.

I built it to practise evaluating a model honestly on messy real data. Most of the work went into the
ways such an evaluation goes wrong (leaking the future, choosing a target that nearly every row meets,
believing a random split) and into reporting what came out even where it was unflattering.

**The data is real and public; only the small arrays in the unit tests are synthetic.** It is the
Department of Health and Mental Hygiene's [Restaurant Inspection Results](https://data.cityofnewyork.us/d/43nn-pn8j)
on NYC Open Data, downloaded once without an account on 2026-10-24 (294,406 rows, one per
violation cited). The script asks only for the columns it uses, so there are no restaurant names,
addresses or phone numbers in this repository, and the dashboard shows aggregates only.

## What it does

```
data/inspections.csv.gz ─> features (records dated before each inspection only)
    ─> boosted trees (one model per seed), logistic regression, two baselines
    ─> calibration ─> each test week ranked, top of the list scored ─> results/, models/
    ─> docs/data/dashboard.json ─> docs/index.html (GitHub Pages)
```

| script | what it does |
|---|---|
| `scripts/fetch_data.py` | downloads the dataset once, drops names, addresses, phone numbers and coordinates, writes `data/inspections.csv.gz` and `data/MANIFEST.json` (times, row counts, hashes) |
| `scripts/profile_data.py` | measures what the publisher's rules leave in the file; `results/data_profile.json` is the basis of the design notes |
| `scripts/run_pipeline.py` | every measurement: models, baselines, calibration, capacity sweep, random-split comparison, ablations, drift; writes `results/` and `models/` |
| `scripts/drift_matched_history.py` | repeats the drift comparison on inspections that see the same history in every window |
| `scripts/score_month.py` | scores one month's initial inspections from the saved models, with a drift report for that month |
| `scripts/time_batch.py` | times `score_month.py` |
| `scripts/check_reproducible.py` | reruns the pipeline into a temporary folder and compares every file with the committed one |
| `scripts/build_dashboard_data.py`, `scripts/make_figures.py`, `scripts/render_docs.py` | write the dashboard's data file, the memo's figures, and this README and the memo from `results/` |

Why the target is two critical violations and not one, how the time windows were chosen, and what the
publisher's rules do to the data: [docs/design.md](docs/design.md). The same findings written for a
non-technical reader: [docs/analysis.md](docs/analysis.md).

## Results

The test window is 26 weeks of routine ("initial cycle") inspections, 2026-02-02 to
2026-08-01, 8,124 inspections, 51.1% of them hits. A hit is an inspection that cites
at least two critical violations. The capacity is my assumption, not something in the data: 100 inspections put
first per week, which is about 32% of an average week's pool of 312. Median of
5 seeds, with the range (min-max) in brackets:

| model | precision | recall | lift | found per week |
|---|---|---|---|---|
| No model (random order) | 51.4% (50.1-52.0) | 32.2% (31.4-32.5) | 1.01 (0.98-1.02) | 51.4 (50.1-52.0) |
| Previous-result rule | 60.2% (60.0-60.7) | 37.7% (37.6-38.0) | 1.18 (1.17-1.19) | 60.2 (60.0-60.7) |
| Logistic regression | 66.5% (66.5-66.5) | 41.6% (41.6-41.6) | 1.30 (1.30-1.30) | 66.5 (66.5-66.5) |
| Boosted trees (LightGBM) | 66.0% (66.0-66.4) | 41.4% (41.3-41.6) | 1.29 (1.29-1.30) | 66.0 (66.0-66.4) |

What this says, and what it does not:

- **Ranking beats random order by a modest amount.** The boosted model finds 14.6 more hits a
  week than a random order. Over the test weeks resampled 2,000 times, its precision is between
  63.5% and 68.3% (95% interval). It was ahead of that week's random baseline in
  26 of 26 weeks.
- **Most of it comes from a one-line rule.** Putting first the restaurants whose previous routine
  inspection had at least two critical violations gets 60.2%. The boosted model is ahead of it by
  5.6 points (interval 3.8 to 7.2).
- **Boosting does not beat a logistic regression.** The difference is -0.4 points (interval
  -1.3 to 0.5), and the boosted model finished ahead in 19%
  of the resamples. The settings search picked the smallest trees on offer (2 leaves, at least
  500 inspections per leaf), which is the edge of the grid. I widened the grid twice, before reading any
  test number, and then stopped (see the design notes).
- **The scores rank better than they predict.** Calibrating with Platt scaling, fitted on the
  validation window, made the test-window calibration worse: expected calibration error 0.025 before and
  0.063 after, Brier score 0.2313 before and 0.2347 after. The share of hits was
  45.1% in the validation window and 51.1% in the test window, and the model's mean score does
  not follow it.
- **A fixed score cut-off does not hold the workload.** A cut-off of 0.495, set to flag 100 a
  week on average over the validation weeks, flagged 115 a week on average in the test window
  and between 65 and 175 in single weeks.
- **A random split would not have flattered the model on this data.** AUC 0.659 on the time split,
  0.659 on a random split by restaurant, 0.660 on a random split by inspection. I did not find out why.
- **Most feature drift alarms come from the publisher's window moving.** 299 of 360 feature-months
  show a major shift against the training window, but the score distribution stays stable (largest monthly
  index 0.023). Restricted to inspections with the same visible history in every window, the major
  shifts fall from 75 of 90 to 17 of 90 feature-windows. The columns that
  accumulate over the whole history still shift.

As first posed, the question asked about at least one critical violation. 88.5% of the training
weeks' routine inspections already cite one, so no ranking can lift precision by more than 1.13
times. On that target the boosted model's top 100 are hits 94.6% of the time against
88.5% for the pool; the numbers are in `results/summary.json` under `secondary_target`.

The capacity curves, the weekly view, the calibration chart and the drift chart are in the memo. The
[dashboard](https://muhammad-musa-ml.github.io/restaurant-inspection-risk/) has the same views, and the
capacity can be moved there.

### How these were measured

- **Windows.** Train, validation and test are consecutive Monday-aligned stretches, so the model is always
  scored on later weeks than it was trained or tuned on. The fourth window is only monitored.

| window | first inspection | last inspection | inspections | weeks | with a hit |
|---|---|---|---|---|---|
| train | 2024-10-21 | 2025-10-04 | 11,655 | 50 | 49.2% |
| validation | 2025-10-06 | 2026-01-31 | 4,668 | 17 | 45.1% |
| test | 2026-02-02 | 2026-08-01 | 8,124 | 26 | 51.1% |
| after the test | 2026-08-03 | 2026-10-03 | 3,199 | 9 | 53.5% |

- **Seeds.** The boosting seed (row and column sampling) and the random tie-breaks of the baselines were run
  with seeds 0, 1, 2, 3, 4; each cell above is the median with the lowest and highest. The seeds do not capture
  which weeks fell in the test window, which is what the resampling of weeks adds.
- **History.** The publisher keeps only inspections up to three years before each restaurant's latest, so
  how far back a restaurant's visible past reaches depends on what happens after an inspection. Every
  feature therefore ignores records before 2023-10-22 for every restaurant. Without that rule the
  boosted model's precision is 66.7%; without any history features, 61.8%.
  Dropping the most recent 30, 60 or 90 days of history to approximate hearing delays gives 66.2%,
  66.2% and 66.1%.
- **Features.** 30 columns: the previous routine inspection's score and violation counts, how
  long ago it was, how often the restaurant has been inspected lately, shares of earlier days that cited each
  of 7 violation families, smoothed hit rates for the cuisine and the ZIP code from earlier
  inspections only, borough, month, weekday and the cuisine. The largest share of the boosted model's gain is
  `cuisine_prior_rate` at 32.2%:

| column | what it is | share of gain, median (min-max) |
|---|---|---|
| `cuisine_prior_rate` | share of the cuisine's earlier initial inspections that were hits (smoothed) | 32.2% (29.3-34.8) |
| `prev_ci_score` | inspection score at the restaurant's previous initial inspection | 20.1% (19.6-21.8) |
| `last_score` | score at the restaurant's latest earlier inspection of any kind | 14.0% (13.8-14.7) |
| `cuisine_group` | the cuisine (the most common ones, the rest as one group) | 8.4% (6.9-11.1) |
| `zip_prior_rate` | share of the ZIP code's earlier initial inspections that were hits (smoothed) | 7.8% (7.6-8.0) |

- **Batch scoring.** Scoring 2026-09's 1,504 initial inspections from the saved models takes
  13.6 (13.0-13.8) s end to end in a fresh process (median of 5 runs, range in brackets). Inside one process,
  reading the snapshot takes 2.4 (2.4-2.5) s, building the inspection table 5.6 (5.3-6.5) s, loading the models
  0.03 (0.03-0.04) s, and building features and scoring 0.6 (0.6-0.8) s. Measured 2026-10-24 09:06 UTC on Windows 11 (10.0.26200), AMD64, wall
  clock. The machine was shared: the Docker containers running at the time (44) are listed in
  `results/timing_docker_ps.txt`, and processes outside Docker were not recorded. Every run is in
  `results/timing_batch_scoring.json`.

## Limits

- The file holds only restaurants that were active on the day I downloaded it, and only violations that
  were upheld or not yet heard. Older months miss restaurants that closed later, and the newest labels are
  not final. With one snapshot I cannot say how much either moves the results.
- The department's real schedule is not public, so "the week's pool" is the routine inspections that
  actually happened that week, and the capacity is assumed.
- Cuisine, ZIP code and borough are the current values, not the values on an old inspection's day.
- The share of inspections whose previous routine inspection is visible grows from 45%
  in the training window to 80% in the test window, because later windows see more of the
  publisher's three years. That is why the feature drift alarms are mostly not about restaurants changing.
- These are measurements on one city's data for one stretch of time. The features predict; they say nothing
  about why a restaurant has violations.

## Running it

The committed results were produced with Python 3.13.5. The pinned packages are in
`requirements.txt`:

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt        # on Windows: .venv\Scripts\pip
python scripts/run_pipeline.py                   # a minute or two; rewrites results/ and models/
python scripts/check_reproducible.py             # reruns into a temp folder and compares every file
python scripts/score_month.py --month 2026-09    # score one month, with its drift report
```

`data/inspections.csv.gz` is the snapshot every number here comes from. `scripts/fetch_data.py` downloads
a fresh one, but the publisher updates the dataset daily, so a fresh download changes every number.

The packages used for the committed results:

| package | version |
|---|---|
| `pandas` | `3.1.0` |
| `numpy` | `2.5.4` |
| `scikit-learn` | `1.9.1` |
| `lightgbm` | `4.7.0` |
| `matplotlib` | `3.11.2` |
| `scipy` | `1.18.1` |

## Tests

```
python -m pytest
```

The tests build small synthetic tables and also run the real feature code on the snapshot. The ones that
matter most: truncating the data at an inspection's date must not change that inspection's features;
blanking an inspection's own outcome columns must not change them either; and four deliberately leaky copies
of the feature builder (counting same-day records, skipping the history floor, reading the outcome, using the
future in a neighbourhood rate) must each be caught. The rest cover the metrics, the calibration choice, PSI,
the time windows, the batch scorer (it must reproduce the pipeline's scores for a month) and this README:
`tests/test_docs.py` checks that the README and the memo equal a fresh render from `results/`, and that
their templates contain no number typed by hand.

`.github/workflows/ci.yml` runs the same tests on GitHub Actions.

## Dashboard

`docs/index.html`, `docs/style.css` and `docs/app.js` are plain HTML, CSS and JavaScript with inline SVG, no
build step and no external requests. They read `docs/data/dashboard.json`, which
`scripts/build_dashboard_data.py` writes from `results/`; every figure on the page is computed in the browser
from that file. To look at it locally:

```
python -m http.server --directory docs
```

and open the address it prints. Opened directly from the file system the page shows a message instead of
the data, because browsers block reading a file that way. It is published with GitHub Pages from `docs/`:
[muhammad-musa-ml.github.io/restaurant-inspection-risk](https://muhammad-musa-ml.github.io/restaurant-inspection-risk/).

## Repository layout

```
inspection_risk/   the package: data, features, metrics, drift, modelling, pipeline, scoring
scripts/           fetch, profile, run, score a month, time, check, build the dashboard data and the docs
scripts/templates/ the README and memo with their numbers left as placeholders
data/              the snapshot (inspections.csv.gz), its manifest, the violation code list
models/            one boosted-tree model per seed (gzip-compressed text), the model card (calibrator, threshold, drift reference)
results/           summary.json, the tables, predictions for the validation, test and later windows, timing
docs/              design notes, memo, figures, and the dashboard (index.html, app.js, style.css, data/)
tests/             unit and integration tests
```
