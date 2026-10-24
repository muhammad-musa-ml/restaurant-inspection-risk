# Design notes

Written before the model results existed. It says what I decided and why, and what I will measure.
Numbers below are copied from `results/data_profile.json` (written by `scripts/profile_data.py`) and
`data/MANIFEST.json`; anything the modelling will produce is marked "not measured yet".

## The question

A health inspection team has a fixed number of visits it can put first each week. Which of the
inspections due that week should it put first, if the aim is to find food-safety problems?

I treat it as a ranking problem inside a week: given the restaurants whose initial inspection falls in a
week, order them by the chance that the inspection finds a pattern of critical violations, and see
what the top of the list buys at a stated capacity. The team's real schedule is not in the data. In
the backtest the week's pool is the set of initial inspections that actually happened that week.

## Data

NYC Department of Health and Mental Hygiene (DOHMH), "Restaurant Inspection Results", on NYC Open Data
(dataset `43nn-pn8j`, https://data.cityofnewyork.us/d/43nn-pn8j). Fetched once without an account or
token on 2026-10-24 (06:52-06:55 UTC); the publisher's own last update of the rows was 2026-10-23
22:05 UTC. 294,406 citation rows, one per violation (or one row for an inspection that cited nothing).
`scripts/fetch_data.py` requests only the columns the analysis uses: no restaurant name, address, building
number or phone number, so the published file has no per-business contact details. `data/MANIFEST.json`
holds the fetch times, row counts and SHA-256.

The publisher's description of the dataset says three things that change what can be done with it:

1. **Only restaurants active on the pull date are included.** A restaurant that closed after an
   inspection is missing from that inspection's month. Older months therefore lack the restaurants that
   later closed.
2. **Only violations "sustained or not yet adjudicated" are included.** A citation dismissed at a
   hearing is removed from old inspections but still present on recent ones. The last months' counts
   are not final, and I have one snapshot, so I cannot measure how much they change.
3. **Only inspections up to three years before each restaurant's most recent inspection are included.**
   I checked this against the data: the longest span between a restaurant's earliest and latest record
   is 1,095 days, and no restaurant exceeds it (`visibility` in the profile).

The third one matters most for leakage. How far back a restaurant's visible history reaches depends on
when it was last inspected, which for an early inspection is information from the future. Restaurants
that get inspected often (usually because of poor results) lose their older records sooner. A feature such
as "number of earlier inspections" would then carry future information without any row of the future
appearing in it. The fix is a floor date that is the same for every restaurant: records before
2023-10-22 (the latest inspection date in the file, 2026-10-21, minus 1,095 days) are ignored for all
restaurants, because from that date on every active restaurant's records are complete.

## What is ranked, and what counts as a hit

**Population:** initial cycle inspections (`Cycle Inspection / Initial Inspection`), the routine
inspection a team can choose to bring forward. Re-inspections, compliance visits, pre-permit
inspections and other programs follow from an earlier result or from an application, so they are not
ranked; they are used as history.

**Target:** the inspection cites at least **two** critical violations. I first looked at "at least
one critical violation", the wording of the question. In the training window 88.5% of initial inspections
cite at least one, so even a perfect ranking could not lift precision above 1/0.885 = 1.13 times the
base rate, and a team cannot prioritise on something that nearly every inspection finds. In the same
window 49.2% cite at least two, 28.5% at least three, 14.8% at least four (`target_choice_from_training_window_only`).
I chose two because it splits the training window almost evenly and still means more than one critical
citation. The first-posed target (at least one) is reported alongside, with the ceiling stated. The cut-off
was chosen from the training window only. `TARGET_MIN_CRITICAL` in `inspection_risk/config.py` changes it.

## Split by time

Modelled inspections start on Monday 2024-10-21, the first date with a full year (365 days) of visible
history after the floor. All boundaries are Mondays and the intervals are half-open:

| window | from | to (exclusive) | inspections | weeks | mean per week |
|---|---|---|---|---|---|
| train | 2024-10-21 | 2025-10-06 | 11,655 | 50 | 233.1 |
| validation | 2025-10-06 | 2026-02-02 | 4,668 | 17 | 274.6 |
| test | 2026-02-02 | 2026-08-03 | 8,124 | 26 | 312.5 |
| after the test (monitoring only) | 2026-08-03 | 2026-10-05 | 3,199 | 9 | 355.4 |

I set these dates from the monthly row counts, before looking at any label rate by window. The window
after the test is never used to fit, tune or choose anything; it is where the batch scoring script and the
drift check are demonstrated on months the model has not seen. Inspections after 2026-10-05 are left out
(a partial week, and the least final labels).

Two properties of this data will show up as drift, and I report them instead of hiding them:

- **The share of rows whose previous initial inspection is visible rises from 44.6% in the training
  window to 79.6% in the test window**, because the visible history grows as the windows move later
  (the median gap between two initial inspections of one restaurant is 492 days, so a short window often
  misses the previous one). The test window looks most like a deployment with the publisher's full history.
- The share of inspections with at least two critical violations differs between windows (49.2%, 45.1%,
  51.1%, 53.5% for train, validation, test and after-test). Survivorship (point 1) and unadjudicated recent
  citations (point 2) can both push it, and I cannot separate them with one snapshot.

## Features

Every feature for an inspection on date d uses records dated before d only, and on or after the floor.
Four rules, tested by truncating the data and comparing (`tests/test_features_leakage.py`):

1. Strictly before d: the day itself is excluded whatever its type, and an inspection's own outcome
   columns are never read.
2. The common floor above.
3. History as "nearest previous" values and rates, not as counts that grow with the window: the previous
   initial inspection's critical count, violation count, score and closure flag; days since it; the
   previous inspection of any type; inspections in the last 365 days; whether the restaurant was ever
   closed; for each violation family (first two characters of the code: temperature, contamination and
   pests, facilities, hygiene, and so on) the share of earlier inspection days that cited it and whether the
   previous initial inspection did.
4. Neighbourhood (ZIP code) and cuisine rates are smoothed shares of earlier initial inspections with the
   target, so they know nothing about the day's results.

Static columns are borough, cuisine (the 25 most common in the training window, the rest "Other"),
month and weekday. The snapshot gives one ZIP code, cuisine and borough per restaurant (none of the
22,606 restaurants with two or more visible events has more than one), so these are current values, not
values at the time of an old inspection.

I do not use the inspection's own score, grade, action, violations or any date after it.

## Models and baselines

- **Base rate**: every inspection gets the same score, so the order inside a week is random.
- **Previous-result rule**: put first the restaurants whose previous initial inspection cited at least
  two critical violations (previous inspection of any type if there is none); random order inside each
  group.
- **Logistic regression** on the same features (median imputation with missing indicators, scaling, one-hot).
- **Gradient-boosted trees** (LightGBM) with early stopping on the validation log loss, and a twelve-point
  grid (leaves 3/7/15/31, minimum rows per leaf 20/80/200) chosen on validation log loss. The first run
  used leaves 7/15/31 and minimum rows 20/80; its best point sat in a corner of that grid, so I widened the
  grid to bracket it before reading any test-window number.

Anything with randomness is run with seeds 0-4 and reported as the median and the range. The random tie
breaks of the two rule baselines and the boosting seed (row and column sampling) are the randomness.
A resampling of the test weeks gives a second interval, because the seeds do not capture which weeks
happened to be in the test window.

## Calibration

Platt scaling and isotonic regression are fitted on the validation predictions. The choice between them is
made on validation only, by fitting on the first half of the validation weeks and scoring the second half.
ECE (10 equal-width bins) and Brier score are reported before and after on the test window. Both the early
stopping and the calibrator use the validation window; the test window is used for neither.

## Capacity and the operating threshold

Stated assumption, mine, not from the data: the team can put **100 inspections a week** first. The
test window pools about 312 inspections a week, so that is about one in three. Two rules are reported:

- *top 100 per week*: rank each week's pool and take the first 100. This fills the capacity exactly.
- *a fixed probability threshold*: the calibrated probability cut-off that flags 100 a week on average over
  the validation weeks, applied unchanged to the test weeks. This shows how well a threshold set in
  advance carries over, including how many inspections it flags in a test week.

Both give precision (hits among flagged), recall (hits among all positives in the pool) and lift
(precision over the pool's base rate). A sweep from 25 to 300 a week is also computed and drives the dashboard.

## Random split, for comparison

The same rows are also split at random into the same sizes as the three windows, five seeds, and the
same model is trained and scored. The difference to the time split is how much a random split would have
overstated. Not measured yet.

## Drift

PSI per feature and for the prediction distribution, month by month, with bins fixed on a reference window
(the training window for features, the validation window for predictions, because training-window
predictions are in-sample). PSI between random same-size samples of the reference gives the floor sampling
noise alone produces at one month's size. The usual cut-offs (0.10 moderate, 0.25 major) are a rule of
thumb, not something derived here.

## Batch scoring

`scripts/score_month.py --month YYYY-MM` builds the features for that month's initial inspections from
records before each inspection's date, scores them with the saved models and calibrator, writes the
ranked list with the weekly top-100 flag, and writes the month's drift report. A test checks that the
scores do not change when the month's own outcome columns are blanked.

## Dashboard and CI

A static page in `docs/` (plain HTML and JavaScript reading one JSON file written from the results; no build
step, no server, no other account) published with GitHub Pages from this repository. It shows aggregates
only, never a named restaurant. GitHub Actions runs the tests on every push.

## What I am not doing

- No restaurant names in any output: a ranking of named businesses by "likely to have violations" is not
  something to publish from public data.
- No claim about how much an inspection team would find, only about ranking inside a week's pool.
- No causal claim: the features predict the outcome; they do not say why restaurants have violations.
