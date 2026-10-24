"""Fill README.md and docs/analysis.md from scripts/templates/ with the numbers in results/.

    python scripts/render_docs.py            # rewrite README.md and docs/analysis.md
    python scripts/render_docs.py --check    # exit 1 if either file differs from a fresh render

Every figure in those two documents is a {{placeholder}} filled here from a results file, so no number in
the prose is typed by hand. tests/test_docs.py checks that the committed files equal a fresh render and
that the templates hold no digits outside code, links and placeholders.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "scripts" / "templates"
OUTPUTS = {"README.md.tmpl": ROOT / "README.md", "analysis.md.tmpl": ROOT / "docs" / "analysis.md"}
PLACEHOLDER = re.compile(r"\{\{([a-z0-9_]+)\}\}")

MODELS = (("base_rate", "No model (random order)"), ("previous_result_rule", "Previous-result rule"),
          ("logistic", "Logistic regression"), ("gbm", "Boosted trees (LightGBM)"))
FEATURE_TEXT = {
    "cuisine_prior_rate": "share of the cuisine's earlier initial inspections that were hits (smoothed)",
    "prev_ci_score": "inspection score at the restaurant's previous initial inspection",
    "last_score": "score at the restaurant's latest earlier inspection of any kind",
    "cuisine_group": "the cuisine (the most common ones, the rest as one group)",
    "zip_prior_rate": "share of the ZIP code's earlier initial inspections that were hits (smoothed)",
    "prev_ci_n_crit": "critical violations at the previous initial inspection",
    "prev_ci_n_viol": "violations of any kind at the previous initial inspection",
}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def pct(x: float, d: int = 1) -> str:
    return f"{x * 100:.{d}f}%"


def stat_pct(s: dict) -> str:
    return f"{pct(s['median'])} ({s['min'] * 100:.1f}-{s['max'] * 100:.1f})"


def stat_num(s: dict, d: int) -> str:
    return f"{s['median']:.{d}f} ({s['min']:.{d}f}-{s['max']:.{d}f})"


def month_name(ym: str) -> str:
    names = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
             "November", "December"]
    return f"{names[int(ym[5:7]) - 1]} {ym[:4]}"


def table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def build_values(root: Path = ROOT) -> dict[str, str]:
    r = root / "results"
    summary, profile = load(r / "summary.json"), load(r / "data_profile.json")
    manifest, card = load(root / "data" / "MANIFEST.json"), load(root / "models" / "model_card.json")
    matched, timing = load(r / "drift_matched_history.json"), load(r / "timing_batch_scoring.json")
    weekly = pd.read_csv(r / "weekly_test.csv")
    importance = pd.read_csv(r / "feature_importance.csv")
    monthly = pd.read_csv(r / "monthly_calibration.csv")
    psi = pd.read_csv(r / "drift_monthly.csv")
    v: dict[str, str] = {}

    # ---- data ------------------------------------------------------------------------------------
    snap, fetch = profile["snapshot"], manifest["fetch"]
    v["rows_total"] = f"{manifest['files']['inspections.csv.gz']['rows']:,}"
    v["fetched_date"] = fetch["started_utc"][:10]
    v["fetch_start"], v["fetch_end"] = fetch["started_utc"][11:16], fetch["finished_utc"][11:16]
    v["publisher_updated"] = f"{fetch['api_rows_updated_at_utc'][:10]} {fetch['api_rows_updated_at_utc'][11:16]} UTC"
    v["restaurants_n"], v["events_n"] = f"{snap['restaurants_with_a_real_inspection']:,}", f"{snap['inspection_events']:,}"
    v["placeholder_rows"] = f"{snap['placeholder_rows_never_inspected']:,}"
    v["first_event_date"], v["last_event_date"] = snap["earliest_inspection_date"], snap["latest_inspection_date"]
    vis = profile["visibility"]
    v["floor_date"], v["max_span"] = vis["history_floor_date"], f"{vis['max_span_days_between_a_restaurants_earliest_and_latest_record']:,}"
    v["first_full_year_date"] = vis["first_date_with_min_window"]
    ini = profile["initial_cycle_inspections_since_floor"]
    v["initial_since_floor"] = f"{ini['inspections']:,}"
    v["gap_median"] = f"{ini['gap_days_between_consecutive_initial_inspections']['p50']:.0f}"
    target = profile["target_choice_from_training_window_only"]
    v["train_ge1"], v["train_ge2"] = pct(target["share_at_least_1_critical"]), pct(target["share_at_least_2_critical"])
    v["train_ge3"], v["train_ge4"] = pct(target["share_at_least_3_critical"]), pct(target["share_at_least_4_critical"])
    v["lift_ceiling"] = f"{1 / target['share_at_least_1_critical']:.2f}"
    v["n_features"] = str(len(card["numeric_columns"]) + len(card["categorical_columns"]))
    v["n_families"], v["n_cuisines"] = str(len(card["families"])), str(len(card["cuisine_levels"]))
    v["family_list"] = ", ".join(f"`{f}`" for f in card["families"])
    setup = summary["setup"]
    v["n_seeds"] = str(len(setup["seeds"]))
    v["seed_list"] = ", ".join(str(s) for s in setup["seeds"])
    v["capacity"] = str(setup["capacity_per_week"])
    v["pool_mean"] = f"{setup['mean_pool_per_week_test']:.0f}"
    v["capacity_share"] = pct(setup["capacity_share_of_test_pool"], 0)
    rows_all = 0
    window_rows = []
    for key, name in (("train", "train"), ("valid", "validation"), ("test", "test"), ("post", "after the test")):
        w, p = setup["windows"][key], profile["splits"][key]
        rows_all += w["rows"]
        v[f"{key}_rows"], v[f"{key}_weeks"] = f"{w['rows']:,}", str(w["weeks"])
        v[f"{key}_first"], v[f"{key}_last"] = w["first_date"], w["last_date"]
        v[f"{key}_pos"] = pct(w["positive_share"])
        v[f"{key}_prev_visible"] = pct(p["share_with_previous_initial_inspection_visible"], 0)
        v[f"{key}_mean_week"] = f"{p['mean_rows_per_week']:.1f}"
        window_rows.append([name, w["first_date"], w["last_date"], f"{w['rows']:,}", str(w["weeks"]), pct(w["positive_share"])])
    v["modelled_rows"] = f"{rows_all:,}"
    v["windows_table"] = table(["window", "first inspection", "last inspection", "inspections", "weeks", "with a hit"], window_rows)

    # ---- headline at capacity ---------------------------------------------------------------------
    rank = summary["ranking_at_capacity"]["test"]
    head_rows = []
    for key, label in MODELS:
        m = rank[key]
        head_rows.append([label, stat_pct(m["precision"]), stat_pct(m["recall"]), stat_num(m["lift"], 2), stat_num(m["hits_per_week"], 1)])
    v["headline_table"] = table(["model", "precision", "recall", "lift", "found per week"], head_rows)
    for key, short in (("gbm", "gbm"), ("previous_result_rule", "rule"), ("logistic", "logistic"), ("base_rate", "base")):
        m = rank[key]
        v[f"{short}_precision"], v[f"{short}_recall"] = pct(m["precision"]["median"]), pct(m["recall"]["median"])
        v[f"{short}_lift"], v[f"{short}_hits"] = f"{m['lift']['median']:.2f}", f"{m['hits_per_week']['median']:.1f}"
        v[f"{short}_precision_range"] = f"{m['precision']['min'] * 100:.1f}-{m['precision']['max'] * 100:.1f}"
    v["gbm_ens_precision"] = pct(rank["gbm_5_seed_average"]["precision"])
    words = {1: "one", 2: "two", 3: "three", 4: "four"}
    v["target_words"] = f"at least {words[card['target']['min_critical_violations']]} critical violations"
    v["gain_hits_base"] = f"{rank['gbm']['hits_per_week']['median'] - rank['base_rate']['hits_per_week']['median']:.1f}"
    v["gain_hits_rule"] = f"{rank['gbm']['hits_per_week']['median'] - rank['previous_result_rule']['hits_per_week']['median']:.1f}"
    v["logistic_minus_rule"] = f"{(rank['logistic']['precision']['median'] - rank['previous_result_rule']['precision']['median']) * 100:.1f}"
    boot = summary["bootstrap_over_test_weeks"]
    b = boot["gbm_5_seed_average"]["precision"]
    v["boot_low"], v["boot_high"] = pct(b["p2.5"]), pct(b["p97.5"])
    v["ci_level"] = f"{97.5 - 2.5:.0f}%"
    v["boot_draws"] = f"{boot['gbm_5_seed_average']['draws']:,}"
    d = boot["gbm_minus_previous_result_rule"]
    v["diff_rule"] = f"{d['precision_difference']['median'] * 100:.1f}"
    v["diff_rule_low"], v["diff_rule_high"] = (f"{d['precision_difference'][k] * 100:.1f}" for k in ("p2.5", "p97.5"))
    v["ahead_rule"] = pct(d["share_of_resamples_a_ahead"], 0)
    d = boot["gbm_minus_logistic"]
    v["diff_logistic"] = f"{d['precision_difference']['median'] * 100:.1f}"
    v["diff_logistic_low"], v["diff_logistic_high"] = (f"{d['precision_difference'][k] * 100:.1f}" for k in ("p2.5", "p97.5"))
    v["ahead_logistic"] = pct(d["share_of_resamples_a_ahead"], 0)
    n = len(weekly)
    v["wk_total"] = str(n)
    v["wk_gbm_gt_base"] = str(int(((weekly["hits_gbm"] / weekly["visits"]) > (weekly["positives"] / weekly["pool"])).sum()))
    v["wk_gbm_gt_rule"] = str(int((weekly["hits_gbm"] > weekly["hits_previous_result_rule"]).sum()))
    v["wk_gbm_gt_logistic"] = str(int((weekly["hits_gbm"] > weekly["hits_logistic"]).sum()))
    v["wk_logistic_gt_gbm"] = str(int((weekly["hits_logistic"] > weekly["hits_gbm"]).sum()))
    v["wk_pool_min"], v["wk_pool_max"] = str(int(weekly["pool"].min())), str(int(weekly["pool"].max()))
    weekly_precision = weekly["hits_gbm"] / weekly["visits"]
    v["wk_gbm_min"], v["wk_gbm_max"] = pct(weekly_precision.min(), 0), pct(weekly_precision.max(), 0)

    # ---- model selection ----------------------------------------------------------------------------
    sel = summary["model_selection"]
    v["leaves"], v["min_rows"] = str(sel["chosen"]["params"]["num_leaves"]), str(sel["chosen"]["params"]["min_child_samples"])
    v["grid_points"] = str(len(sel["grid_on_validation_log_loss"]))
    v["iters_min"], v["iters_max"] = str(min(sel["best_iterations_by_seed"])), str(max(sel["best_iterations_by_seed"]))

    # ---- calibration and the fixed threshold ------------------------------------------------------------
    cal = summary["calibration"]
    method = cal["chosen_method"]
    block = cal["windows"]["test"]["gbm_5_seed_average"]
    v["calibrator"] = method.capitalize()
    v["ece_raw"], v["ece_scaled"] = f"{block['before']['ece']:.3f}", f"{block[method]['ece']:.3f}"
    v["brier_raw"], v["brier_scaled"] = f"{block['before']['brier']:.4f}", f"{block[method]['brier']:.4f}"
    test_months = monthly[monthly["split"] == "test"]
    peak = test_months.loc[test_months["observed_rate"].idxmax()]
    v["peak_month"], v["peak_observed"], v["peak_raw"] = month_name(peak["month"]), pct(peak["observed_rate"]), pct(peak["mean_raw_probability"])
    v["test_obs_first"], v["test_obs_last"] = pct(test_months["observed_rate"].iloc[0]), pct(test_months["observed_rate"].iloc[-1])
    v["test_raw_min"], v["test_raw_max"] = pct(test_months["mean_raw_probability"].min()), pct(test_months["mean_raw_probability"].max())
    op = summary["operating_threshold"]["gbm_5_seed_average"]
    v["thr_value"] = f"{op['threshold']:.3f}"
    v["thr_visits_mean"], v["thr_visits_min"], v["thr_visits_max"] = f"{op['visits_per_week']:.0f}", str(op["visits_per_week_min"]), str(op["visits_per_week_max"])
    v["thr_precision"], v["thr_recall"], v["thr_hits"] = pct(op["precision"]), pct(op["recall"]), f"{op['hits_per_week']:.1f}"
    v["thr_valid_fraction"] = pct(op["validation_flag_fraction"], 0)

    # ---- time split vs random splits ------------------------------------------------------------------------
    cmp = summary["random_vs_time_split"]
    schemes = (("time_split_test_window", "Time split (the test window)"), ("random_by_restaurant", "Random split by restaurant"),
               ("random_by_inspection", "Random split by inspection"))
    v["split_table"] = table(["scheme", "AUC", "lift at the same share of the pool"],
                             [[label, stat_num(cmp[k]["auc"], 3), stat_num(cmp[k]["lift"], 2)] for k, label in schemes])
    v["auc_time"], v["auc_rest"], v["auc_insp"] = (f"{cmp[k]['auc']['median']:.3f}" for k, _ in schemes)
    v["lift_time"], v["lift_rest"], v["lift_insp"] = (f"{cmp[k]['lift']['median']:.2f}" for k, _ in schemes)

    # ---- ablations, embargo, secondary target ---------------------------------------------------------------------
    ab = summary["ablations"]
    v["abl_nohist_precision"], v["abl_nohist_lift"] = pct(ab["no_history_features"]["precision"]["median"]), f"{ab['no_history_features']['lift']['median']:.2f}"
    v["abl_nofloor_precision"], v["abl_nofloor_lift"] = pct(ab["history_not_floored"]["precision"]["median"]), f"{ab['history_not_floored']['lift']['median']:.2f}"
    v["n_history_cols"] = str(len(ab["dropped_columns_in_no_history"]))
    for days in ("30", "60", "90"):
        v[f"emb{days}"] = pct(summary["history_embargo_days"][days]["precision"]["median"])
    sec = summary["secondary_target"]
    v["sec_base"], v["sec_gbm_precision"] = pct(sec["base_rate_test"]), pct(sec["gbm"]["precision"]["median"])
    v["sec_rule_precision"], v["sec_gbm_lift"] = pct(sec["previous_result_rule"]["precision"]["median"]), f"{sec['gbm']['lift']['median']:.2f}"
    v["sec_ceiling"] = f"{sec['lift_ceiling_at_capacity']:.2f}"

    # ---- drift ---------------------------------------------------------------------------------------------------------------
    feat = psi[psi["feature"] != "__prediction__"]
    v["drift_total"] = str(len(feat))
    for status in ("major", "moderate", "stable"):
        v[f"drift_{status}"] = str(int((feat["status"] == status).sum()))
    v["pred_psi_max"] = f"{psi.loc[psi['feature'] == '__prediction__', 'psi'].max():.3f}"
    dr = summary["drift"]
    v["noise_median"], v["noise_max"] = f"{dr['noise_floor']['median_over_features_and_seeds']:.4f}", f"{dr['noise_floor']['max_over_features_and_seeds']:.3f}"
    v["noise_size"] = f"{dr['noise_floor']['sample_size']:,}"
    v["psi_moderate"], v["psi_major"] = str(dr["cutoffs"]["moderate"]), str(dr["cutoffs"]["major"])
    v["match_days"] = re.search(r"(\d+) days", matched["rule"]).group(1)
    v["m_all_major"], v["m_all_total"] = str(matched["all_rows"]["all"]["major"]), str(matched["all_rows"]["feature_windows"])
    v["m_major"], v["m_total"] = str(matched["matched"]["all"]["major"]), str(matched["matched"]["feature_windows"])
    prev, hist = "previous initial inspection", "earlier history"
    v["m_prev_from"], v["m_prev_to"] = str(matched["all_rows"]["by_group"][prev]["major"]), str(matched["matched"]["by_group"][prev]["major"])
    v["m_prev_total"] = str(matched["matched"]["by_group"][prev]["feature_windows"])
    v["m_hist_from"], v["m_hist_to"] = str(matched["all_rows"]["by_group"][hist]["major"]), str(matched["matched"]["by_group"][hist]["major"])
    v["m_hist_total"] = str(matched["matched"]["by_group"][hist]["feature_windows"])
    v["match_test_share"] = pct(matched["windows"]["test"]["matched_share"], 0)
    v["match_post_rows"] = str(matched["windows"]["post"]["rows_matched"])
    v["match_post_noise_max"] = f"{matched['sampling_noise_floor_matched']['post']['max']:.2f}"
    group_rows = [[g.capitalize(), f"{matched['all_rows']['by_group'][g]['major']} of {matched['all_rows']['by_group'][g]['feature_windows']}",
                   f"{matched['matched']['by_group'][g]['major']} of {matched['matched']['by_group'][g]['feature_windows']}"]
                  for g in matched["all_rows"]["by_group"]]
    v["drift_group_table"] = table(["feature group", "major shifts, all inspections", "major shifts, same visible history"], group_rows)

    # ---- importance ---------------------------------------------------------------------------------------------------------------
    top = importance.head(5)
    v["importance_table"] = table(["column", "what it is", "share of gain, median (min-max)"], [
        [f"`{row.feature}`", FEATURE_TEXT.get(row.feature, row.feature), f"{row.gain_share_median * 100:.1f}% ({row.gain_share_min * 100:.1f}-{row.gain_share_max * 100:.1f})"]
        for row in top.itertuples()])
    v["imp1"], v["imp1_share"] = f"`{top.iloc[0]['feature']}`", pct(top.iloc[0]["gain_share_median"])
    v["imp_top3_share"] = pct(top.head(3)["gain_share_median"].sum(), 0)
    v["imp_zero_n"] = str(int((importance["gain_share_median"] == 0).sum()))

    # ---- timing --------------------------------------------------------------------------------------------------------------------
    e2e, st = timing["seconds_end_to_end_fresh_process"], timing["seconds_by_stage_in_one_process"]
    v["t_month"], v["t_inspections"] = timing["month"], f"{timing['inspections_scored']:,}"
    v["t_reps"] = str(timing["repetitions"])
    v["t_e2e"] = stat_num(e2e, 1)
    v["t_read"], v["t_events"] = stat_num(st["read_snapshot"], 1), stat_num(st["build_inspection_table"], 1)
    v["t_models"], v["t_feat"] = stat_num(st["load_models"], 2), stat_num(st["features_and_scoring"], 1)
    v["t_when"] = timing["meta"]["timestamp_utc"]

    # ---- environment -----------------------------------------------------------------------------------------------------------------
    meta = summary["meta"]
    v["run_date"], v["machine"], v["python_version"] = meta["timestamp_utc"][:10], meta["machine"], meta["python"]
    v["tools_table"] = table(["package", "version"], [[f"`{k}`", f"`{ver}`"] for k, ver in meta["packages"].items()])
    return v


def render(template: str, values: dict[str, str]) -> str:
    missing = sorted(set(PLACEHOLDER.findall(template)) - set(values))
    if missing:
        raise KeyError(f"placeholders without a value: {missing}")
    return PLACEHOLDER.sub(lambda m: values[m.group(1)], template)


def render_all(root: Path = ROOT) -> dict[Path, str]:
    values = build_values(root)
    return {out: render((TEMPLATES / name).read_text(encoding="utf-8"), values) for name, out in OUTPUTS.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = render_all()
    stale = [out for out, text in rendered.items() if not out.exists() or out.read_text(encoding="utf-8") != text]
    if args.check:
        for out in stale:
            print(f"differs from a fresh render: {out.relative_to(ROOT)}")
        return 1 if stale else 0
    for out, text in rendered.items():
        out.write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
