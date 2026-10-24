"""Turn results/ into docs/data/dashboard.json, the only file the dashboard page reads.

    python scripts/build_dashboard_data.py

Everything in the output is copied or aggregated from results/ and models/model_card.json: medians and ranges
over the 5 seeds for the curves, the monthly PSI table, the reliability bins, the weekly series. No
restaurant name or address is read or written (the snapshot does not contain them).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inspection_risk import runinfo  # noqa: E402

LABELS = {
    "gbm": "Boosted model",
    "previous_result_rule": "Previous-result rule",
    "logistic": "Logistic regression",
    "base_rate": "No model (base rate)",
    "perfect_ranking": "Perfect ranking",
}


def curve_block(curve: pd.DataFrame) -> dict:
    out = {}
    capacities = sorted(curve["capacity"].unique().tolist())
    for model, frame in curve.groupby("model"):
        if model == "gbm_5_seed_average":
            continue
        block = {"label": LABELS[model]}
        for metric in ("precision", "recall", "lift", "hits_per_week"):
            wide = frame.pivot(index="capacity", columns="seed", values=metric).loc[capacities]
            block[metric] = wide.median(axis=1).round(6).tolist()
            block[metric + "_min"] = wide.min(axis=1).round(6).tolist()
            block[metric + "_max"] = wide.max(axis=1).round(6).tolist()
        out[model] = block
    return {"capacities": capacities, "models": out}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="docs/data/dashboard.json")
    args = parser.parse_args()
    results = Path(args.results)
    summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    card = json.loads((ROOT / "models" / "model_card.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "data" / "MANIFEST.json").read_text(encoding="utf-8"))
    curve = pd.read_csv(results / "capacity_curve_test.csv")
    weekly = pd.read_csv(results / "weekly_test.csv")
    psi = pd.read_csv(results / "drift_monthly.csv")
    months = pd.read_csv(results / "monthly_calibration.csv")
    importance = pd.read_csv(results / "feature_importance.csv")

    profile = json.loads((results / "data_profile.json").read_text(encoding="utf-8"))
    matched = json.loads((results / "drift_matched_history.json").read_text(encoding="utf-8"))
    coverage = {name: block["share_with_previous_initial_inspection_visible"]
                for name, block in profile["splits"].items()}
    setup = summary["setup"]
    cal = summary["calibration"]
    chosen = cal["chosen_method"]
    test_cal = cal["windows"]["test"]["gbm_5_seed_average"]
    window_names = {"valid": "validation", "test": "test", "post": "after the test"}

    month_order = months["month"].tolist()
    features = [f for f in psi["feature"].unique() if f != "__prediction__"]
    peak = psi[psi["feature"].isin(features)].groupby("feature")["psi"].max()
    features = sorted(features, key=lambda f: -peak[f])
    grid = psi.pivot(index="feature", columns="month", values="psi")
    matrix = [[None if pd.isna(grid.loc[f, m]) else round(float(grid.loc[f, m]), 5) for m in month_order]
              for f in features]
    prediction_row = [None if pd.isna(grid.loc["__prediction__", m]) else round(float(grid.loc["__prediction__", m]), 5)
                      for m in month_order]

    comparison = summary["random_vs_time_split"]
    dashboard = {
        "about": {
            "dataset": manifest["source"]["name"], "publisher": manifest["source"]["publisher"],
            "dataset_page": manifest["source"]["page"],
            "snapshot_fetched_utc": manifest["fetch"]["started_utc"],
            "publisher_rows_updated_utc": manifest["fetch"]["api_rows_updated_at_utc"],
            "citation_rows": manifest["files"]["inspections.csv.gz"]["rows"],
            "target": setup["target"],
            "windows": {window_names.get(k, k): {**v, "previous_initial_visible_share": coverage[k]}
                        for k, v in setup["windows"].items()},
            "repository": "https://github.com/muhammad-musa-ml/restaurant-inspection-risk",
        },
        "capacity": {"default": setup["capacity_per_week"], "mean_pool_per_week_test": setup["mean_pool_per_week_test"],
                     "share_of_test_pool": setup["capacity_share_of_test_pool"],
                     "test_base_rate": setup["windows"]["test"]["positive_share"]},
        "headline": summary["ranking_at_capacity"]["test"],
        "curve": curve_block(curve),
        "weekly": weekly.to_dict("records"),
        "calibration": {
            "method": chosen, "ece_before": test_cal["before"]["ece"], "ece_after": test_cal[chosen]["ece"],
            "brier_before": test_cal["before"]["brier"], "brier_after": test_cal[chosen]["brier"],
            "reliability_before": cal["reliability_test_5_seed_average"]["before"],
            "reliability_after": cal["reliability_test_5_seed_average"]["after"],
            "monthly": months.round(6).to_dict("records")},
        "split_comparison": {
            "capacity_fraction": comparison["capacity_fraction_of_weekly_pool"],
            "schemes": [{"scheme": name, "label": label,
                         **{metric: comparison[name][metric] for metric in ("auc", "lift", "precision")}}
                        for name, label in (("time_split_test_window", "Time split (test window)"),
                                            ("random_by_restaurant", "Random split by restaurant"),
                                            ("random_by_inspection", "Random split by inspection"))]},
        "threshold": summary["operating_threshold"],
        "drift": {"months": month_order, "splits": months["split"].tolist(), "rows": months["rows"].tolist(),
                  "features": features, "psi": matrix, "prediction_psi": prediction_row,
                  "cutoffs": summary["drift"]["cutoffs"], "noise_floor": summary["drift"]["noise_floor"],
                  "matched_history": {key: matched[key] for key in ("rule", "windows", "all_rows", "matched")}},
        "importance": importance.head(12).round(6).to_dict("records"),
        "model": {"num_leaves": card["lightgbm"]["params"]["num_leaves"],
                  "min_child_samples": card["lightgbm"]["params"]["min_child_samples"],
                  "seeds": card["lightgbm"]["seeds"]},
    }
    runinfo.write_json(args.out, dashboard)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
