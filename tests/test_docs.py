"""The README, the memo and the dashboard data must be copies of what is in results/, not retyped numbers."""
import gzip
import importlib.util
import json
import re
import subprocess
import sys

import pandas as pd
import pytest


def load_script(root, name):
    spec = importlib.util.spec_from_file_location(name, root / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def render_docs(root):
    return load_script(root, "render_docs")


FENCE = re.compile(r"```.*?```", re.S)
CODE = re.compile(r"`[^`\n]*`")
LINK_TARGET = re.compile(r"\]\([^)\n]*\)")
URL = re.compile(r"https?://\S+")
PLACEHOLDER = re.compile(r"\{\{[a-z0-9_]+\}\}")


def typed_numbers(text: str) -> list[str]:
    """Digit runs left in prose after removing code, links and placeholders."""
    for pattern in (FENCE, CODE, LINK_TARGET, URL, PLACEHOLDER):
        text = pattern.sub(" ", text)
    return re.findall(r"\S*\d\S*", text)


# ------------------------------------------------------------------ the guard itself, with negative cases
def test_the_lint_flags_a_hand_typed_figure_and_ignores_code_and_placeholders():
    assert typed_numbers("the model found 66.0% of them") == ["66.0%"]
    assert typed_numbers("about 3 in 10 weeks") == ["3", "10"]
    assert typed_numbers("it found {{gbm_precision}} of them, see `results/summary.json` and "
                         "[the page](https://example.org/a1) or ```\nrun 2026-09\n```") == []


def test_rendering_refuses_a_placeholder_with_no_value(render_docs):
    with pytest.raises(KeyError):
        render_docs.render("a {{not_a_value}} b", {"other": "x"})


def test_rendering_fills_every_placeholder(render_docs):
    assert render_docs.render("a {{x}} b {{x}}", {"x": "1"}) == "a 1 b 1"


# ------------------------------------------------------------------ the real documents
@pytest.mark.parametrize("name", ["README.md.tmpl", "analysis.md.tmpl"])
def test_templates_hold_no_hand_typed_numbers(root, name):
    text = (root / "scripts" / "templates" / name).read_text(encoding="utf-8")
    assert typed_numbers(text) == []


def test_every_placeholder_in_the_templates_has_a_value(root, render_docs):
    values = render_docs.build_values(root)
    for name in render_docs.OUTPUTS:
        used = set(render_docs.PLACEHOLDER.findall((render_docs.TEMPLATES / name).read_text(encoding="utf-8")))
        assert used <= set(values), sorted(used - set(values))


def test_the_committed_documents_equal_a_fresh_render(root, render_docs):
    for path, text in render_docs.render_all(root).items():
        assert path.read_text(encoding="utf-8") == text, f"{path.name} is stale: run python scripts/render_docs.py"


def test_no_unfilled_placeholder_is_left_in_the_documents(root):
    for path in (root / "README.md", root / "docs" / "analysis.md"):
        assert "{{" not in path.read_text(encoding="utf-8")


def test_the_figures_the_memo_links_exist(root):
    memo = (root / "docs" / "analysis.md").read_text(encoding="utf-8")
    links = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", memo)
    assert links
    for link in links:
        assert (root / "docs" / link).is_file(), link


# ------------------------------------------------------------------ dashboard data
def test_dashboard_data_equals_a_fresh_build(root, tmp_path):
    out = tmp_path / "dashboard.json"
    done = subprocess.run([sys.executable, str(root / "scripts" / "build_dashboard_data.py"), "--out", str(out)],
                          cwd=root, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert json.loads(out.read_text(encoding="utf-8")) == json.loads((root / "docs" / "data" / "dashboard.json").read_text(encoding="utf-8"))


def test_dashboard_numbers_agree_with_the_summary(root):
    dash = json.loads((root / "docs" / "data" / "dashboard.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "results" / "summary.json").read_text(encoding="utf-8"))
    cap = dash["capacity"]["default"]
    index = dash["curve"]["capacities"].index(cap)
    head = summary["ranking_at_capacity"]["test"]
    for key in ("gbm", "previous_result_rule", "logistic", "base_rate"):
        for metric in ("precision", "recall", "lift", "hits_per_week"):
            assert dash["curve"]["models"][key][metric][index] == pytest.approx(head[key][metric]["median"], abs=1e-5), (key, metric)
    assert dash["calibration"]["ece_before"] == summary["calibration"]["windows"]["test"]["gbm_5_seed_average"]["before"]["ece"]
    flat = [v for row in dash["drift"]["psi"] for v in row if v is not None]
    cut = dash["drift"]["cutoffs"]
    major = sum(v >= cut["major"] for v in flat)
    psi = pd.read_csv(root / "results" / "drift_monthly.csv")
    assert major == int(((psi["feature"] != "__prediction__") & (psi["status"] == "major")).sum())
    assert len(flat) == int((psi["feature"] != "__prediction__").sum())
    weekly = pd.read_csv(root / "results" / "weekly_test.csv")
    assert [w["hits_gbm"] for w in dash["weekly"]] == weekly["hits_gbm"].tolist()


def test_the_dashboard_page_holds_no_typed_figures(root):
    html = (root / "docs" / "index.html").read_text(encoding="utf-8")
    body = re.sub(r"<(script|style)\b.*?</\1>", " ", html, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", body)
    assert typed_numbers(text) == []


# ------------------------------------------------------------------ the extra drift measurement and the timing
def test_matched_history_counts_match_their_own_table(root):
    result = json.loads((root / "results" / "drift_matched_history.json").read_text(encoding="utf-8"))
    table = pd.read_csv(root / "results" / "drift_matched_history.csv")
    for key, used in (("all_rows", "all"), ("matched", "matched")):
        part = table[table["rows_used"] == used]
        block = result[key]
        assert block["feature_windows"] == len(part)
        assert block["all"] == {s: int((part["status"] == s).sum()) for s in ("stable", "moderate", "major")}
        assert sum(g["feature_windows"] for g in block["by_group"].values()) == len(part)
    assert result["matched"]["all"]["major"] < result["all_rows"]["all"]["major"]


def test_timing_file_is_consistent_with_the_predictions(root):
    timing = json.loads((root / "results" / "timing_batch_scoring.json").read_text(encoding="utf-8"))
    with gzip.open(root / "results" / "predictions.csv.gz", "rt", encoding="utf-8") as handle:
        dates = pd.read_csv(handle, usecols=["date"])["date"]
    assert timing["inspections_scored"] == int(dates.str.startswith(timing["month"]).sum())
    assert timing["repetitions"] == len(timing["seconds_end_to_end_fresh_process"]["values"]) == 5
    assert all(v > 0 for v in timing["seconds_end_to_end_fresh_process"]["values"])
    assert (root / "results" / "timing_docker_ps.txt").is_file()


def test_the_docker_listing_beside_the_timing_is_well_formed_and_holds_nothing_private(root):
    lines = (root / "results" / "timing_docker_ps.txt").read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("taken ") and lines[1].split() == ["NAMES", "IMAGE", "STATUS", "PORTS"]
    assert len(lines) > 2 and all(line.strip() for line in lines[2:])
    text = "\n".join(lines)
    assert not re.search(r"[A-Za-z]:\\|/Users/|/home/|@|\b\d{12}\b", text)
