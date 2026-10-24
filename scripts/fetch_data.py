"""Fetch the NYC restaurant inspection results once and freeze them in data/.

Source: NYC Open Data, "DOHMH New York City Restaurant Inspection Results" (Socrata dataset
43nn-pn8j), https://data.cityofnewyork.us/d/43nn-pn8j . No account or token is used.

Writes
  data/inspections.csv.gz   one row per violation citation (or per inspection with none), selected columns
  data/violation_codes.csv  violation code -> description (most common wording per code)
  data/MANIFEST.json        source, fetch time (UTC), the API's own refresh time, row counts, SHA-256

The restaurant name, street address, building number and phone number are not requested: nothing in
the analysis needs them, and leaving them out keeps the published file free of per-business contact
details.

Run once. Every later step reads the frozen file; run it again only to take a new snapshot (the
numbers in README.md and results/ then no longer match until the pipeline is re-run).

    python scripts/fetch_data.py [--out-dir data] [--page-size 50000]
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import gzip
import hashlib
import io
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

DATASET_ID = "43nn-pn8j"
DOMAIN = "data.cityofnewyork.us"
RESOURCE_URL = f"https://{DOMAIN}/resource/{DATASET_ID}.json"
VIEW_URL = f"https://{DOMAIN}/api/views/{DATASET_ID}.json"

# Columns written to data/inspections.csv.gz, in this order.
COLUMNS = [
    "camis", "boro", "zipcode", "cuisine_description", "inspection_date", "inspection_type", "action",
    "violation_code", "critical_flag", "score", "grade", "grade_date", "record_date",
]
DATE_COLUMNS = ("inspection_date", "grade_date", "record_date")
# Requested from the API in addition to COLUMNS: the text of the violation, moved to its own table.
API_COLUMNS = COLUMNS + ["violation_description"]
USER_AGENT = "restaurant-inspection-risk/fetch_data (public-data research; no account)"


def http_json(url: str, tries: int = 5):
    for attempt in range(1, tries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # network errors and 5xx are retried; the last one is raised
            if attempt == tries:
                raise
            wait = 2 ** attempt
            print(f"  request failed ({exc!r}); retry {attempt}/{tries - 1} in {wait}s", file=sys.stderr)
            time.sleep(wait)


def api_count() -> int:
    query = urllib.parse.urlencode({"$select": "count(*)"})
    return int(http_json(f"{RESOURCE_URL}?{query}")[0]["count"])


def api_rows_updated_at() -> str:
    seconds = http_json(VIEW_URL)["rowsUpdatedAt"]
    return dt.datetime.fromtimestamp(seconds, dt.timezone.utc).isoformat()


def repair_mojibake(text: str) -> tuple[str, bool]:
    """Defensive: if the API ever returns UTF-8 text that was decoded once as cp1252, reverse it, but
    only when the strict round trip succeeds; anything else is returned untouched. The snapshot taken
    for this repository needed no repair (MANIFEST.json counts 0)."""
    try:
        fixed = text.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text, False
    return fixed, fixed != text


def clean_date(value: str | None) -> str:
    """'2025-08-29T00:00:00.000' -> '2025-08-29'. These are calendar dates in New York, not instants:
    no time zone is attached and none is applied."""
    return (value or "")[:10]


def fetch_all(page_size: int):
    rows = []
    offset = 0
    select = ",".join(API_COLUMNS)
    while True:
        query = urllib.parse.urlencode(
            {"$select": select, "$order": ":id", "$limit": page_size, "$offset": offset})
        page = http_json(f"{RESOURCE_URL}?{query}")
        rows.extend(page)
        print(f"  fetched {len(rows)} rows", file=sys.stderr)
        if len(page) < page_size:
            return rows
        offset += page_size


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--page-size", type=int, default=50000)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    started = dt.datetime.now(dt.timezone.utc)
    count_before = api_count()
    updated_before = api_rows_updated_at()
    print(f"API reports {count_before} rows, last row update {updated_before}", file=sys.stderr)
    rows = fetch_all(args.page_size)
    count_after = api_count()
    updated_after = api_rows_updated_at()
    finished = dt.datetime.now(dt.timezone.utc)
    if updated_before != updated_after or count_before != count_after or len(rows) != count_before:
        print(f"REFUSE: the dataset changed or paging lost rows (count {count_before}->{count_after}, "
              f"fetched {len(rows)}, updated {updated_before}->{updated_after}); run again.", file=sys.stderr)
        return 1

    descriptions: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    repaired = 0
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(COLUMNS)
    for row in rows:
        code = (row.get("violation_code") or "").strip()
        text = row.get("violation_description")
        if code and text:
            fixed, changed = repair_mojibake(" ".join(text.split()))
            repaired += changed
            descriptions[code][fixed] += 1
        out = []
        for name in COLUMNS:
            value = (row.get(name) or "").strip()
            out.append(clean_date(value) if name in DATE_COLUMNS else value)
        writer.writerow(out)
    csv_bytes = buffer.getvalue().encode("utf-8")

    gz_path = out_dir / "inspections.csv.gz"
    with open(gz_path, "wb") as handle:
        # mtime=0 makes the compressed bytes depend only on the content
        with gzip.GzipFile(fileobj=handle, mode="wb", compresslevel=9, mtime=0) as gz:
            gz.write(csv_bytes)

    codes_path = out_dir / "violation_codes.csv"
    with open(codes_path, "w", encoding="utf-8", newline="") as handle:
        code_writer = csv.writer(handle, lineterminator="\n")
        code_writer.writerow(["violation_code", "description", "citations"])
        for code in sorted(descriptions):
            wording, _ = descriptions[code].most_common(1)[0]
            code_writer.writerow([code, wording, sum(descriptions[code].values())])

    manifest = {
        "source": {
            "name": "DOHMH New York City Restaurant Inspection Results",
            "publisher": "New York City Department of Health and Mental Hygiene (DOHMH), via NYC Open Data",
            "dataset_id": DATASET_ID,
            "page": f"https://{DOMAIN}/d/{DATASET_ID}",
            "api": RESOURCE_URL,
            "credentials": "none (unauthenticated Socrata API)",
        },
        "fetch": {
            "started_utc": started.isoformat(),
            "finished_utc": finished.isoformat(),
            "api_rows_updated_at_utc": updated_before,
            "api_row_count_before": count_before,
            "api_row_count_after": count_after,
            "page_size": args.page_size,
            "order": ":id",
        },
        "files": {
            "inspections.csv.gz": {
                "rows": len(rows),
                "columns": COLUMNS,
                "sha256_of_gzip": hashlib.sha256(gz_path.read_bytes()).hexdigest(),
                "sha256_of_csv": hashlib.sha256(csv_bytes).hexdigest(),
                "dates": "calendar dates (YYYY-MM-DD) in New York; no time zone applied",
            },
            "violation_codes.csv": {
                "rows": len(descriptions),
                "sha256": hashlib.sha256(codes_path.read_bytes()).hexdigest(),
                "descriptions_repaired_from_double_decoded_text": repaired,
            },
        },
        "not_requested": ["dba (name)", "building", "street", "phone", "latitude", "longitude", "bin", "bbl",
                          "nta", "community_board", "council_district", "census_tract", "location"],
    }
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {gz_path} ({gz_path.stat().st_size} bytes), {codes_path}, MANIFEST.json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
