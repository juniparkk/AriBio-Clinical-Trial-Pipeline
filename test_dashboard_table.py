# ============================================================
# TESTS for the rendered dashboard TABLE DATA (pipeline_overview.html)
#
# These validate the JSON blob embedded in the generated HTML — the
# exact data the browser's JS renders into the visible drug table —
# rather than drug_classification.py's pure functions (see
# test_classification.py for those). Run AFTER pipeline_viz.py:
#
#     .venv/bin/python pipeline_viz.py
#     .venv/bin/python test_dashboard_table.py
#
# What this can't test from Python alone: actual browser-rendered
# behavior (e.g. that a blank official_source_url really produces no
# <a href> in the live DOM) — that logic lives in JS template literals
# that only execute in a browser. Where noted below, the check is a
# structural one against the HTML/JS source instead of a live-DOM
# assertion — there's no headless-browser test runner set up in this
# project.
# ============================================================

import json
import re

HTML_PATH = "pipeline_overview.html"
UNRESOLVED_CSV_PATH = "pipeline_unresolved_trials.csv"

with open(HTML_PATH, encoding="utf-8") as f:
    HTML_SOURCE = f.read()

_match = re.search(
    r'<script id="drug-data" type="application/json">(.*?)</script>', HTML_SOURCE, re.DOTALL
)
assert _match, "Could not find the drug-data JSON blob in pipeline_overview.html — run pipeline_viz.py first"
TABLE_ROWS = json.loads(_match.group(1))

RAW_ENUM_VALUES = {
    "sponsor_developed_therapeutic",
    "investigational_therapeutic_unverified",
    "pipeline_record_match_without_source",
    "confirmed_official_match",
    "no_match",
}

# curated diagnostic/imaging tracer + procedure keywords, used to scan
# the rendered table for anything that should have been excluded
_DIAGNOSTIC_NAME_FRAGMENTS = ["florbetapir", "flortaucipir", "flutemetamol", "florbetaben", "amyvid", "tauvid"]
_PROCEDURE_NAME_FRAGMENTS = ["spect scan", "pet scan", "ct scan", "mri scan"]


def test_no_diagnostic_tracer_names_in_table():
    for row in TABLE_ROWS:
        name = str(row["display_name"]).lower()
        for fragment in _DIAGNOSTIC_NAME_FRAGMENTS:
            assert fragment not in name, f"diagnostic tracer name leaked into table: {row['display_name']}"


def test_no_procedure_only_entries_in_table():
    for row in TABLE_ROWS:
        name = str(row["display_name"]).lower()
        for fragment in _PROCEDURE_NAME_FRAGMENTS:
            assert fragment not in name, f"procedure-only entry leaked into table: {row['display_name']}"


def test_ar1001_row_exists():
    names = [row["display_name"] for row in TABLE_ROWS]
    assert "AR1001" in names


def test_wujia_yizhi_granules_row_exists():
    names = [row["display_name"] for row in TABLE_ROWS]
    assert "Wujia Yizhi granules" in names


def test_trx0237_occurs_exactly_once():
    matches = [row for row in TABLE_ROWS if row["display_name"] == "TRx0237"]
    assert len(matches) == 1, f"expected exactly one TRx0237 row, found {len(matches)}"


def test_lecanemab_and_ban2401_unified_into_one_row():
    lecanemab_rows = [row for row in TABLE_ROWS if row["display_name"] == "Lecanemab"]
    ban2401_rows = [row for row in TABLE_ROWS if row["display_name"] == "BAN2401"]
    assert len(lecanemab_rows) == 1, f"expected exactly one Lecanemab row, found {len(lecanemab_rows)}"
    assert len(ban2401_rows) == 0, "BAN2401 should be unified into the Lecanemab row, not a separate one"


def test_official_source_url_field_is_always_a_safe_string():
    # "Official source" is no longer rendered in the row-detail panel
    # (removed per request), but official_source_url is still carried in
    # the underlying data — confirm it stays a safe string (never null),
    # in case it's ever displayed again.
    for row in TABLE_ROWS:
        assert isinstance(row["official_source_url"], str), \
            f"official_source_url must be a string (never null) for {row['display_name']!r}"


def test_readable_verification_labels_not_raw_enum_values():
    for row in TABLE_ROWS:
        assert row["verification_label"] not in RAW_ENUM_VALUES, \
            f"raw enum value leaked as verification_label: {row['verification_label']!r}"
        assert row["confidence_label"] not in RAW_ENUM_VALUES
    # and the raw values ARE still present as separate, non-displayed
    # fields (used only for filtering/sorting) — not removed entirely
    assert all("verification_status" in row for row in TABLE_ROWS)
    assert all("classification_confidence" in row for row in TABLE_ROWS)


def test_multi_sponsor_row_retains_every_sponsor():
    multi_sponsor_rows = [row for row in TABLE_ROWS if "; " in str(row["sponsor"])]
    assert len(multi_sponsor_rows) > 0, "expected at least one multi-sponsor row in the real dataset"
    for row in multi_sponsor_rows:
        full_sponsors = row["sponsor"].split("; ")
        assert len(full_sponsors) >= 2
        # the compact display value must be a shortened form, not a
        # silent single-sponsor substitution — and needs_manual_review
        # must be flagged, per the multi-sponsor-ownership requirement
        assert row["sponsor_display"] != row["sponsor"] or len(full_sponsors) <= 1
        assert row["needs_manual_review"] is True or row["needs_manual_review"] == 1


def test_unresolved_trials_do_not_enter_the_table():
    import csv
    with open(UNRESOLVED_CSV_PATH, encoding="utf-8") as f:
        unresolved_nct_ids = {row["NCT Number"] for row in csv.DictReader(f)}

    for row in TABLE_ROWS:
        row_nct_ids = set(str(row.get("nct_ids", "")).split("; ")) if row.get("nct_ids") else set()
        overlap = row_nct_ids & unresolved_nct_ids
        assert not overlap, f"unresolved trial(s) {overlap} leaked into table row {row['display_name']!r}"


ALL_TESTS = [
    test_no_diagnostic_tracer_names_in_table,
    test_no_procedure_only_entries_in_table,
    test_ar1001_row_exists,
    test_wujia_yizhi_granules_row_exists,
    test_trx0237_occurs_exactly_once,
    test_lecanemab_and_ban2401_unified_into_one_row,
    test_official_source_url_field_is_always_a_safe_string,
    test_readable_verification_labels_not_raw_enum_values,
    test_multi_sponsor_row_retains_every_sponsor,
    test_unresolved_trials_do_not_enter_the_table,
]


def run_test(test_fn):
    try:
        test_fn()
        print(f"PASS  {test_fn.__name__}")
        return True
    except AssertionError as e:
        print(f"FAIL  {test_fn.__name__}  -- {e}")
        return False
    except Exception as e:
        print(f"ERROR {test_fn.__name__}  -- {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    results = [run_test(t) for t in ALL_TESTS]
    passed = sum(results)
    total = len(results)
    print()
    print(f"{passed}/{total} tests passed")
    if passed != total:
        raise SystemExit(1)
