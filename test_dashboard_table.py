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

import csv
import json
import re

import pandas as pd

HTML_PATH = "pipeline_overview.html"
UNRESOLVED_CSV_PATH = "pipeline_unresolved_trials.csv"
DRUGS_CSV_PATH = "pipeline_drugs.csv"
PIPELINE_VIZ_PATH = "pipeline_viz.py"

with open(HTML_PATH, encoding="utf-8") as f:
    HTML_SOURCE = f.read()

with open(PIPELINE_VIZ_PATH, encoding="utf-8") as f:
    PIPELINE_VIZ_SOURCE = f.read()

_match = re.search(
    r'<script id="drug-data" type="application/json">(.*?)</script>', HTML_SOURCE, re.DOTALL
)
assert _match, "Could not find the drug-data JSON blob in pipeline_overview.html — run pipeline_viz.py first"
TABLE_ROWS = json.loads(_match.group(1))

DRUGS_CSV_ROWS = list(csv.DictReader(open(DRUGS_CSV_PATH, encoding="utf-8")))


def extract_plotly_traces(html, div_id):
    """
    Pull the list of trace dicts out of a `Plotly.newPlot("<div_id>", [...traces...], {...layout...})`
    call embedded in the HTML (this is what plotly.io.to_html() emits).
    Not a regex-with-.*? match — that breaks on nested brackets inside
    the trace JSON (e.g. "labels":[...] inside a trace inside the outer
    traces array) — instead this scans character-by-character, tracking
    bracket depth and skipping over string literals, to find the exact
    end of the traces array, then json.loads() just that substring.
    """
    marker = f'"{div_id}",'
    start = html.index(marker) + len(marker)
    while html[start] in " \t\n":
        start += 1
    assert html[start] == "[", f"expected traces array to start with '[' for div {div_id!r}"
    depth = 0
    i = start
    while True:
        ch = html[i]
        if ch == '"':
            i += 1
            while not (html[i] == '"' and html[i - 1] != "\\"):
                i += 1
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                break
        i += 1
    return json.loads(html[start:i + 1])

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


# ============================================================
# PHASE 0 — dashboard data-source consolidation reconciliation tests
#
# Every one of these checks that a specific VISIBLE dashboard component
# (KPI tiles, heatmap, Phase 3 leaderboard, drug-type/target pies)
# actually agrees with resolved_drugs_df (here represented by
# TABLE_ROWS, parsed straight from the table's own embedded JSON, and
# cross-checked against pipeline_drugs.csv — both are resolved_drugs_df,
# just two different serializations of it).
# ============================================================

def get_therapeutic_rows():
    # Phase 1A: TABLE_ROWS is now the BROADER resolved_drugs_df population
    # (every pipeline_scope except Exclude/Placebo or Comparator, which
    # never get a row at all — see build_resolved_drugs_dataframe()).
    # KPI tiles/heatmap/leaderboard/drug-type/target pies all narrow to
    # THIS subset (pipeline_viz.py's therapeutic_drugs_df) — the table's
    # DEFAULT view does too (via the JS scope toggle), but the raw JSON
    # blob itself intentionally keeps the broader set so the "reveal
    # non-therapeutic records" filter has real data to show.
    return [r for r in TABLE_ROWS if r.get("pipeline_scope") == "Therapeutic Drug"]


def test_kpi_total_drugs_equals_resolved_drugs_df_length():
    therapeutic_rows = get_therapeutic_rows()
    total_drugs = len(therapeutic_rows)
    kpi_match = re.search(r'<div class="kpi-value">(\d+)</div>', HTML_SOURCE)
    assert kpi_match, "could not find the 'Total therapeutic drugs' KPI value in the HTML"
    assert int(kpi_match.group(1)) == total_drugs

    topbar_match = re.search(r"(\d+) therapeutic drugs \((\d+) total resolved records\)", HTML_SOURCE)
    assert topbar_match, "could not find the topbar's 'N therapeutic drugs (M total resolved records)' text"
    assert int(topbar_match.group(1)) == total_drugs
    assert int(topbar_match.group(2)) == len(TABLE_ROWS)

    # the table's full JSON payload (broader) must still match
    # pipeline_drugs.csv row count — both are the SAME resolved_drugs_df,
    # just two different serializations of it
    assert len(TABLE_ROWS) == len(DRUGS_CSV_ROWS), "table row count must also match pipeline_drugs.csv row count"
    # and pipeline_drugs.csv must carry the same Therapeutic Drug subset
    csv_therapeutic = [r for r in DRUGS_CSV_ROWS if r.get("pipeline_scope") == "Therapeutic Drug"]
    assert len(csv_therapeutic) == total_drugs


def test_kpi_phase_counts_computed_from_resolved_drugs_df():
    therapeutic_rows = get_therapeutic_rows()
    # 5 colored numeric KPI tiles, in dashboard order: FDA approved,
    # Phase 3/2/1 agents, High relevance. (Total therapeutic drugs is
    # the one KPI tile left uncolored.)
    phase_kpi_matches = re.findall(r'<div class="kpi-value" style="color:[^"]*">(\d+)</div>', HTML_SOURCE)
    assert len(phase_kpi_matches) == 5, "expected 5 colored numeric KPI tiles (FDA approved, Phase 3/2/1 agents, High relevance)"
    _fda_kpi, phase3_kpi, phase2_kpi, phase1_kpi, high_relevance_kpi = (int(v) for v in phase_kpi_matches)

    # "Active Phase N agents": only rows with status_summary == "Active"
    # count -- a plain phase_reached tally would include drugs that
    # reached that phase but have since finished/been dropped.
    phase_counts = {"Phase 1": 0, "Phase 2": 0, "Phase 3": 0}
    high_relevance_count = 0
    for row in therapeutic_rows:
        if row["phase_reached"] in phase_counts and row["status_summary"] == "Active":
            phase_counts[row["phase_reached"]] += 1
        if row["aribio_relevance_score"] >= 70 and not row["is_aribio"]:
            high_relevance_count += 1

    assert phase3_kpi == phase_counts["Phase 3"]
    assert phase2_kpi == phase_counts["Phase 2"]
    assert phase1_kpi == phase_counts["Phase 1"]
    assert high_relevance_kpi == high_relevance_count


TARGET_ORDER = ["Amyloid", "Tau", "Inflammation", "Neuroprotection", "Metabolism", "Symptomatic", "Neuropsychiatric"]
PHASES_ASC = ["NA", "Early Phase 1", "Phase 1", "Phase 1/Phase 2", "Phase 2", "Phase 2/Phase 3", "Phase 3", "Phase 4"]


def test_heatmap_all_tab_reconciles_to_resolved_drug_counts():
    therapeutic_rows = get_therapeutic_rows()
    traces = extract_plotly_traces(HTML_SOURCE, "heatmapAll")
    assert len(traces) == 1, "expected a single go.Heatmap trace in the 'All' tab"
    z = traces[0]["z"]

    # Phase 1A: the heatmap is built from therapeutic_drugs_df (pipeline_scope
    # == "Therapeutic Drug" only), not the broader resolved_drugs_df/TABLE_ROWS.
    expected = [
        [sum(1 for r in therapeutic_rows if r["target"] == t and r["phase_reached"] == p) for p in PHASES_ASC]
        for t in TARGET_ORDER
    ]
    assert z == expected

    # every drug counted in the heatmap must be a real THERAPEUTIC drug —
    # i.e. the grid total can never exceed len(therapeutic_rows), and
    # only falls short of it for drugs whose target isn't one of the
    # 7 curated pathway buckets (target "Other"/"Unknown" isn't
    # plotted — a pre-existing, deliberate heatmap scope, not a bug)
    grid_total = sum(sum(row) for row in z)
    assert grid_total <= len(therapeutic_rows)
    eligible = sum(1 for r in therapeutic_rows if r["target"] in TARGET_ORDER)
    assert grid_total == eligible


def test_phase3_leaderboard_names_are_subset_of_resolved_drugs_df():
    table_match = re.search(r'<table class="phase3-table">(.*?)</table>', HTML_SOURCE, re.DOTALL)
    assert table_match, "could not find the Phase 3 leaderboard table"
    leaderboard_names = re.findall(r'<a href="[^"]*"[^>]*>([^<]+)</a>', table_match.group(1))
    assert len(leaderboard_names) > 0, "expected at least one Phase 3 leaderboard row in the real dataset"

    resolved_names = {row["display_name"] for row in TABLE_ROWS}
    for name in leaderboard_names:
        # the leaderboard wraps AriBio's own row in "<name> ★" (a
        # star suffix added by phase3_row_html) — strip it before comparing
        clean_name = name.replace(" ★", "").strip()
        assert clean_name in resolved_names, f"leaderboard name {clean_name!r} not found in resolved_drugs_df"


def test_phase3_leaderboard_sorted_by_relevance_and_excludes_ar1001():
    table_match = re.search(r'<table class="phase3-table">.*?<tbody>(.*?)</tbody>', HTML_SOURCE, re.DOTALL)
    assert table_match, "could not find the Phase 3 leaderboard table body"
    row_html_blocks = re.findall(r'<tr>(.*?)</tr>', table_match.group(1), re.DOTALL)
    assert len(row_html_blocks) > 0, "expected at least one Phase 3 leaderboard row in the real dataset"

    names = []
    scores = []
    for block in row_html_blocks:
        name_match = re.search(r'<a href="[^"]*"[^>]*>([^<]+)</a>', block)
        names.append(name_match.group(1).strip())
        score_match = re.search(r'font-weight:700;">(\d+)</span>', block)
        assert score_match, f"expected a numeric relevance score cell in leaderboard row: {block!r}"
        scores.append(int(score_match.group(1)))

    assert "AR1001" not in names, "AR1001 itself should not appear in its own relevance ranking"
    assert scores == sorted(scores, reverse=True), "leaderboard rows must be sorted by relevance score descending"


def test_pipeline_table_names_are_subset_of_pipeline_drugs_csv():
    resolved_names = {row["display_name"] for row in DRUGS_CSV_ROWS}
    for row in TABLE_ROWS:
        assert row["display_name"] in resolved_names


def test_no_dashboard_calculation_references_legacy_drugs_df():
    # legacy_drugs_df, and its whole upstream chain, must have zero
    # remaining CODE references (comments mentioning the retired name
    # for historical context are fine and expected — this checks for
    # actual Python usage: assignment, indexing, or attribute access)
    assert re.search(r"legacy_drugs_df\s*=", PIPELINE_VIZ_SOURCE) is None
    assert re.search(r"legacy_drugs_df\[", PIPELINE_VIZ_SOURCE) is None
    assert re.search(r"legacy_drugs_df\.\w", PIPELINE_VIZ_SOURCE) is None
    for removed_def in ["def primary_intervention_name", "def canonical_drug_key", "def summarize_drug", "def mode_or_first"]:
        assert removed_def not in PIPELINE_VIZ_SOURCE, f"{removed_def} should have been removed entirely in Phase 0"


# ============================================================
# PHASE 1A — intervention-scope gap closure reconciliation tests
# ============================================================

GAP_AUDIT_CSV_PATH = "outputs/classification_gap_audit.csv"
INTERVENTIONS_CSV_PATH = "pipeline_interventions.csv"


def test_pipeline_scope_field_present_on_every_table_row():
    assert len(TABLE_ROWS) > 0
    for row in TABLE_ROWS:
        assert "pipeline_scope" in row
        assert "scope_reason" in row
        assert row["pipeline_scope"] in {
            "Therapeutic Drug", "Diagnostic Agent", "Non-Drug Intervention",
            "Supportive Treatment", "Needs Review",
        }


def test_no_non_therapeutic_scopes_present_in_table_data():
    # resolved_drugs_df (and therefore the table's JSON payload) now
    # only ever contains "Therapeutic Drug" scope rows — a record only
    # enters it if its primary investigational intervention resolved to
    # a real drug/biologic. The "reveal non-therapeutic records" toggle
    # still exists in the UI (unchanged) but has nothing left to
    # reveal; that's an intentional consequence of this narrowing.
    non_therapeutic = [r for r in TABLE_ROWS if r["pipeline_scope"] != "Therapeutic Drug"]
    assert len(non_therapeutic) == 0, f"expected zero non-therapeutic rows, found {len(non_therapeutic)}"
    assert len(TABLE_ROWS) == len(get_therapeutic_rows())


def test_placebo_or_comparator_never_in_table_rows():
    for row in TABLE_ROWS:
        assert row["pipeline_scope"] != "Placebo or Comparator", \
            f"Placebo or Comparator must never appear in the table data, even under the optional filter: {row['display_name']!r}"


def test_dietary_supplement_confirmed_leakage_examples_excluded_entirely():
    # confirmed leakage examples from the Phase 1A audit — dietary
    # supplements are neither DRUG nor BIOLOGICAL type, so they're now
    # excluded from resolved_drugs_df/TABLE_ROWS entirely (not merely
    # hidden from the default Therapeutic Drug view as before). Still
    # fully auditable in pipeline_annotated.csv / pipeline_
    # interventions.csv / outputs/non_drug_exclusion_audit.csv.
    table_names = {r["display_name"] for r in TABLE_ROWS}
    for name in ["lutein/zeaxanthin", "Curcumin C3 Complex"]:
        assert name not in table_names, f"{name!r} must not appear anywhere in resolved_drugs_df/TABLE_ROWS"


def test_scope_toggle_control_removed_from_html():
    # The "Show non-therapeutic / needs-review records" checkbox was
    # removed per request -- the table now always filters to
    # THERAPEUTIC_SCOPE, with no UI path to reveal the rest.
    assert 'id="scope-toggle"' not in HTML_SOURCE
    assert "showNonTherapeutic" not in HTML_SOURCE
    assert "THERAPEUTIC_SCOPE" in HTML_SOURCE


def test_gap_audit_csv_exists_with_expected_columns():
    with open(GAP_AUDIT_CSV_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 0
    expected_columns = {
        "raw_intervention_name", "normalized_name", "nct_ids", "clinicaltrials_intervention_type",
        "previous_drug_type", "new_pipeline_scope", "scope_reason", "scope_method",
        "scope_confidence", "dashboard_eligible", "manual_review_required",
    }
    assert expected_columns.issubset(set(rows[0].keys()))


def test_gap_audit_csv_flags_confirmed_leakage_examples_as_not_dashboard_eligible():
    with open(GAP_AUDIT_CSV_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_name = {r["raw_intervention_name"]: r for r in rows}
    for name in ["Curcumin C3 Complex", "lutein/zeaxanthin"]:
        if name in by_name:
            assert by_name[name]["dashboard_eligible"] == "False"
            assert by_name[name]["new_pipeline_scope"] != "Therapeutic Drug"
    for name in ["Blood Test", "Cerebrospinal fluid (CSF) Biomarkers"]:
        if name in by_name:
            assert by_name[name]["new_pipeline_scope"] == "Exclude"
            assert by_name[name]["dashboard_eligible"] == "False"


def test_pipeline_interventions_csv_preserves_scope_columns_and_row_count():
    # traceability requirement: every parsed intervention stays in this
    # CSV regardless of pipeline_scope — including Exclude/Placebo or
    # Comparator, which never get a resolved_drugs_df row at all
    with open(INTERVENTIONS_CSV_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) > 0
    assert {"pipeline_scope", "scope_reason", "scope_method", "scope_confidence", "manual_review_required"}.issubset(rows[0].keys())
    scopes_present = {r["pipeline_scope"] for r in rows}
    assert "Placebo or Comparator" in scopes_present, "placebo/comparator records must still be preserved in pipeline_interventions.csv"


def test_pipeline_drugs_csv_carries_pipeline_scope_column():
    assert "pipeline_scope" in DRUGS_CSV_ROWS[0]
    assert "scope_reason" in DRUGS_CSV_ROWS[0]


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
    test_kpi_total_drugs_equals_resolved_drugs_df_length,
    test_kpi_phase_counts_computed_from_resolved_drugs_df,
    test_heatmap_all_tab_reconciles_to_resolved_drug_counts,
    test_phase3_leaderboard_names_are_subset_of_resolved_drugs_df,
    test_phase3_leaderboard_sorted_by_relevance_and_excludes_ar1001,
    test_pipeline_table_names_are_subset_of_pipeline_drugs_csv,
    test_no_dashboard_calculation_references_legacy_drugs_df,
    test_pipeline_scope_field_present_on_every_table_row,
    test_no_non_therapeutic_scopes_present_in_table_data,
    test_placebo_or_comparator_never_in_table_rows,
    test_dietary_supplement_confirmed_leakage_examples_excluded_entirely,
    test_scope_toggle_control_removed_from_html,
    test_gap_audit_csv_exists_with_expected_columns,
    test_gap_audit_csv_flags_confirmed_leakage_examples_as_not_dashboard_eligible,
    test_pipeline_interventions_csv_preserves_scope_columns_and_row_count,
    test_pipeline_drugs_csv_carries_pipeline_scope_column,
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
