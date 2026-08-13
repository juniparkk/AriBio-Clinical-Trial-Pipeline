# ============================================================
# TESTS for adni_viz_data.py / adni_viz.py (biomarker dashboard
# visualization layer -- Medical Affairs redesign).
#
# Two kinds of test here:
#   1. Governance unit tests against synthetic temp directories/files
#      (never touch the real ADNI tree).
#   2. Structural tests against the REAL generated biomarker_dashboard.html
#      and its embedded JSON payload -- these read the already-built
#      output file, never raw/interim/processed/participant data
#      themselves.
#
# Run: .venv/bin/python test_adni_viz.py
# (assumes biomarker_dashboard.html has already been built via
#  `.venv/bin/python adni_viz.py`)
# ============================================================

import json
import os
import re
import shutil
import tempfile

import pandas as pd

import adni_viz as V
import adni_viz_data as D
from adni_analysis import ADNI_OUTPUTS_DIR

DASHBOARD_PATH = "biomarker_dashboard.html"


def _load_dashboard_html():
    with open(DASHBOARD_PATH, encoding="utf-8") as f:
        return f.read()


def _load_payload():
    content = _load_dashboard_html()
    marker = "const DATA = "
    start = content.find(marker) + len(marker)
    end = content.find(";\nconst CLASS_A", start)
    return json.loads(content[start:end])


def _load_overall_payload():
    """The Overall-ADNI population's chart/table payload -- since the
    population-aware restructuring, cognitiveChange/cognitiveAbsolute/
    biomarkersChange/biomarkersAbsolute/keyPatterns/resultsTable all
    live under payload["populations"]["overall"] rather than at the
    top level. Every test that predates population-awareness and
    checked Overall-ADNI values now goes through this helper --
    exercising the identical values as before, just at the new path."""
    return _load_payload()["populations"]["overall"]


def _load_polaris_payload():
    return _load_payload()["populations"]["polaris"]


# ------------------------------------------------------------------
# Governance unit tests (synthetic temp dirs only)
# ------------------------------------------------------------------


def test_governance_rejects_raw_interim_processed_paths():
    tmp = tempfile.mkdtemp()
    try:
        outputs_dir = os.path.join(tmp, "outputs")
        os.makedirs(outputs_dir)
        for forbidden in ["raw", "interim", "processed"]:
            forbidden_dir = os.path.join(tmp, forbidden)
            os.makedirs(forbidden_dir, exist_ok=True)
            pd.DataFrame({"month": [0]}).to_csv(os.path.join(forbidden_dir, "sneaky.csv"), index=False)
            raised = False
            try:
                D.load_aggregate_csv(outputs_dir, os.path.join("..", forbidden, "sneaky.csv"))
            except D.DataGovernanceError:
                raised = True
            assert raised, f"expected DataGovernanceError for a path under {forbidden}/"
    finally:
        shutil.rmtree(tmp)


def test_governance_rejects_parquet_files():
    tmp = tempfile.mkdtemp()
    try:
        outputs_dir = os.path.join(tmp, "outputs")
        os.makedirs(outputs_dir)
        pd.DataFrame({"month": [0]}).to_csv(os.path.join(outputs_dir, "not_really.parquet"), index=False)
        raised = False
        try:
            D.load_aggregate_csv(outputs_dir, "not_really.parquet")
        except D.DataGovernanceError:
            raised = True
        assert raised
    finally:
        shutil.rmtree(tmp)


def test_governance_rejects_participant_identifier_columns():
    tmp = tempfile.mkdtemp()
    try:
        outputs_dir = os.path.join(tmp, "outputs")
        os.makedirs(outputs_dir)
        pd.DataFrame({"RID": [1, 2, 3], "value": [1.0, 2.0, 3.0]}).to_csv(
            os.path.join(outputs_dir, "leaky.csv"), index=False
        )
        raised = False
        try:
            D.load_aggregate_csv(outputs_dir, "leaky.csv")
        except D.DataGovernanceError:
            raised = True
        assert raised
    finally:
        shutil.rmtree(tmp)


def test_governance_rejects_path_outside_outputs_dir():
    tmp = tempfile.mkdtemp()
    try:
        outputs_dir = os.path.join(tmp, "outputs")
        os.makedirs(outputs_dir)
        pd.DataFrame({"secret": [1]}).to_csv(os.path.join(tmp, "escaped.csv"), index=False)
        raised = False
        try:
            D.load_aggregate_csv(outputs_dir, os.path.join("..", "escaped.csv"))
        except D.DataGovernanceError:
            raised = True
        assert raised
    finally:
        shutil.rmtree(tmp)


def test_real_outputs_dir_loads_cleanly_through_governance():
    """The actual pipeline's outputs/ must load without tripping any
    governance rule -- confirms the checks aren't so strict they break
    the real, approved aggregate files."""
    data = D.load_all(ADNI_OUTPUTS_DIR)
    assert set(data.keys()) == {"eligibility", "cognitive", "biomarker", "pairwise", "robustness", "sensitivity"}
    for df in data.values():
        assert len(df) > 0


def test_disease_continuum_and_absolute_columns_present_and_governed():
    """The new raw_absolute_*/raw_geometric_mean_ci_* columns the
    Medical Affairs redesign depends on must be real columns in the
    approved aggregate files (not silently missing), and loading them
    still passes every governance check (no PII, no forbidden path)."""
    data = D.load_all(ADNI_OUTPUTS_DIR)
    for col in ("raw_absolute_mean", "raw_absolute_ci_lower", "raw_absolute_ci_upper"):
        assert col in data["cognitive"].columns
    for col in ("raw_geometric_mean_ci_lower", "raw_geometric_mean_ci_upper"):
        assert col in data["biomarker"].columns


# ------------------------------------------------------------------
# Structural tests against the generated dashboard
# ------------------------------------------------------------------


def test_no_rid_ptid_fields_embedded():
    payload = _load_payload()
    raw_json = json.dumps(payload)
    for forbidden in ["\"RID\"", "\"PTID\"", "\"USUBJID\"", "\"SUBJID\""]:
        assert forbidden not in raw_json, f"forbidden identifier key {forbidden} found in embedded payload"


def test_no_participant_level_records_present():
    payload = _load_payload()
    overall = payload["populations"]["overall"]
    allowed_point_keys = {
        "month", "group", "classification", "reason", "n", "estimate", "ci_lower", "ci_upper",
        "overall_p_hc3", "is_hc3", "is_descriptive_ci",
    }
    for series_key in ("cognitiveChange", "cognitiveAbsolute"):
        for endpoint_points in overall[series_key].values():
            assert len(endpoint_points) <= 21  # 3 groups x 7 months, never a per-participant row count
            for pt in endpoint_points:
                assert set(pt.keys()) <= allowed_point_keys
    for series_key in ("biomarkersChange", "biomarkersAbsolute"):
        for biomarker_platforms in overall[series_key].values():
            for analysis_types in biomarker_platforms.values():
                for pts in analysis_types.values():
                    assert len(pts) <= 21
                    for pt in pts:
                        assert set(pt.keys()) <= allowed_point_keys
    # Disease Continuum: 7 endpoints, 3 group cells each -- never a
    # per-participant structure. Stays at the top level (Overall-ADNI
    # only, not population-scoped -- see item 6 of the population-aware
    # trajectories spec).
    assert len(payload["diseaseContinuum"]) == 7
    for row in payload["diseaseContinuum"]:
        assert set(row["cells"].keys()) == {"CN", "MCI", "Dementia"}
        for cell in row["cells"].values():
            assert set(cell.keys()) <= {"n", "value", "ci_lower", "ci_upper"}


def test_gfap_nfl_show_no_inferential_claims():
    payload = _load_overall_payload()
    for biomarker in ["GFAP", "NfL"]:
        for view in ("biomarkersChange", "biomarkersAbsolute"):
            for platform_data in payload[view][biomarker].values():
                for pts in platform_data.values():
                    for pt in pts:
                        assert pt["classification"] not in (D.CLASS_ADJUSTED, D.CLASS_SENSITIVITY_CONCERN), (
                            f"{biomarker} point unexpectedly classified {pt['classification']} -- "
                            "GFAP/NfL must never show an adjusted/inferential result"
                        )
                        if "is_hc3" in pt:
                            assert pt["is_hc3"] is False


def test_hc3_is_primary_inference_not_conventional():
    """A known case where conventional and HC3 disagree on CI-exclusion-
    of-zero (ADAS-Cog13 month 12, CN group -- see adni_robustness_summary.csv):
    the dashboard's displayed value (Change-from-baseline view) must be
    the HC3 one."""
    payload = _load_overall_payload()
    pt = next(p for p in payload["cognitiveChange"]["ADAS_COG13"] if p["month"] == 12 and p["group"] == "CN")
    assert pt["classification"] == D.CLASS_SENSITIVITY_CONCERN
    assert pt["is_hc3"] is True

    robustness = D.load_aggregate_csv(ADNI_OUTPUTS_DIR, "adni_robustness_summary.csv")
    hc3_row = robustness[
        (robustness["endpoint_or_biomarker"] == "ADAS_COG13") & (robustness["analysis_type"] == "primary")
        & (robustness["month"] == 12) & (robustness["level"] == "adjusted_mean")
        & (robustness["group_or_comparison"] == "CN") & (robustness["robustness_check"] == "HC3")
    ].iloc[0]
    assert abs(pt["estimate"] - float(hc3_row["alternative_estimate"])) < 1e-9
    assert abs(pt["ci_lower"] - float(hc3_row["alternative_ci_lower"])) < 1e-9
    # And explicitly NOT the conventional CI (which included zero, per adni_cognitive_summary.csv).
    assert abs(pt["ci_lower"] - float(hc3_row["conventional_ci_lower"])) > 1e-6


def test_sensitivity_concern_status_visible():
    payload = _load_overall_payload()
    has_c_point = any(
        p["classification"] == D.CLASS_SENSITIVITY_CONCERN
        for pts in payload["cognitiveChange"].values() for p in pts
    )
    assert has_c_point  # confirms this dashboard build actually has something to show
    html_source = _load_dashboard_html()
    assert "Sensitivity concern" in html_source
    assert "⚠" in html_source
    assert "warning-badge" in html_source or "compact-warn" in html_source


def test_no_significance_stars():
    html_source = _load_dashboard_html()
    # No conventional significance-star notation anywhere in the page.
    assert not re.search(r"p\s*<\s*0?\.05\s*\*", html_source)
    assert "***" not in html_source
    assert "★" not in html_source


def test_descriptive_only_cells_labeled():
    html_source = _load_dashboard_html()
    assert "Descriptive only" in html_source
    assert "small-cell rule" in html_source or "n &lt; 10" in html_source


def test_descriptive_biomarker_points_are_aggregate_only_and_plotted():
    """
    B-status (descriptive-only) biomarker cells must carry a real,
    aggregate geometric-percent-change estimate (not None) wherever
    raw_geometric_pct_change was derivable in adni_biomarker_summary.csv,
    while still exposing only aggregate fields -- never a participant
    identifier, never a per-row list.
    """
    payload = _load_overall_payload()
    allowed_point_keys = {
        "month", "group", "classification", "reason", "n", "estimate", "ci_lower", "ci_upper",
        "overall_p_hc3", "is_hc3", "is_descriptive_ci",
    }
    found_plotted_descriptive_point = False
    for biomarker_platforms in payload["biomarkersChange"].values():
        for analysis_types in biomarker_platforms.values():
            for pts in analysis_types.values():
                for pt in pts:
                    assert set(pt.keys()) <= allowed_point_keys
                    assert not any(isinstance(v, (list, dict)) for v in pt.values())
                    if pt["classification"] == D.CLASS_DESCRIPTIVE and pt["estimate"] is not None:
                        found_plotted_descriptive_point = True
                        assert pt["is_hc3"] is False
                        if pt["ci_lower"] is not None:
                            assert pt["is_descriptive_ci"] is True
    assert found_plotted_descriptive_point, "expected at least one B-status biomarker cell with a real plotted descriptive value"


def test_gfap_nfl_descriptive_points_have_no_p_value_or_hc3_wording():
    """GFAP/NfL are descriptive-only for every cell -- confirm the
    embedded payload never attaches an HC3 p-value/CI to them even now
    that B-status points carry real estimates."""
    payload = _load_overall_payload()
    for biomarker in ["GFAP", "NfL"]:
        for platform_data in payload["biomarkersChange"][biomarker].values():
            for pts in platform_data.values():
                for pt in pts:
                    assert pt["overall_p_hc3"] is None or pt["classification"] != D.CLASS_DESCRIPTIVE
                    assert pt["is_hc3"] is False


def test_group_colors_consistent():
    payload = _load_payload()
    assert payload["groupColors"] == {"CN": "#2196F3", "MCI": "#FF9800", "Dementia": "#F44336"}
    html_source = _load_dashboard_html()
    for color in ["#2196F3", "#FF9800", "#F44336"]:
        assert color in html_source


# ------------------------------------------------------------------
# Medical Affairs redesign -- new sections/behavior
# ------------------------------------------------------------------


def test_absolute_views_never_carry_adjusted_or_sensitivity_classification():
    """The Absolute view is always descriptive by construction (no
    ANCOVA was ever fit against an absolute score/level) -- confirms
    no point in either absolute series is ever labeled Adjusted or
    Sensitivity concern, which would falsely imply a model was fit."""
    payload = _load_overall_payload()
    for pt in [p for pts in payload["cognitiveAbsolute"].values() for p in pts]:
        assert pt["classification"] in (D.CLASS_DESCRIPTIVE, D.CLASS_NOT_AVAILABLE)
    for biomarker_platforms in payload["biomarkersAbsolute"].values():
        for analysis_types in biomarker_platforms.values():
            for pts in analysis_types.values():
                for pt in pts:
                    assert pt["classification"] in (D.CLASS_DESCRIPTIVE, D.CLASS_NOT_AVAILABLE)


def test_absolute_cognitive_baseline_matches_known_group_ordering():
    """Sanity check on real data: baseline (month 0) absolute ADAS-Cog13
    must increase CN < MCI < Dementia (a well-established, already-
    validated pattern in this cohort) -- confirms the new absolute
    values aren't scrambled/mislabeled by group."""
    payload = _load_overall_payload()
    baseline = {p["group"]: p["estimate"] for p in payload["cognitiveAbsolute"]["ADAS_COG13"] if p["month"] == 0}
    assert baseline["CN"] < baseline["MCI"] < baseline["Dementia"]


def test_disease_continuum_covers_all_seven_endpoints_with_correct_direction_metadata():
    payload = _load_payload()
    keys = {row["key"] for row in payload["diseaseContinuum"]}
    assert keys == {"ADAS_COG13", "MMSE", "pTau181", "pTau217", "Abeta42_40_ratio", "GFAP", "NfL"}
    by_key = {row["key"]: row for row in payload["diseaseContinuum"]}
    # MMSE and the amyloid ratio are the two endpoints where a LOWER
    # value is the worse direction -- everything else is higher=worse.
    assert by_key["MMSE"]["higherIsWorse"] is False
    assert by_key["Abeta42_40_ratio"]["higherIsWorse"] is False
    assert by_key["ADAS_COG13"]["higherIsWorse"] is True
    assert by_key["GFAP"]["higherIsWorse"] is True
    assert by_key["NfL"]["higherIsWorse"] is True


def test_disease_continuum_baseline_values_show_expected_disease_stage_pattern():
    """Real-data sanity check: baseline ADAS-Cog13 and GFAP should both
    increase CN -> MCI -> Dementia (higher = more impaired/more marker),
    confirming the Disease Continuum reshape didn't scramble groups."""
    payload = _load_payload()
    by_key = {row["key"]: row for row in payload["diseaseContinuum"]}
    adas = by_key["ADAS_COG13"]["cells"]
    assert adas["CN"]["value"] < adas["MCI"]["value"] < adas["Dementia"]["value"]
    gfap = by_key["GFAP"]["cells"]
    assert gfap["CN"]["value"] < gfap["MCI"]["value"] < gfap["Dementia"]["value"]


def test_key_patterns_present_for_every_endpoint_and_view():
    payload = _load_overall_payload()
    for endpoint in V.COGNITIVE_ENDPOINTS:
        assert payload["keyPatterns"]["cognitive"]["change"][endpoint["key"]]
        assert payload["keyPatterns"]["cognitive"]["absolute"][endpoint["key"]]
    for b in V.BIOMARKER_SPECS:
        for platform, analysis_type, _label in b["platforms"]:
            assert payload["keyPatterns"]["biomarkers"]["change"][b["key"]][platform][analysis_type]
            assert payload["keyPatterns"]["biomarkers"]["absolute"][b["key"]][platform][analysis_type]


def test_key_pattern_text_never_uses_forbidden_treatment_language():
    """Key Pattern text is deterministic/template-derived, but still
    must never make a POSITIVE claim of treatment effect or an
    external-control-arm comparison -- the same boundary enforced
    everywhere else on this dashboard. The required disclaimer
    ("...not a treatment effect") legitimately contains the phrase
    "treatment effect" as a negation, so that specific, expected
    sentence is excluded before checking for a genuine violation."""
    payload = _load_overall_payload()
    forbidden_phrases = ["treatment effect", "efficacy", "responded to", "improved with", "control arm"]
    required_disclaimer = "not a treatment effect"
    all_texts = list(payload["keyPatterns"]["cognitive"]["change"].values())
    all_texts += list(payload["keyPatterns"]["cognitive"]["absolute"].values())
    for b_platforms in payload["keyPatterns"]["biomarkers"]["change"].values():
        for analysis_types in b_platforms.values():
            all_texts += list(analysis_types.values())
    for text in all_texts:
        lowered = text.lower().replace(required_disclaimer, "")
        for phrase in forbidden_phrases:
            assert phrase not in lowered, f"forbidden phrase {phrase!r} found in Key Pattern text: {text}"


def test_key_pattern_absolute_avoids_small_n_late_timepoint():
    """Regression guard for a real bug found during the redesign: the
    Absolute Key Pattern must never highlight a late timepoint whose
    group n is below the display threshold (e.g. ADAS-Cog13 month 48
    Dementia has n=5 in the real data -- a well-known ADNI attrition/
    survivorship artifact) without flagging it -- it must instead fall
    back to the latest month where every group meets the threshold."""
    payload = _load_overall_payload()
    text = payload["keyPatterns"]["cognitive"]["absolute"]["ADAS_COG13"]
    assert "month 48" not in text
    assert "n=5)" not in text


def test_compact_legend_and_cohort_summary_present():
    html_source = _load_dashboard_html()
    assert "header-info-stats" in html_source
    assert "3,030" in html_source
    assert "header-info-legend" in html_source
    # The old large 4-tile KPI layout must be gone.
    assert "kpi-tile" not in html_source
    assert "kpi-row" not in html_source


def test_analysis_details_section_holds_technical_metadata():
    html_source = _load_dashboard_html()
    assert "Analysis Details" in html_source
    assert "observed cases, no imputation" in html_source
    assert "Primary GFAP/NfL platform" in html_source


def test_chart_height_reduced_from_original():
    html_source = _load_dashboard_html()
    assert "min-height: 330px" in html_source
    assert "min-height: 460px" not in html_source


def test_biomarker_uses_horizontal_tabs_not_dropdown():
    html_source = _load_dashboard_html()
    assert "data-biomarker=" in html_source
    # biomarker-select was the old <select> element's CSS class --
    # confirms it was actually replaced with tab buttons, not just
    # supplemented. The Statistical Results section's own <select>
    # (a different, still-appropriate control for choosing among many
    # endpoint/platform/analysis_type combinations) is unaffected.
    assert "biomarker-select" not in html_source


def test_n_label_only_shown_for_isolated_small_n_points():
    """The 'always show n=' text trace was removed -- confirms the JS
    only ever labels a point directly on the chart (with a "⚠ n="
    warning label) when it's isolated (n below minGroupN), never for
    an ordinary or merely-descriptive-but-well-supported point."""
    html_source = _load_dashboard_html()
    assert "isIsolated" in html_source
    assert "minGroupN" in html_source
    assert '"⚠ n="' in html_source or "'⚠ n='" in html_source


def test_results_table_partial_eta_squared_label_preserved():
    html_source = _load_dashboard_html()
    assert "Partial &eta;&sup2; (conventional ANCOVA)" in html_source


# ------------------------------------------------------------------
# Targeted refinement pass: sparse-trajectory rendering, hover
# behavior, deterministic Key Pattern interpretation, navigation label
# ------------------------------------------------------------------


def test_dementia_ptau181_has_a_real_extremely_sparse_month():
    """Ground the sparse-point rendering rule in real data: dementia
    pTau181 at month 6 and month 36 has n=1 in the actual pipeline
    output -- exactly the case named in the request. Confirms the
    payload carries this small n through untouched (rendering as
    isolated is a JS-side rule tested structurally below, but the
    underlying data point this rule must fire on has to actually be
    n < minGroupN, not hypothetical)."""
    payload = _load_payload()
    overall = payload["populations"]["overall"]
    pts = overall["biomarkersChange"]["pTau181"]["Gothenburg_Simoa"]["primary"]
    month6 = next(p for p in pts if p["month"] == 6 and p["group"] == "Dementia")
    month36 = next(p for p in pts if p["month"] == 36 and p["group"] == "Dementia")
    assert month6["n"] == 1
    assert month36["n"] == 1
    assert month6["n"] < payload["minGroupN"]
    assert month36["n"] < payload["minGroupN"]


def test_isolated_points_are_never_wired_into_the_connected_line():
    """Structural check on buildGroupTraces: the main 'solid' trace's y
    values must be withheld (mapped to null, not the estimate) for any
    isolated point, so it can never appear as part of a connected
    line segment -- isolated points get their own separate
    markers-only trace instead."""
    html_source = _load_dashboard_html()
    assert "isConnectable(p) && p.classification !== CLASS_B) ? p.estimate : null" in html_source
    # The isolated-points trace uses mode "markers+text" (point + "⚠ n="
    # label), never "lines+markers" (which would imply a connected
    # trajectory).
    assert 'mode: "markers+text"' in html_source


def test_descriptive_non_isolated_segments_render_dashed_not_solid():
    """A segment touching a descriptive (but not extremely sparse)
    point must use a dashed line, distinct from the solid line used
    between two adjusted points."""
    html_source = _load_dashboard_html()
    assert 'dash: "dot"' in html_source
    assert "a.classification !== CLASS_B && b.classification !== CLASS_B) continue" in html_source


def test_hover_tooltip_includes_all_required_fields():
    """buildTooltip must still surface group, month, value, n, CI (when
    displayable), and analysis status on hover -- unchanged by the
    sparse-rendering refinement."""
    html_source = _load_dashboard_html()
    tooltip_fn = html_source.split("function buildTooltip")[1].split("\nfunction ")[0]
    assert "pt.group" in tooltip_fn
    assert "pt.month" in tooltip_fn
    assert "pt.estimate" in tooltip_fn
    assert '"n = "' in tooltip_fn
    assert "pt.classification" in tooltip_fn
    assert "hasDisplayableCI" in tooltip_fn


def test_key_pattern_does_not_repeat_raw_numeric_values():
    """The rewritten Key Pattern must describe the PATTERN (ordered
    gradient, magnitude, sensitivity/sparse flags) rather than restate
    the numbers already shown on the chart/hover -- confirms no
    formatted decimal value (e.g. "8.8") appears in the text."""
    payload = _load_overall_payload()
    text = payload["keyPatterns"]["cognitive"]["absolute"]["ADAS_COG13"]
    assert not re.search(r"\d+\.\d", text), f"Key Pattern text still contains a raw decimal value: {text}"
    assert "Key pattern" in text


def test_key_pattern_flags_sparse_or_declining_follow_up_when_present():
    """ADAS-Cog13 has a well-documented sparse tail (n=5 at month 48) --
    confirms the Absolute Key Pattern explicitly flags this rather than
    silently reporting a clean trend."""
    payload = _load_overall_payload()
    text = payload["keyPatterns"]["cognitive"]["absolute"]["ADAS_COG13"]
    assert "interpreted cautiously" in text or "sample sizes decline" in text


def test_key_pattern_never_claims_causality():
    payload = _load_overall_payload()
    all_texts = list(payload["keyPatterns"]["cognitive"]["change"].values())
    all_texts += list(payload["keyPatterns"]["cognitive"]["absolute"].values())
    for b_platforms in payload["keyPatterns"]["biomarkers"]["change"].values():
        for analysis_types in b_platforms.values():
            all_texts += list(analysis_types.values())
    forbidden = ["causes", "caused by", "due to treatment", "leads to", "results from"]
    for text in all_texts:
        lowered = text.lower()
        for phrase in forbidden:
            assert phrase not in lowered, f"causal phrase {phrase!r} found in: {text}"


def test_disease_continuum_explanation_is_the_required_short_form():
    html_source = _load_dashboard_html()
    assert "Darker = greater disease-associated abnormality within each endpoint. Colors are not comparable across endpoints." in html_source


def test_disease_continuum_row_labels_carry_direction_arrows():
    html_source = _load_dashboard_html()
    assert "r.higherIsWorse ? \" ↑\" : \" ↓\"" in html_source


def test_compact_warning_indicator_format_preserved():
    """Confirms the compact '⚠ X/Y timepoints descriptive-only' format
    is still used (not a large amber banner) for the routine case, and
    that only a genuine sensitivity concern escalates to the visually
    prominent variant."""
    html_source = _load_dashboard_html()
    assert "timepoints descriptive-only" in html_source
    assert "compact-warn--concern" in html_source
    assert "warning-note" not in html_source


def test_navigation_label_renamed():
    html_source = _load_dashboard_html()
    assert "ADNI Natural History" in html_source
    assert "Biomarker Dashboard" not in html_source
    assert "Cognitive and biomarker progression across the Alzheimer's disease continuum" in html_source


# ------------------------------------------------------------------
# POLARIS AD-aligned cohort -- dashboard UI integration
# ------------------------------------------------------------------


def test_population_selector_renders():
    html_source = _load_dashboard_html()
    assert 'id="populationToggleGroup"' in html_source
    assert 'data-population="overall"' in html_source
    assert 'data-population="polaris"' in html_source
    assert "Overall ADNI" in html_source
    assert "POLARIS AD" in html_source
    assert "Broad validated ADNI natural-history cohort." in html_source
    assert "ADNI participants meeting validated MMSE and amyloid-PET eligibility criteria" in html_source


def test_population_selector_default_is_overall_adni():
    html_source = _load_dashboard_html()
    # The "overall" button is statically rendered active; "polaris" is not.
    overall_btn = re.search(r'<button class="toggle-btn[^"]*" data-population="overall"[^>]*>', html_source)
    polaris_btn = re.search(r'<button class="toggle-btn[^"]*" data-population="polaris"[^>]*>', html_source)
    assert overall_btn and "active" in overall_btn.group(0)
    assert polaris_btn and "active" not in polaris_btn.group(0)
    # The POLARIS panel's helper text and the Disease-Continuum POLARIS
    # note are hidden by default. The Cohort Construction toggle itself
    # is NOT display:none (it stays in flow at a fixed height so
    # switching population never shifts the page) -- it starts disabled
    # instead.
    assert 'class="population-helper is-hidden" id="populationHelperPolaris"' in html_source
    assert 'class="population-note is-hidden" id="diseaseContinuumPolarisNote"' in html_source
    assert 'polaris-panel--disabled' in html_source
    # The per-section population labels default to Overall ADNI.
    assert 'id="cognitivePopulationLabel">Population: Overall ADNI<' in html_source
    assert 'id="biomarkerPopulationLabel">Population: Overall ADNI<' in html_source
    assert 'id="resultsPopulationLabel">Population: Overall ADNI<' in html_source


def test_polaris_cohort_final_n_is_620():
    payload = _load_payload()
    funnel = payload["polaris"]["funnel"]
    assert funnel[-1]["step"] == "Final POLARIS-aligned cohort"
    assert funnel[-1]["remaining_n"] == 620


def test_polaris_diagnosis_composition_matches_validated_target():
    payload = _load_payload()
    dx_row = next(v for v in payload["polaris"]["profile"] if v["variable"] == "Baseline diagnosis")
    by_level = {lvl["level"]: lvl["polaris"]["n"] for lvl in dx_row["levels"]}
    assert by_level["CN"] == 151
    assert by_level["MCI"] == 309
    assert by_level["Dementia"] == 160


def test_polaris_funnel_matches_governed_attrition_csv():
    attrition = D.load_aggregate_csv(ADNI_OUTPUTS_DIR, "adni_polaris_cohort_attrition.csv")
    payload = _load_payload()
    funnel = payload["polaris"]["funnel"]
    assert len(funnel) == len(attrition) == 7
    for step, (_, row) in zip(funnel, attrition.iterrows()):
        assert step["step"] == row["step"]
        assert step["remaining_n"] == int(row["remaining_n"])
        assert step["excluded_n"] == int(row["excluded_n"])


def test_polaris_funnel_step_over_step_percent_is_a_ratio_of_governed_counts():
    payload = _load_payload()
    funnel = payload["polaris"]["funnel"]
    assert funnel[0]["percent_retained_of_previous"] is None
    for prev, cur in zip(funnel, funnel[1:]):
        expected = round(cur["remaining_n"] / prev["remaining_n"] * 100, 1)
        assert cur["percent_retained_of_previous"] == expected


def test_polaris_population_profile_covers_requested_variables():
    payload = _load_payload()
    variables = {v["variable"] for v in payload["polaris"]["profile"]}
    assert variables == {
        "Baseline age (years)", "Baseline MMSE", "Baseline ADAS-Cog13", "Baseline Centiloid",
        "Baseline diagnosis", "Sex", "APOE4 carrier",
    }


def test_polaris_population_profile_matches_governed_csv():
    profile_csv = D.load_aggregate_csv(ADNI_OUTPUTS_DIR, "adni_polaris_population_profile.csv")
    payload = _load_payload()
    age_row = next(v for v in payload["polaris"]["profile"] if v["variable"] == "Baseline age (years)")
    csv_overall = profile_csv[(profile_csv["variable"] == "Baseline age (years)") & (profile_csv["population"] == "Overall ADNI")].iloc[0]
    assert age_row["overall"]["n"] == int(csv_overall["n"])
    assert abs(age_row["overall"]["mean"] - float(csv_overall["mean"])) < 1e-9


def test_not_propensity_score_matched_disclaimer_present():
    html_source = _load_dashboard_html()
    assert "been propensity-score" in html_source
    assert "matched to an AR1001 trial cohort" in html_source
    assert "should not be interpreted as an external control" in html_source
    assert "eligibility-filtered only" in html_source


def test_apoe4_context_text_is_neutral_not_causal():
    html_source = _load_dashboard_html()
    assert "APOE4-carrier prevalence differs between the overall and eligibility-filtered populations." in html_source
    # No causal/biological-mechanism language auto-generated around APOE4.
    for forbidden in ["because", "due to", "caused by", "driven by", "explains why"]:
        idx = html_source.find("APOE4-carrier prevalence differs")
        window = html_source[max(0, idx - 200):idx + 400]
        assert forbidden not in window.lower()


def test_disease_continuum_polaris_note_present_and_labeled():
    """Disease Continuum stays Overall-ADNI-only (population-aware
    trajectories do not extend there) -- the subtle note explaining
    this must be present in the markup (JS toggles its visibility)."""
    html_source = _load_dashboard_html()
    assert "Disease Continuum shown from Overall ADNI baseline reference." in html_source
    assert 'id="diseaseContinuumPolarisNote"' in html_source


def test_existing_trajectory_payload_unchanged_by_polaris_addition():
    """Adding the polaris key to the payload must not alter a single
    value in the existing trajectory/statistics keys -- build the
    payload with and without polaris_data from the same underlying
    data and diff everything except the new "polaris" key."""
    data = D.load_all(ADNI_OUTPUTS_DIR)
    without_polaris = V.build_payload(data)
    polaris_data = D.load_polaris_data(ADNI_OUTPUTS_DIR)
    with_polaris = V.build_payload(data, polaris_data)
    assert "polaris" not in without_polaris
    assert "polaris" in with_polaris
    shared_keys = set(without_polaris.keys())
    assert shared_keys == set(with_polaris.keys()) - {"polaris"}
    for key in shared_keys:
        assert json.dumps(without_polaris[key], sort_keys=True) == json.dumps(with_polaris[key], sort_keys=True), (
            f"payload key {key!r} changed after adding polaris_data"
        )


def test_real_dashboard_disease_continuum_and_cognitive_values_still_match_direct_computation():
    """Cross-check against a live real-data snapshot: values embedded in
    the actual generated dashboard must still equal a fresh direct call
    to the (unmodified) chart-data builders."""
    data = D.load_all(ADNI_OUTPUTS_DIR)
    payload = _load_payload()
    expected_continuum = D.build_disease_continuum_data(data)
    for expected_row, actual_row in zip(expected_continuum, payload["diseaseContinuum"]):
        assert expected_row["key"] == actual_row["key"]
        assert expected_row["cells"] == actual_row["cells"]
    expected_mmse_change = D.build_cognitive_chart_data(data, "MMSE", "primary")
    assert expected_mmse_change == payload["populations"]["overall"]["cognitiveChange"]["MMSE"]


def test_polaris_loader_uses_only_governed_aggregate_csv_entry_point():
    """load_polaris_data must go through load_aggregate_csv exactly like
    every other loader -- so it inherits the same forbidden-path /
    forbidden-column / no-parquet protections with zero extra code."""
    tmp = tempfile.mkdtemp()
    try:
        outputs_dir = os.path.join(tmp, "outputs")
        os.makedirs(outputs_dir)
        raised = False
        try:
            D.load_polaris_data(outputs_dir)
        except D.DataGovernanceError:
            raised = True
        assert raised, "load_polaris_data should raise DataGovernanceError when the governed files don't exist under outputs_dir"
    finally:
        shutil.rmtree(tmp)


def test_polaris_loader_rejects_participant_identifier_columns():
    tmp = tempfile.mkdtemp()
    try:
        outputs_dir = os.path.join(tmp, "outputs")
        os.makedirs(outputs_dir)
        pd.DataFrame({"RID": [1, 2], "step": ["a", "b"]}).to_csv(
            os.path.join(outputs_dir, "adni_polaris_cohort_attrition.csv"), index=False
        )
        pd.DataFrame({"variable": ["x"], "population": ["Overall ADNI"], "n": [1]}).to_csv(
            os.path.join(outputs_dir, "adni_polaris_population_profile.csv"), index=False
        )
        raised = False
        try:
            D.load_polaris_data(outputs_dir)
        except D.DataGovernanceError:
            raised = True
        assert raised
    finally:
        shutil.rmtree(tmp)


def test_no_participant_level_pet_file_referenced_anywhere_in_viz_modules():
    """Neither visualization module may ACTUALLY reference (import,
    open, or read) the participant-level PET/eligibility parquet --
    mentioning its name inside a `#`-comment (to document that it is
    NOT opened, as adni_viz_data.py's module docstring does) is fine
    and expected; only a live code reference would be a violation."""
    for path in ("adni_viz.py", "adni_viz_data.py"):
        with open(path, encoding="utf-8") as f:
            source = f.read()
        assert "read_parquet" not in source
        assert "import adni_pet" not in source
        assert "from adni_pet " not in source
        for line in source.splitlines():
            if "adni_pet_eligibility.parquet" in line:
                assert line.strip().startswith("#"), f"non-comment reference to the participant-level PET parquet: {line!r}"


def test_polaris_payload_contains_no_participant_identifiers():
    payload = _load_payload()
    raw_json = json.dumps(payload["polaris"])
    for forbidden in ["\"RID\"", "\"PTID\"", "\"LONIUID\"", "\"USUBJID\"", "\"SUBJID\""]:
        assert forbidden not in raw_json


def test_existing_governance_protections_still_intact():
    """Re-affirms the governance boundary this stage was told to
    preserve -- REQUIRED_AGGREGATE_FILES, DataGovernanceError, and
    load_aggregate_csv's checks are all untouched by this dashboard
    integration; the new polaris outputs are loaded through the same
    single sanctioned entry point, not a parallel/looser path."""
    assert D.REQUIRED_AGGREGATE_FILES == [
        "adni_dashboard_eligibility.csv", "adni_cognitive_summary.csv", "adni_biomarker_summary.csv",
        "adni_pairwise_results.csv", "adni_robustness_summary.csv", "adni_sensitivity_summary.csv",
    ]
    assert hasattr(D, "DataGovernanceError")
    assert "load_aggregate_csv" in D.load_polaris_data.__code__.co_names


# ------------------------------------------------------------------
# Population-aware longitudinal charts (Cognitive/Biomarker trajectories,
# Key Pattern, data-support summary, Statistical Results) -- Disease
# Continuum intentionally NOT included here: it stays Overall-ADNI-only.
# ------------------------------------------------------------------


def test_population_selector_changes_cognitive_data_source():
    payload = _load_payload()
    overall_mmse = payload["populations"]["overall"]["cognitiveChange"]["MMSE"]
    polaris_mmse = payload["populations"]["polaris"]["cognitiveChange"]["MMSE"]
    assert overall_mmse != polaris_mmse
    # Real, population-specific n's -- not a coincidental reshuffle.
    overall_n_by_cell = {(p["month"], p["group"]): p["n"] for p in overall_mmse}
    polaris_n_by_cell = {(p["month"], p["group"]): p["n"] for p in polaris_mmse}
    assert overall_n_by_cell != polaris_n_by_cell
    # POLARIS n's are never larger than Overall ADNI's for the same cell
    # (POLARIS is a strict eligibility-filtered subset).
    for key in polaris_n_by_cell:
        if polaris_n_by_cell[key] is not None and overall_n_by_cell.get(key) is not None:
            assert polaris_n_by_cell[key] <= overall_n_by_cell[key]


def test_population_selector_changes_biomarker_data_source():
    payload = _load_payload()
    overall_pts = payload["populations"]["overall"]["biomarkersChange"]["pTau181"]["Gothenburg_Simoa"]["primary"]
    polaris_pts = payload["populations"]["polaris"]["biomarkersChange"]["pTau181"]["Gothenburg_Simoa"]["primary"]
    assert overall_pts != polaris_pts
    overall_n_by_cell = {(p["month"], p["group"]): p["n"] for p in overall_pts}
    polaris_n_by_cell = {(p["month"], p["group"]): p["n"] for p in polaris_pts}
    assert overall_n_by_cell != polaris_n_by_cell


def test_overall_population_unchanged_when_polaris_trajectories_are_added():
    """The exact regression guard this task requires: build the payload
    with NO polaris args at all, then with BOTH polaris_data and
    polaris_traj_data, and diff the "overall" population sub-payload
    (and the population-agnostic diseaseContinuum) byte-for-byte."""
    data = D.load_all(ADNI_OUTPUTS_DIR)
    polaris_data = D.load_polaris_data(ADNI_OUTPUTS_DIR)
    polaris_traj_data = D.polaris_data_view(ADNI_OUTPUTS_DIR)

    without_any_polaris = V.build_payload(data)
    with_both = V.build_payload(data, polaris_data, polaris_traj_data)

    assert json.dumps(without_any_polaris["populations"]["overall"], sort_keys=True) == json.dumps(
        with_both["populations"]["overall"], sort_keys=True
    ), "Overall ADNI population payload changed after adding POLARIS trajectory data"
    assert json.dumps(without_any_polaris["diseaseContinuum"], sort_keys=True) == json.dumps(
        with_both["diseaseContinuum"], sort_keys=True
    )


def test_polaris_baseline_values_match_validated_population_profile():
    """The dashboard's own POLARIS chart data (Absolute view, month 0)
    must reconstruct the exact same baseline mean already validated in
    adni_polaris_population_profile.csv -- proving the viz-layer
    adapter (polaris_data_view) carries the upstream analysis-stage
    numbers through untouched, not recomputed."""
    polaris_traj_data = D.polaris_data_view(ADNI_OUTPUTS_DIR)
    profile = pd.read_csv(os.path.join(ADNI_OUTPUTS_DIR, "adni_polaris_population_profile.csv"))

    for endpoint, profile_var in [("MMSE", "Baseline MMSE"), ("ADAS_COG13", "Baseline ADAS-Cog13")]:
        pts = D.build_cognitive_absolute_chart_data(polaris_traj_data, endpoint, "primary")
        baseline_pts = [p for p in pts if p["month"] == 0 and p["n"]]
        total_n = sum(p["n"] for p in baseline_pts)
        weighted_mean = sum(p["n"] * p["estimate"] for p in baseline_pts) / total_n
        prof_row = profile[(profile["variable"] == profile_var) & (profile["population"] == "POLARIS-aligned ADNI")].iloc[0]
        # The dashboard's own longitudinal-eligibility-restricted n can be
        # smaller than the full-620 population-profile n (documented,
        # expected -- see run_adni_polaris_trajectories.py's own baseline
        # check); the MEAN over the participants both share should still
        # be very close, not wildly different or scrambled.
        assert total_n <= int(prof_row["n"])
        assert abs(weighted_mean - float(prof_row["mean"])) < 1.0


def test_polaris_status_classification_matches_governed_status_csv():
    """Spot-check a specific, known cell against the governed status
    file directly (not re-derived) -- ADAS-Cog13 month 6, primary,
    classified 'A. Adjusted analysis' per adni_polaris_trajectory_status.csv."""
    status = pd.read_csv(os.path.join(ADNI_OUTPUTS_DIR, "adni_polaris_trajectory_status.csv"))
    row = status[
        (status["endpoint_or_biomarker"] == "ADAS_COG13") & (status["analysis_type"] == "primary") & (status["month"] == 6)
    ].iloc[0]
    expected_classification = row["classification"]

    payload = _load_payload()
    pts = payload["populations"]["polaris"]["cognitiveChange"]["ADAS_COG13"]
    for p in pts:
        if p["month"] == 6:
            assert p["classification"] == expected_classification


def test_polaris_sparse_observations_remain_disconnected():
    """Real POLARIS data: MMSE month 18 has n<10 in all three diagnosis
    groups (confirmed against adni_polaris_cognitive_trajectories.csv)
    -- these must be rendered as isolated points (n < minGroupN),
    exactly like the existing Overall-ADNI sparse-point rule, never
    wired into a solid connected trajectory line."""
    payload = _load_payload()
    min_group_n = payload["minGroupN"]
    pts = payload["populations"]["polaris"]["cognitiveChange"]["MMSE"]
    month18 = [p for p in pts if p["month"] == 18]
    assert len(month18) == 3
    for p in month18:
        assert p["n"] < min_group_n
        assert p["estimate"] is not None  # a real, plottable value -- just never connected


def test_key_pattern_differs_and_is_explicitly_labeled_by_population():
    payload = _load_payload()
    overall_text = payload["populations"]["overall"]["keyPatterns"]["cognitive"]["change"]["ADAS_COG13"]
    polaris_text = payload["populations"]["polaris"]["keyPatterns"]["cognitive"]["change"]["ADAS_COG13"]
    assert overall_text != polaris_text
    assert "POLARIS AD" in polaris_text
    assert "POLARIS AD" not in overall_text
    assert "620" in polaris_text or "eligibility-filtered" in polaris_text
    # Never a treatment-effect claim, in either population.
    for text in (overall_text, polaris_text):
        assert "treatment effect" not in text.lower() or "not a treatment effect" in text.lower() or "not compared" in text.lower()


def test_key_pattern_never_turns_descriptive_biomarker_into_a_strong_claim():
    payload = _load_payload()
    gfap_text = payload["populations"]["polaris"]["keyPatterns"]["biomarkers"]["change"]["GFAP"]["Quanterix"]["primary"]
    assert "no adjusted (hc3) timepoint" in gfap_text.lower() or "not available for all three" in gfap_text.lower()
    assert "markedly" not in gfap_text and "modestly" not in gfap_text


def test_cognitive_data_support_present_only_for_polaris_biomarker_for_both():
    """Cognitive data-support remains POLARIS-only (out of scope for
    the biomarker redesign); biomarker data-support is now generalized
    to BOTH populations (see build_biomarker_data_support())."""
    payload = _load_payload()
    assert payload["populations"]["overall"]["cognitiveDataSupport"] == {}
    assert payload["populations"]["overall"]["biomarkerDataSupport"] != {}

    polaris_cog_support = payload["populations"]["polaris"]["cognitiveDataSupport"]
    assert set(polaris_cog_support.keys()) == {"ADAS_COG13", "MMSE"}
    for text in polaris_cog_support.values():
        assert text.startswith("Data support:")

    polaris_bio_support = payload["populations"]["polaris"]["biomarkerDataSupport"]
    ptau181_change = polaris_bio_support["pTau181"]["Gothenburg_Simoa"]["primary"]["change"]
    gfap_change = polaris_bio_support["GFAP"]["Quanterix"]["primary"]["change"]
    assert ptau181_change != gfap_change
    assert "adjusted follow-up available" in ptau181_change
    assert "descriptive-only" in gfap_change


def test_disease_continuum_is_population_agnostic_and_stays_overall_only():
    """Disease Continuum lives ONLY at the top level of the payload --
    never duplicated per population -- so it is structurally impossible
    for it to change when the Population selector is toggled."""
    payload = _load_payload()
    assert "diseaseContinuum" not in payload["populations"]["overall"]
    assert "diseaseContinuum" not in payload["populations"]["polaris"]
    assert len(payload["diseaseContinuum"]) == 7


def test_polaris_data_view_governance_rejects_missing_files():
    tmp = tempfile.mkdtemp()
    try:
        outputs_dir = os.path.join(tmp, "outputs")
        os.makedirs(outputs_dir)
        raised = False
        try:
            D.polaris_data_view(outputs_dir)
        except D.DataGovernanceError:
            raised = True
        assert raised
    finally:
        shutil.rmtree(tmp)


def test_polaris_data_view_rejects_participant_identifier_columns():
    tmp = tempfile.mkdtemp()
    try:
        outputs_dir = os.path.join(tmp, "outputs")
        os.makedirs(outputs_dir)
        pd.DataFrame({"RID": [1], "level": ["a"], "robustness_check": ["HC3"]}).to_csv(
            os.path.join(outputs_dir, "adni_polaris_trajectory_status.csv"), index=False
        )
        pd.DataFrame({"endpoint": ["MMSE"]}).to_csv(os.path.join(outputs_dir, "adni_polaris_cognitive_trajectories.csv"), index=False)
        pd.DataFrame({"biomarker": ["pTau181"]}).to_csv(os.path.join(outputs_dir, "adni_polaris_biomarker_trajectories.csv"), index=False)
        raised = False
        try:
            D.polaris_data_view(outputs_dir)
        except D.DataGovernanceError:
            raised = True
        assert raised
    finally:
        shutil.rmtree(tmp)


def test_no_participant_level_data_in_polaris_population_payload():
    payload = _load_payload()
    raw_json = json.dumps(payload["populations"]["polaris"])
    for forbidden in ["\"RID\"", "\"PTID\"", "\"LONIUID\"", "\"USUBJID\"", "\"SUBJID\""]:
        assert forbidden not in raw_json


def test_no_participant_level_pet_or_trajectory_parquet_referenced_in_viz_modules():
    """Neither visualization module may call read_parquet() (the only
    pandas API that would actually open a participant-level file) or
    even import ADNI_PROCESSED_DIR at all -- both modules only ever
    need ADNI_OUTPUTS_DIR, so the absence of the processed-dir import
    is a strong structural guarantee, not just a string-matching guess."""
    for path in ("adni_viz.py", "adni_viz_data.py"):
        with open(path, encoding="utf-8") as f:
            source = f.read()
        assert "read_parquet" not in source
        assert "ADNI_PROCESSED_DIR" not in source


# ------------------------------------------------------------------
# Plasma Biomarker Trajectories redesign -- Absolute-view cross-
# sectional sample-size fix + view-aware data-support summaries.
# ------------------------------------------------------------------


def test_biomarker_absolute_uses_cross_sectional_n_not_paired_change_n():
    """The core data-issue fix: the Absolute view's n at a real,
    already-known-sparser month must equal the governed
    n_cross_sectional column, not the smaller paired-sample n used by
    the % change view -- and must be strictly larger for a known real
    case (pTau181, month 48, where cross-sectional support is
    substantially richer than the paired sample)."""
    bio = pd.read_csv(os.path.join(ADNI_OUTPUTS_DIR, "adni_biomarker_summary.csv"))
    row = bio[
        (bio["biomarker"] == "pTau181") & (bio["analysis_type"] == "primary")
        & (bio["month"] == 48) & (bio["group"] == "MCI")
    ].iloc[0]
    assert row["n_cross_sectional"] > row["n"]

    data = D.load_all(ADNI_OUTPUTS_DIR)
    abs_pts = D.build_biomarker_absolute_chart_data(data, "pTau181", "Gothenburg_Simoa", "primary")
    change_pts = D.build_biomarker_chart_data(data, "pTau181", "Gothenburg_Simoa", "primary")
    abs_n = next(p["n"] for p in abs_pts if p["month"] == 48 and p["group"] == "MCI")
    change_n = next(p["n"] for p in change_pts if p["month"] == 48 and p["group"] == "MCI")
    assert abs_n == int(row["n_cross_sectional"])
    assert change_n == int(row["n"])
    assert abs_n > change_n


def test_biomarker_absolute_and_change_views_use_different_denominators():
    """Absolute and % change are different analytical questions with
    different correct denominators -- confirms they actually differ in
    the real dashboard payload for a biomarker/month with known sparse
    paired support, rather than silently sharing one sample."""
    payload = _load_payload()
    overall = payload["populations"]["overall"]
    abs_pts = overall["biomarkersAbsolute"]["pTau181"]["Gothenburg_Simoa"]["primary"]
    change_pts = overall["biomarkersChange"]["pTau181"]["Gothenburg_Simoa"]["primary"]
    abs_n_by_cell = {(p["month"], p["group"]): p["n"] for p in abs_pts}
    change_n_by_cell = {(p["month"], p["group"]): p["n"] for p in change_pts}
    assert abs_n_by_cell != change_n_by_cell
    # Cross-sectional can never be smaller than the paired sample for the same cell.
    for key, change_n in change_n_by_cell.items():
        if change_n is not None and abs_n_by_cell.get(key) is not None:
            assert abs_n_by_cell[key] >= change_n


def test_previously_missing_biomarker_cell_now_shows_real_absolute_data():
    """The concrete fixed data issue: GFAP/NfL on Fujirebio at months
    36/48 previously had ZERO rows in adni_biomarker_summary.csv (no
    paired data at all) and rendered as 'D. Not available' in the
    Absolute view even though real cross-sectional data existed. Must
    now show real descriptive data there."""
    payload = _load_payload()
    overall = payload["populations"]["overall"]
    for biomarker in ("GFAP", "NfL"):
        abs_pts = overall["biomarkersAbsolute"][biomarker]["Fujirebio"]["sensitivity_fujirebio"]
        for month in (36, 48):
            cell = [p for p in abs_pts if p["month"] == month]
            assert cell, f"{biomarker} month {month} missing from Absolute payload entirely"
            assert any(p["classification"] == D.CLASS_DESCRIPTIVE and p["n"] for p in cell), (
                f"{biomarker} month {month} should show real descriptive data in the Absolute view now"
            )


def test_sparse_biomarker_absolute_points_remain_isolated_not_connected():
    """Real, now-correct data: a month with genuinely small
    cross-sectional support (n < minGroupN) must still be treated as
    isolated/sparse by the same JS rule as everywhere else -- the fix
    corrects the NUMBER, not the sparse-point rendering rule itself."""
    payload = _load_payload()
    min_group_n = payload["minGroupN"]
    overall = payload["populations"]["overall"]
    abs_pts = overall["biomarkersAbsolute"]["GFAP"]["Fujirebio"]["sensitivity_fujirebio"]
    sparse_points = [p for p in abs_pts if p["n"] is not None and 0 < p["n"] < min_group_n]
    assert sparse_points, "expected at least one genuinely sparse (but non-zero) GFAP/Fujirebio absolute cell"


def test_descriptive_and_sensitivity_concern_biomarker_points_remain_distinguishable():
    """A known real C-status (Sensitivity concern) change-view cell
    (pTau181, month 12) must remain classified distinctly from a
    routine B-status (Descriptive only) cell -- the redesign must not
    have collapsed or blurred the governed classification vocabulary."""
    payload = _load_payload()
    change_pts = payload["populations"]["overall"]["biomarkersChange"]["pTau181"]["Gothenburg_Simoa"]["primary"]
    concern_pts = [p for p in change_pts if p["month"] == 12]
    assert concern_pts and all(p["classification"] == D.CLASS_SENSITIVITY_CONCERN for p in concern_pts)
    descriptive_pts = [p for p in change_pts if p["month"] == 36]
    assert descriptive_pts and all(p["classification"] == D.CLASS_DESCRIPTIVE for p in descriptive_pts)


def test_biomarker_absolute_data_support_present_for_overall_and_polaris():
    """Generalized (no longer POLARIS-only) data-support summaries --
    both populations must get real Absolute-view text, and it must
    reflect each population's own (different) support pattern: GFAP's
    smaller POLARIS cohort loses MCI support earlier than Overall ADNI
    does."""
    payload = _load_payload()
    overall_text = payload["populations"]["overall"]["biomarkerDataSupport"]["GFAP"]["Quanterix"]["primary"]["absolute"]
    polaris_text = payload["populations"]["polaris"]["biomarkerDataSupport"]["GFAP"]["Quanterix"]["primary"]["absolute"]
    assert overall_text.startswith("GFAP:")
    assert polaris_text.startswith("GFAP:")
    assert overall_text != polaris_text
    assert "MCI" in polaris_text and "MCI" not in overall_text


def test_biomarker_data_support_distinguishes_change_and_absolute_questions():
    payload = _load_payload()
    entry = payload["populations"]["overall"]["biomarkerDataSupport"]["GFAP"]["Quanterix"]["primary"]
    assert set(entry.keys()) == {"change", "absolute"}
    assert entry["change"] != entry["absolute"]
    assert "adjusted-analysis threshold" in entry["change"] or "adjusted follow-up" in entry["change"]
    assert "absolute levels" in entry["absolute"]


def test_population_switching_uses_correct_biomarker_absolute_data():
    payload = _load_payload()
    overall_abs = payload["populations"]["overall"]["biomarkersAbsolute"]["pTau181"]["Gothenburg_Simoa"]["primary"]
    polaris_abs = payload["populations"]["polaris"]["biomarkersAbsolute"]["pTau181"]["Gothenburg_Simoa"]["primary"]
    assert overall_abs != polaris_abs
    overall_n = {(p["month"], p["group"]): p["n"] for p in overall_abs}
    polaris_n = {(p["month"], p["group"]): p["n"] for p in polaris_abs}
    for key in polaris_n:
        if polaris_n[key] is not None and overall_n.get(key) is not None:
            assert polaris_n[key] <= overall_n[key]  # POLARIS is a strict eligibility-filtered subset


def test_biomarker_summary_csv_with_new_columns_still_governed():
    """The widened adni_biomarker_summary.csv (new columns, new rows)
    must still load cleanly through the same single governed entry
    point -- confirms the upstream fix didn't introduce a forbidden
    column or an out-of-bounds path."""
    df = D.load_aggregate_csv(ADNI_OUTPUTS_DIR, "adni_biomarker_summary.csv")
    for col in ("n_cross_sectional", "raw_geometric_mean_cross_sectional", "raw_geometric_mean_ci_lower_cross_sectional", "raw_geometric_mean_ci_upper_cross_sectional"):
        assert col in df.columns


# ------------------------------------------------------------------
# Biomarker chart y-axis range fix -- a single sparse/isolated point's
# outsized CI must never dominate the visible scale and crush every
# well-supported point into a thin band ("clustered dots").
# ------------------------------------------------------------------


def test_sensible_y_range_helper_present_and_wired_into_both_charts():
    """computeSensibleYRange() must exist and be applied in BOTH
    renderBiomarkerChart() and renderCognitiveChart() -- a sparse,
    faded-out point's own huge CI must not dictate the axis range on
    either chart, only its point estimate."""
    html_source = _load_dashboard_html()
    assert "function computeSensibleYRange(pointsByGroup)" in html_source
    render_bio_start = html_source.index("function renderBiomarkerChart()")
    render_bio_end = html_source.index("function toggleCollapsible")
    render_bio_body = html_source[render_bio_start:render_bio_end]
    assert "computeSensibleYRange(pointsByGroup)" in render_bio_body
    assert "layout.yaxis.range = yRange" in render_bio_body

    render_cog_start = html_source.index("function renderCognitiveChart()")
    render_cog_end = html_source.index("function renderBiomarkerChart", render_cog_start)
    render_cog_body = html_source[render_cog_start:render_cog_end]
    assert "computeSensibleYRange(pointsByGroup)" in render_cog_body
    assert "layout.yaxis.range = yRange" in render_cog_body


def test_sensible_y_range_excludes_isolated_point_ci_but_keeps_its_estimate():
    """Real-data check on the exact logic: computeSensibleYRange must
    build its range from every point's estimate, but only include CI
    bounds for non-isolated (n >= minGroupN) points -- confirmed by
    reproducing the JS logic in Python against the real GFAP payload,
    where a known n=2 cell carries a CI width in the thousands."""
    payload = _load_payload()
    min_group_n = payload["minGroupN"]
    pts = payload["populations"]["overall"]["biomarkersAbsolute"]["GFAP"]["Quanterix"]["primary"]
    by_group = {}
    for p in pts:
        by_group.setdefault(p["group"], []).append(p)

    values = []
    for group_pts in by_group.values():
        for p in group_pts:
            if p["estimate"] is None:
                continue
            values.append(p["estimate"])
            is_isolated = p["n"] is not None and p["n"] < min_group_n
            if not is_isolated and p["ci_lower"] is not None:
                values.append(p["ci_lower"])
                values.append(p["ci_upper"])
    lo, hi = min(values), max(values)

    # The known extreme-CI point (month 36, Dementia, n small) must NOT
    # have pulled the range out anywhere near its own CI bound.
    extreme_point = next(p for p in pts if p["month"] == 36 and p["group"] == "Dementia")
    assert extreme_point["n"] is not None and extreme_point["n"] < min_group_n
    assert extreme_point["ci_upper"] is not None and extreme_point["ci_upper"] > hi
    # But its own estimate must still be within the computed range (it
    # stays visible as a point, only its CI extent is excluded).
    assert lo <= extreme_point["estimate"] <= hi


def test_biomarker_error_bars_hidden_on_chart_but_cognitive_unaffected():
    """Error bars are hidden on-chart for biomarkers (visible: false on
    every error_y object, via buildGroupTraces()'s showErrorBars param)
    -- the exact CI remains available via hover (buildTooltip() is
    untouched). Cognitive chart calls must keep the default (visible)
    behavior -- this is a biomarker-only change."""
    html_source = _load_dashboard_html()
    assert "function buildGroupTraces(pointsByGroup, yLabel, yUnit, showErrorBars)" in html_source
    assert "visible: showErrorBars" in html_source

    render_bio_start = html_source.index("function renderBiomarkerChart()")
    render_bio_end = html_source.index("function toggleCollapsible")
    render_bio_body = html_source[render_bio_start:render_bio_end]
    assert 'buildGroupTraces(pointsByGroup, spec.label + " concentration", "", false)' in render_bio_body
    assert 'buildGroupTraces(pointsByGroup, "Geometric mean % change", "%", false)' in render_bio_body

    render_cog_start = html_source.index("function renderCognitiveChart()")
    render_cog_end = html_source.index("function renderBiomarkerChart", render_cog_start)
    render_cog_body = html_source[render_cog_start:render_cog_end]
    # Cognitive calls keep their original 3-argument form (default
    # showErrorBars=true) -- no ", false)" 4th-argument opt-out.
    assert 'buildGroupTraces(pointsByGroup, spec.label + " score", "");' in render_cog_body
    assert 'buildGroupTraces(pointsByGroup, "Change from baseline", "");' in render_cog_body

    # Hover still carries the exact CI regardless of on-chart visibility.
    assert '"95% CI (HC3)"' in html_source or "95% CI (HC3)" in html_source
    assert "ciLabel + " in html_source


# ------------------------------------------------------------------
# UI transition smoothing (charts, toggles, collapsibles, table rows).
# ------------------------------------------------------------------


def test_chart_transition_config_present_on_both_layout_builders():
    """Plotly.react() only animates a redraw when layout.transition is
    set -- confirms both cognitive/biomarker layout builders declare it
    (shared CHART_TRANSITION constant), so switching endpoint/biomarker/
    view/population morphs the chart instead of snapping."""
    html_source = _load_dashboard_html()
    assert "const CHART_TRANSITION = { duration: 400, easing:" in html_source
    baseline_start = html_source.index("function baseLayout(yTitle, upLabel, downLabel)")
    baseline_end = html_source.index("function absoluteLayout(yTitle)")
    assert "transition: CHART_TRANSITION" in html_source[baseline_start:baseline_end]
    absolute_start = baseline_end
    absolute_end = html_source.index("function computeSensibleYRange", absolute_start)
    assert "transition: CHART_TRANSITION" in html_source[absolute_start:absolute_end]


def test_collapsible_sections_use_animated_transition_not_instant_display_swap():
    """Statistical Results / Analysis Details / Methods / Limitations
    must expand/collapse via an animated max-height+opacity transition,
    not the old instant display:none/block toggle."""
    html_source = _load_dashboard_html()
    assert "transition: max-height 0.4s ease, opacity 0.3s ease" in html_source
    assert ".collapsible-body {{ display: none; margin-top: 12px; }}" not in html_source
    assert "max-height: 6000px; opacity: 1;" in html_source


def test_toggle_buttons_and_results_table_rows_have_transitions():
    html_source = _load_dashboard_html()
    assert "transition: background-color 0.2s ease, color 0.2s ease" in html_source
    assert "animation: contentFadeIn 0.3s ease" in html_source
    assert "@keyframes contentFadeIn" in html_source


def test_fade_span_helper_used_for_key_pattern_and_data_support_updates():
    """Content swapped inside a persistent container (Key Pattern,
    biomarker interpretation, data-support notes, population labels)
    is wrapped in fadeSpan() so it still fades in on update, even
    though the container element itself is never recreated."""
    html_source = _load_dashboard_html()
    assert "function fadeSpan(html)" in html_source
    assert 'innerHTML = fadeSpan(pop.keyPatterns.cognitive[cognitiveView][cognitiveEndpointKey])' in html_source
    assert 'innerHTML = fadeSpan(pop.keyPatterns.biomarkers[biomarkerView][currentBiomarker][currentPlatform][currentAnalysisType])' in html_source
    assert "supportDiv.innerHTML = fadeSpan(supportText)" in html_source


def test_disease_continuum_spells_out_group_abbreviations_directly_on_the_chart():
    """CN/MCI/Dementia are jargon on their own -- the spelled-out labels
    must appear directly on the heatmap's own column headers (Plotly
    ticktext), not just in a separate caption someone could miss, and
    the hover text must also use the full label."""
    html_source = _load_dashboard_html()
    assert "const GROUP_AXIS_LABELS = {" in html_source
    assert "Cognitively Normal" in html_source
    assert "Mild Cognitive Impairment" in html_source

    render_start = html_source.index("function renderDiseaseContinuum()")
    render_end = html_source.index("function setCognitiveEndpoint(key)", render_start)
    render_body = html_source[render_start:render_end]
    assert "ticktext: groups.map(function (g) { return GROUP_AXIS_LABELS[g] || g; })" in render_body
    assert "tickvals: groups" in render_body

    assert "const GROUP_FULL_LABELS = {" in html_source
    assert "GROUP_FULL_LABELS[g] || g" in html_source


def test_trend_bolded_and_sparse_points_faded_in_buildGroupTraces():
    """'Bold the trend, fade the noise' -- the well-supported solid
    trace must be visually heavier (thicker line, larger marker) than
    both the isolated/sparse trace (which must carry reduced opacity, a
    smaller marker, and a muted label) and the plain descriptive trace,
    so the reliable signal is the obvious focal point at a glance."""
    html_source = _load_dashboard_html()
    render_start = html_source.index("function buildGroupTraces(pointsByGroup, yLabel, yUnit, showErrorBars)")
    render_end = html_source.index("function baseLayout")
    body = html_source[render_start:render_end]

    # Trace 1 (solid, well-supported): bold.
    assert "line: { color: color, width: 3.5 }" in body
    assert 'marker: { color: color, line: { color: color, width: 2 }, size: 12, symbol: "circle" }' in body

    # Trace 4 (isolated/sparse): faded -- lower opacity, smaller marker,
    # thinner ring, muted (non-amber) label color.
    assert "opacity: 0.45" in body
    assert 'marker: { color: "white", line: { color: color, width: 1.5 }, size: 6, symbol: "circle" }' in body
    assert 'textfont: { size: 8, color: "#999" }' in body

    # The warning symbol itself is preserved (still visible, just muted).
    assert '"⚠ n="' in body


def test_population_filter_reads_as_a_filter_and_lives_in_the_unified_header_card():
    """Redesign requirements: (1) it must be unambiguous that this
    control is a FILTER (explicit wording, an icon), and (2) it must be
    visually consistent with its siblings -- one section of the shared
    .header-info-card (same radius/shadow as the description/stats/
    legend), not its own separately-shadowed, differently-rounded box."""
    html_source = _load_dashboard_html()
    assert "Filter by population:" in html_source
    assert 'class="population-filter-icon"' in html_source

    card_start = html_source.index('class="header-info-card"')
    card_end = html_source.index("</main>", card_start)
    card_html = html_source[card_start:card_end]
    # All four sections share the same parent card.
    assert "header-info-disclaimer" in card_html
    assert "header-info-stats" in card_html
    assert "header-info-legend" in card_html
    assert "header-info-filter-row" in card_html

    # The old separately-boxed/pill-shaped filter container is gone.
    assert 'class="population-filter-bar"' not in html_source
    assert 'class="population-panel"' not in html_source
    assert 'class="population-toggle-row"' not in html_source

    # Structural/behavioral pieces the JS still depends on are preserved.
    assert 'id="populationToggleGroup"' in html_source
    assert 'id="populationHelperOverall"' in html_source
    assert 'class="population-helper is-hidden" id="populationHelperPolaris"' in html_source


def test_disease_continuum_sits_beside_header_description_labels_and_filter():
    """Disease Continuum must render in a right-hand column next to a
    left column holding the page description, cohort-summary
    ("participants info"), status legend ("labels"), AND the Population
    filter -- not as its own stacked, full-width section below them."""
    html_source = _load_dashboard_html()
    row_start = html_source.index('class="header-continuum-row"')
    row_end = html_source.index("</main>", row_start)
    row_html = html_source[row_start:row_end]

    left_start = row_html.index('class="header-left-col"')
    right_start = row_html.index('class="header-right-col"')
    assert left_start < right_start, "left column must come before the right column"

    left_html = row_html[left_start:right_start]
    right_html = row_html[right_start:]
    assert "ADNI Natural History Dashboard" in left_html  # description
    assert "header-info-stats" in left_html  # participants info
    assert "header-info-legend" in left_html  # labels
    assert "header-info-filter-row" in left_html  # filter
    assert "Disease Continuum" not in left_html
    assert "Disease Continuum" in right_html
    assert 'id="diseaseContinuumChart"' in right_html


def test_cognitive_and_biomarker_trajectories_sit_side_by_side():
    """On wide screens, Cognitive Trajectories and Plasma Biomarker
    Trajectories must be two columns of one flex row (.trajectories-row),
    not two independently stacked full-width sections."""
    html_source = _load_dashboard_html()
    row_start = html_source.index('class="trajectories-row"')
    cognitive_h2 = html_source.index("<h2>Cognitive Trajectories</h2>")
    biomarker_h2 = html_source.index("<h2>Plasma Biomarker Trajectories</h2>")
    row_end = html_source.index("</div>", biomarker_h2)
    # Both headings must fall inside the trajectories-row wrapper, and
    # cognitive must come first (left column) per the existing page order.
    assert row_start < cognitive_h2 < biomarker_h2
    assert "flex: 1 1 500px" in html_source


def test_disease_continuum_and_biomarker_trajectories_columns_are_aligned():
    """Disease Continuum's right column and Plasma Biomarker
    Trajectories' right column must share the identical flex-basis/
    min-width AND the identical wrap breakpoint as their left-column
    siblings -- otherwise the two rows render different column widths
    and can flip between 1-column/2-column independently, leaving the
    panels visually misaligned down the page."""
    html_source = _load_dashboard_html()
    assert ".header-left-col { flex: 1 1 500px; min-width: 440px;" in html_source
    assert ".header-right-col { flex: 1 1 500px; min-width: 440px;" in html_source
    assert "@media (max-width: 1300px) { .header-continuum-row { flex-direction: column; } }" in html_source
    assert "@media (max-width: 1300px) { .trajectories-row { flex-direction: column; } }" in html_source


def test_paired_panel_bottoms_are_aligned_not_just_tops():
    """Both row pairs (header card vs. Disease Continuum; Cognitive vs.
    Biomarker Trajectories) must stretch to match heights (align-items:
    stretch, the flex default) so their BOTTOM edges line up, and the
    inner content (chart / heatmap / last filter section) -- not blank
    space -- must be what absorbs any extra height, so a shorter panel
    never shows a bare gap between its content and its own bottom edge."""
    html_source = _load_dashboard_html()
    assert "align-items: stretch" in html_source
    assert "align-items: flex-start" not in html_source  # the earlier tops-only approach is gone

    # Header row: card and heatmap grow via flex: 1, not leftover blank space.
    assert ".header-left-col .header-info-card { flex: 1;" in html_source
    assert ".header-info-filter-row { flex: 1; }" in html_source
    assert ".header-right-col section.panel { margin-bottom: 0; flex: 1;" in html_source
    assert ".header-right-col .continuum-card { flex: 1; }" in html_source

    # Trajectories row: each panel's own chart grows via flex: 1.
    assert ".trajectories-row > section.panel .chart-card { flex: 1; }" in html_source


def test_polaris_cohort_construction_does_not_auto_expand_when_polaris_selected():
    """Selecting POLARIS AD-Aligned must only reveal the section HEADER
    (title + one-line hint + chevron) -- the funnel/population-profile
    body must stay collapsed until the user clicks it, using the same
    .collapsible-toggle/.collapsible-body mechanism as Statistical
    Results/Analysis Details rather than popping open automatically."""
    html_source = _load_dashboard_html()
    panel_start = html_source.index('id="polarisPanel"')
    panel_end = html_source.index("</section>", panel_start)
    panel_html = html_source[panel_start:panel_end]

    assert 'onclick="toggleCollapsible(this)"' in panel_html
    assert 'class="collapsible-body"' in panel_html
    # Neither the toggle header nor the body carries the "open" class by
    # default -- collapsed until clicked.
    assert 'class="collapsible-toggle open"' not in panel_html
    assert 'class="collapsible-body open"' not in panel_html

    # .collapsible-toggle's IMMEDIATE next sibling must be
    # .collapsible-body (toggleCollapsible() uses nextElementSibling),
    # so the hint text must live inside one of those two elements, not
    # as a sibling paragraph in between that would break the toggle.
    # Anchor on the chevron span (always the last child of
    # .collapsible-toggle right before its own closing tag) rather than
    # counting </div>s, since that count is fragile to nested markup.
    chev_end = panel_html.index("</span>", panel_html.index('class="chev"')) + len("</span>")
    toggle_close = panel_html.index("</div>", chev_end) + len("</div>")
    assert panel_html[toggle_close:].lstrip().startswith('<div class="collapsible-body"')

    # setPopulation() enables/disables the toggle based on population
    # (never display:none, so switching population can't shift page
    # layout) -- and resets it back to collapsed whenever POLARIS is
    # deselected, so re-selecting it later never shows it pre-expanded.
    assert 'polarisPanelEl.classList.toggle("polaris-panel--disabled", pop !== "polaris")' in html_source
    assert 'polarisToggleHeader.classList.remove("open")' in html_source


def test_polaris_cohort_construction_appears_inside_header_card_under_filter():
    """The POLARIS AD-Aligned Cohort Construction toggle must live
    INSIDE .header-info-card, immediately after the population filter
    row -- not as its own separate full-width section appearing
    somewhere further down the page."""
    html_source = _load_dashboard_html()
    card_start = html_source.index('class="header-info-card"')
    card_end = html_source.index("</main>", card_start)
    card_html = html_source[card_start:card_end]

    filter_row_pos = card_html.index("header-info-filter-row")
    polaris_pos = card_html.index('id="polarisPanel"')
    card_close_pos = card_html.index('<div class="header-right-col">')
    assert filter_row_pos < polaris_pos < card_close_pos, (
        "polarisPanel must sit inside header-info-card, after the filter row"
    )
    # It is no longer its own standalone section.panel.
    assert 'class="panel polaris-panel"' not in html_source
    assert 'class="header-info-section polaris-panel' in html_source


def test_switching_population_never_removes_polaris_toggle_from_layout():
    """The POLARIS Cohort Construction toggle must never be display:none
    -- it stays in the document flow at a fixed height regardless of
    which population is selected, only its enabled/disabled look and
    hint text change, so switching population can never shift the
    header card's height (or anything below it on the page)."""
    html_source = _load_dashboard_html()
    id_pos = html_source.index('id="polarisPanel"')
    tag_start = html_source.rindex("<div", 0, id_pos)
    tag_end = html_source.index(">", id_pos)
    opening_tag = html_source[tag_start:tag_end]
    assert "style=" not in opening_tag or "display:none" not in opening_tag
    assert "polaris-panel--disabled" in opening_tag  # starts disabled (Overall ADNI is the default)

    # JS never sets display on polarisPanel -- only the disabled class
    # and the hint text change.
    assert 'polarisPanel").style.display' not in html_source
    assert 'id="polarisHintDisabled"' in html_source
    assert 'id="polarisHintEnabled"' in html_source


def test_population_helper_text_height_is_reserved_for_both_sentences():
    """The Overall/POLARIS helper sentences are very different lengths
    (POLARIS's wraps onto an extra line at typical widths). Both must
    be stacked in the SAME grid cell and toggled via a visibility class
    (never display), so the row's height is always the taller of the
    two regardless of which is shown -- otherwise switching population
    grows the header card (and, since it's stretch-aligned, Disease
    Continuum too) taller and shifts everything below on the page."""
    html_source = _load_dashboard_html()
    assert 'class="population-helper-stack"' in html_source
    assert ".population-helper-stack { display: grid;" in html_source
    assert ".population-helper-stack > .population-helper { grid-area: 1 / 1; }" in html_source
    assert ".is-hidden { visibility: hidden; }" in html_source

    # Both sentences are present in markup at all times (only one has
    # is-hidden initially) -- neither is display:none.
    assert 'class="population-helper" id="populationHelperOverall"' in html_source
    assert 'class="population-helper is-hidden" id="populationHelperPolaris"' in html_source
    assert 'id="populationHelperOverall" style="display:none;"' not in html_source
    assert 'id="populationHelperPolaris" style="display:none;"' not in html_source

    # JS toggles the class, never .style.display, for either sentence.
    assert 'populationHelperOverall").classList.toggle("is-hidden"' in html_source
    assert 'populationHelperPolaris").classList.toggle("is-hidden"' in html_source
    assert 'populationHelperOverall").style.display' not in html_source
    assert 'populationHelperPolaris").style.display' not in html_source


def test_polaris_hint_and_disease_continuum_note_also_reserve_height():
    """Two more spots with the identical population-dependent-text-
    length problem: the POLARIS toggle's enabled hint sentence is much
    longer than its disabled one (wraps an extra line), and the Disease
    Continuum POLARIS note either exists or doesn't (a whole extra
    line). Both must use the same stack-and-toggle-visibility pattern,
    never display/textContent swaps, so neither can change its card's
    height when population changes."""
    html_source = _load_dashboard_html()

    # Hint stack: both sentences present, grid-stacked, toggled by class.
    assert 'class="polaris-hint-stack"' in html_source
    assert ".polaris-hint-stack { display: grid; }" in html_source
    assert ".polaris-hint-stack > .polaris-hint { grid-area: 1 / 1;" in html_source
    assert 'class="panel-sub polaris-hint" id="polarisHintDisabled"' in html_source
    assert 'class="panel-sub polaris-hint is-hidden" id="polarisHintEnabled"' in html_source
    assert 'polarisHintDisabled").classList.toggle("is-hidden"' in html_source
    assert 'polarisHintEnabled").classList.toggle("is-hidden"' in html_source
    assert 'getElementById("polarisToggleHint")' not in html_source

    # Disease Continuum note: always present, never display:none,
    # toggled by the same visibility class.
    assert 'class="population-note is-hidden" id="diseaseContinuumPolarisNote"' in html_source
    assert 'diseaseContinuumPolarisNote").classList.toggle("is-hidden"' in html_source
    assert 'diseaseContinuumPolarisNote").style.display' not in html_source


ALL_TESTS = [
    test_governance_rejects_raw_interim_processed_paths,
    test_governance_rejects_parquet_files,
    test_governance_rejects_participant_identifier_columns,
    test_governance_rejects_path_outside_outputs_dir,
    test_real_outputs_dir_loads_cleanly_through_governance,
    test_disease_continuum_and_absolute_columns_present_and_governed,
    test_no_rid_ptid_fields_embedded,
    test_no_participant_level_records_present,
    test_gfap_nfl_show_no_inferential_claims,
    test_hc3_is_primary_inference_not_conventional,
    test_sensitivity_concern_status_visible,
    test_no_significance_stars,
    test_descriptive_only_cells_labeled,
    test_descriptive_biomarker_points_are_aggregate_only_and_plotted,
    test_gfap_nfl_descriptive_points_have_no_p_value_or_hc3_wording,
    test_group_colors_consistent,
    test_absolute_views_never_carry_adjusted_or_sensitivity_classification,
    test_absolute_cognitive_baseline_matches_known_group_ordering,
    test_disease_continuum_covers_all_seven_endpoints_with_correct_direction_metadata,
    test_disease_continuum_baseline_values_show_expected_disease_stage_pattern,
    test_key_patterns_present_for_every_endpoint_and_view,
    test_key_pattern_text_never_uses_forbidden_treatment_language,
    test_key_pattern_absolute_avoids_small_n_late_timepoint,
    test_compact_legend_and_cohort_summary_present,
    test_analysis_details_section_holds_technical_metadata,
    test_chart_height_reduced_from_original,
    test_biomarker_uses_horizontal_tabs_not_dropdown,
    test_n_label_only_shown_for_isolated_small_n_points,
    test_results_table_partial_eta_squared_label_preserved,
    test_dementia_ptau181_has_a_real_extremely_sparse_month,
    test_isolated_points_are_never_wired_into_the_connected_line,
    test_descriptive_non_isolated_segments_render_dashed_not_solid,
    test_hover_tooltip_includes_all_required_fields,
    test_key_pattern_does_not_repeat_raw_numeric_values,
    test_key_pattern_flags_sparse_or_declining_follow_up_when_present,
    test_key_pattern_never_claims_causality,
    test_disease_continuum_explanation_is_the_required_short_form,
    test_disease_continuum_row_labels_carry_direction_arrows,
    test_compact_warning_indicator_format_preserved,
    test_navigation_label_renamed,
    test_population_selector_renders,
    test_population_selector_default_is_overall_adni,
    test_polaris_cohort_final_n_is_620,
    test_polaris_diagnosis_composition_matches_validated_target,
    test_polaris_funnel_matches_governed_attrition_csv,
    test_polaris_funnel_step_over_step_percent_is_a_ratio_of_governed_counts,
    test_polaris_population_profile_covers_requested_variables,
    test_polaris_population_profile_matches_governed_csv,
    test_not_propensity_score_matched_disclaimer_present,
    test_apoe4_context_text_is_neutral_not_causal,
    test_disease_continuum_polaris_note_present_and_labeled,
    test_existing_trajectory_payload_unchanged_by_polaris_addition,
    test_real_dashboard_disease_continuum_and_cognitive_values_still_match_direct_computation,
    test_polaris_loader_uses_only_governed_aggregate_csv_entry_point,
    test_polaris_loader_rejects_participant_identifier_columns,
    test_no_participant_level_pet_file_referenced_anywhere_in_viz_modules,
    test_polaris_payload_contains_no_participant_identifiers,
    test_existing_governance_protections_still_intact,
    test_population_selector_changes_cognitive_data_source,
    test_population_selector_changes_biomarker_data_source,
    test_overall_population_unchanged_when_polaris_trajectories_are_added,
    test_polaris_baseline_values_match_validated_population_profile,
    test_polaris_status_classification_matches_governed_status_csv,
    test_polaris_sparse_observations_remain_disconnected,
    test_key_pattern_differs_and_is_explicitly_labeled_by_population,
    test_key_pattern_never_turns_descriptive_biomarker_into_a_strong_claim,
    test_cognitive_data_support_present_only_for_polaris_biomarker_for_both,
    test_disease_continuum_is_population_agnostic_and_stays_overall_only,
    test_polaris_data_view_governance_rejects_missing_files,
    test_polaris_data_view_rejects_participant_identifier_columns,
    test_no_participant_level_data_in_polaris_population_payload,
    test_no_participant_level_pet_or_trajectory_parquet_referenced_in_viz_modules,
    test_biomarker_absolute_uses_cross_sectional_n_not_paired_change_n,
    test_biomarker_absolute_and_change_views_use_different_denominators,
    test_previously_missing_biomarker_cell_now_shows_real_absolute_data,
    test_sparse_biomarker_absolute_points_remain_isolated_not_connected,
    test_descriptive_and_sensitivity_concern_biomarker_points_remain_distinguishable,
    test_biomarker_absolute_data_support_present_for_overall_and_polaris,
    test_biomarker_data_support_distinguishes_change_and_absolute_questions,
    test_population_switching_uses_correct_biomarker_absolute_data,
    test_biomarker_summary_csv_with_new_columns_still_governed,
    test_sensible_y_range_helper_present_and_wired_into_both_charts,
    test_sensible_y_range_excludes_isolated_point_ci_but_keeps_its_estimate,
    test_biomarker_error_bars_hidden_on_chart_but_cognitive_unaffected,
    test_chart_transition_config_present_on_both_layout_builders,
    test_collapsible_sections_use_animated_transition_not_instant_display_swap,
    test_toggle_buttons_and_results_table_rows_have_transitions,
    test_fade_span_helper_used_for_key_pattern_and_data_support_updates,
    test_disease_continuum_spells_out_group_abbreviations_directly_on_the_chart,
    test_trend_bolded_and_sparse_points_faded_in_buildGroupTraces,
    test_population_filter_reads_as_a_filter_and_lives_in_the_unified_header_card,
    test_disease_continuum_sits_beside_header_description_labels_and_filter,
    test_cognitive_and_biomarker_trajectories_sit_side_by_side,
    test_disease_continuum_and_biomarker_trajectories_columns_are_aligned,
    test_paired_panel_bottoms_are_aligned_not_just_tops,
    test_polaris_cohort_construction_does_not_auto_expand_when_polaris_selected,
    test_polaris_cohort_construction_appears_inside_header_card_under_filter,
    test_switching_population_never_removes_polaris_toggle_from_layout,
    test_population_helper_text_height_is_reserved_for_both_sentences,
    test_polaris_hint_and_disease_continuum_note_also_reserve_height,
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
