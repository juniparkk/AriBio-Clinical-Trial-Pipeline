# ============================================================
# TESTS for nih_reference.py (Phase 1B — NIH reference dataset audit)
#
# Plain-Python tests (no pytest install needed) — run with:
#     .venv/bin/python test_nih_reference.py
#
# Kept in its own file, separate from test_classification.py and
# test_dashboard_table.py, so Phase 1B adds zero risk of disturbing
# either of those — this file imports nih_reference.py only, never
# pipeline_viz.py (which has import-time side effects) and never
# modifies the two existing test files.
# ============================================================

import io
import os
import tempfile

import pandas as pd

from nih_reference import (
    extract_canonical_and_aliases,
    parse_nih_dataset,
    profile_nih_dataset,
    summarize_nih_dataset_shape,
    build_dashboard_name_lookup,
    match_nih_row_to_dashboard,
    bucket_unmatched_dashboard_drug,
    build_nih_match_audit,
    map_cadro_to_target,
    infer_nih_target,
    build_nih_conflict_audit,
    MATCH_TIERS,
)

NIH_CSV_PATH = os.path.join(os.path.dirname(__file__), "nih_data.csv")


# ------------------------------------------------------------
# extract_canonical_and_aliases()
# ------------------------------------------------------------

def test_extract_simple_agent_no_alias():
    canonical, aliases, components = extract_canonical_and_aliases("Buntanetap")
    assert canonical == "Buntanetap"
    assert aliases == []
    assert components == [{"name": "Buntanetap", "alias": ""}]


def test_extract_agent_with_parenthetical_alias():
    canonical, aliases, components = extract_canonical_and_aliases("Zervimesine (CT1812)")
    assert canonical == "Zervimesine (CT1812)"
    assert aliases == ["CT1812"]
    assert components == [{"name": "Zervimesine", "alias": "CT1812"}]


def test_extract_combination_agent_with_two_aliases():
    canonical, aliases, components = extract_canonical_and_aliases("Etalanetug (E2814) + Lecanemab (BAN2401)")
    assert canonical == "Etalanetug (E2814) + Lecanemab (BAN2401)"
    assert aliases == ["E2814", "BAN2401"]
    assert components == [
        {"name": "Etalanetug", "alias": "E2814"},
        {"name": "Lecanemab", "alias": "BAN2401"},
    ]


def test_extract_combination_agent_without_aliases():
    canonical, aliases, components = extract_canonical_and_aliases("Dasatinib + Quercetin")
    assert canonical == "Dasatinib + Quercetin"
    assert aliases == []
    assert len(components) == 2
    assert components[0]["name"] == "Dasatinib"
    assert components[1]["name"] == "Quercetin"


def test_extract_does_not_treat_long_descriptive_parenthetical_as_alias():
    # a parenthetical that reads as a description, not a short code,
    # must not be mistaken for a development-code alias
    canonical, aliases, components = extract_canonical_and_aliases("SomeDrug (a long descriptive phrase here)")
    assert aliases == []
    assert components == [{"name": "SomeDrug (a long descriptive phrase here)", "alias": ""}]


# ------------------------------------------------------------
# parse_nih_dataset()
# ------------------------------------------------------------

_SAMPLE_NIH_CSV = (
    "Phase 3,,,,,,,\n"
    "Agent,Therapeutic purpose,CADRO,Mechanism of action,Clinical trial,Lead sponsor,Start date,Primary completion date\n"
    'AR1001,DTT; small molecule,Synaptic plasticity/ neuroprotection,PDE5 inhibitor,NCT05531526,"AriBio Co., Ltd.","Dec, 2022","May, 2026"\n'
    'Multi Trial Drug,DTT; biologic,Amyloid‐beta,Some mechanism,"NCT001\nNCT002","Sponsor A\nSponsor B","Jan, 2020\nFeb, 2021","Jan, 2025\nFeb, 2026"\n'
    "Phase 2,,,,,,,\n"
    ",,,,,,,\n"
    ",,,,,,,\n"
    "Agent,Therapeutic purpose,CADRO,Mechanism of action,Clinical trial,Lead sponsor,Start date,Primary completion date\n"
    "SomeAgent,STT; cognition enhancer,Neurotransmitter receptors,Some other mechanism,NCT003,Some Sponsor,\"Jan, 2021\",\"Jan, 2026\"\n"
)


def _write_sample_csv():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write(_SAMPLE_NIH_CSV)
        return f.name


def test_parse_nih_dataset_basic_structure():
    path = _write_sample_csv()
    try:
        df = parse_nih_dataset(path)
    finally:
        os.remove(path)
    assert len(df) == 3
    assert set(df["phase"]) == {"Phase 3", "Phase 2"}
    assert list(df["canonical_name"]) == ["AR1001", "Multi Trial Drug", "SomeAgent"]


def test_parse_nih_dataset_multi_trial_row_splits_parallel_lists():
    path = _write_sample_csv()
    try:
        df = parse_nih_dataset(path)
    finally:
        os.remove(path)
    row = df[df["canonical_name"] == "Multi Trial Drug"].iloc[0]
    assert row["nct_ids"] == ["NCT001", "NCT002"]
    assert row["lead_sponsors"] == ["Sponsor A", "Sponsor B"]
    assert len(row["nct_ids"]) == len(row["lead_sponsors"])


def test_parse_nih_dataset_purpose_class_and_detail():
    path = _write_sample_csv()
    try:
        df = parse_nih_dataset(path)
    finally:
        os.remove(path)
    ar1001 = df[df["canonical_name"] == "AR1001"].iloc[0]
    assert ar1001["purpose_class"] == "DTT"
    assert ar1001["purpose_detail"] == "small molecule"

    some_agent = df[df["canonical_name"] == "SomeAgent"].iloc[0]
    assert some_agent["purpose_class"] == "STT"
    assert some_agent["purpose_detail"] == "cognition enhancer"


def test_parse_nih_dataset_skips_header_and_blank_rows():
    path = _write_sample_csv()
    try:
        df = parse_nih_dataset(path)
    finally:
        os.remove(path)
    assert "Agent" not in df["canonical_name"].values
    assert all(df["canonical_name"].astype(str).str.strip() != "")


def test_parse_real_nih_csv_row_count_and_phases():
    df = parse_nih_dataset(NIH_CSV_PATH)
    assert len(df) == 165
    assert df["phase"].value_counts().to_dict() == {"Phase 2": 84, "Phase 1": 45, "Phase 3": 36}
    for col in ("canonical_name", "cadro", "therapeutic_purpose"):
        assert df[col].astype(str).str.strip().ne("").all(), f"{col} should never be blank"


def test_parse_real_nih_csv_ar1001_present_and_correct():
    df = parse_nih_dataset(NIH_CSV_PATH)
    row = df[df["canonical_name"] == "AR1001"]
    assert len(row) == 1
    row = row.iloc[0]
    assert row["phase"] == "Phase 3"
    assert row["nct_ids"] == ["NCT05531526"]
    assert "AriBio" in row["lead_sponsors"][0]


# ------------------------------------------------------------
# profile_nih_dataset() / summarize_nih_dataset_shape()
# ------------------------------------------------------------

def test_profile_nih_dataset_has_one_row_per_field():
    df = parse_nih_dataset(NIH_CSV_PATH)
    profile = profile_nih_dataset(df)
    assert len(profile) > 0
    assert set(profile["field"]) <= set(df.columns) | {"purpose_class"}
    assert {"non_null_count", "total_rows", "pct_populated", "distinct_values", "supports"} <= set(profile.columns)
    assert (profile["pct_populated"] <= 100.0).all()
    assert (profile["pct_populated"] >= 0.0).all()


def test_summarize_nih_dataset_shape_finds_known_duplicates():
    df = parse_nih_dataset(NIH_CSV_PATH)
    shape = summarize_nih_dataset_shape(df)
    assert shape["total_agent_rows"] == 165
    assert shape["duplicate_canonical_name_count"] >= 1
    assert "semaglutide" in shape["duplicate_canonical_names"]
    assert shape["has_date_or_version_field_for_the_dataset_itself"] is False


# ------------------------------------------------------------
# matching
# ------------------------------------------------------------

def _resolved_drugs_fixture():
    return pd.DataFrame([
        {"display_name": "AR1001", "phase_reached": "Phase 3", "drug_type": "Small Molecule", "target": "Neuroprotection",
         "sponsor": "AriBio Co., Ltd.", "status_summary": "Recruiting", "pipeline_scope": "Therapeutic Drug"},
        {"display_name": "Lecanemab", "phase_reached": "Phase 3", "drug_type": "Biologic", "target": "Amyloid",
         "sponsor": "Eisai Inc.", "status_summary": "FDA Approved", "pipeline_scope": "Therapeutic Drug"},
        {"display_name": "Some Discontinued Drug", "phase_reached": "Phase 2", "drug_type": "Small Molecule", "target": "Other",
         "sponsor": "Some Sponsor", "status_summary": "Discontinued", "pipeline_scope": "Therapeutic Drug"},
        {"display_name": "Fish Oil", "phase_reached": "Phase 2", "drug_type": "Dietary Supplement", "target": "Other",
         "sponsor": "Some Sponsor", "status_summary": "Recruiting", "pipeline_scope": "Non-Drug Intervention"},
    ])


def test_build_dashboard_name_lookup():
    lookup = build_dashboard_name_lookup(_resolved_drugs_fixture())
    assert lookup["ar1001"] == "AR1001"
    assert lookup["lecanemab"] == "Lecanemab"


def test_match_exact_canonical():
    resolved = _resolved_drugs_fixture()
    lookup = build_dashboard_name_lookup(resolved)
    nih_row = {"canonical_name": "AR1001", "aliases": [], "components": [{"name": "AR1001", "alias": ""}]}
    result = match_nih_row_to_dashboard(nih_row, resolved["display_name"].tolist(), lookup)
    assert result["match_tier"] == "exact_canonical"
    assert result["matched_dashboard_name"] == "AR1001"


def test_match_exact_alias():
    resolved = _resolved_drugs_fixture()
    lookup = build_dashboard_name_lookup(resolved)
    nih_row = {"canonical_name": "Lecanemab (BAN2401)", "aliases": ["Lecanemab"],
               "components": [{"name": "Lecanemab", "alias": "Lecanemab"}]}
    result = match_nih_row_to_dashboard(nih_row, resolved["display_name"].tolist(), lookup)
    assert result["match_tier"] == "exact_alias"
    assert result["matched_dashboard_name"] == "Lecanemab"


def test_match_normalized_exact():
    resolved = _resolved_drugs_fixture()
    lookup = build_dashboard_name_lookup(resolved)
    nih_row = {"canonical_name": "ar1001", "aliases": [], "components": [{"name": "ar1001", "alias": ""}]}
    result = match_nih_row_to_dashboard(nih_row, resolved["display_name"].tolist(), lookup)
    assert result["match_tier"] == "normalized_exact"
    assert result["matched_dashboard_name"] == "AR1001"


def test_match_fuzzy_suggestion_not_auto_accepted():
    resolved = _resolved_drugs_fixture()
    lookup = build_dashboard_name_lookup(resolved)
    # "AR1001X" is close (difflib ratio ~0.92) but not equal after
    # normalization — must be a suggestion only, never an auto-match
    nih_row = {"canonical_name": "AR1001X", "aliases": [], "components": [{"name": "AR1001X", "alias": ""}]}
    result = match_nih_row_to_dashboard(nih_row, resolved["display_name"].tolist(), lookup)
    assert result["match_tier"] == "fuzzy_suggestion"
    assert result["matched_dashboard_name"] == "AR1001"


def test_match_unmatched_when_nothing_close():
    resolved = _resolved_drugs_fixture()
    lookup = build_dashboard_name_lookup(resolved)
    nih_row = {"canonical_name": "TotallyDifferentCompoundXYZ", "aliases": [], "components": [{"name": "TotallyDifferentCompoundXYZ", "alias": ""}]}
    result = match_nih_row_to_dashboard(nih_row, resolved["display_name"].tolist(), lookup)
    assert result["match_tier"] == "unmatched"
    assert result["matched_dashboard_name"] == ""


def test_match_tier_priority_prefers_stronger_tier_across_candidates():
    # canonical name only fuzzy-matches, but an alias exact-matches —
    # the function must return the STRONGER tier, not the first one tried
    resolved = _resolved_drugs_fixture()
    lookup = build_dashboard_name_lookup(resolved)
    nih_row = {"canonical_name": "Lecanemab-ish (AR1001)", "aliases": ["AR1001"],
               "components": [{"name": "Lecanemab-ish", "alias": "AR1001"}]}
    result = match_nih_row_to_dashboard(nih_row, resolved["display_name"].tolist(), lookup)
    assert result["match_tier"] == "exact_alias"
    assert result["matched_dashboard_name"] == "AR1001"


# ------------------------------------------------------------
# bucket_unmatched_dashboard_drug()
# ------------------------------------------------------------

def test_bucket_non_therapeutic_scope_first():
    row = {"pipeline_scope": "Non-Drug Intervention", "status_summary": "Discontinued"}
    assert bucket_unmatched_dashboard_drug(row, has_fuzzy_suggestion=True) == "non_therapeutic_or_ambiguous"


def test_bucket_fuzzy_suggestion_before_discontinued():
    row = {"pipeline_scope": "Therapeutic Drug", "status_summary": "Discontinued"}
    assert bucket_unmatched_dashboard_drug(row, has_fuzzy_suggestion=True) == "unresolved_naming_alias_issue"


def test_bucket_historical_discontinued():
    row = {"pipeline_scope": "Therapeutic Drug", "status_summary": "Discontinued"}
    assert bucket_unmatched_dashboard_drug(row, has_fuzzy_suggestion=False) == "historical_or_discontinued"


def test_bucket_current_missing_from_nih():
    row = {"pipeline_scope": "Therapeutic Drug", "status_summary": "Recruiting"}
    assert bucket_unmatched_dashboard_drug(row, has_fuzzy_suggestion=False) == "current_missing_from_nih"


# ------------------------------------------------------------
# build_nih_match_audit() — the two-sided reconciliation table
# ------------------------------------------------------------

def _small_nih_df():
    return pd.DataFrame([
        {"canonical_name": "AR1001", "phase": "Phase 3", "aliases": [], "components": [{"name": "AR1001", "alias": ""}],
         "lead_sponsors": ["AriBio Co., Ltd."], "nct_ids": ["NCT05531526"]},
        {"canonical_name": "Unmatched Agent", "phase": "Phase 2", "aliases": [], "components": [{"name": "Unmatched Agent", "alias": ""}],
         "lead_sponsors": ["Some Sponsor"], "nct_ids": ["NCT999"]},
    ])


def test_build_nih_match_audit_row_counts_and_directions():
    resolved = _resolved_drugs_fixture()
    audit = build_nih_match_audit(_small_nih_df(), resolved)
    nih_rows = audit[audit["record_type"] == "nih_record"]
    dashboard_rows = audit[audit["record_type"] == "dashboard_only"]
    assert len(nih_rows) == 2
    # AR1001 matched -> excluded from dashboard_only; the other 3 fixture
    # drugs (Lecanemab, Some Discontinued Drug, Fish Oil) remain unmatched
    assert len(dashboard_rows) == 3
    assert set(dashboard_rows["name"]) == {"Lecanemab", "Some Discontinued Drug", "Fish Oil"}


def test_build_nih_match_audit_buckets_dashboard_only_rows_correctly():
    resolved = _resolved_drugs_fixture()
    audit = build_nih_match_audit(_small_nih_df(), resolved)
    dashboard_rows = audit[audit["record_type"] == "dashboard_only"].set_index("name")
    assert dashboard_rows.loc["Some Discontinued Drug", "dashboard_bucket"] == "historical_or_discontinued"
    assert dashboard_rows.loc["Fish Oil", "dashboard_bucket"] == "non_therapeutic_or_ambiguous"
    assert dashboard_rows.loc["Lecanemab", "dashboard_bucket"] == "current_missing_from_nih"


def test_build_nih_match_audit_no_records_silently_dropped():
    resolved = _resolved_drugs_fixture()
    audit = build_nih_match_audit(_small_nih_df(), resolved)
    assert len(audit) == 2 + 3  # every NIH row + every unmatched dashboard row accounted for


# ------------------------------------------------------------
# CADRO -> target mapping / conflict audit
# ------------------------------------------------------------

def test_map_cadro_to_target_known_categories():
    assert map_cadro_to_target("Amyloid‐beta") == "Amyloid"
    assert map_cadro_to_target("Tau") == "Tau"
    assert map_cadro_to_target("Inflammation") == "Inflammation"
    assert map_cadro_to_target("Metabolism and bioenergetics") == "Metabolism"


def test_map_cadro_to_target_unmapped_category_returns_none():
    # deliberately NOT force-mapped — surfacing the taxonomy gap honestly
    assert map_cadro_to_target("Proteostasis/proteinopathies") is None
    assert map_cadro_to_target("Circadian rhythm") is None
    assert map_cadro_to_target("Undisclosed") is None


def test_infer_nih_target_prefers_neuropsychiatric_purpose_detail():
    # CADRO alone would map "Neurotransmitter receptors" -> "Symptomatic",
    # but an STT row whose purpose_detail says "neuropsychiatric (...)"
    # should resolve to "Neuropsychiatric" instead
    result = infer_nih_target("Neurotransmitter receptors", "STT", "neuropsychiatric (agitation)")
    assert result == "Neuropsychiatric"


def test_infer_nih_target_falls_back_to_cadro_for_non_neuropsychiatric_stt():
    result = infer_nih_target("Neurotransmitter receptors", "STT", "cognition enhancer")
    assert result == "Symptomatic"


def test_infer_nih_target_dtt_rows_use_cadro_directly():
    result = infer_nih_target("Amyloid‐beta", "DTT", "biologic")
    assert result == "Amyloid"


def test_build_nih_conflict_audit_flags_drug_type_conflict():
    resolved = pd.DataFrame([
        {"display_name": "Semaglutide", "phase_reached": "Phase 3", "drug_type": "Small Molecule", "target": "Metabolism",
         "sponsor": "Novo Nordisk A/S", "status_summary": "Recruiting", "pipeline_scope": "Therapeutic Drug"},
    ])
    nih_df = pd.DataFrame([
        {"canonical_name": "Semaglutide", "phase": "Phase 3", "aliases": [], "components": [{"name": "Semaglutide", "alias": ""}],
         "lead_sponsors": ["Novo Nordisk A/S"], "nct_ids": ["NCT1"], "purpose_class": "DTT", "purpose_detail": "biologic",
         "cadro": "Metabolism and bioenergetics", "mechanism_of_action": "GLP-1 agonist"},
    ])
    match_audit = build_nih_match_audit(nih_df, resolved)
    conflict_audit = build_nih_conflict_audit(nih_df, resolved, match_audit)
    assert len(conflict_audit) == 1
    row = conflict_audit.iloc[0]
    assert row["drug_type_conflict"] == True
    assert row["nih_modality"] == "biologic"
    assert row["dashboard_drug_type"] == "Small Molecule"
    assert row["target_conflict"] == False  # both map to Metabolism
    assert row["phase_conflict"] == False


def test_build_nih_conflict_audit_excludes_fuzzy_matches():
    resolved = pd.DataFrame([
        # "AR1001X" is close to NIH's "AR1001" (difflib ratio ~0.92) but
        # not normalization-equal — must land as fuzzy_suggestion, not a
        # confirmed match
        {"display_name": "AR1001X", "phase_reached": "Phase 3", "drug_type": "Small Molecule", "target": "Neuroprotection",
         "sponsor": "AriBio", "status_summary": "Recruiting", "pipeline_scope": "Therapeutic Drug"},
    ])
    nih_df = pd.DataFrame([
        {"canonical_name": "AR1001", "phase": "Phase 3", "aliases": [], "components": [{"name": "AR1001", "alias": ""}],
         "lead_sponsors": ["AriBio"], "nct_ids": ["NCT1"], "purpose_class": "DTT", "purpose_detail": "small molecule",
         "cadro": "Synaptic plasticity/neuroprotection", "mechanism_of_action": "PDE5 inhibitor"},
    ])
    match_audit = build_nih_match_audit(nih_df, resolved)
    assert (match_audit["match_tier"] == "fuzzy_suggestion").any()
    conflict_audit = build_nih_conflict_audit(nih_df, resolved, match_audit)
    assert len(conflict_audit) == 0  # fuzzy-only match must not produce a "confirmed" conflict row


def test_build_nih_conflict_audit_company_conflict_uses_company_matches_logic():
    resolved = pd.DataFrame([
        {"display_name": "DrugX", "phase_reached": "Phase 2", "drug_type": "Small Molecule", "target": "Other",
         "sponsor": "Totally Unrelated University", "status_summary": "Recruiting", "pipeline_scope": "Therapeutic Drug"},
    ])
    nih_df = pd.DataFrame([
        {"canonical_name": "DrugX", "phase": "Phase 2", "aliases": [], "components": [{"name": "DrugX", "alias": ""}],
         "lead_sponsors": ["Some Pharma Inc."], "nct_ids": ["NCT1"], "purpose_class": "DTT", "purpose_detail": "small molecule",
         "cadro": "Undisclosed", "mechanism_of_action": ""},
    ])
    match_audit = build_nih_match_audit(nih_df, resolved)
    conflict_audit = build_nih_conflict_audit(nih_df, resolved, match_audit)
    assert conflict_audit.iloc[0]["company_conflict"] == True


ALL_TESTS = [
    test_extract_simple_agent_no_alias,
    test_extract_agent_with_parenthetical_alias,
    test_extract_combination_agent_with_two_aliases,
    test_extract_combination_agent_without_aliases,
    test_extract_does_not_treat_long_descriptive_parenthetical_as_alias,
    test_parse_nih_dataset_basic_structure,
    test_parse_nih_dataset_multi_trial_row_splits_parallel_lists,
    test_parse_nih_dataset_purpose_class_and_detail,
    test_parse_nih_dataset_skips_header_and_blank_rows,
    test_parse_real_nih_csv_row_count_and_phases,
    test_parse_real_nih_csv_ar1001_present_and_correct,
    test_profile_nih_dataset_has_one_row_per_field,
    test_summarize_nih_dataset_shape_finds_known_duplicates,
    test_build_dashboard_name_lookup,
    test_match_exact_canonical,
    test_match_exact_alias,
    test_match_normalized_exact,
    test_match_fuzzy_suggestion_not_auto_accepted,
    test_match_unmatched_when_nothing_close,
    test_match_tier_priority_prefers_stronger_tier_across_candidates,
    test_bucket_non_therapeutic_scope_first,
    test_bucket_fuzzy_suggestion_before_discontinued,
    test_bucket_historical_discontinued,
    test_bucket_current_missing_from_nih,
    test_build_nih_match_audit_row_counts_and_directions,
    test_build_nih_match_audit_buckets_dashboard_only_rows_correctly,
    test_build_nih_match_audit_no_records_silently_dropped,
    test_map_cadro_to_target_known_categories,
    test_map_cadro_to_target_unmapped_category_returns_none,
    test_infer_nih_target_prefers_neuropsychiatric_purpose_detail,
    test_infer_nih_target_falls_back_to_cadro_for_non_neuropsychiatric_stt,
    test_infer_nih_target_dtt_rows_use_cadro_directly,
    test_build_nih_conflict_audit_flags_drug_type_conflict,
    test_build_nih_conflict_audit_excludes_fuzzy_matches,
    test_build_nih_conflict_audit_company_conflict_uses_company_matches_logic,
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
