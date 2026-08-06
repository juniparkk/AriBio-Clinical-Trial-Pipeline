# ============================================================
# TESTS for scientific_classification.py (Phase 2 — drug-centric
# modality/target-pathway resolution)
#
# Plain-Python tests (no pytest install needed) — run with:
#     .venv/bin/python test_scientific_classification.py
#
# Kept in its own file, separate from test_classification.py,
# test_dashboard_table.py, and test_nih_reference.py, so Phase 2 adds
# zero risk of disturbing any of them.
# ============================================================

import os
import tempfile

import pandas as pd

from scientific_classification import (
    load_drug_classification_overrides,
    KNOWN_COMPOUND_TARGETS,
    KNOWN_COMPOUND_MODALITY,
    TARGET_KEYWORDS,
    TARGET_ORDER,
    gather_structured_evidence_for_drug,
    infer_modality_from_structured_evidence,
    infer_target_pathways_from_structured_evidence,
    build_nih_name_lookup,
    match_drug_to_nih,
    claude_infer_classification,
    resolve_drug_classification,
    build_official_pipeline_classification_lookup,
    build_classification_conflicts_dataframe,
    _derive_therapeutic_purpose_category,
    classify_pipeline_quadrant,
    PIPELINE_QUADRANTS,
)
from nih_reference import parse_nih_dataset
from drug_classification import load_official_pipeline

NIH_CSV_PATH = os.path.join(os.path.dirname(__file__), "nih_data.csv")
PIPELINE_CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "official_pipeline.csv")


# ------------------------------------------------------------
# load_drug_classification_overrides()
# ------------------------------------------------------------

def test_load_drug_classification_overrides_missing_file_returns_empty_dict():
    assert load_drug_classification_overrides("does/not/exist.csv") == {}


def test_load_drug_classification_overrides_parses_semicolon_lists():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(
            "normalized_drug_name,modality,target_pathways,molecular_targets,mechanism_of_action,"
            "reason,source,reviewer,verified_date\n"
            'testdrug,Small Molecule,"Amyloid; Tau",PDE5,Some mechanism,test reason,test source,'
            "Test Reviewer,2026-01-01\n"
        )
        path = f.name
    try:
        overrides = load_drug_classification_overrides(path)
    finally:
        os.remove(path)
    assert "testdrug" in overrides
    entry = overrides["testdrug"]
    assert entry["modality"] == "Small Molecule"
    assert entry["target_pathways"] == ["Amyloid", "Tau"]
    assert entry["molecular_targets"] == ["PDE5"]


def test_load_drug_classification_overrides_missing_column_raises():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("normalized_drug_name,modality\ntestdrug,Small Molecule\n")
        path = f.name
    try:
        raised = False
        try:
            load_drug_classification_overrides(path)
        except ValueError:
            raised = True
        assert raised
    finally:
        os.remove(path)


def test_the_real_drug_classification_overrides_csv_has_ar1001():
    path = os.path.join(os.path.dirname(__file__), "data", "reference", "drug_classification_overrides.csv")
    overrides = load_drug_classification_overrides(path)
    assert "ar1001" in overrides
    assert overrides["ar1001"]["modality"] == "Small Molecule"
    assert set(overrides["ar1001"]["target_pathways"]) == {"Amyloid", "Tau", "Neuroprotection"}


# ------------------------------------------------------------
# gather_structured_evidence_for_drug() — verified-evidence isolation
# ------------------------------------------------------------

def _interventions_fixture():
    return pd.DataFrame([
        {"nct_id": "NCT1", "title": "A Study of DrugX for Amyloid Clearance", "original_type": "DRUG",
         "original_name": "DrugX 10 mg", "candidate_name": "DrugX", "classification": "investigational_therapeutic_unverified"},
        {"nct_id": "NCT1", "title": "A Study of DrugX for Amyloid Clearance", "original_type": "OTHER",
         "original_name": "Placebo", "candidate_name": "Placebo", "classification": "placebo_or_sham"},
        {"nct_id": "NCT1", "title": "A Study of DrugX for Amyloid Clearance", "original_type": "DEVICE",
         "original_name": "Some Unrelated Device", "candidate_name": "Some Unrelated Device", "classification": "device"},
        {"nct_id": "NCT2", "title": "DrugX Extension Study", "original_type": "DRUG",
         "original_name": "DrugX 20 mg", "candidate_name": "DrugX", "classification": "investigational_therapeutic_unverified"},
    ])


def test_gather_structured_evidence_isolates_verified_rows_only():
    evidence = gather_structured_evidence_for_drug(_interventions_fixture(), "drugx")
    assert len(evidence) == 2
    names = {e["name"] for e in evidence}
    assert names == {"DrugX 10 mg", "DrugX 20 mg"}
    # the sibling placebo/device rows in the SAME trial must never appear —
    # this is the key correctness fix over legacy guess_drug_type()
    assert "Placebo" not in names
    assert "Some Unrelated Device" not in names


def test_gather_structured_evidence_carries_trial_title():
    evidence = gather_structured_evidence_for_drug(_interventions_fixture(), "drugx")
    assert all("Amyloid" in e["title"] or "DrugX" in e["title"] for e in evidence)


def test_gather_structured_evidence_empty_for_no_match():
    evidence = gather_structured_evidence_for_drug(_interventions_fixture(), "totally_unrelated_name")
    assert evidence == []


def test_gather_structured_evidence_empty_dataframe():
    assert gather_structured_evidence_for_drug(pd.DataFrame(), "anything") == []


# ------------------------------------------------------------
# infer_modality_from_structured_evidence() / infer_target_pathways...()
# ------------------------------------------------------------

def test_infer_modality_biologic_from_name_keyword():
    evidence = [{"type": "DRUG", "name": "Some-mab antibody", "title": ""}]
    assert infer_modality_from_structured_evidence(evidence) == "Biologic"


def test_infer_modality_biologic_from_type():
    evidence = [{"type": "BIOLOGICAL", "name": "Some Compound", "title": ""}]
    assert infer_modality_from_structured_evidence(evidence) == "Biologic"


def test_infer_modality_small_molecule_from_drug_type():
    evidence = [{"type": "DRUG", "name": "Some Compound", "title": ""}]
    assert infer_modality_from_structured_evidence(evidence) == "Small Molecule"


def test_infer_modality_device_and_dietary_supplement():
    assert infer_modality_from_structured_evidence([{"type": "DEVICE", "name": "X", "title": ""}]) == "Device"
    assert infer_modality_from_structured_evidence([{"type": "DIETARY_SUPPLEMENT", "name": "X", "title": ""}]) == "Dietary Supplement"


def test_infer_modality_empty_evidence_returns_blank_not_unknown():
    # blank ("no tier-5 evidence") must be distinguishable from a real
    # "Unknown"/"Other" outcome by the resolver
    assert infer_modality_from_structured_evidence([]) == ""


def test_infer_target_pathways_from_name_text():
    evidence = [{"type": "DRUG", "name": "Some amyloid-targeting compound", "title": ""}]
    assert infer_target_pathways_from_structured_evidence(evidence) == ["Amyloid"]


def test_infer_target_pathways_from_title_text_not_just_name():
    # the drug's own name carries no pathway keyword, but the trial
    # TITLE studying this specific (verified) drug does
    evidence = [{"type": "DRUG", "name": "AVP-786", "title": "AVP-786 for Agitation in Alzheimer's Disease"}]
    assert "Neuropsychiatric" in infer_target_pathways_from_structured_evidence(evidence)


def test_infer_target_pathways_supports_multiple_pathways():
    evidence = [{"type": "DRUG", "name": "Compound targeting amyloid and tau pathology", "title": ""}]
    result = infer_target_pathways_from_structured_evidence(evidence)
    assert "Amyloid" in result
    assert "Tau" in result
    assert len(result) >= 2
    # TARGET_ORDER order preserved
    assert result == [p for p in TARGET_ORDER if p in result]


def test_infer_target_pathways_empty_evidence():
    assert infer_target_pathways_from_structured_evidence([]) == []


# ------------------------------------------------------------
# NIH matching (tier 2)
# ------------------------------------------------------------

def test_build_nih_name_lookup_solo_agent_wins_over_combination_component():
    # regression test for a real bug found during Phase 2 development:
    # "Etalanetug (E2814) + Lecanemab (BAN2401)" (a combination row) has
    # "Lecanemab" as a component name — a plain lookup that indexed
    # combination components with equal priority would attribute the
    # COMBO's mechanism/CADRO text to solo Lecanemab, which is wrong
    # (the combo's mechanism describes BOTH an anti-tau and an
    # anti-amyloid antibody; Lecanemab alone is only anti-amyloid).
    nih_df = parse_nih_dataset(NIH_CSV_PATH)
    lookup = build_nih_name_lookup(nih_df)
    solo_row = lookup["lecanemab"]
    assert solo_row["canonical_name"] == "Lecanemab"
    assert "anti" in solo_row["mechanism_of_action"].lower()
    assert "tau" not in solo_row["mechanism_of_action"].lower()  # solo Lecanemab's own mechanism, not the combo's


def test_match_drug_to_nih_by_display_name():
    nih_df = parse_nih_dataset(NIH_CSV_PATH)
    lookup = build_nih_name_lookup(nih_df)
    row = match_drug_to_nih("AR1001", [], lookup)
    assert row is not None
    assert row["canonical_name"] == "AR1001"


def test_match_drug_to_nih_by_synonym():
    nih_df = parse_nih_dataset(NIH_CSV_PATH)
    lookup = build_nih_name_lookup(nih_df)
    row = match_drug_to_nih("Some Internal Display Name", ["AR1001"], lookup)
    assert row is not None
    assert row["canonical_name"] == "AR1001"


def test_match_drug_to_nih_no_match_returns_none():
    nih_df = parse_nih_dataset(NIH_CSV_PATH)
    lookup = build_nih_name_lookup(nih_df)
    assert match_drug_to_nih("TotallyUnknownCompoundXYZ123", [], lookup) is None


# ------------------------------------------------------------
# resolve_drug_classification() — the main resolver
# ------------------------------------------------------------

def _nih_lookup():
    return build_nih_name_lookup(parse_nih_dataset(NIH_CSV_PATH))


def test_resolve_tier1_curated_override_wins_outright():
    overrides = {"ar1001": {
        "modality": "Small Molecule", "target_pathways": ["Amyloid", "Tau", "Neuroprotection"],
        "molecular_targets": ["PDE5"], "mechanism_of_action": "PDE5 inhibitor",
        "reason": "curated", "source": "test",
    }}
    result = resolve_drug_classification("AR1001", [], [], overrides=overrides, nih_name_lookup=_nih_lookup())
    assert result["classification_source"] == "curated_override"
    assert result["classification_confidence"] == "high"
    assert set(result["target_pathways"]) == {"Amyloid", "Tau", "Neuroprotection"}


def test_resolve_biologic_previously_labeled_small_molecule():
    # Remternetug: a real anti-amyloid monoclonal antibody NIH lists as
    # "DTT; biologic" — a drug with no strong tier-4/5 modality signal
    # of its own should pick up "Biologic" from NIH, correcting what the
    # legacy heuristic (mislabeling many antibodies "Small Molecule" per
    # the Phase 1B audit) would have gotten wrong
    result = resolve_drug_classification("Remternetug", [], [], overrides={}, nih_name_lookup=_nih_lookup())
    assert result["modality"] == "Biologic"
    assert result["classification_source"] == "nih_reference"


def test_resolve_cadro_target_correction_via_nih():
    # Baricitinib: NIH CADRO = Inflammation — a drug with no tier-4/5
    # target evidence should pick up the correct pathway from NIH
    result = resolve_drug_classification("Baricitinib", [], [], overrides={}, nih_name_lookup=_nih_lookup())
    assert result["target_pathways"] == ["Inflammation"]


def test_resolve_multi_target_drug_via_structured_evidence():
    evidence = [{"type": "DRUG", "name": "Dual amyloid and tau aggregation inhibitor", "title": ""}]
    result = resolve_drug_classification("SomeDualCompound", [], evidence, overrides={}, nih_name_lookup={})
    assert set(result["target_pathways"]) == {"Amyloid", "Tau"}
    assert len(result["target_pathways"]) > 1  # never forced into a single pathway


def test_resolve_nih_match_high_confidence():
    result = resolve_drug_classification("Lecanemab", ["BAN2401"], [], overrides={}, nih_name_lookup=_nih_lookup())
    assert result["classification_confidence"] == "high"
    assert result["modality"] == "Biologic"
    assert result["target_pathways"] == ["Amyloid"]


def test_resolve_non_nih_drug_uses_existing_evidence_without_confidence_penalty():
    # per the explicit requirement: "Do not downgrade confidence solely
    # because NIH is missing" — a drug with a strong existing-tier match
    # (known-compound exact target match) and no NIH match at all must
    # still get a real (non-"low") confidence and the correct pathway
    result = resolve_drug_classification(
        "Donepezil", [], [{"type": "DRUG", "name": "Donepezil", "title": ""}],
        overrides={}, nih_name_lookup={},  # empty lookup == "NIH has no match"
    )
    assert result["target_pathways"] == ["Symptomatic"]  # from KNOWN_COMPOUND_TARGETS, tier 4
    assert result["classification_confidence"] != "low"


def test_resolve_conflicting_nih_and_existing_evidence_flagged():
    # fabricate a scenario where NIH says one target and existing
    # (structured evidence) says a clearly different one
    fake_nih_lookup = {
        "conflicttestdrug": pd.Series({
            "canonical_name": "ConflictTestDrug", "purpose_class": "DTT", "purpose_detail": "biologic",
            "cadro": "Tau", "mechanism_of_action": "anti-tau antibody",
        }),
    }
    evidence = [{"type": "DRUG", "name": "amyloid-targeting small molecule", "title": ""}]
    result = resolve_drug_classification(
        "ConflictTestDrug", [], evidence, overrides={}, nih_name_lookup=fake_nih_lookup,
    )
    assert "disagree" in result["classification_reason"]
    assert result["manual_review_required"] is True
    assert result["classification_confidence"] in ("medium", "low")
    # NIH (tier 2) outranks existing (tiers 3-5) on disagreement
    assert result["target_pathways"] == ["Tau"]


def test_resolve_unresolved_drug_falls_through_to_needs_review():
    result = resolve_drug_classification("CompletelyUnknownDrugXYZ", [], [], overrides={}, nih_name_lookup={})
    assert result["classification_source"] == "needs_review"
    assert result["modality"] == "Unknown"
    assert result["manual_review_required"] is True
    assert result["classification_confidence"] == "low"


def test_claude_inference_tier_is_a_documented_noop():
    # tier 6 must never fabricate a classification — always defers
    assert claude_infer_classification("AnyDrug", "any evidence summary") is None


def test_resolve_confidence_scoring_agree_vs_disagree_vs_single_source():
    lookup = _nih_lookup()
    # single source (existing only, structured evidence) -> medium (not penalized, not "high" either
    # since it's not a curated/official source)
    single = resolve_drug_classification(
        "SomeObscureCompound", [], [{"type": "DRUG", "name": "SomeObscureCompound", "title": ""}],
        overrides={}, nih_name_lookup={},
    )
    assert single["classification_confidence"] == "medium"

    # NIH + existing agree -> high
    agree = resolve_drug_classification("Lecanemab", [], [], overrides={}, nih_name_lookup=lookup)
    assert agree["classification_confidence"] == "high"


# ------------------------------------------------------------
# therapeutic_purpose_class/category + CADRO (display-only NIH fields)
# ------------------------------------------------------------

def test_derive_therapeutic_purpose_category_strips_parenthetical():
    assert _derive_therapeutic_purpose_category("neuropsychiatric (agitation)") == "Neuropsychiatric"
    assert _derive_therapeutic_purpose_category("cognition enhancer") == "Cognition Enhancer"


def test_derive_therapeutic_purpose_category_dtt_modality_values():
    assert _derive_therapeutic_purpose_category("small molecule") == "Small Molecule"
    assert _derive_therapeutic_purpose_category("biologic") == "Biologic"


def test_derive_therapeutic_purpose_category_takes_first_of_multiple_purposes():
    # a few real NIH cells pack two purposes separated by a newline
    # (see nih_reference.py's profile notes) — only the first is used
    assert _derive_therapeutic_purpose_category("cognition enhancer\nSTT; neuropsychiatric (agitation)") == "Cognition Enhancer"


def test_derive_therapeutic_purpose_category_blank_input():
    assert _derive_therapeutic_purpose_category("") == ""
    assert _derive_therapeutic_purpose_category(None) == ""


def test_resolve_attaches_nih_purpose_and_cadro_for_normal_nih_match():
    result = resolve_drug_classification("Baricitinib", [], [], overrides={}, nih_name_lookup=_nih_lookup())
    assert result["therapeutic_purpose_class"] == "DTT"
    assert result["therapeutic_purpose_category"] == "Small Molecule"
    assert result["cadro"] == "Inflammation"


def test_resolve_attaches_nih_purpose_and_cadro_even_when_curated_override_wins():
    # AR1001 has its own real NIH row (DTT; small molecule) even though
    # its target_pathways come from the curated override, not NIH — the
    # supplementary NIH cross-reference must still surface
    overrides = {"ar1001": {
        "modality": "Small Molecule", "target_pathways": ["Amyloid", "Tau", "Neuroprotection"],
        "molecular_targets": ["PDE5"], "mechanism_of_action": "PDE5 inhibitor",
        "reason": "curated", "source": "test",
    }}
    result = resolve_drug_classification("AR1001", [], [], overrides=overrides, nih_name_lookup=_nih_lookup())
    assert result["classification_source"] == "curated_override"
    assert result["therapeutic_purpose_class"] == "DTT"
    assert result["therapeutic_purpose_category"] == "Small Molecule"
    assert result["cadro"] != ""


def test_resolve_no_nih_match_leaves_purpose_and_cadro_blank_not_fabricated():
    result = resolve_drug_classification(
        "CompletelyUnmatchedDrugXYZ", [], [{"type": "DRUG", "name": "CompletelyUnmatchedDrugXYZ", "title": ""}],
        overrides={}, nih_name_lookup={},
    )
    assert result["therapeutic_purpose_class"] == ""
    assert result["therapeutic_purpose_category"] == ""
    assert result["cadro"] == ""


# ------------------------------------------------------------
# classify_pipeline_quadrant() — the 4-category "pipeline chart" drug_type scheme
# ------------------------------------------------------------

def test_quadrant_uses_nih_purpose_directly_when_it_maps_cleanly():
    quadrant, source, inferred = classify_pipeline_quadrant("Small Molecule", "Amyloid", "DTT", "Small Molecule")
    assert quadrant == "Disease-Targeted Small Molecule"
    assert source == "nih_reference"
    assert inferred is False

    quadrant, source, inferred = classify_pipeline_quadrant("Biologic", "Amyloid", "DTT", "Biologic")
    assert quadrant == "Disease-Targeted Biologic"
    assert inferred is False

    quadrant, source, inferred = classify_pipeline_quadrant("Small Molecule", "Symptomatic", "STT", "Cognition Enhancer")
    assert quadrant == "Cognition Enhancer"
    assert inferred is False

    quadrant, source, inferred = classify_pipeline_quadrant("Small Molecule", "Neuropsychiatric", "STT", "Neuropsychiatric")
    assert quadrant == "Neuropsychiatric Symptom Tx"
    assert inferred is False


def test_quadrant_infers_neuropsychiatric_from_target_when_no_nih_match():
    quadrant, source, inferred = classify_pipeline_quadrant("Small Molecule", "Neuropsychiatric", "", "")
    assert quadrant == "Neuropsychiatric Symptom Tx"
    assert source == "inferred_from_target_pathway"
    assert inferred is True


def test_quadrant_infers_cognition_enhancer_from_symptomatic_target():
    quadrant, source, inferred = classify_pipeline_quadrant("Small Molecule", "Symptomatic", "", "")
    assert quadrant == "Cognition Enhancer"
    assert source == "inferred_from_target_pathway"
    assert inferred is True


def test_quadrant_infers_disease_targeted_biologic_from_modality():
    quadrant, source, inferred = classify_pipeline_quadrant("Biologic", "Amyloid", "", "")
    assert quadrant == "Disease-Targeted Biologic"
    assert source == "inferred_from_modality"
    assert inferred is True

    # Cell/Gene Therapy is regulated as a biologic — same bucket
    quadrant, _, _ = classify_pipeline_quadrant("Cell/Gene Therapy", "Amyloid", "", "")
    assert quadrant == "Disease-Targeted Biologic"


def test_quadrant_defaults_to_disease_targeted_small_molecule():
    quadrant, source, inferred = classify_pipeline_quadrant("Small Molecule", "Amyloid", "", "")
    assert quadrant == "Disease-Targeted Small Molecule"
    assert source == "inferred_default"
    assert inferred is True

    # edge-case modalities with no symptomatic target signal fall through
    # to the same default rather than being left uncategorized
    quadrant, _, _ = classify_pipeline_quadrant("Dietary Supplement", "Other", "", "")
    assert quadrant == "Disease-Targeted Small Molecule"
    quadrant, _, _ = classify_pipeline_quadrant("Unknown", "Other", "", "")
    assert quadrant == "Disease-Targeted Small Molecule"


def test_quadrant_always_returns_one_of_the_four_canonical_values():
    samples = [
        ("Small Molecule", "Amyloid", "DTT", "Small Molecule"),
        ("Biologic", "Tau", "DTT", "Biologic"),
        ("Small Molecule", "Symptomatic", "STT", "Cognition Enhancer"),
        ("Small Molecule", "Neuropsychiatric", "STT", "Neuropsychiatric"),
        ("Device", "Other", "", ""),
        ("Cell/Gene Therapy", "Inflammation", "", ""),
        ("Unknown", "Metabolism", "", ""),
    ]
    for modality, target, purpose_class, purpose_category in samples:
        quadrant, _, _ = classify_pipeline_quadrant(modality, target, purpose_class, purpose_category)
        assert quadrant in PIPELINE_QUADRANTS


# ------------------------------------------------------------
# build_official_pipeline_classification_lookup() (tier 3)
# ------------------------------------------------------------

def test_official_pipeline_classification_lookup_reads_real_csv():
    pipeline_records = load_official_pipeline(PIPELINE_CSV_PATH)
    lookup = build_official_pipeline_classification_lookup(pipeline_records)
    assert "lecanemab" in lookup
    assert lookup["lecanemab"]["modality"] == "Biologic"
    assert lookup["lecanemab"]["target_pathways"] == ["Amyloid"]


def test_official_pipeline_classification_lookup_skips_rows_without_modality_or_target():
    records = [{"drug_name": "SomeDrug", "modality": "", "target_pathways": ""}]
    lookup = build_official_pipeline_classification_lookup(records)
    assert lookup == {}


# ------------------------------------------------------------
# build_classification_conflicts_dataframe()
# ------------------------------------------------------------

def test_conflicts_dataframe_flags_modality_and_target_changes():
    records = [{
        "canonical_drug_name": "DrugA", "previous_modality": "Small Molecule", "previous_target": "Other",
        "new_modality": "Biologic", "new_target_pathways": ["Amyloid"],
        "classification_source": "nih_reference", "classification_confidence": "high",
        "classification_reason": "NIH reference evidence", "manual_review_required": False,
    }]
    df = build_classification_conflicts_dataframe(records)
    row = df.iloc[0]
    assert row["manual_review_required"] == True
    assert "modality corrected" in row["conflict_reason"]
    assert "target corrected" in row["conflict_reason"]


def test_conflicts_dataframe_no_spurious_flag_when_target_stays_other():
    # regression test for a real bug found during Phase 2 development:
    # an EMPTY new_target_pathways list (-> displays as "Other") must
    # NOT be flagged as "changed" when previous_target was ALREADY "Other"
    records = [{
        "canonical_drug_name": "DrugB", "previous_modality": "Small Molecule", "previous_target": "Other",
        "new_modality": "Small Molecule", "new_target_pathways": [],
        "classification_source": "needs_review", "classification_confidence": "low",
        "classification_reason": "no evidence found", "manual_review_required": True,
    }]
    df = build_classification_conflicts_dataframe(records)
    row = df.iloc[0]
    assert "target corrected" not in row["conflict_reason"]


def test_conflicts_dataframe_no_change_case():
    records = [{
        "canonical_drug_name": "DrugC", "previous_modality": "Biologic", "previous_target": "Amyloid",
        "new_modality": "Biologic", "new_target_pathways": ["Amyloid"],
        "classification_source": "nih_reference", "classification_confidence": "high",
        "classification_reason": "NIH reference evidence (agrees with existing evidence)", "manual_review_required": False,
    }]
    df = build_classification_conflicts_dataframe(records)
    row = df.iloc[0]
    assert row["manual_review_required"] == False
    assert row["conflict_reason"] == "no change from legacy classification"


def test_conflicts_dataframe_output_columns():
    df = build_classification_conflicts_dataframe([])
    expected = {
        "canonical_drug_name", "previous_modality", "new_modality", "previous_target",
        "new_target_pathways", "classification_source", "confidence", "conflict_reason",
        "manual_review_required",
    }
    assert set(df.columns) == expected


ALL_TESTS = [
    test_load_drug_classification_overrides_missing_file_returns_empty_dict,
    test_load_drug_classification_overrides_parses_semicolon_lists,
    test_load_drug_classification_overrides_missing_column_raises,
    test_the_real_drug_classification_overrides_csv_has_ar1001,
    test_gather_structured_evidence_isolates_verified_rows_only,
    test_gather_structured_evidence_carries_trial_title,
    test_gather_structured_evidence_empty_for_no_match,
    test_gather_structured_evidence_empty_dataframe,
    test_infer_modality_biologic_from_name_keyword,
    test_infer_modality_biologic_from_type,
    test_infer_modality_small_molecule_from_drug_type,
    test_infer_modality_device_and_dietary_supplement,
    test_infer_modality_empty_evidence_returns_blank_not_unknown,
    test_infer_target_pathways_from_name_text,
    test_infer_target_pathways_from_title_text_not_just_name,
    test_infer_target_pathways_supports_multiple_pathways,
    test_infer_target_pathways_empty_evidence,
    test_build_nih_name_lookup_solo_agent_wins_over_combination_component,
    test_match_drug_to_nih_by_display_name,
    test_match_drug_to_nih_by_synonym,
    test_match_drug_to_nih_no_match_returns_none,
    test_resolve_tier1_curated_override_wins_outright,
    test_resolve_biologic_previously_labeled_small_molecule,
    test_resolve_cadro_target_correction_via_nih,
    test_resolve_multi_target_drug_via_structured_evidence,
    test_resolve_nih_match_high_confidence,
    test_resolve_non_nih_drug_uses_existing_evidence_without_confidence_penalty,
    test_resolve_conflicting_nih_and_existing_evidence_flagged,
    test_resolve_unresolved_drug_falls_through_to_needs_review,
    test_claude_inference_tier_is_a_documented_noop,
    test_resolve_confidence_scoring_agree_vs_disagree_vs_single_source,
    test_derive_therapeutic_purpose_category_strips_parenthetical,
    test_derive_therapeutic_purpose_category_dtt_modality_values,
    test_derive_therapeutic_purpose_category_takes_first_of_multiple_purposes,
    test_derive_therapeutic_purpose_category_blank_input,
    test_resolve_attaches_nih_purpose_and_cadro_for_normal_nih_match,
    test_resolve_attaches_nih_purpose_and_cadro_even_when_curated_override_wins,
    test_resolve_no_nih_match_leaves_purpose_and_cadro_blank_not_fabricated,
    test_quadrant_uses_nih_purpose_directly_when_it_maps_cleanly,
    test_quadrant_infers_neuropsychiatric_from_target_when_no_nih_match,
    test_quadrant_infers_cognition_enhancer_from_symptomatic_target,
    test_quadrant_infers_disease_targeted_biologic_from_modality,
    test_quadrant_defaults_to_disease_targeted_small_molecule,
    test_quadrant_always_returns_one_of_the_four_canonical_values,
    test_official_pipeline_classification_lookup_reads_real_csv,
    test_official_pipeline_classification_lookup_skips_rows_without_modality_or_target,
    test_conflicts_dataframe_flags_modality_and_target_changes,
    test_conflicts_dataframe_no_spurious_flag_when_target_stays_other,
    test_conflicts_dataframe_no_change_case,
    test_conflicts_dataframe_output_columns,
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
