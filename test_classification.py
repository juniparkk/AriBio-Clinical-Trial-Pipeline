# ============================================================
# TESTS for drug_classification.py
#
# Plain-Python tests (no pytest install needed) — run with:
#     .venv/bin/python test_classification.py
#
# Each test is a small function that calls a function from
# drug_classification.py and checks the result with assert.
# run_test() catches failures so ALL tests run and report, instead
# of stopping at the first failure.
#
# Tests are grouped by function:
#   1. normalize_text()
#   2. parse_interventions()
#   3. load_official_pipeline() / match_official_pipeline()
#   4. classify_intervention()
#
# For classify_intervention() tests, pipeline_records is [] unless the
# test is specifically about official-pipeline matching — that keeps
# the classification-priority tests independent of data/official_pipeline.csv's
# contents, which will grow over time.
# ============================================================

import os

import pandas as pd

from drug_classification import (
    normalize_text,
    parse_interventions,
    load_official_pipeline,
    match_official_pipeline,
    classify_intervention,
    build_interventions_dataframe,
    resolve_developed_drug,
    normalize_intervention_candidate_name,
    _contains_phrase,
    build_resolved_drugs_dataframe,
    build_unresolved_trials_dataframe,
    build_target_phase_counts,
    build_resolved_drug_trial_links_df,
    classify_pipeline_scope,
    load_scope_overrides,
    build_scope_audit_dataframe,
    PIPELINE_SCOPE_LABELS,
    build_drug_date_rollup,
    _is_isotope_labeled_name,
    determine_diagnostic_subtype,
    build_diagnostic_agent_audit_dataframe,
    _is_extended_non_drug_activity,
    _has_known_therapeutic_evidence,
    build_resolved_drugs_exclusion_audit_dataframe,
    RESOLVED_DRUGS_DF_ELIGIBLE_SCOPES,
    EXTENDED_NON_DRUG_REASON,
    _is_diagnostic_challenge_or_probe_purpose,
    _is_deprescribing_or_procedural_support,
)

PIPELINE_CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "official_pipeline.csv")


def make_record(company, drug_name, synonyms=None, source_url=""):
    """Test-only fixture builder — constructs a pipeline record dict by
    hand, in the same shape load_official_pipeline() produces, without
    needing a real CSV file. Used for synthetic edge cases (ambiguous
    matches, short-name substring safety) that shouldn't depend on
    data/official_pipeline.csv's real contents changing over time."""
    synonyms = synonyms or []
    return {
        "company": company,
        "drug_name": drug_name,
        "synonyms": synonyms,
        "source_url": source_url,
        "notes": "",
        "company_normalized": normalize_text(company),
        "drug_name_normalized": normalize_text(drug_name),
        "synonyms_normalized": [normalize_text(s) for s in synonyms],
    }


# ------------------------------------------------------------
# normalize_text()
# ------------------------------------------------------------

def test_normalize_lowercases():
    assert normalize_text("Leqembi") == "leqembi"


def test_normalize_strips_punctuation():
    assert normalize_text("BMS-708163") == "bms 708163"


def test_normalize_strips_registered_trademark_symbol():
    assert normalize_text("Leqembi®") == "leqembi"


def test_normalize_collapses_extra_whitespace():
    assert normalize_text("  AR1001   Placebo  ") == "ar1001 placebo"


def test_normalize_handles_none():
    assert normalize_text(None) == ""


def test_normalize_handles_nan():
    assert normalize_text(pd.NA) == ""
    assert normalize_text(float("nan")) == ""


def test_normalize_handles_empty_string():
    assert normalize_text("") == ""


# ------------------------------------------------------------
# parse_interventions()
# ------------------------------------------------------------

def test_parse_ar1001_plus_placebo():
    result = parse_interventions("DRUG: AR1001|OTHER: Placebo")
    assert result == [
        {"type": "DRUG", "name": "AR1001"},
        {"type": "OTHER", "name": "Placebo"},
    ]


def test_parse_wujia_yizhi_granules_plus_placebo():
    result = parse_interventions("DRUG: Wujia Yizhi granules|OTHER: Placebo")
    assert result == [
        {"type": "DRUG", "name": "Wujia Yizhi granules"},
        {"type": "OTHER", "name": "Placebo"},
    ]


def test_parse_sar110894_donepezil_placebo_keeps_all_three():
    result = parse_interventions("DRUG: SAR110894|DRUG: Donepezil|OTHER: Placebo")
    assert result == [
        {"type": "DRUG", "name": "SAR110894"},
        {"type": "DRUG", "name": "Donepezil"},
        {"type": "OTHER", "name": "Placebo"},
    ]


def test_parse_florbetapir_alone():
    result = parse_interventions("DRUG: Florbetapir F18")
    assert result == [{"type": "DRUG", "name": "Florbetapir F18"}]


def test_parse_ct_scan_procedure():
    result = parse_interventions("PROCEDURE: CT scan")
    assert result == [{"type": "PROCEDURE", "name": "CT scan"}]


def test_parse_nintendo_wii_behavioral():
    result = parse_interventions("BEHAVIORAL: Exercise with Nintendo Wii")
    assert result == [{"type": "BEHAVIORAL", "name": "Exercise with Nintendo Wii"}]


def test_parse_entry_without_type_prefix_is_kept():
    result = parse_interventions("Some Untyped Entry|DRUG: AR1001")
    assert result == [
        {"type": None, "name": "Some Untyped Entry"},
        {"type": "DRUG", "name": "AR1001"},
    ]


def test_parse_handles_none():
    assert parse_interventions(None) == []


def test_parse_handles_nan():
    assert parse_interventions(pd.NA) == []
    assert parse_interventions(float("nan")) == []


def test_parse_handles_empty_string():
    assert parse_interventions("") == []


# ------------------------------------------------------------
# normalize_intervention_candidate_name()
# ------------------------------------------------------------

def test_normalize_candidate_strips_dose_per_day():
    assert normalize_intervention_candidate_name("TRx0237 150 mg/day") == "TRx0237"
    assert normalize_intervention_candidate_name("TRx0237 250 mg/day") == "TRx0237"


def test_normalize_candidate_strips_plain_mg_dose():
    assert normalize_intervention_candidate_name("AR1001 30 mg") == "AR1001"


def test_normalize_candidate_strips_dose_and_route_and_frequency():
    assert normalize_intervention_candidate_name("Donepezil 10 mg once daily") == "Donepezil"
    assert normalize_intervention_candidate_name("Lecanemab 10 mg/kg IV every 2 weeks") == "Lecanemab"


def test_normalize_candidate_preserves_avp786_hyphenated_code():
    # "AVP-786-18" has no recognized dose/unit suffix (no mg/mcg/g/iu/ml
    # anywhere) — it must pass through completely unchanged. This is
    # the regression case for "do not strip meaningful digits that are
    # part of a development code."
    assert normalize_intervention_candidate_name("AVP-786-18") == "AVP-786-18"
    assert normalize_intervention_candidate_name("AVP-786-28") == "AVP-786-28"
    assert normalize_intervention_candidate_name("AVP-786-42.63") == "AVP-786-42.63"


def test_normalize_candidate_handles_none_and_empty():
    assert normalize_intervention_candidate_name(None) == ""
    assert normalize_intervention_candidate_name("") == ""


# ------------------------------------------------------------
# load_official_pipeline()
# ------------------------------------------------------------

def test_load_official_pipeline_reads_seed_file():
    records = load_official_pipeline(PIPELINE_CSV_PATH)
    assert len(records) == 6
    ariBio_rows = [r for r in records if r["company"] == "AriBio"]
    assert len(ariBio_rows) == 1
    assert ariBio_rows[0]["drug_name"] == "AR1001"
    assert ariBio_rows[0]["synonyms"] == ["mirodenafil"]
    assert ariBio_rows[0]["company_normalized"] == "aribio"
    assert ariBio_rows[0]["drug_name_normalized"] == "ar1001"
    assert ariBio_rows[0]["synonyms_normalized"] == ["mirodenafil"]
    # source_url is intentionally blank in the seed template
    assert ariBio_rows[0]["source_url"] == ""


def test_load_official_pipeline_does_not_modify_the_file():
    before = open(PIPELINE_CSV_PATH, "rb").read()
    load_official_pipeline(PIPELINE_CSV_PATH)
    after = open(PIPELINE_CSV_PATH, "rb").read()
    assert before == after


def test_load_official_pipeline_missing_file_returns_empty_list():
    assert load_official_pipeline("does/not/exist.csv") == []


def test_load_official_pipeline_missing_column_raises():
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("company,drug_name\nAcme,Foo\n")
        path = f.name
    try:
        raised = False
        try:
            load_official_pipeline(path)
        except ValueError:
            raised = True
        assert raised
    finally:
        os.remove(path)


# ------------------------------------------------------------
# match_official_pipeline()
# ------------------------------------------------------------

# Loaded once — used by tests that are specifically about the real
# seed file's content (Lecanemab/Eisai synonyms, AR1001/AriBio, etc.)
PIPELINE_RECORDS = load_official_pipeline(PIPELINE_CSV_PATH)


def test_match_ar1001_aribio():
    result = match_official_pipeline("AR1001", "AriBio", PIPELINE_RECORDS)
    assert result["matched"] is True
    assert result["matched_company"] == "AriBio"
    assert result["matched_drug_name"] == "AR1001"
    assert result["source_url"] == ""
    assert result["verification_status"] == "pipeline_record_match_without_source"


def test_match_ar1001_unrelated_university_sponsor_does_not_match():
    # AR1001 appears in the pipeline file, but for a DIFFERENT sponsor —
    # a drug-name match without a sponsor/company match must not count.
    result = match_official_pipeline("AR1001", "Seoul National University Hospital", PIPELINE_RECORDS)
    assert result["matched"] is False
    assert result["verification_status"] == "no_match"


def test_match_ban2401_eisai_synonym():
    result = match_official_pipeline("BAN2401", "Eisai", PIPELINE_RECORDS)
    assert result["matched"] is True
    assert result["matched_drug_name"] == "Lecanemab"
    assert result["matched_alias"] == "BAN2401"
    assert result["verification_status"] == "pipeline_record_match_without_source"


def test_match_leqembi_registered_trademark_eisai():
    result = match_official_pipeline("LEQEMBI®", "Eisai", PIPELINE_RECORDS)
    assert result["matched"] is True
    assert result["matched_drug_name"] == "Lecanemab"
    assert result["matched_alias"] == "Leqembi"


def test_match_lecanemab_biogen_does_not_match_without_a_biogen_record():
    # The seed file's Lecanemab row lists Eisai as the company. There is
    # no Biogen row, so — even though Biogen co-developed lecanemab in
    # real life — this function must not infer that; it only knows what
    # is actually in official_pipeline.csv.
    result = match_official_pipeline("Lecanemab", "Biogen", PIPELINE_RECORDS)
    assert result["matched"] is False
    assert result["verification_status"] == "no_match"


def test_match_ambiguous_when_two_company_records_match_same_drug_name():
    r1 = make_record("Eisai", "TestDrug", source_url="")
    r2 = make_record("Eisai Pharmaceuticals", "TestDrug", source_url="https://example.com/testdrug")
    result = match_official_pipeline("TestDrug", "Eisai Inc", [r1, r2])
    assert result["matched"] is False
    assert result["verification_status"] == "ambiguous_multiple_matches"
    assert len(result["candidate_matches"]) == 2


def test_match_short_name_does_not_substring_match_longer_official_name():
    # AR100 (missing a digit) must not match the real AR1001 record.
    result = match_official_pipeline("AR100", "AriBio", PIPELINE_RECORDS)
    assert result["matched"] is False
    # AR10011 (extra digit) must not match either.
    result2 = match_official_pipeline("AR10011", "AriBio", PIPELINE_RECORDS)
    assert result2["matched"] is False


def test_match_longer_name_does_not_substring_match_shorter_official_name():
    # A short official record ("AR100") must not be triggered by a
    # longer intervention name that merely contains it as a substring.
    record = make_record("TestCo", "AR100", source_url="")
    result = match_official_pipeline("AR1001", "TestCo", [record])
    assert result["matched"] is False


def test_match_confirmed_when_source_url_present():
    record = make_record("Acme Pharma", "AcmeDrug", source_url="https://acme.example.com/acmedrug")
    result = match_official_pipeline("AcmeDrug", "Acme Pharma Inc", [record])
    assert result["matched"] is True
    assert result["verification_status"] == "confirmed_official_match"


# ------------------------------------------------------------
# classify_intervention()
# ------------------------------------------------------------
# pipeline_records is [] for tests that are purely about classification
# priority, so they don't depend on data/official_pipeline.csv's
# real (evolving) contents.

def test_classify_ar1001_aribio_is_sponsor_developed_medium_confidence():
    result = classify_intervention("DRUG", "AR1001", "AriBio", [], PIPELINE_RECORDS)
    assert result["classification"] == "sponsor_developed_therapeutic"
    assert result["confidence"] == "medium"
    assert result["verification_status"] == "pipeline_record_match_without_source"
    assert result["needs_manual_review"] is True
    assert result["official_pipeline_match"] is True


def test_classify_ar1001_unrelated_sponsor_is_not_sponsor_developed():
    result = classify_intervention("DRUG", "AR1001", "Seoul National University Hospital", [], PIPELINE_RECORDS)
    assert result["classification"] != "sponsor_developed_therapeutic"
    # AR1001 is still development-code-shaped, so it's a plausible
    # unverified investigational candidate — just not confirmed as
    # sponsor-developed merely because the name appears in the file.
    assert result["classification"] == "investigational_therapeutic_unverified"


def test_classify_florbetapir_f18_typed_drug_is_diagnostic():
    result = classify_intervention("DRUG", "Florbetapir F18", "Some Sponsor", [], [])
    assert result["classification"] == "diagnostic_or_imaging_agent"


def test_classify_all_curated_tracers_are_diagnostic():
    for tracer_name in ["Florbetaben", "Flutemetamol", "Pittsburgh Compound B", "PiB", "FDG"]:
        result = classify_intervention("DRUG", tracer_name, "Some Sponsor", [], [])
        assert result["classification"] == "diagnostic_or_imaging_agent", tracer_name


def test_classify_expanded_amyloid_tracer_aliases_are_diagnostic():
    for name in ["Amyvid", "AV-45", "av45", "Neuraceq", "Vizamyl", "11C-PiB", "NAV4694", "AZD4694"]:
        result = classify_intervention("DRUG", name, "Some Sponsor", [], [])
        assert result["classification"] == "diagnostic_or_imaging_agent", name


def test_classify_flortaucipir_and_av1451_are_diagnostic_even_typed_drug():
    # Real trials.csv row NCT02016560 types Flortaucipir F18 as DRUG.
    # Before this checkpoint it fell through to
    # investigational_therapeutic_unverified — it's a Tau PET tracer,
    # not a therapeutic, and must be excluded regardless of ct.gov's
    # (misleading) type label, just like Florbetapir already was.
    for name in ["Flortaucipir F18", "Flortaucipir", "AV-1451", "av1451"]:
        result = classify_intervention("DRUG", name, "Avid Radiopharmaceuticals", [], [])
        assert result["classification"] == "diagnostic_or_imaging_agent", name


def test_classify_expanded_tau_tracer_aliases_are_diagnostic():
    for name in ["Tauvid", "T807", "MK-6240", "mk6240", "PI-2620", "pi2620",
                 "RO948", "RO6958948", "GTP1", "Genentech Tau Probe 1",
                 "PM-PBB3", "PBB3", "THK-5317", "THK-5351", "THK-5117"]:
        result = classify_intervention("DRUG", name, "Some Sponsor", [], [])
        assert result["classification"] == "diagnostic_or_imaging_agent", name


def test_classify_metabolic_tracer_aliases_are_diagnostic():
    for name in ["FDG", "18F-FDG", "Fluorodeoxyglucose", "FDG PET"]:
        result = classify_intervention("DRUG", name, "Some Sponsor", [], [])
        assert result["classification"] == "diagnostic_or_imaging_agent", name


def test_classify_unknown_name_containing_f18_is_not_excluded_as_diagnostic():
    result = classify_intervention("DRUG", "XYZ-9284 (F18-labeled)", "Some Sponsor", [], [])
    assert result["classification"] != "diagnostic_or_imaging_agent"


# ------------------------------------------------------------
# real-data imaging false-positive fixes (found via browser testing:
# NCT05043675/NCT04141150/NCT05542953/NCT07422857 [18F]APN-1607,
# NCT04604600/NCT03542656 "amyloid PET", NCT07611357 duplicated
# "DRUG: Drug: [18F]F-AraG (PET tracer)")
# ------------------------------------------------------------

def test_classify_apn1607_aliases_are_diagnostic():
    for name in ["[18F]APN-1607", "[18F]-APN-1607", "APN-1607"]:
        result = classify_intervention("DRUG", name, "Some Sponsor", [], [])
        assert result["classification"] == "diagnostic_or_imaging_agent", name


def test_classify_bare_amyloid_and_tau_pet_are_diagnostic():
    for name in ["amyloid PET", "tau PET", "brain PET"]:
        result = classify_intervention("DRUG", name, "Some Sponsor", [], [])
        assert result["classification"] == "diagnostic_or_imaging_agent", name


def test_classify_f_ara_g_diagnostic_only_with_explicit_pet_tracer_wording():
    # Real row: NCT07611357, "DRUG: Drug: [18F]F-AraG (PET tracer)" —
    # explicit "(PET tracer)" wording makes this diagnostic.
    explicit = classify_intervention("DRUG", "[18F]F-AraG (PET tracer)", "Some Sponsor", [], [])
    assert explicit["classification"] == "diagnostic_or_imaging_agent"

    # But F-AraG must NOT be assumed diagnostic from isotope notation
    # alone — bare "[18F]F-AraG" with no imaging-context wording at all
    # is not in any curated list, so it correctly falls through to
    # manual review rather than being silently excluded or included.
    bare = classify_intervention("DRUG", "[18F]F-AraG", "Some Sponsor", [], [])
    assert bare["classification"] != "diagnostic_or_imaging_agent"


def test_classify_tau_pet_typed_other_is_diagnostic():
    result = classify_intervention("OTHER", "tau PET", "Some Sponsor", [], [])
    assert result["classification"] == "diagnostic_or_imaging_agent"


def test_classify_novel_f18_compound_without_imaging_wording_not_automatically_diagnostic():
    result = classify_intervention("DRUG", "novel compound with F18 in its name", "Some Sponsor", [], [])
    assert result["classification"] != "diagnostic_or_imaging_agent"


def test_classify_pet_scan_procedure_typed_procedure_stays_procedure():
    result = classify_intervention("PROCEDURE", "PET scan", "Some Sponsor", [], [])
    assert result["classification"] == "procedure"


def test_classify_amyloid_and_brain_pet_scan_remain_procedure_not_diagnostic():
    # The scan/procedure form must NOT be caught by the bare "amyloid
    # pet"/"brain pet" diagnostic phrase match, even though it's a
    # prefix of the longer "... pet scan" sequence.
    for name in ["amyloid PET scan", "brain PET scan"]:
        result = classify_intervention("RADIATION", name, "Some Sponsor", [], [])
        assert result["classification"] == "procedure", name


def test_classify_amyloid_and_tau_pet_imaging_are_diagnostic():
    for name in ["amyloid PET imaging", "tau PET imaging"]:
        result = classify_intervention("DRUG", name, "Some Sponsor", [], [])
        assert result["classification"] == "diagnostic_or_imaging_agent", name


def test_classify_pet_ligand_is_diagnostic():
    result = classify_intervention("DRUG", "PET ligand", "Some Sponsor", [], [])
    assert result["classification"] == "diagnostic_or_imaging_agent"


def test_classify_pm_pbb3_still_diagnostic():
    result = classify_intervention("DRUG", "PM-PBB3", "Some Sponsor", [], [])
    assert result["classification"] == "diagnostic_or_imaging_agent"


def test_parse_duplicated_drug_prefix_is_stripped():
    result = parse_interventions("DRUG: Drug: Florbetapir F18")
    assert result == [{"type": "DRUG", "name": "Florbetapir F18"}]
    assert not result[0]["name"].lower().startswith("drug:")


def test_parse_duplicated_prefix_real_f_ara_g_row():
    # Real row: NCT07611357
    result = parse_interventions("DRUG: Drug: [18F]F-AraG (PET tracer)")
    assert result == [{"type": "DRUG", "name": "[18F]F-AraG (PET tracer)"}]


def test_classify_f_ara_g_with_duplicated_prefix_end_to_end():
    # Full pipeline: parse (strips the duplicated prefix) then classify
    parsed = parse_interventions("DRUG: Drug: [18F]F-AraG (PET tracer)")
    entry = parsed[0]
    result = classify_intervention(entry["type"], entry["name"], "Some Sponsor", [], [])
    assert result["classification"] == "diagnostic_or_imaging_agent"
    assert not result["original_name"].lower().startswith("drug:")


def test_classify_unknown_name_containing_c11_is_not_excluded_as_diagnostic():
    result = classify_intervention("DRUG", "ABC-4471 [C11]", "Some Sponsor", [], [])
    assert result["classification"] != "diagnostic_or_imaging_agent"


def test_classify_pet_scan_is_procedure():
    result = classify_intervention("DIAGNOSTIC_TEST", "PET scan", "Some Sponsor", [], [])
    assert result["classification"] == "procedure"


def test_classify_ct_scan_typed_radiation_is_procedure():
    result = classify_intervention("RADIATION", "CT scan", "Some Sponsor", [], [])
    assert result["classification"] == "procedure"


def test_classify_spect_scan_is_procedure_not_via_the_ct_scan_false_match():
    # Real trials.csv row: NCT00605046, "DRUG: SPECT scan". SPECT is now
    # a recognized imaging-procedure keyword (added this checkpoint), so
    # this SHOULD classify as "procedure" — but via the legitimate
    # "spect" whole-token match, never via "ct scan" accidentally
    # matching as a substring of "spect scan" (the tail of "spe-CT" +
    # " SCAN" spells "ct scan" by coincidence). Assert both things
    # directly: the classification is correct, AND the token-sequence
    # phrase matcher specifically does not fire on "ct scan" for this
    # input (so a future edit can't silently reintroduce the bug behind
    # a still-passing top-level assertion).
    result = classify_intervention("DRUG", "SPECT scan", "Some Sponsor", [], [])
    assert result["classification"] == "procedure"
    assert not _contains_phrase("spect scan", "ct scan")


def test_classify_pet_ct_mri_spect_scan_are_all_procedures():
    for name in ["PET scan", "CT scan", "MRI scan", "SPECT scan"]:
        result = classify_intervention("DRUG", name, "Some Sponsor", [], [])
        assert result["classification"] == "procedure", name


def test_classify_placebo_for_sar110894_trial_is_placebo():
    result = classify_intervention("OTHER", "Placebo", "Some Sponsor", [], [])
    assert result["classification"] == "placebo_or_sham"


def test_classify_matched_placebo_sham_and_vehicle_control_are_placebo():
    for name in ["Matched Placebo", "Sham", "Vehicle Control (saline)"]:
        result = classify_intervention("OTHER", name, "Some Sponsor", [], [])
        assert result["classification"] == "placebo_or_sham", name


def test_classify_expanded_placebo_phrase_list():
    for name in ["Placebo", "Placebos", "Matched Placebo", "Matching Placebo",
                 "Placebo Control", "Placebo Comparator", "Sham", "Sham Control",
                 "Vehicle Control"]:
        result = classify_intervention("DRUG", name, "Some Sponsor", [], [])
        assert result["classification"] == "placebo_or_sham", name


def test_classify_bromocriptine_mesilate_plus_placebos_plural():
    # Real trials.csv row: NCT04413344, "DRUG: Bromocriptine
    # Mesilate|DRUG: Placebos". Before this checkpoint, "Placebos"
    # (plural) wasn't recognized, so both interventions looked like
    # "2 unresolved candidates" and Bromocriptine Mesilate was wrongly
    # marked "uncertain" instead of being recognized as the trial's
    # sole real candidate.
    bromocriptine = {"type": "DRUG", "name": "Bromocriptine Mesilate"}
    placebos = {"type": "DRUG", "name": "Placebos"}

    placebos_result = classify_intervention("DRUG", "Placebos", "Kyoto University", [bromocriptine], [])
    assert placebos_result["classification"] == "placebo_or_sham"

    bromocriptine_result = classify_intervention("DRUG", "Bromocriptine Mesilate", "Kyoto University", [placebos], [])
    assert bromocriptine_result["classification"] == "investigational_therapeutic_unverified"


def test_classify_no_intervention_alone_is_other_not_investigational():
    # Real trials.csv row: NCT07177352, "OTHER: No Intervention" (the
    # trial's only intervention). Must never become
    # investigational_therapeutic_unverified.
    result = classify_intervention("OTHER", "No Intervention", "Hoffmann-La Roche", [], [])
    assert result["classification"] == "other"
    assert result["classification"] != "investigational_therapeutic_unverified"
    assert "non-treatment control" in result["reason"]


def test_classify_untreated_is_other():
    result = classify_intervention("OTHER", "Untreated", "Some Sponsor", [], [])
    assert result["classification"] == "other"


def test_classify_bare_usual_care_and_standard_of_care_are_other():
    for name in ["Usual Care", "Standard of Care"]:
        result = classify_intervention("OTHER", name, "Some Sponsor", [], [])
        assert result["classification"] == "other", name


def test_classify_drug_plus_no_intervention_no_intervention_not_a_sibling_candidate():
    # "No Intervention" must not count as a plausible therapeutic
    # sibling — a real drug alongside it should still resolve via the
    # sole-candidate rule (as if "No Intervention" weren't there at all).
    no_intervention = {"type": "OTHER", "name": "No Intervention"}
    drug_result = classify_intervention("DRUG", "Wujia Yizhi granules", "Some Sponsor", [no_intervention], [])
    assert drug_result["classification"] == "investigational_therapeutic_unverified"

    wujia = {"type": "DRUG", "name": "Wujia Yizhi granules"}
    ni_result = classify_intervention("OTHER", "No Intervention", "Some Sponsor", [wujia], [])
    assert ni_result["classification"] == "other"


def test_classify_drug_plus_placebo_plus_no_intervention():
    drug = {"type": "DRUG", "name": "SAR110894"}
    placebo = {"type": "OTHER", "name": "Placebo"}
    no_intervention = {"type": "OTHER", "name": "No Intervention"}

    drug_result = classify_intervention("DRUG", "SAR110894", "Sanofi", [placebo, no_intervention], [])
    placebo_result = classify_intervention("OTHER", "Placebo", "Sanofi", [drug, no_intervention], [])
    ni_result = classify_intervention("OTHER", "No Intervention", "Sanofi", [drug, placebo], [])

    assert drug_result["classification"] == "investigational_therapeutic_unverified"
    assert placebo_result["classification"] == "placebo_or_sham"
    assert ni_result["classification"] == "other"


def test_classify_bare_vehicle_alone_is_not_automatically_placebo():
    # Per requirement D.1: "vehicle" alone (no "control") must not be
    # auto-classified as placebo — it could be part of an active
    # formulation's own name.
    result = classify_intervention("DRUG", "Vehicle", "Some Sponsor", [], [])
    assert result["classification"] != "placebo_or_sham"


def test_classify_nintendo_wii_exercise_is_behavioral():
    result = classify_intervention("BEHAVIORAL", "Exercise with Nintendo Wii", "Some Sponsor", [], [])
    assert result["classification"] == "behavioral"


def test_classify_sar110894_donepezil_placebo_trial():
    sponsor = "Sanofi"
    sar = {"type": "DRUG", "name": "SAR110894"}
    donepezil = {"type": "DRUG", "name": "Donepezil"}
    placebo = {"type": "OTHER", "name": "Placebo"}

    sar_result = classify_intervention("DRUG", "SAR110894", sponsor, [donepezil, placebo], [])
    assert sar_result["classification"] == "investigational_therapeutic_unverified"

    donepezil_result = classify_intervention("DRUG", "Donepezil", sponsor, [sar, placebo], [])
    assert donepezil_result["classification"] == "comparator_or_background_therapy"

    placebo_result = classify_intervention("OTHER", "Placebo", sponsor, [sar, donepezil], [])
    assert placebo_result["classification"] == "placebo_or_sham"


def test_classify_donepezil_alone_with_placebo_is_investigational_unverified():
    sponsor = "Some Academic Sponsor"
    placebo = {"type": "OTHER", "name": "Placebo"}
    result = classify_intervention("DRUG", "Donepezil", sponsor, [placebo], [])
    assert result["classification"] == "investigational_therapeutic_unverified"
    assert result["needs_manual_review"] is True


def test_classify_donepezil_vs_memantine_head_to_head_is_uncertain():
    sponsor = "Some Academic Sponsor"
    donepezil = {"type": "DRUG", "name": "Donepezil"}
    memantine = {"type": "DRUG", "name": "Memantine"}

    donepezil_result = classify_intervention("DRUG", "Donepezil", sponsor, [memantine], [])
    memantine_result = classify_intervention("DRUG", "Memantine", sponsor, [donepezil], [])

    assert donepezil_result["classification"] == "uncertain"
    assert donepezil_result["needs_manual_review"] is True
    assert memantine_result["classification"] == "uncertain"
    assert memantine_result["needs_manual_review"] is True


def test_classify_wujia_yizhi_granules_with_placebo():
    # Wujia Yizhi granules is a real, named herbal formulation, not
    # code-shaped and not in the curated KNOWN_COMPOUND_NAMES list. The
    # sole-plausible-therapeutic-candidate rule (added this checkpoint)
    # catches this: its only sibling is Placebo, which fails the
    # therapeutic gate, so it's the trial's one real candidate and gets
    # promoted to investigational_therapeutic_unverified rather than
    # being discarded as merely "uncertain".
    placebo = {"type": "OTHER", "name": "Placebo"}
    result = classify_intervention("DRUG", "Wujia Yizhi granules", "Some Sponsor", [placebo], [])
    assert result["classification"] == "investigational_therapeutic_unverified"
    assert result["confidence"] == "medium"
    assert result["needs_manual_review"] is True

    placebo_result = classify_intervention("OTHER", "Placebo", "Some Sponsor", [{"type": "DRUG", "name": "Wujia Yizhi granules"}], [])
    assert placebo_result["classification"] == "placebo_or_sham"


def test_classify_sole_candidate_rule_does_not_fire_with_two_unresolved_candidates():
    # Two named, non-code-shaped, non-approved-background candidates in
    # the same trial — the sole-candidate rule must NOT promote either
    # of them; both should fall through to "uncertain" instead.
    other_herbal = {"type": "DRUG", "name": "Ginkgo Biloba Extract"}
    wujia = {"type": "DRUG", "name": "Wujia Yizhi granules"}

    wujia_result = classify_intervention("DRUG", "Wujia Yizhi granules", "Some Sponsor", [other_herbal], [])
    other_result = classify_intervention("DRUG", "Ginkgo Biloba Extract", "Some Sponsor", [wujia], [])

    assert wujia_result["classification"] == "uncertain"
    assert other_result["classification"] == "uncertain"


def test_classify_ar1001_aricept_placebo_aricept_is_comparator():
    ar1001 = {"type": "DRUG", "name": "AR1001"}
    placebo = {"type": "OTHER", "name": "Placebo"}
    result = classify_intervention("DRUG", "Aricept", "AriBio", [ar1001, placebo], [])
    assert result["classification"] == "comparator_or_background_therapy"


def test_classify_namenda_alone_with_placebo_is_investigational_unverified():
    placebo = {"type": "OTHER", "name": "Placebo"}
    result = classify_intervention("DRUG", "Namenda", "Some Academic Sponsor", [placebo], [])
    assert result["classification"] == "investigational_therapeutic_unverified"
    assert result["needs_manual_review"] is True


def test_classify_exelon_vs_razadyne_head_to_head_is_uncertain():
    exelon = {"type": "DRUG", "name": "Exelon"}
    razadyne = {"type": "DRUG", "name": "Razadyne"}

    exelon_result = classify_intervention("DRUG", "Exelon", "Some Academic Sponsor", [razadyne], [])
    razadyne_result = classify_intervention("DRUG", "Razadyne", "Some Academic Sponsor", [exelon], [])

    assert exelon_result["classification"] == "uncertain"
    assert exelon_result["needs_manual_review"] is True
    assert razadyne_result["classification"] == "uncertain"
    assert razadyne_result["needs_manual_review"] is True


def test_classify_fdg_as_whole_word_is_diagnostic_but_fdg_inside_longer_word_is_not():
    fdg_result = classify_intervention("DRUG", "FDG", "Some Sponsor", [], [])
    assert fdg_result["classification"] == "diagnostic_or_imaging_agent"

    fdg_pet_result = classify_intervention("DRUG", "FDG PET", "Some Sponsor", [], [])
    assert fdg_pet_result["classification"] == "diagnostic_or_imaging_agent"

    # "Fdgotinib" is a made-up therapeutic-sounding name that merely
    # contains the letters "fdg" as part of one longer word — it must
    # NOT be excluded as a diagnostic tracer just because of that.
    unrelated_result = classify_intervention("DRUG", "Fdgotinib", "Some Sponsor", [], [])
    assert unrelated_result["classification"] != "diagnostic_or_imaging_agent"


def test_classify_ambiguous_pipeline_match_forces_manual_review():
    r1 = make_record("Eisai", "TestDrug", source_url="")
    r2 = make_record("Eisai Pharmaceuticals", "TestDrug", source_url="https://example.com/testdrug")
    result = classify_intervention("DRUG", "TestDrug", "Eisai Inc", [], [r1, r2])
    assert result["classification"] != "sponsor_developed_therapeutic"
    assert result["official_pipeline_match"] is False
    assert result["verification_status"] == "ambiguous_multiple_matches"
    assert result["needs_manual_review"] is True


# ------------------------------------------------------------
# resolve_developed_drug()
# ------------------------------------------------------------

def test_resolve_one_confirmed_official_match():
    record = make_record("Acme Pharma", "AcmeDrug", source_url="https://acme.example.com/acmedrug")
    drug = {"type": "DRUG", "name": "AcmeDrug"}
    placebo = {"type": "OTHER", "name": "Placebo"}
    classified = [
        classify_intervention("DRUG", "AcmeDrug", "Acme Pharma", [placebo], [record]),
        classify_intervention("OTHER", "Placebo", "Acme Pharma", [drug], [record]),
    ]
    result = resolve_developed_drug(classified)
    assert result["developed_drug"] == "AcmeDrug"
    assert result["drug_classification"] == "sponsor_developed_therapeutic"
    assert result["classification_confidence"] == "high"
    assert result["needs_manual_review"] is False
    assert result["official_pipeline_match"] is True
    assert result["official_source_url"] == "https://acme.example.com/acmedrug"


def test_resolve_one_pipeline_match_without_source():
    ar1001 = {"type": "DRUG", "name": "AR1001"}
    placebo = {"type": "OTHER", "name": "Placebo"}
    classified = [
        classify_intervention("DRUG", "AR1001", "AriBio", [placebo], PIPELINE_RECORDS),
        classify_intervention("OTHER", "Placebo", "AriBio", [ar1001], PIPELINE_RECORDS),
    ]
    result = resolve_developed_drug(classified)
    assert result["developed_drug"] == "AR1001"
    assert result["drug_classification"] == "sponsor_developed_therapeutic"
    assert result["classification_confidence"] == "medium"
    assert result["needs_manual_review"] is True


def test_resolve_one_unverified_therapeutic():
    sar = {"type": "DRUG", "name": "SAR110894"}
    placebo = {"type": "OTHER", "name": "Placebo"}
    classified = [
        classify_intervention("DRUG", "SAR110894", "Sanofi", [placebo], []),
        classify_intervention("OTHER", "Placebo", "Sanofi", [sar], []),
    ]
    result = resolve_developed_drug(classified)
    assert result["developed_drug"] == "SAR110894"
    assert result["drug_classification"] == "investigational_therapeutic_unverified"
    assert result["classification_confidence"] == "medium"
    assert result["needs_manual_review"] is True


def test_resolve_multiple_confirmed_candidates():
    r1 = make_record("Acme Pharma", "AcmeDrug", source_url="https://acme.example.com/a")
    r2 = make_record("Acme Pharma", "AnotherDrug", source_url="https://acme.example.com/b")
    acme = {"type": "DRUG", "name": "AcmeDrug"}
    another = {"type": "DRUG", "name": "AnotherDrug"}
    classified = [
        classify_intervention("DRUG", "AcmeDrug", "Acme Pharma", [another], [r1, r2]),
        classify_intervention("DRUG", "AnotherDrug", "Acme Pharma", [acme], [r1, r2]),
    ]
    result = resolve_developed_drug(classified)
    assert result["drug_classification"] == "sponsor_developed_therapeutic"
    assert "AcmeDrug" in result["developed_drug"]
    assert "AnotherDrug" in result["developed_drug"]
    assert result["classification_confidence"] == "low"
    assert result["needs_manual_review"] is True


def test_resolve_multiple_unverified_candidates():
    sar = {"type": "DRUG", "name": "SAR110894"}
    xyz = {"type": "DRUG", "name": "XYZ204"}
    classified = [
        classify_intervention("DRUG", "SAR110894", "Some Sponsor", [xyz], []),
        classify_intervention("DRUG", "XYZ204", "Some Sponsor", [sar], []),
    ]
    result = resolve_developed_drug(classified)
    assert result["drug_classification"] == "investigational_therapeutic_unverified"
    assert "SAR110894" in result["developed_drug"]
    assert "XYZ204" in result["developed_drug"]
    assert result["classification_confidence"] == "low"
    assert result["needs_manual_review"] is True


def test_resolve_placebo_only_trial():
    classified = [classify_intervention("OTHER", "Placebo", "Some Sponsor", [], [])]
    result = resolve_developed_drug(classified)
    assert result["developed_drug"] == ""
    assert result["drug_classification"] == "no_therapeutic_candidate"
    assert result["classification_confidence"] == "high"
    assert result["needs_manual_review"] is False


def test_resolve_diagnostic_only_trial():
    classified = [classify_intervention("DRUG", "Florbetapir F18", "Some Sponsor", [], [])]
    result = resolve_developed_drug(classified)
    assert result["developed_drug"] == ""
    assert result["drug_classification"] == "no_therapeutic_candidate"
    assert result["needs_manual_review"] is False


def test_resolve_therapeutic_plus_placebo_plus_diagnostic_tracer():
    sar = {"type": "DRUG", "name": "SAR110894"}
    placebo = {"type": "OTHER", "name": "Placebo"}
    tracer = {"type": "DRUG", "name": "Florbetapir F18"}
    classified = [
        classify_intervention("DRUG", "SAR110894", "Sanofi", [placebo, tracer], []),
        classify_intervention("OTHER", "Placebo", "Sanofi", [sar, tracer], []),
        classify_intervention("DRUG", "Florbetapir F18", "Sanofi", [sar, placebo], []),
    ]
    result = resolve_developed_drug(classified)
    assert result["developed_drug"] == "SAR110894"
    assert result["drug_classification"] == "investigational_therapeutic_unverified"
    assert result["needs_manual_review"] is True


def test_resolve_approved_background_alongside_investigational_candidate():
    sar = {"type": "DRUG", "name": "SAR110894"}
    donepezil = {"type": "DRUG", "name": "Donepezil"}
    classified = [
        classify_intervention("DRUG", "SAR110894", "Sanofi", [donepezil], []),
        classify_intervention("DRUG", "Donepezil", "Sanofi", [sar], []),
    ]
    # Donepezil should have resolved to comparator, not a candidate
    assert classified[1]["classification"] == "comparator_or_background_therapy"
    result = resolve_developed_drug(classified)
    assert result["developed_drug"] == "SAR110894"
    assert result["drug_classification"] == "investigational_therapeutic_unverified"
    assert result["needs_manual_review"] is True


def test_resolve_trx0237_dose_variants_resolve_to_one_candidate():
    # Real trials.csv row: NCT01689246, "DRUG: TRx0237 150 mg/day|DRUG:
    # TRx0237 250 mg/day|DRUG: Placebo". Before dose-normalization, both
    # TRx0237 rows failed the development-code regex (the trailing
    # "150 mg/day"/"250 mg/day" text broke the match), so they showed
    # up as 2 separate "uncertain" candidates instead of one recognized
    # drug at two doses.
    dose150 = {"type": "DRUG", "name": "TRx0237 150 mg/day"}
    dose250 = {"type": "DRUG", "name": "TRx0237 250 mg/day"}
    placebo = {"type": "OTHER", "name": "Placebo"}

    classified = [
        classify_intervention("DRUG", "TRx0237 150 mg/day", "TauRx Therapeutics Ltd", [dose250, placebo], []),
        classify_intervention("DRUG", "TRx0237 250 mg/day", "TauRx Therapeutics Ltd", [dose150, placebo], []),
        classify_intervention("OTHER", "Placebo", "TauRx Therapeutics Ltd", [dose150, dose250], []),
    ]
    assert classified[0]["classification"] == "investigational_therapeutic_unverified"
    assert classified[1]["classification"] == "investigational_therapeutic_unverified"

    result = resolve_developed_drug(classified)
    assert result["developed_drug"] == "TRx0237"
    assert result["drug_classification"] == "investigational_therapeutic_unverified"
    assert result["needs_manual_review"] is True
    # the key regression: must NOT be flagged as multiple/ambiguous
    assert "multiple" not in result["classification_reason"].lower()


def test_resolve_ar1001_dose_variants_resolve_to_one_candidate():
    dose30 = {"type": "DRUG", "name": "AR1001 30 mg"}
    dose60 = {"type": "DRUG", "name": "AR1001 60 mg"}
    placebo = {"type": "OTHER", "name": "Placebo"}

    classified = [
        classify_intervention("DRUG", "AR1001 30 mg", "Some Academic Sponsor", [dose60, placebo], []),
        classify_intervention("DRUG", "AR1001 60 mg", "Some Academic Sponsor", [dose30, placebo], []),
        classify_intervention("OTHER", "Placebo", "Some Academic Sponsor", [dose30, dose60], []),
    ]
    result = resolve_developed_drug(classified)
    assert result["developed_drug"] == "AR1001"
    assert result["drug_classification"] == "investigational_therapeutic_unverified"


def test_resolve_avp786_variants_are_not_truncated_and_remain_distinct():
    # AVP-786-18/-28/-42.63 are NOT simple dose suffixes (no
    # recognized mg/mcg/g/iu/ml unit text) — normalize_intervention_candidate_name
    # correctly leaves each one fully distinct, so they remain 3
    # separate candidates rather than being incorrectly collapsed into one.
    v18 = {"type": "DRUG", "name": "AVP-786-18"}
    v28 = {"type": "DRUG", "name": "AVP-786-28"}
    v4263 = {"type": "DRUG", "name": "AVP-786-42.63"}
    placebo = {"type": "OTHER", "name": "Placebo"}

    classified = [
        classify_intervention("DRUG", "AVP-786-18", "Otsuka Pharmaceutical", [v28, v4263, placebo], []),
        classify_intervention("DRUG", "AVP-786-28", "Otsuka Pharmaceutical", [v18, v4263, placebo], []),
        classify_intervention("DRUG", "AVP-786-42.63", "Otsuka Pharmaceutical", [v18, v28, placebo], []),
        classify_intervention("OTHER", "Placebo", "Otsuka Pharmaceutical", [v18, v28, v4263], []),
    ]
    for row in classified[:3]:
        assert row["candidate_name"] == row["original_name"]


def test_resolve_wujia_yizhi_granules_plus_placebo():
    wujia = {"type": "DRUG", "name": "Wujia Yizhi granules"}
    placebo = {"type": "OTHER", "name": "Placebo"}
    classified = [
        classify_intervention("DRUG", "Wujia Yizhi granules", "Some Sponsor", [placebo], []),
        classify_intervention("OTHER", "Placebo", "Some Sponsor", [wujia], []),
    ]
    result = resolve_developed_drug(classified)
    assert result["developed_drug"] == "Wujia Yizhi granules"
    assert result["drug_classification"] == "investigational_therapeutic_unverified"
    assert result["needs_manual_review"] is True


# ------------------------------------------------------------
# integration test: build_interventions_dataframe() + resolve_developed_drug()
# over a small in-memory multi-trial dataset
# ------------------------------------------------------------

def test_integration_multi_trial_dataframe():
    trials_df = pd.DataFrame([
        {
            "nct_id": "NCT00000001", "sponsor": "AriBio", "title": "AR1001 Phase 3",
            "interventions": "DRUG: AR1001|OTHER: Placebo",
        },
        {
            "nct_id": "NCT00000002", "sponsor": "Sanofi", "title": "SAR110894 vs Donepezil",
            "interventions": "DRUG: SAR110894|DRUG: Donepezil|OTHER: Placebo",
        },
        {
            "nct_id": "NCT00000003", "sponsor": "Some Academic Sponsor", "title": "Wujia Yizhi granules Study",
            "interventions": "DRUG: Wujia Yizhi granules|OTHER: Placebo",
        },
        {
            "nct_id": "NCT00000004", "sponsor": "Imaging Sponsor", "title": "Florbetapir imaging substudy",
            "interventions": "DRUG: Florbetapir F18",
        },
    ])

    interventions_df = build_interventions_dataframe(trials_df, PIPELINE_RECORDS)

    # nothing silently discarded: 2 + 3 + 2 + 1 = 8 individual interventions in, 8 rows out
    assert len(interventions_df) == 8
    assert set(interventions_df["nct_id"]) == {"NCT00000001", "NCT00000002", "NCT00000003", "NCT00000004"}

    resolved_by_trial = {}
    for nct_id, group in interventions_df.groupby("nct_id"):
        resolved_by_trial[nct_id] = resolve_developed_drug(group.to_dict("records"))

    assert resolved_by_trial["NCT00000001"]["developed_drug"] == "AR1001"
    assert resolved_by_trial["NCT00000001"]["drug_classification"] == "sponsor_developed_therapeutic"

    assert resolved_by_trial["NCT00000002"]["developed_drug"] == "SAR110894"
    assert resolved_by_trial["NCT00000002"]["drug_classification"] == "investigational_therapeutic_unverified"

    assert resolved_by_trial["NCT00000003"]["developed_drug"] == "Wujia Yizhi granules"
    assert resolved_by_trial["NCT00000003"]["drug_classification"] == "investigational_therapeutic_unverified"

    assert resolved_by_trial["NCT00000004"]["developed_drug"] == ""
    assert resolved_by_trial["NCT00000004"]["drug_classification"] == "no_therapeutic_candidate"

    # every row has the fields pipeline_interventions.csv will need
    for _, row in interventions_df.iterrows():
        assert "original_name" in row and "classification" in row and "verification_status" in row


# ------------------------------------------------------------
# build_resolved_drugs_dataframe() / build_unresolved_trials_dataframe()
#
# These test the DRUG-LEVEL ROLLUP in isolation, using synthetic
# trial-level rows shaped like what STEP 3.6 in pipeline_viz.py merges
# onto `df` (one row per trial, already carrying resolve_developed_drug()'s
# output) — NOT re-deriving them via classify_intervention(), since
# trial-level resolution correctness was already covered by last
# checkpoint's resolve_developed_drug() tests. This checkpoint's new
# behavior is purely about how multiple TRIALS roll up into one drug row.
# ------------------------------------------------------------

def make_trial_row(nct_id, sponsor, developed_drug, drug_classification, verification_status,
                    classification_confidence, needs_manual_review, phase="Phase 2", status="Recruiting",
                    drug_type="Small Molecule", target="Amyloid", enrollment=100, is_aribio=False,
                    title="Test Trial", interventions="DRUG: X", classification_reason="test reason",
                    pipeline_scope="Therapeutic Drug", scope_reason="test scope reason",
                    scope_method="rule_classification", scope_confidence="high", manual_review_required=None):
    return {
        "nct_id": nct_id, "sponsor": sponsor, "title": title, "interventions": interventions,
        "phase_clean": phase, "status_clean": status, "drug_type": drug_type, "target": target,
        "enrollment": enrollment, "is_aribio": is_aribio,
        "developed_drug": developed_drug,
        "developed_drug_normalized": normalize_text(developed_drug),
        "drug_classification": drug_classification,
        "classification_reason": classification_reason,
        "verification_status": verification_status,
        "classification_confidence": classification_confidence,
        "needs_manual_review": needs_manual_review,
        # Phase 1A: defaults to "Therapeutic Drug" so every PRE-EXISTING
        # test built on this fixture (written before Phase 1A existed)
        # keeps behaving exactly as before — build_resolved_drugs_dataframe()
        # now also gates on pipeline_scope, so a row with no scope info at
        # all would otherwise be silently dropped by that new filter.
        "pipeline_scope": pipeline_scope,
        "scope_reason": scope_reason,
        "scope_method": scope_method,
        "scope_confidence": scope_confidence,
        "manual_review_required": manual_review_required if manual_review_required is not None else needs_manual_review,
    }


def test_rollup_placebo_only_trial_produces_no_drug_row():
    trials_df = pd.DataFrame([
        make_trial_row("NCT1", "Some Sponsor", "", "no_therapeutic_candidate", "not_applicable", "high", False),
    ])
    result = build_resolved_drugs_dataframe(trials_df)
    assert len(result) == 0


def test_rollup_diagnostic_only_trial_produces_no_drug_row():
    trials_df = pd.DataFrame([
        make_trial_row("NCT2", "Avid Radiopharmaceuticals", "", "no_therapeutic_candidate", "not_applicable", "high", False),
    ])
    result = build_resolved_drugs_dataframe(trials_df)
    assert len(result) == 0


def test_rollup_device_only_trial_produces_no_drug_row():
    trials_df = pd.DataFrame([
        make_trial_row("NCT3", "Device Co", "", "no_therapeutic_candidate", "not_applicable", "high", False),
    ])
    result = build_resolved_drugs_dataframe(trials_df)
    assert len(result) == 0


def test_rollup_ar1001_confirmed_trial_produces_one_row():
    trials_df = pd.DataFrame([
        make_trial_row("NCT4", "AriBio", "AR1001", "sponsor_developed_therapeutic",
                        "pipeline_record_match_without_source", "medium", True, phase="Phase 3"),
    ])
    result = build_resolved_drugs_dataframe(trials_df)
    assert len(result) == 1
    assert result.iloc[0]["display_name"] == "AR1001"
    assert result.iloc[0]["confirmed_trial_count"] == 1
    assert result.iloc[0]["unverified_trial_count"] == 0


# --- regression: every phase_clean value pipeline_viz.py's clean_phase()
# can produce (not just Phase 1/2/3) must have a _DRUG_ROLLUP_PHASE_RANK
# entry, or a drug whose trials are ALL one of the newer values (NA,
# Early Phase 1, Phase 4, or a combined Phase 1/Phase 2 / Phase 2/Phase 3
# designation) would make g["phase_rank"].max() NaN, leaving top_rows
# empty and raising an IndexError building the rollup — this used to be
# unreachable because those trials were filtered out entirely upstream.

def test_rollup_drug_with_only_na_phase_trial_does_not_crash():
    trials_df = pd.DataFrame([
        make_trial_row("NCT20", "Some Sponsor", "SomeDrug", "investigational_therapeutic_unverified",
                        "no_match", "medium", True, phase="NA"),
    ])
    result = build_resolved_drugs_dataframe(trials_df)
    assert len(result) == 1
    assert result.iloc[0]["phase_reached"] == "NA"


def test_rollup_drug_with_only_phase4_trial_does_not_crash():
    trials_df = pd.DataFrame([
        make_trial_row("NCT21", "Some Sponsor", "SomeDrug", "investigational_therapeutic_unverified",
                        "no_match", "medium", True, phase="Phase 4"),
    ])
    result = build_resolved_drugs_dataframe(trials_df)
    assert len(result) == 1
    assert result.iloc[0]["phase_reached"] == "Phase 4"


def test_rollup_drug_with_only_early_phase1_trial_does_not_crash():
    trials_df = pd.DataFrame([
        make_trial_row("NCT22", "Some Sponsor", "SomeDrug", "investigational_therapeutic_unverified",
                        "no_match", "medium", True, phase="Early Phase 1"),
    ])
    result = build_resolved_drugs_dataframe(trials_df)
    assert len(result) == 1
    assert result.iloc[0]["phase_reached"] == "Early Phase 1"


def test_rollup_combined_dual_phase_values_do_not_crash():
    trials_df = pd.DataFrame([
        make_trial_row("NCT23", "Some Sponsor", "SomeDrug", "investigational_therapeutic_unverified",
                        "no_match", "medium", True, phase="Phase 1/Phase 2"),
    ])
    result = build_resolved_drugs_dataframe(trials_df)
    assert len(result) == 1
    assert result.iloc[0]["phase_reached"] == "Phase 1/Phase 2"


def test_rollup_phase4_outranks_na_for_the_same_drug():
    trials_df = pd.DataFrame([
        make_trial_row("NCT24", "Some Sponsor", "SomeDrug", "investigational_therapeutic_unverified",
                        "no_match", "medium", True, phase="NA"),
        make_trial_row("NCT25", "Some Sponsor", "SomeDrug", "investigational_therapeutic_unverified",
                        "no_match", "medium", True, phase="Phase 4"),
    ])
    result = build_resolved_drugs_dataframe(trials_df)
    assert len(result) == 1
    assert result.iloc[0]["phase_reached"] == "Phase 4"
    assert result.iloc[0]["trial_count"] == 2


def test_rollup_wujia_unverified_trial_produces_one_row():
    trials_df = pd.DataFrame([
        make_trial_row("NCT5", "Some Sponsor", "Wujia Yizhi granules", "investigational_therapeutic_unverified",
                        "no_match", "medium", True),
    ])
    result = build_resolved_drugs_dataframe(trials_df)
    assert len(result) == 1
    assert result.iloc[0]["display_name"] == "Wujia Yizhi granules"
    assert result.iloc[0]["unverified_trial_count"] == 1
    assert result.iloc[0]["confirmed_trial_count"] == 0


def test_rollup_same_drug_across_two_trials_collapses_to_one_row():
    # Trial-level dose-arm collapsing ("TRx0237 150 mg/day" + "TRx0237
    # 250 mg/day" -> one candidate) was already tested at the
    # resolve_developed_drug() level last checkpoint. This checkpoint's
    # new behavior is that the DRUG ROLLUP also collapses the same
    # resolved drug name across DIFFERENT trials into one row.
    trials_df = pd.DataFrame([
        make_trial_row("NCT6", "TauRx Therapeutics Ltd", "TRx0237", "investigational_therapeutic_unverified",
                        "no_match", "medium", True),
        make_trial_row("NCT7", "TauRx Therapeutics Ltd", "TRx0237", "investigational_therapeutic_unverified",
                        "no_match", "medium", True),
    ])
    result = build_resolved_drugs_dataframe(trials_df)
    assert len(result) == 1
    assert result.iloc[0]["display_name"] == "TRx0237"
    assert result.iloc[0]["trial_count"] == 2
    assert result.iloc[0]["unverified_trial_count"] == 2


def test_rollup_case_variant_duplicate_names_collapse():
    trials_df = pd.DataFrame([
        make_trial_row("NCT8", "Some Sponsor", "AR1001", "investigational_therapeutic_unverified", "no_match", "medium", True),
        make_trial_row("NCT9", "Some Sponsor", "ar1001", "investigational_therapeutic_unverified", "no_match", "medium", True),
    ])
    result = build_resolved_drugs_dataframe(trials_df)
    assert len(result) == 1
    assert result.iloc[0]["display_name"].lower() == "ar1001"
    assert result.iloc[0]["trial_count"] == 2


def test_rollup_one_confirmed_plus_one_unverified_produces_one_mixed_row():
    trials_df = pd.DataFrame([
        make_trial_row("NCT10", "Cassava Sciences", "Simufilam", "sponsor_developed_therapeutic",
                        "confirmed_official_match", "high", False),
        make_trial_row("NCT11", "Some Academic Sponsor", "simufilam", "investigational_therapeutic_unverified",
                        "no_match", "medium", True),
    ])
    result = build_resolved_drugs_dataframe(trials_df)
    assert len(result) == 1
    row = result.iloc[0]
    assert row["display_name"] == "Simufilam"  # canonical casing preferred
    assert row["confirmed_trial_count"] == 1
    assert row["unverified_trial_count"] == 1
    assert row["verification_status"] == "mixed"
    assert row["classification_confidence"] == "medium"  # min(high, medium)
    # bool(...): pandas returns numpy.bool_ here (post DataFrame round-trip),
    # and numpy.bool_(True) is not the same object as the Python True singleton
    assert bool(row["needs_manual_review"]) is True  # propagated from the unverified trial
    # multiple distinct sponsors preserved, not silently collapsed
    assert "Cassava Sciences" in row["sponsor"]
    assert "Some Academic Sponsor" in row["sponsor"]


def test_rollup_multiple_unresolved_candidates_excluded():
    trials_df = pd.DataFrame([
        make_trial_row("NCT12", "Some Sponsor", "DrugA; DrugB", "sponsor_developed_therapeutic",
                        "multiple_candidates_unresolved", "low", True),
    ])
    result = build_resolved_drugs_dataframe(trials_df)
    assert len(result) == 0


def test_unresolved_trials_csv_includes_ambiguous_and_uncertain_trials():
    trials_df = pd.DataFrame([
        make_trial_row("NCT13", "Some Sponsor", "DrugA; DrugB", "sponsor_developed_therapeutic",
                        "multiple_candidates_unresolved", "low", True,
                        interventions="DRUG: DrugA|DRUG: DrugB"),
        make_trial_row("NCT14", "Some Sponsor", "", "no_therapeutic_candidate", "no_match", "low", True,
                        interventions="DRUG: Donepezil|DRUG: Memantine"),
        # a cleanly-resolved trial must NOT show up here
        make_trial_row("NCT15", "AriBio", "AR1001", "sponsor_developed_therapeutic",
                        "pipeline_record_match_without_source", "medium", True),
        # a cleanly non-therapeutic trial (placebo/diagnostic-only) must NOT show up here either
        make_trial_row("NCT16", "Some Sponsor", "", "no_therapeutic_candidate", "not_applicable", "high", False),
    ])
    result = build_unresolved_trials_dataframe(trials_df)
    nct_numbers = set(result["NCT Number"])
    assert nct_numbers == {"NCT13", "NCT14"}
    assert "Interventions" in result.columns


# ------------------------------------------------------------
# Phase 0 — build_target_phase_counts() / build_resolved_drug_trial_links_df()
# (the pure functions that let the heatmap and the drug<->trial join
# read only from resolved_drugs_df, never legacy_drugs_df or raw df)
# ------------------------------------------------------------

def test_build_target_phase_counts_basic_crosstab():
    resolved = pd.DataFrame([
        {"target": "Amyloid", "phase_reached": "Phase 3"},
        {"target": "Amyloid", "phase_reached": "Phase 3"},
        {"target": "Amyloid", "phase_reached": "Phase 1"},
        {"target": "Tau", "phase_reached": "Phase 2"},
    ])
    z = build_target_phase_counts(resolved, targets=["Amyloid", "Tau"], phases=["Phase 1", "Phase 2", "Phase 3"])
    assert z == [
        [1, 0, 2],  # Amyloid: 1 in Phase 1, 0 in Phase 2, 2 in Phase 3
        [0, 1, 0],  # Tau: 0, 1, 0
    ]


def test_build_target_phase_counts_empty_dataframe():
    resolved = pd.DataFrame(columns=["target", "phase_reached"])
    z = build_target_phase_counts(resolved, targets=["Amyloid"], phases=["Phase 1"])
    assert z == [[0]]


def test_build_target_phase_counts_sum_equals_eligible_drug_count():
    # every row's target is in `targets` and phase in `phases` here, so
    # the full grid must sum to exactly len(resolved) — this is the
    # "heatmap totals reconcile to resolved drug counts" guarantee
    resolved = pd.DataFrame([
        {"target": "Amyloid", "phase_reached": "Phase 1"},
        {"target": "Amyloid", "phase_reached": "Phase 2"},
        {"target": "Tau", "phase_reached": "Phase 3"},
        {"target": "Inflammation", "phase_reached": "Phase 1"},
    ])
    targets = ["Amyloid", "Tau", "Inflammation"]
    phases = ["Phase 1", "Phase 2", "Phase 3"]
    z = build_target_phase_counts(resolved, targets, phases)
    assert sum(sum(row) for row in z) == len(resolved)


def test_build_resolved_drug_trial_links_df_explodes_nct_ids():
    resolved = pd.DataFrame([
        {"display_name": "AR1001", "nct_ids": "NCT001; NCT002"},
        {"display_name": "SAR110894", "nct_ids": "NCT003"},
    ])
    links = build_resolved_drug_trial_links_df(resolved)
    assert len(links) == 3
    assert set(links[links["display_name"] == "AR1001"]["nct_id"]) == {"NCT001", "NCT002"}
    assert set(links[links["display_name"] == "SAR110894"]["nct_id"]) == {"NCT003"}


def test_build_resolved_drug_trial_links_df_handles_blank_nct_ids():
    resolved = pd.DataFrame([{"display_name": "SomeDrug", "nct_ids": ""}])
    links = build_resolved_drug_trial_links_df(resolved)
    assert len(links) == 0


def test_build_resolved_drug_trial_links_df_row_count_matches_trial_count_sum():
    # every trial resolves to at most ONE drug (resolve_developed_drug()
    # never splits a trial across two drugs), so the number of links
    # must equal the number of DISTINCT trials referenced — never more
    resolved = pd.DataFrame([
        {"display_name": "AR1001", "nct_ids": "NCT001; NCT002", "trial_count": 2},
        {"display_name": "SAR110894", "nct_ids": "NCT003", "trial_count": 1},
    ])
    links = build_resolved_drug_trial_links_df(resolved)
    assert len(links) == resolved["trial_count"].sum()
    assert links["nct_id"].nunique() == len(links)


# ------------------------------------------------------------
# build_drug_date_rollup() — earliest start / latest primary completion
# per canonical drug, across ALL contributing trials
# ------------------------------------------------------------

def test_build_drug_date_rollup_basic():
    links = pd.DataFrame([
        {"display_name": "AR1001", "nct_id": "NCT001"},
        {"display_name": "AR1001", "nct_id": "NCT002"},
        {"display_name": "SAR110894", "nct_id": "NCT003"},
    ])
    trials = pd.DataFrame([
        {"nct_id": "NCT001", "start_date_parsed": pd.Timestamp("2020-01-01"), "primary_completion_date_parsed": pd.Timestamp("2023-06-01")},
        {"nct_id": "NCT002", "start_date_parsed": pd.Timestamp("2019-03-01"), "primary_completion_date_parsed": pd.Timestamp("2026-12-01")},
        {"nct_id": "NCT003", "start_date_parsed": pd.Timestamp("2022-05-01"), "primary_completion_date_parsed": pd.Timestamp("2024-01-01")},
    ])
    result = build_drug_date_rollup(links, trials).set_index("display_name")
    # AR1001: earliest of the two starts, latest of the two completions —
    # never just whichever trial happens to be listed first
    assert result.loc["AR1001", "earliest_start_date"] == pd.Timestamp("2019-03-01")
    assert result.loc["AR1001", "latest_primary_completion_date"] == pd.Timestamp("2026-12-01")
    assert result.loc["SAR110894", "earliest_start_date"] == pd.Timestamp("2022-05-01")


def test_build_drug_date_rollup_missing_dates_become_nat_not_fabricated():
    links = pd.DataFrame([{"display_name": "DrugX", "nct_id": "NCT001"}])
    trials = pd.DataFrame([
        {"nct_id": "NCT001", "start_date_parsed": pd.NaT, "primary_completion_date_parsed": pd.NaT},
    ])
    result = build_drug_date_rollup(links, trials).set_index("display_name")
    assert pd.isna(result.loc["DrugX", "earliest_start_date"])
    assert pd.isna(result.loc["DrugX", "latest_primary_completion_date"])


def test_build_drug_date_rollup_empty_links_returns_empty_with_columns():
    result = build_drug_date_rollup(pd.DataFrame(columns=["display_name", "nct_id"]), pd.DataFrame())
    assert len(result) == 0
    assert list(result.columns) == ["display_name", "earliest_start_date", "latest_primary_completion_date"]


# ------------------------------------------------------------
# Phase 1A — classify_pipeline_scope() / load_scope_overrides() /
# build_scope_audit_dataframe()
#
# classify_intervention()'s own classification/CLASSIFICATION_LABELS are
# untouched by Phase 1A (every test above this section still exercises
# the exact same behavior it always has) — these are the NEW gap-closure
# layer's tests: confirmed dietary-supplement/generic-diagnostic-test/
# combination-product/genetic leakage, and the curated-override escape
# hatch.
# ------------------------------------------------------------

def test_scope_dietary_supplement_investigational_excluded_from_therapeutic():
    # confirmed leakage example: "lutein/zeaxanthin"
    r = classify_pipeline_scope("DIETARY_SUPPLEMENT", "lutein/zeaxanthin", "investigational_therapeutic_unverified", "no_match")
    assert r["pipeline_scope"] != "Therapeutic Drug"
    assert r["pipeline_scope"] == "Non-Drug Intervention"
    assert r["scope_method"] == "rule_type"


def test_scope_dietary_supplement_curcumin_c3_complex_excluded():
    r = classify_pipeline_scope("DIETARY_SUPPLEMENT", "Curcumin C3 Complex", "investigational_therapeutic_unverified", "no_match")
    assert r["pipeline_scope"] == "Non-Drug Intervention"


def test_scope_behavioral_excluded_from_therapeutic():
    r = classify_pipeline_scope("BEHAVIORAL", "Cognitive Training", "behavioral", "not_applicable")
    assert r["pipeline_scope"] == "Non-Drug Intervention"


def test_scope_device_excluded_from_therapeutic():
    r = classify_pipeline_scope("DEVICE", "TMS Device", "device", "not_applicable")
    assert r["pipeline_scope"] == "Non-Drug Intervention"


def test_scope_blood_test_is_not_a_drug():
    # Blood Test/CSF Biomarkers commonly land as classify_intervention()'s
    # "uncertain" (multi-candidate ambiguity resolves before the ct.gov
    # type is ever consulted) — the scope layer must still catch them via
    # its type-agnostic generic-description net, not just via type.
    for classification in ("uncertain", "investigational_therapeutic_unverified"):
        r = classify_pipeline_scope("DIAGNOSTIC_TEST", "Blood Test", classification, "no_match")
        assert r["pipeline_scope"] == "Exclude", f"Blood Test must never be a drug (classification={classification})"


def test_scope_csf_biomarkers_is_not_a_drug():
    for classification in ("uncertain", "investigational_therapeutic_unverified"):
        r = classify_pipeline_scope("DIAGNOSTIC_TEST", "Cerebrospinal fluid (CSF) Biomarkers", classification, "no_match")
        assert r["pipeline_scope"] == "Exclude"


def test_scope_cbti_with_application_is_not_a_drug():
    # real trials.csv typed this COMBINATION_PRODUCT even though it's a
    # digital cognitive-behavioral-therapy program, not a pharmaceutical —
    # the type-agnostic keyword net must catch it regardless of type.
    r = classify_pipeline_scope("COMBINATION_PRODUCT", "CBTi with Application", "investigational_therapeutic_unverified", "no_match")
    assert r["pipeline_scope"] != "Therapeutic Drug"
    assert r["pipeline_scope"] == "Non-Drug Intervention"


def test_scope_generic_diagnostic_description_wins_over_generic_non_drug_net():
    # a name matching BOTH nets (unlikely in practice, but the priority
    # must be deterministic) — diagnostic/biomarker phrasing is checked
    # first, so it always wins
    r = classify_pipeline_scope("OTHER", "Blood Test", "uncertain", "no_match")
    assert r["pipeline_scope"] == "Exclude"


def test_scope_placebo_is_placebo_or_comparator_not_a_drug():
    r = classify_pipeline_scope("OTHER", "Placebo", "placebo_or_sham", "not_applicable")
    assert r["pipeline_scope"] == "Placebo or Comparator"
    assert r["pipeline_scope"] != "Therapeutic Drug"


def test_scope_comparator_background_therapy_is_placebo_or_comparator():
    r = classify_pipeline_scope("DRUG", "Donepezil", "comparator_or_background_therapy", "no_match")
    assert r["pipeline_scope"] == "Placebo or Comparator"


def test_scope_diagnostic_imaging_agent_is_diagnostic_agent():
    r = classify_pipeline_scope("DRUG", "Florbetapir F18", "diagnostic_or_imaging_agent", "not_applicable")
    assert r["pipeline_scope"] == "Diagnostic Agent"


def test_scope_radiation_with_imaging_wording_is_diagnostic_agent():
    r = classify_pipeline_scope("RADIATION", "Amyloid PET scan", "procedure", "not_applicable")
    assert r["pipeline_scope"] == "Diagnostic Agent"


def test_scope_radiation_without_imaging_wording_is_non_drug_and_flagged():
    r = classify_pipeline_scope("RADIATION", "Radiation therapy", "procedure", "not_applicable")
    assert r["pipeline_scope"] == "Non-Drug Intervention"
    assert r["manual_review_required"] is True


# ------------------------------------------------------------
# Uncurated isotope-labeled PET/SPECT tracer leakage
# (diagnostic_agent_audit.csv fix — see drug_classification.py's
# _is_isotope_labeled_name / _has_diagnostic_study_context)
# ------------------------------------------------------------

def test_isotope_labeled_name_matches_bracketed_18f():
    assert _is_isotope_labeled_name("[F-18]Flornaptitril") is True


def test_isotope_labeled_name_matches_bracketed_11c():
    assert _is_isotope_labeled_name("[11C]MK-6884") is True


def test_isotope_labeled_name_matches_prefix_without_brackets():
    assert _is_isotope_labeled_name("11C-ER176") is True
    assert _is_isotope_labeled_name("18F-92") is True


def test_isotope_labeled_name_matches_fused_no_separator():
    # "18F" immediately fused onto a compound code with no hyphen/space/
    # bracket at all -- the real trials.csv example that first exposed
    # the gap (18F + AV45, AV45 already a known amyloid tracer token,
    # but the fused single token "18fav45" doesn't equal "av45").
    assert _is_isotope_labeled_name("18FAV45") is True


def test_isotope_labeled_name_false_positive_guard_embedded_digits():
    # SSR180711C is a real, unrelated investigational therapeutic (a
    # nicotinic alpha7 agonist) -- its "11C" is embedded mid-code
    # (immediately preceded by "0", not a word boundary), not a
    # radiochemistry isotope label. This is the exact case the audit
    # requirement calls out: don't assume diagnostic from name alone.
    assert _is_isotope_labeled_name("SSR180711C") is False


def test_isotope_labeled_name_false_positive_guard_ordinary_names():
    for name in ("Donepezil", "AR1001", "Lecanemab", "BMS-708163", "Memantine"):
        assert _is_isotope_labeled_name(name) is False, name


def test_scope_isotope_name_with_diagnostic_primary_purpose_is_diagnostic_agent():
    design = "Allocation: NA | Intervention Model: SINGLE_GROUP | Masking: NONE () | Primary Purpose: DIAGNOSTIC"
    r = classify_pipeline_scope(
        "DRUG", "18F-92", "investigational_therapeutic_unverified", "no_match",
        brief_summary="18F-92 molecular probe is a novel molecularly targeted imaging agent for amyloid-beta.",
        study_title="Amyloid-beta PET Imaging With 18F-92 in Alzheimer's Disease",
        study_design=design,
    )
    assert r["pipeline_scope"] == "Diagnostic Agent"
    assert r["diagnostic_subtype"] == "Amyloid PET tracer"


def test_scope_isotope_name_with_pet_wording_in_summary_is_diagnostic_agent_even_without_diagnostic_purpose():
    # Real dataset finding: ct.gov's own Primary Purpose is inconsistently
    # tagged (BASIC_SCIENCE/OTHER/TREATMENT/blank) for genuine tracer-
    # validation trials -- explicit PET/radioligand wording in the title
    # or summary must be an equally-sufficient signal, not a fallback.
    r = classify_pipeline_scope(
        "DRUG", "[11C]MK-6884", "investigational_therapeutic_unverified", "no_match",
        brief_summary="investigate the safety and efficacy of [11C]MK-6884 as a positron emission "
                       "tomography (PET) imaging agent for quantifying muscarinic 4 (M4) receptors",
        study_title="[11C]MK-6884 Positron Emission Tomography (PET) Tracer Validation Trial",
        study_design="Allocation: NA | Intervention Model: SINGLE_GROUP | Masking: NONE () | Primary Purpose: BASIC_SCIENCE",
    )
    assert r["pipeline_scope"] == "Diagnostic Agent"


def test_scope_isotope_name_with_biodistribution_wording_is_diagnostic_agent():
    r = classify_pipeline_scope(
        "DRUG", "[11C]MPC6827", "investigational_therapeutic_unverified", "no_match",
        brief_summary="This is a phase 0 study that will enable an assessment of biodistribution and "
                       "estimation of absorbed dose in humans based on data collected from five healthy volunteers",
        study_title="Exploratory Evaluation of [11C]MPC6827",
        study_design="Allocation: NA | Intervention Model: SINGLE_GROUP | Masking: NONE () | Primary Purpose: BASIC_SCIENCE",
    )
    assert r["pipeline_scope"] == "Diagnostic Agent"


def test_scope_isotope_name_without_any_diagnostic_context_is_not_reclassified():
    # Isotope-looking name alone, with NO corroborating study-level
    # evidence at all, must NOT be swept into Diagnostic Agent --
    # exactly the "don't assume diagnostic solely from name" rule.
    r = classify_pipeline_scope(
        "DRUG", "18F-92", "investigational_therapeutic_unverified", "no_match",
        brief_summary="", study_title="", study_design="",
    )
    assert r["pipeline_scope"] == "Therapeutic Drug"


def test_scope_ssr180711c_stays_therapeutic_despite_embedded_11c_substring():
    r = classify_pipeline_scope(
        "DRUG", "SSR180711C", "investigational_therapeutic_unverified", "no_match",
        brief_summary="assess the effect of 3 doses of SSR180711C on cognitive performance in patients "
                       "with mild Alzheimer's Disease",
        study_title="Effect on Cognitive Performance and Safety/Tolerability of SSR180711C in Mild Alzheimer's Disease",
        study_design="Allocation: RANDOMIZED | Intervention Model: PARALLEL | Masking: QUADRUPLE (PARTICIPANT, "
                      "CARE_PROVIDER, INVESTIGATOR, OUTCOMES_ASSESSOR) | Primary Purpose: TREATMENT",
    )
    assert r["pipeline_scope"] == "Therapeutic Drug"


def test_scope_sponsor_developed_therapeutic_never_overridden_by_isotope_check():
    # An official-pipeline-confirmed match is real, verified evidence --
    # never second-guessed by a name/summary heuristic, even if the name
    # happens to carry isotope notation and the trial looks diagnostic.
    r = classify_pipeline_scope(
        "DRUG", "18F-SomeConfirmedAsset", "sponsor_developed_therapeutic", "confirmed_official_match",
        brief_summary="a PET imaging agent", study_title="PET imaging study",
        study_design="Primary Purpose: DIAGNOSTIC",
    )
    assert r["pipeline_scope"] == "Therapeutic Drug"


def test_diagnostic_subtype_amyloid():
    assert determine_diagnostic_subtype("18F-92", "amyloid-beta imaging probe", "Amyloid PET") == "Amyloid PET tracer"


def test_diagnostic_subtype_tau():
    assert determine_diagnostic_subtype("F-18 PMPBB3", "a tau targeted radiopharmaceutical", "Tau PET") == "Tau PET tracer"


def test_diagnostic_subtype_tspo():
    assert determine_diagnostic_subtype("11C-PBR28", "TSPO binding, neuroinflammation", "Imaging Inflammation") == "TSPO/neuroinflammation PET tracer"


def test_diagnostic_subtype_generic_fallback():
    assert determine_diagnostic_subtype("[18F]MNI-1126", "assess synaptic density loss", "Imaging Marker") == "PET tracer"


def test_diagnostic_subtype_prioritizes_own_name_over_shared_trial_summary():
    # A trial can dose sibling tracers for different pathways in the
    # same protocol; the shared Brief Summary must not leak one
    # sibling's pathway wording onto the other's subtype.
    subtype = determine_diagnostic_subtype(
        "18F-Florbetapir",  # amyloid tracer by name
        "Evaluation of a tau radioligand alongside an amyloid tracer in the same cohort",
        "Dual tau and amyloid PET imaging study",
    )
    assert subtype == "Amyloid PET tracer"


def test_build_diagnostic_agent_audit_dataframe_empty_input():
    result = build_diagnostic_agent_audit_dataframe(pd.DataFrame())
    assert list(result.columns) == [
        "name", "nct_ids", "trial_count", "evidence_used",
        "previous_pipeline_scope", "new_pipeline_scope", "diagnostic_subtype",
        "confidence", "previously_leaked_into_therapeutic_dashboard",
    ]
    assert len(result) == 0


def _fake_interventions_row(nct_id, name, normalized_name, pipeline_scope, scope_reason, scope_method,
                             scope_confidence="high", diagnostic_subtype=""):
    return {
        "nct_id": nct_id, "original_name": name, "normalized_name": normalized_name,
        "pipeline_scope": pipeline_scope, "scope_reason": scope_reason, "scope_method": scope_method,
        "scope_confidence": scope_confidence, "diagnostic_subtype": diagnostic_subtype,
    }


def test_build_diagnostic_agent_audit_dataframe_flags_newly_caught_as_leaked():
    interventions_df = pd.DataFrame([
        _fake_interventions_row(
            "NCT00000001", "18F-92", "18f 92", "Diagnostic Agent",
            "isotope-labeled tracer name (18F/11C-style notation) combined with diagnostic study context",
            "rule_keyword", diagnostic_subtype="Amyloid PET tracer",
        ),
    ])
    result = build_diagnostic_agent_audit_dataframe(interventions_df)
    assert len(result) == 1
    row = result.iloc[0]
    assert row["previous_pipeline_scope"] == "Therapeutic Drug"
    assert row["new_pipeline_scope"] == "Diagnostic Agent"
    assert row["previously_leaked_into_therapeutic_dashboard"] == True
    assert row["diagnostic_subtype"] == "Amyloid PET tracer"


def test_build_diagnostic_agent_audit_dataframe_pre_existing_diagnostic_agent_not_flagged_as_leaked():
    interventions_df = pd.DataFrame([
        _fake_interventions_row(
            "NCT00000002", "Florbetapir F18", "florbetapir f18", "Diagnostic Agent",
            "matches a curated diagnostic/imaging tracer name", "rule_classification",
        ),
    ])
    result = build_diagnostic_agent_audit_dataframe(interventions_df)
    assert len(result) == 1
    row = result.iloc[0]
    assert row["previous_pipeline_scope"] == "Diagnostic Agent"
    assert row["previously_leaked_into_therapeutic_dashboard"] == False


def test_build_diagnostic_agent_audit_dataframe_uncertain_case_flagged_low_confidence():
    interventions_df = pd.DataFrame([
        _fake_interventions_row(
            "NCT00000003", "[11C]SomeNewCode", "11c somenewcode", "Therapeutic Drug",
            "investigational therapeutic candidate; drug/biological intervention type with no disqualifying evidence found",
            "rule_classification",
        ),
    ])
    result = build_diagnostic_agent_audit_dataframe(interventions_df)
    assert len(result) == 1
    row = result.iloc[0]
    assert row["confidence"] == "low"
    assert row["previous_pipeline_scope"] == row["new_pipeline_scope"] == "Therapeutic Drug"
    assert row["previously_leaked_into_therapeutic_dashboard"] == True
    assert "needs manual review" in row["evidence_used"]


def test_build_diagnostic_agent_audit_dataframe_non_suspect_row_excluded():
    interventions_df = pd.DataFrame([
        _fake_interventions_row(
            "NCT00000004", "Donepezil", "donepezil", "Therapeutic Drug",
            "confirmed or unsourced official sponsor-pipeline match", "rule_classification",
        ),
    ])
    result = build_diagnostic_agent_audit_dataframe(interventions_df)
    assert len(result) == 0


def test_build_diagnostic_agent_audit_dataframe_aggregates_nct_ids_across_trials():
    interventions_df = pd.DataFrame([
        _fake_interventions_row(
            "NCT00000001", "11C-PBR28", "11c pbr28", "Diagnostic Agent",
            "isotope-labeled tracer name (18F/11C-style notation) combined with diagnostic study context",
            "rule_keyword", diagnostic_subtype="TSPO/neuroinflammation PET tracer",
        ),
        _fake_interventions_row(
            "NCT00000002", "11C-PBR28", "11c pbr28", "Diagnostic Agent",
            "isotope-labeled tracer name (18F/11C-style notation) combined with diagnostic study context",
            "rule_keyword", diagnostic_subtype="TSPO/neuroinflammation PET tracer",
        ),
    ])
    result = build_diagnostic_agent_audit_dataframe(interventions_df)
    assert len(result) == 1
    assert result.iloc[0]["trial_count"] == 2
    assert result.iloc[0]["nct_ids"] == "NCT00000001; NCT00000002"


# ------------------------------------------------------------
# Non-drug intervention exclusion (source-level filtering) — devices,
# neuromodulation/electrical stimulation, digital/apps, extended
# behavioral/cognitive/educational/exercise, observational/monitoring.
# resolved_drugs_df now only ever contains "Therapeutic Drug" scope
# rows; see RESOLVED_DRUGS_DF_ELIGIBLE_SCOPES and
# _is_extended_non_drug_activity.
# ------------------------------------------------------------

def test_resolved_drugs_df_eligible_scopes_is_therapeutic_drug_only():
    assert RESOLVED_DRUGS_DF_ELIGIBLE_SCOPES == ["Therapeutic Drug"]


def test_extended_net_catches_neuromodulation_device():
    for name in ("transcranial magnetic stimulation", "tDCS", "Deep brain stimulation",
                 "Active transcutaneous vagus nerve stimulation of the auricle"):
        assert _is_extended_non_drug_activity(normalize_text(name)), name


def test_extended_net_catches_digital_app_intervention():
    for name in ("Mobile app", "Cultural pathways App", "A digital self-help self-compassion app intervention"):
        assert _is_extended_non_drug_activity(normalize_text(name)), name


def test_extended_net_catches_exercise_variants():
    for name in ("Aerobic exercises", "Adapted Physical Activity", "Physical training"):
        assert _is_extended_non_drug_activity(normalize_text(name)), name


def test_extended_net_catches_educational_intervention():
    for name in ("Patient Education", "Dementia Education", "Educational Materials"):
        assert _is_extended_non_drug_activity(normalize_text(name)), name


def test_extended_net_catches_observational_monitoring():
    for name in ("Questionnaire and Physical Exam", "Actigraphy and video recording signal",
                 "Remote activity monitoring system"):
        assert _is_extended_non_drug_activity(normalize_text(name)), name


def test_extended_net_does_not_match_ordinary_drug_names():
    for name in ("Donepezil", "AR1001", "Lecanemab", "BMS-708163"):
        assert not _is_extended_non_drug_activity(normalize_text(name)), name


def test_extended_net_does_not_use_bare_therapy_token():
    # "therapy" alone must never be a trigger -- it would wrongly
    # exclude real modalities like "Stem Cell Therapy"/"Gene Therapy".
    assert not _is_extended_non_drug_activity(normalize_text("Stem Cell Therapy - Experimental"))
    assert not _is_extended_non_drug_activity(normalize_text("Gene Therapy"))


def test_classify_intervention_vagus_nerve_stimulation_is_behavioral_non_drug():
    r = classify_intervention(
        "OTHER",
        "Active transcutaneous vagus nerve stimulation respiratory-gated non-painful electrical "
        "stimulation of the auricle for 10 minute sessions",
        "Massachusetts General Hospital", [], [],
    )
    assert r["classification"] == "behavioral"
    assert r["reason"] == EXTENDED_NON_DRUG_REASON


def test_classify_intervention_cognitive_stimulation_therapy_is_behavioral_non_drug():
    r = classify_intervention("OTHER", "Cognitive Stimulation Therapy", "Some Sponsor", [], [])
    assert r["classification"] == "behavioral"


def test_scope_extended_net_match_becomes_non_drug_intervention():
    r = classify_pipeline_scope("OTHER", "transcranial magnetic stimulation", "behavioral", "not_applicable")
    assert r["pipeline_scope"] == "Non-Drug Intervention"


# --- false-positive protection: known-compound override guard -------

def test_has_known_therapeutic_evidence_true_for_curated_compound_substring():
    assert _has_known_therapeutic_evidence(normalize_text("etanercept and repeated contrast ultrasound"))


def test_has_known_therapeutic_evidence_false_for_unrelated_text():
    assert not _has_known_therapeutic_evidence(normalize_text("transcranial magnetic stimulation"))


def test_classify_intervention_real_drug_combined_with_procedure_word_stays_therapeutic():
    # "etanercept" (a real, KNOWN_COMPOUND_NAMES-listed biologic)
    # combined in one string with a trailing procedure word
    # ("ultrasound") must NOT be excluded by the extended non-drug net.
    r = classify_intervention("DRUG", "etanercept and repeated contrast ultrasound", "Some Sponsor", [], [])
    assert r["classification"] == "investigational_therapeutic_unverified"


def test_classify_intervention_aln_app_is_not_misread_as_an_application():
    # ALN-APP (Alnylam Pharmaceuticals) is a real RNAi therapeutic
    # targeting Amyloid Precursor Protein -- "APP" here is the gene/
    # protein target, not "application". Confirmed false positive found
    # during the real-data audit for this fix; protected via the
    # curated KNOWN_COMPOUND_NAMES list (data/reference precedent).
    r = classify_intervention("DRUG", "ALN-APP", "Alnylam Pharmaceuticals", [], [])
    assert r["classification"] == "investigational_therapeutic_unverified"
    assert r["reason"] != EXTENDED_NON_DRUG_REASON


# --- resolved_drugs_df exclusion audit (outputs/non_drug_exclusion_audit.csv) ---

def test_exclusion_audit_includes_scope_level_exclusion():
    interventions_df = pd.DataFrame([
        _fake_interventions_row(
            "NCT00000001", "Florbetapir F18", "florbetapir f18", "Diagnostic Agent",
            "matches a curated diagnostic/imaging tracer name", "rule_classification",
        ),
    ])
    interventions_df["classification"] = "investigational_therapeutic_unverified"
    interventions_df["reason"] = "no confirmed pipeline match"
    interventions_df["original_type"] = "DRUG"
    result = build_resolved_drugs_exclusion_audit_dataframe(interventions_df)
    assert len(result) == 1
    assert result.iloc[0]["pipeline_scope"] == "Diagnostic Agent"


def test_exclusion_audit_includes_extended_net_classification_level_exclusion():
    # This population is classified "behavioral" from the start (never
    # investigational_therapeutic_unverified), so it would NOT be
    # caught by a scope-only filter -- confirms the audit's second
    # (classification-level) population actually fires.
    interventions_df = pd.DataFrame([
        _fake_interventions_row(
            "NCT00000002", "transcranial magnetic stimulation", "transcranial magnetic stimulation",
            "Non-Drug Intervention", "behavioral/non-drug activity", "rule_type",
        ),
    ])
    interventions_df["classification"] = "behavioral"
    interventions_df["reason"] = EXTENDED_NON_DRUG_REASON
    interventions_df["original_type"] = "OTHER"
    result = build_resolved_drugs_exclusion_audit_dataframe(interventions_df)
    assert len(result) == 1
    assert result.iloc[0]["exclusion_reason"] == EXTENDED_NON_DRUG_REASON


def test_exclusion_audit_excludes_unambiguous_placebo_arms():
    interventions_df = pd.DataFrame([
        _fake_interventions_row(
            "NCT00000003", "Placebo", "placebo", "Placebo or Comparator",
            "placebo/sham/vehicle-control arm", "rule_classification",
        ),
    ])
    interventions_df["classification"] = "placebo_or_sham"
    interventions_df["reason"] = "name indicates a placebo/sham/vehicle-control arm"
    result = build_resolved_drugs_exclusion_audit_dataframe(interventions_df)
    assert len(result) == 0


def test_exclusion_audit_empty_input():
    result = build_resolved_drugs_exclusion_audit_dataframe(pd.DataFrame())
    assert list(result.columns) == [
        "name", "clinicaltrials_intervention_type", "nct_ids", "trial_count",
        "classification", "pipeline_scope", "exclusion_reason", "confidence",
    ]
    assert len(result) == 0


def test_rollup_excludes_extended_net_matches_from_resolved_drugs_df():
    trials_df = pd.DataFrame([
        make_trial_row("NCT206", "Some Sponsor", "transcranial magnetic stimulation",
                        "investigational_therapeutic_unverified", "no_match", "medium", True,
                        pipeline_scope="Non-Drug Intervention"),
    ])
    result = build_resolved_drugs_dataframe(trials_df)
    assert len(result) == 0


# ------------------------------------------------------------
# Non-therapeutic DRUG PURPOSE — a real drug/biologic used only for a
# pharmacological challenge/probe, diagnostic-tool/contrast-imaging
# development, or deprescribing/procedural-support role within a
# specific trial, not as an investigational AD treatment. Every
# "should exclude" case below is a real ClinicalTrials.gov example
# found during this fix's data audit; every "should keep" case is a
# real genuine investigational-drug Phase 1 study that must NOT be
# swept in just because it happens to also carry Primary Purpose:
# DIAGNOSTIC (ct.gov's own field is not reliable enough alone — see
# _is_diagnostic_challenge_or_probe_purpose's module docstring).
# ------------------------------------------------------------

def test_isotope_labeled_name_matches_123i_notation():
    for name in ("[123I] AV 39", "[123I]AV94", "123-I MNI-340", "123-I MNI-187", "[123I]CLINDE"):
        assert _is_isotope_labeled_name(name), name


def test_diagnostic_challenge_probe_pramlintide_challenge_test():
    r = _is_diagnostic_challenge_or_probe_purpose(
        "Pramlintide challenge test",
        "a simple blood-based test for early detection of Alzheimer's disease...single injection of Pramlintide",
        "Multi-Center Development of a Novel Diagnostic Test for Alzheimer's Disease",
        "Primary Purpose: DIAGNOSTIC",
    )
    assert r is True


def test_diagnostic_challenge_probe_scopolamine_eeg_diagnostic_tool():
    r = _is_diagnostic_challenge_or_probe_purpose(
        "Scopolamine",
        "compare EEG responses...to scopolamine...to develop a diagnostic tool for AD",
        "The Use of EEG in Alzheimer's Disease, With and Without Scopolamine - A Pilot Study",
        "Primary Purpose: DIAGNOSTIC",
    )
    assert r is True


def test_diagnostic_challenge_probe_pet_imaging_with_challenge_in_title():
    r = _is_diagnostic_challenge_or_probe_purpose(
        "LPS",
        "examine the differences in the capacity to activate microglia",
        "Peripheral Benzodiazepine Receptors (PBR28) Brain PET Imaging With Lipopolysaccharide "
        "Challenge for the Study of Microglia Function in Alzheimer's Disease",
        "Primary Purpose: DIAGNOSTIC",
    )
    assert r is True


def test_diagnostic_challenge_probe_contrast_imaging_agent():
    r = _is_diagnostic_challenge_or_probe_purpose(
        "DSPE-DOTA-Gd Liposomal Injection",
        "establish safety of ADx-001 in healthy volunteers...developed for use in contrast-enabled "
        "MR imaging of amyloid plaques",
        "Proof-of-concept Study of New Imaging Diagnostic in Patients With Suspected Alzheimer's Disease",
        "Primary Purpose: DIAGNOSTIC",
    )
    assert r is True


def test_diagnostic_challenge_probe_requires_diagnostic_purpose_not_just_a_token_match():
    # "challenge"/"probe" bare tokens are only consulted AFTER Primary
    # Purpose: DIAGNOSTIC already gates the check -- without that gate,
    # unrelated context mentioning "challenge" must NOT trigger exclusion.
    r = _is_diagnostic_challenge_or_probe_purpose(
        "SomeRealDrug", "this trial presents a real challenge for patients", "A Study of SomeRealDrug",
        "Primary Purpose: TREATMENT",
    )
    assert r is False


def test_diagnostic_challenge_probe_name_alone_is_sufficient_without_context():
    # "challenge test" in the intervention's OWN name is unambiguous on
    # its own -- no Primary Purpose/context corroboration required.
    r = _is_diagnostic_challenge_or_probe_purpose("Pramlintide challenge test", "", "", "")
    assert r is True


def test_diagnostic_challenge_probe_protects_real_investigational_drug_phase1_studies():
    # Real Phase 1 safety/PK/immunogenicity studies of genuine
    # investigational AD candidates must NOT be excluded, even though
    # some of them are (apparently inconsistently) tagged Primary
    # Purpose: DIAGNOSTIC in ct.gov's own data.
    cases = [
        ("LY450139 dihydrate",
         "safety of LY450139 dihydrate...how much should be given...effect on a protein found in blood, called A beta",
         "Effects of LY450139 Dihydrate on Subjects With Mild to Moderate Alzheimer's Disease"),
        ("TC-5619",
         "Phase 1 study to examine the safety, tolerability and pharmacokinetics of TC-5619",
         "Multiple Ascending Dose Study of TC-5619 in Healthy Elderly Subjects and Subjects With Alzheimer's Disease"),
        ("AMDX-2011P",
         "assess safety, tolerability, plasma pharmacokinetics and biologic activity of a single intravenous dose",
         "A Study of AMDX-2011P in Participants With Alzheimer's Disease"),
        ("V950",
         "test the safety, tolerability and the immune response to an investigational vaccine",
         "A Study of V950 in People With Alzheimer Disease"),
    ]
    for name, summary, title in cases:
        r = _is_diagnostic_challenge_or_probe_purpose(name, summary, title, "Primary Purpose: DIAGNOSTIC")
        assert r is False, name


def test_diagnostic_challenge_probe_protects_real_drug_administered_as_treatment_within_diagnostic_study():
    # Reminyl retard (galantamine, a real FDA-approved AD drug)
    # administered AS TREATMENT while an MRI technique monitors
    # response -- the "diagnostic" framing describes the imaging
    # methodology, not the drug's own role. Must not be excluded.
    r = _is_diagnostic_challenge_or_probe_purpose(
        "Reminyl retard",
        "Examination of the correlation between the cerebral bloodflow and the clinical change under "
        "treatment with Reminyl retard",
        "Continuous Arterial Spin Labeling (CASL) MRI for Monitoring and Prediction of Drug Therapy in "
        "Alzheimers Disease",
        "Primary Purpose: DIAGNOSTIC",
    )
    assert r is False


def test_deprescribing_name_token_detected():
    assert _is_deprescribing_or_procedural_support(normalize_text("Deprescribing of target anticholinergics"))


def test_deprescribing_phrases_detected():
    for phrase in ("medication withdrawal", "drug discontinuation", "dose tapering"):
        assert _is_deprescribing_or_procedural_support(normalize_text(phrase))


def test_procedural_support_sedation_phrases_detected():
    for phrase in ("procedural sedation", "conscious sedation"):
        assert _is_deprescribing_or_procedural_support(normalize_text(phrase))


def test_deprescribing_does_not_match_ordinary_drug_names():
    for name in ("Donepezil", "AR1001", "Lecanemab"):
        assert not _is_deprescribing_or_procedural_support(normalize_text(name))


def test_scope_pramlintide_challenge_test_becomes_diagnostic_agent():
    c = classify_intervention("DRUG", "Pramlintide challenge test", "Some Sponsor", [], [])
    s = classify_pipeline_scope(
        "DRUG", "Pramlintide challenge test", c["classification"], c["verification_status"],
        brief_summary="a simple blood-based test for early detection...single injection of Pramlintide",
        study_title="Multi-Center Development of a Novel Diagnostic Test for Alzheimer's Disease",
        study_design="Primary Purpose: DIAGNOSTIC",
    )
    assert s["pipeline_scope"] == "Diagnostic Agent"


def test_scope_deprescribing_becomes_non_drug_intervention():
    c = classify_intervention("OTHER", "Deprescribing of target anticholinergics", "Some Sponsor", [], [])
    s = classify_pipeline_scope(
        "OTHER", "Deprescribing of target anticholinergics", c["classification"], c["verification_status"],
        brief_summary="evaluate the impact of a deprescribing intervention on cognitive and safety outcomes",
        study_title="Reducing Risk of Dementia Through Deprescribing",
        study_design="Primary Purpose: PREVENTION",
    )
    assert s["pipeline_scope"] == "Non-Drug Intervention"


def test_scope_sponsor_developed_therapeutic_never_overridden_by_diagnostic_purpose_check():
    r = classify_pipeline_scope(
        "DRUG", "SomeConfirmedAsset challenge test", "sponsor_developed_therapeutic", "confirmed_official_match",
        brief_summary="a diagnostic challenge test", study_title="Diagnostic study",
        study_design="Primary Purpose: DIAGNOSTIC",
    )
    assert r["pipeline_scope"] == "Therapeutic Drug"


def test_flyer_excluded_via_extended_net():
    assert _is_extended_non_drug_activity(normalize_text("Flyer"))


def test_classify_intervention_flyer_is_behavioral_non_drug():
    r = classify_intervention("OTHER", "Flyer", "Some Sponsor", [], [])
    assert r["classification"] == "behavioral"


def test_blood_withdrawal_is_a_procedure():
    from drug_classification import _is_procedure
    assert _is_procedure("OTHER", normalize_text("blood withdrawal"))


def test_scope_generic_diagnostic_test_type_still_gets_diagnostic_agent():
    # a DIAGNOSTIC_TEST-type name that ISN'T generic (e.g. a specific
    # named test) should be "Diagnostic Agent", not silently excluded
    r = classify_pipeline_scope("DIAGNOSTIC_TEST", "Retinal fundus photography", "investigational_therapeutic_unverified", "no_match")
    assert r["pipeline_scope"] == "Diagnostic Agent"


def test_scope_combination_product_with_known_compound_stays_therapeutic():
    # a genuine therapeutic combination retains its therapeutic component
    r = classify_pipeline_scope("COMBINATION_PRODUCT", "AR1001 plus Donepezil Combination", "investigational_therapeutic_unverified", "no_match")
    assert r["pipeline_scope"] == "Therapeutic Drug"
    assert r["manual_review_required"] is True  # combined name still needs a human check


def test_scope_combination_product_with_no_recognized_compound_needs_review():
    r = classify_pipeline_scope("COMBINATION_PRODUCT", "Herbal Formula X Plus Y", "investigational_therapeutic_unverified", "no_match")
    assert r["pipeline_scope"] == "Needs Review"
    assert r["manual_review_required"] is True


def test_scope_genetic_testing_is_not_classified_as_gene_therapy():
    r = classify_pipeline_scope("GENETIC", "APOE Genotyping", "investigational_therapeutic_unverified", "no_match")
    assert r["pipeline_scope"] != "Therapeutic Drug"
    assert r["pipeline_scope"] == "Exclude"
    r2 = classify_pipeline_scope("GENETIC", "Genetic Counseling and Testing", "uncertain", "no_match")
    assert r2["pipeline_scope"] == "Exclude"


def test_scope_genetic_gene_therapy_product_detected():
    r = classify_pipeline_scope("GENETIC", "AAV2-NGF gene therapy vector", "investigational_therapeutic_unverified", "no_match")
    assert r["pipeline_scope"] == "Therapeutic Drug"
    assert r["manual_review_required"] is True


def test_scope_genetic_ambiguous_type_needs_review():
    r = classify_pipeline_scope("GENETIC", "Novel Compound XG-14", "investigational_therapeutic_unverified", "no_match")
    assert r["pipeline_scope"] == "Needs Review"


def test_scope_drug_type_candidate_with_no_disqualifying_evidence_stays_therapeutic():
    r = classify_pipeline_scope("DRUG", "AR1001", "sponsor_developed_therapeutic", "confirmed_official_match")
    assert r["pipeline_scope"] == "Therapeutic Drug"
    assert r["scope_confidence"] == "high"

    r2 = classify_pipeline_scope("DRUG", "SAR110894", "investigational_therapeutic_unverified", "no_match")
    assert r2["pipeline_scope"] == "Therapeutic Drug"


def test_scope_uncertain_drug_like_name_with_no_type_evidence_is_needs_review():
    r = classify_pipeline_scope("DRUG", "Unresolved Candidate X", "uncertain", "no_match")
    assert r["pipeline_scope"] == "Needs Review"
    assert r["manual_review_required"] is True


def test_scope_curated_override_can_promote_a_record():
    overrides = {
        "curcumin c3 complex": {
            "pipeline_scope": "Therapeutic Drug", "canonical_name_override": "Curcumin C3",
            "reason": "curated: genuine investigational program per sponsor filing",
            "source": "test", "reviewer": "test", "verified_date": "2026-01-01",
        },
    }
    r = classify_pipeline_scope("DIETARY_SUPPLEMENT", "Curcumin C3 Complex", "investigational_therapeutic_unverified", "no_match", overrides=overrides)
    assert r["pipeline_scope"] == "Therapeutic Drug"
    assert r["scope_method"] == "curated_override"
    assert r["canonical_name_override"] == "Curcumin C3"


def test_scope_curated_override_can_correct_a_record_to_excluded():
    overrides = {
        "ar1001": {
            "pipeline_scope": "Exclude", "canonical_name_override": "",
            "reason": "curated: turned out to be a mislabeled diagnostic kit name in this dataset",
            "source": "test", "reviewer": "test", "verified_date": "2026-01-01",
        },
    }
    r = classify_pipeline_scope("DRUG", "AR1001", "sponsor_developed_therapeutic", "confirmed_official_match", overrides=overrides)
    assert r["pipeline_scope"] == "Exclude"
    assert r["scope_method"] == "curated_override"


def test_scope_all_results_use_a_documented_label():
    # every branch classify_pipeline_scope() can take must return a value
    # from PIPELINE_SCOPE_LABELS — never an ad hoc string
    samples = [
        ("DRUG", "AR1001", "sponsor_developed_therapeutic", "confirmed_official_match"),
        ("OTHER", "Placebo", "placebo_or_sham", "not_applicable"),
        ("DEVICE", "TMS", "device", "not_applicable"),
        ("BEHAVIORAL", "Exercise", "behavioral", "not_applicable"),
        ("DIETARY_SUPPLEMENT", "Fish Oil", "investigational_therapeutic_unverified", "no_match"),
        ("DIAGNOSTIC_TEST", "Blood Test", "uncertain", "no_match"),
        ("COMBINATION_PRODUCT", "Mystery Combo", "investigational_therapeutic_unverified", "no_match"),
        ("GENETIC", "Gene Panel", "investigational_therapeutic_unverified", "no_match"),
        ("PROCEDURE", "MRI", "procedure", "not_applicable"),
        ("RADIATION", "Radiation therapy", "procedure", "not_applicable"),
    ]
    for itype, name, classification, vstatus in samples:
        r = classify_pipeline_scope(itype, name, classification, vstatus)
        assert r["pipeline_scope"] in PIPELINE_SCOPE_LABELS, f"unrecognized scope {r['pipeline_scope']!r} for {name!r}"


def test_load_scope_overrides_missing_file_returns_empty_dict():
    assert load_scope_overrides("data/reference/does_not_exist.csv") == {}


def test_load_scope_overrides_parses_real_columns(tmp_path=None):
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(
            "normalized_intervention_name,pipeline_scope,canonical_name_override,reason,source,reviewer,verified_date\n"
            "curcumin c3 complex,Therapeutic Drug,Curcumin C3,test reason,test source,Test Reviewer,2026-01-01\n"
        )
        path = f.name
    overrides = load_scope_overrides(path)
    os.remove(path)
    assert "curcumin c3 complex" in overrides
    entry = overrides["curcumin c3 complex"]
    assert entry["pipeline_scope"] == "Therapeutic Drug"
    assert entry["canonical_name_override"] == "Curcumin C3"
    assert entry["reviewer"] == "Test Reviewer"


def test_load_scope_overrides_missing_required_column_raises():
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("normalized_intervention_name,pipeline_scope\ncurcumin,Therapeutic Drug\n")
        path = f.name
    try:
        raised = False
        try:
            load_scope_overrides(path)
        except ValueError:
            raised = True
        assert raised
    finally:
        os.remove(path)


def test_the_real_intervention_scope_overrides_csv_loads_without_error():
    # smoke test against the actual project file (should be a valid,
    # possibly-empty curated overrides table)
    path = os.path.join(os.path.dirname(__file__), "data", "reference", "intervention_scope_overrides.csv")
    overrides = load_scope_overrides(path)
    assert isinstance(overrides, dict)


# --- build_interventions_dataframe() carries pipeline_scope end-to-end ---

def test_build_interventions_dataframe_attaches_pipeline_scope_columns():
    trials_df = pd.DataFrame([
        {"nct_id": "NCT100", "sponsor": "AriBio", "title": "AR1001 Phase 3",
         "interventions": "DRUG: AR1001|OTHER: Placebo|DIETARY_SUPPLEMENT: Fish Oil"},
    ])
    interventions_df = build_interventions_dataframe(trials_df, PIPELINE_RECORDS)
    for col in ["pipeline_scope", "scope_reason", "scope_method", "scope_confidence", "manual_review_required"]:
        assert col in interventions_df.columns
    fish_oil_row = interventions_df[interventions_df["original_name"] == "Fish Oil"].iloc[0]
    assert fish_oil_row["pipeline_scope"] != "Therapeutic Drug"


def test_build_interventions_dataframe_applies_scope_overrides():
    trials_df = pd.DataFrame([
        {"nct_id": "NCT101", "sponsor": "Some Sponsor", "title": "Supplement study",
         "interventions": "DIETARY_SUPPLEMENT: Curcumin C3 Complex"},
    ])
    overrides = {
        "curcumin c3 complex": {
            "pipeline_scope": "Therapeutic Drug", "canonical_name_override": "",
            "reason": "curated promotion", "source": "test", "reviewer": "test", "verified_date": "2026-01-01",
        },
    }
    interventions_df = build_interventions_dataframe(trials_df, PIPELINE_RECORDS, overrides)
    assert interventions_df.iloc[0]["pipeline_scope"] == "Therapeutic Drug"
    assert interventions_df.iloc[0]["scope_method"] == "curated_override"


# --- build_resolved_drugs_dataframe() gates on pipeline_scope ---

def test_rollup_excludes_records_whose_scope_is_excluded():
    trials_df = pd.DataFrame([
        make_trial_row("NCT200", "Some Sponsor", "Blood Test", "investigational_therapeutic_unverified",
                        "no_match", "medium", True, pipeline_scope="Exclude",
                        scope_reason="generic diagnostic description"),
    ])
    result = build_resolved_drugs_dataframe(trials_df)
    assert len(result) == 0


def test_rollup_excludes_records_whose_scope_is_placebo_or_comparator():
    trials_df = pd.DataFrame([
        make_trial_row("NCT201", "Some Sponsor", "Donepezil", "sponsor_developed_therapeutic",
                        "confirmed_official_match", "high", False, pipeline_scope="Placebo or Comparator"),
    ])
    result = build_resolved_drugs_dataframe(trials_df)
    assert len(result) == 0


def test_rollup_excludes_non_therapeutic_scopes_entirely():
    # A record only enters resolved_drugs_df if its primary
    # investigational intervention resolved to "Therapeutic Drug"
    # scope — Diagnostic Agent / Non-Drug Intervention / Supportive
    # Treatment / Needs Review are excluded at the source now, not just
    # hidden from the default view. The dashboard's "reveal non-
    # therapeutic records" toggle still exists in the UI but has
    # nothing left to reveal — an intentional consequence, not a bug.
    for scope in ("Diagnostic Agent", "Non-Drug Intervention", "Supportive Treatment", "Needs Review"):
        trials_df = pd.DataFrame([
            make_trial_row("NCT202", "Some Sponsor", "Retinal Fundus Photography", "investigational_therapeutic_unverified",
                            "no_match", "medium", True, pipeline_scope=scope),
        ])
        result = build_resolved_drugs_dataframe(trials_df)
        assert len(result) == 0, f"expected scope {scope!r} to be excluded, got {len(result)} row(s)"


def test_rollup_default_fixture_rows_stay_therapeutic_drug_scope():
    # every pre-Phase-1A rollup test uses the make_trial_row() default
    # (pipeline_scope="Therapeutic Drug") — pin that the resulting row
    # actually carries it, so a future default change doesn't silently
    # break those tests' assumptions unnoticed
    trials_df = pd.DataFrame([
        make_trial_row("NCT203", "AriBio", "AR1001", "sponsor_developed_therapeutic",
                        "pipeline_record_match_without_source", "medium", True, phase="Phase 3"),
    ])
    result = build_resolved_drugs_dataframe(trials_df)
    assert result.iloc[0]["pipeline_scope"] == "Therapeutic Drug"


def test_rollup_non_therapeutic_sibling_trial_does_not_block_the_eligible_one():
    # Two trials share a canonical drug name; one resolves Therapeutic
    # Drug scope, the other Diagnostic Agent. The eligibility filter
    # runs BEFORE grouping, so the non-therapeutic trial's row never
    # reaches the group at all — the result reflects ONLY the eligible
    # trial, not a "scope disagreement" (that mechanism is now
    # unreachable in practice, since every row surviving the filter is
    # already "Therapeutic Drug" by construction).
    trials_df = pd.DataFrame([
        make_trial_row("NCT204", "Sponsor A", "AmbiguousDrug", "investigational_therapeutic_unverified",
                        "no_match", "medium", True, pipeline_scope="Therapeutic Drug"),
        make_trial_row("NCT205", "Sponsor A", "AmbiguousDrug", "investigational_therapeutic_unverified",
                        "no_match", "medium", True, pipeline_scope="Diagnostic Agent"),
    ])
    result = build_resolved_drugs_dataframe(trials_df)
    assert len(result) == 1
    assert result.iloc[0]["pipeline_scope"] == "Therapeutic Drug"
    assert result.iloc[0]["nct_id"] == "NCT204"


def test_resolve_developed_drug_carries_scope_from_winning_intervention():
    trials_df = pd.DataFrame([
        {"nct_id": "NCT300", "sponsor": "Some Sponsor", "title": "Supplement study",
         "interventions": "DIETARY_SUPPLEMENT: Curcumin C3 Complex"},
    ])
    interventions_df = build_interventions_dataframe(trials_df, PIPELINE_RECORDS)
    resolved = resolve_developed_drug(interventions_df.to_dict("records"))
    assert resolved["developed_drug"] == "Curcumin C3 Complex"
    assert resolved["pipeline_scope"] == "Non-Drug Intervention"


# --- build_scope_audit_dataframe() (outputs/classification_gap_audit.csv source) ---

def test_build_scope_audit_dataframe_basic_structure():
    interventions_df = pd.DataFrame([
        {"nct_id": "NCT400", "original_name": "Curcumin C3 Complex", "original_type": "DIETARY_SUPPLEMENT",
         "normalized_name": "curcumin c3 complex", "classification": "investigational_therapeutic_unverified",
         "pipeline_scope": "Non-Drug Intervention", "scope_reason": "dietary supplement",
         "scope_method": "rule_type", "scope_confidence": "medium", "manual_review_required": False},
        {"nct_id": "NCT401", "original_name": "Curcumin C3 Complex", "original_type": "DIETARY_SUPPLEMENT",
         "normalized_name": "curcumin c3 complex", "classification": "investigational_therapeutic_unverified",
         "pipeline_scope": "Non-Drug Intervention", "scope_reason": "dietary supplement",
         "scope_method": "rule_type", "scope_confidence": "medium", "manual_review_required": False},
        {"nct_id": "NCT402", "original_name": "AR1001", "original_type": "DRUG",
         "normalized_name": "ar1001", "classification": "sponsor_developed_therapeutic",
         "pipeline_scope": "Therapeutic Drug", "scope_reason": "confirmed match",
         "scope_method": "rule_classification", "scope_confidence": "high", "manual_review_required": False},
    ])
    audit_df = build_scope_audit_dataframe(interventions_df)
    # two distinct (normalized_name, type) groups: Curcumin C3 Complex (2 NCTs merged into one row), AR1001
    assert len(audit_df) == 2
    curcumin_row = audit_df[audit_df["raw_intervention_name"] == "Curcumin C3 Complex"].iloc[0]
    assert curcumin_row["nct_ids"] == "NCT400; NCT401"
    assert curcumin_row["dashboard_eligible"] == False
    assert curcumin_row["new_pipeline_scope"] == "Non-Drug Intervention"
    ar1001_row = audit_df[audit_df["raw_intervention_name"] == "AR1001"].iloc[0]
    assert ar1001_row["dashboard_eligible"] == True


def test_build_scope_audit_dataframe_empty_input():
    audit_df = build_scope_audit_dataframe(pd.DataFrame())
    assert len(audit_df) == 0
    assert "dashboard_eligible" in audit_df.columns


def test_build_scope_audit_dataframe_dashboard_eligible_matches_therapeutic_scope():
    interventions_df = pd.DataFrame([
        {"nct_id": "NCT500", "original_name": "Blood Test", "original_type": "DIAGNOSTIC_TEST",
         "normalized_name": "blood test", "classification": "uncertain",
         "pipeline_scope": "Exclude", "scope_reason": "generic test", "scope_method": "rule_keyword",
         "scope_confidence": "high", "manual_review_required": False},
    ])
    audit_df = build_scope_audit_dataframe(interventions_df)
    assert audit_df.iloc[0]["dashboard_eligible"] == False
    assert audit_df.iloc[0]["previous_drug_type"] == "uncertain"


# ------------------------------------------------------------
# test runner
# ------------------------------------------------------------

ALL_TESTS = [
    test_normalize_lowercases,
    test_normalize_strips_punctuation,
    test_normalize_strips_registered_trademark_symbol,
    test_normalize_collapses_extra_whitespace,
    test_normalize_handles_none,
    test_normalize_handles_nan,
    test_normalize_handles_empty_string,
    test_parse_ar1001_plus_placebo,
    test_parse_wujia_yizhi_granules_plus_placebo,
    test_parse_sar110894_donepezil_placebo_keeps_all_three,
    test_parse_florbetapir_alone,
    test_parse_ct_scan_procedure,
    test_parse_nintendo_wii_behavioral,
    test_parse_entry_without_type_prefix_is_kept,
    test_parse_handles_none,
    test_parse_handles_nan,
    test_parse_handles_empty_string,
    test_normalize_candidate_strips_dose_per_day,
    test_normalize_candidate_strips_plain_mg_dose,
    test_normalize_candidate_strips_dose_and_route_and_frequency,
    test_normalize_candidate_preserves_avp786_hyphenated_code,
    test_normalize_candidate_handles_none_and_empty,
    test_load_official_pipeline_reads_seed_file,
    test_load_official_pipeline_does_not_modify_the_file,
    test_load_official_pipeline_missing_file_returns_empty_list,
    test_load_official_pipeline_missing_column_raises,
    test_match_ar1001_aribio,
    test_match_ar1001_unrelated_university_sponsor_does_not_match,
    test_match_ban2401_eisai_synonym,
    test_match_leqembi_registered_trademark_eisai,
    test_match_lecanemab_biogen_does_not_match_without_a_biogen_record,
    test_match_ambiguous_when_two_company_records_match_same_drug_name,
    test_match_short_name_does_not_substring_match_longer_official_name,
    test_match_longer_name_does_not_substring_match_shorter_official_name,
    test_match_confirmed_when_source_url_present,
    test_classify_ar1001_aribio_is_sponsor_developed_medium_confidence,
    test_classify_ar1001_unrelated_sponsor_is_not_sponsor_developed,
    test_classify_florbetapir_f18_typed_drug_is_diagnostic,
    test_classify_all_curated_tracers_are_diagnostic,
    test_classify_expanded_amyloid_tracer_aliases_are_diagnostic,
    test_classify_flortaucipir_and_av1451_are_diagnostic_even_typed_drug,
    test_classify_expanded_tau_tracer_aliases_are_diagnostic,
    test_classify_metabolic_tracer_aliases_are_diagnostic,
    test_classify_unknown_name_containing_f18_is_not_excluded_as_diagnostic,
    test_classify_apn1607_aliases_are_diagnostic,
    test_classify_bare_amyloid_and_tau_pet_are_diagnostic,
    test_classify_f_ara_g_diagnostic_only_with_explicit_pet_tracer_wording,
    test_classify_tau_pet_typed_other_is_diagnostic,
    test_classify_novel_f18_compound_without_imaging_wording_not_automatically_diagnostic,
    test_classify_pet_scan_procedure_typed_procedure_stays_procedure,
    test_classify_amyloid_and_brain_pet_scan_remain_procedure_not_diagnostic,
    test_classify_amyloid_and_tau_pet_imaging_are_diagnostic,
    test_classify_pet_ligand_is_diagnostic,
    test_classify_pm_pbb3_still_diagnostic,
    test_parse_duplicated_drug_prefix_is_stripped,
    test_parse_duplicated_prefix_real_f_ara_g_row,
    test_classify_f_ara_g_with_duplicated_prefix_end_to_end,
    test_classify_unknown_name_containing_c11_is_not_excluded_as_diagnostic,
    test_classify_pet_scan_is_procedure,
    test_classify_ct_scan_typed_radiation_is_procedure,
    test_classify_spect_scan_is_procedure_not_via_the_ct_scan_false_match,
    test_classify_pet_ct_mri_spect_scan_are_all_procedures,
    test_classify_placebo_for_sar110894_trial_is_placebo,
    test_classify_matched_placebo_sham_and_vehicle_control_are_placebo,
    test_classify_expanded_placebo_phrase_list,
    test_classify_bromocriptine_mesilate_plus_placebos_plural,
    test_classify_no_intervention_alone_is_other_not_investigational,
    test_classify_untreated_is_other,
    test_classify_bare_usual_care_and_standard_of_care_are_other,
    test_classify_drug_plus_no_intervention_no_intervention_not_a_sibling_candidate,
    test_classify_drug_plus_placebo_plus_no_intervention,
    test_classify_bare_vehicle_alone_is_not_automatically_placebo,
    test_classify_nintendo_wii_exercise_is_behavioral,
    test_classify_sar110894_donepezil_placebo_trial,
    test_classify_donepezil_alone_with_placebo_is_investigational_unverified,
    test_classify_donepezil_vs_memantine_head_to_head_is_uncertain,
    test_classify_wujia_yizhi_granules_with_placebo,
    test_classify_sole_candidate_rule_does_not_fire_with_two_unresolved_candidates,
    test_classify_ar1001_aricept_placebo_aricept_is_comparator,
    test_classify_namenda_alone_with_placebo_is_investigational_unverified,
    test_classify_exelon_vs_razadyne_head_to_head_is_uncertain,
    test_classify_fdg_as_whole_word_is_diagnostic_but_fdg_inside_longer_word_is_not,
    test_classify_ambiguous_pipeline_match_forces_manual_review,
    test_resolve_one_confirmed_official_match,
    test_resolve_one_pipeline_match_without_source,
    test_resolve_one_unverified_therapeutic,
    test_resolve_multiple_confirmed_candidates,
    test_resolve_multiple_unverified_candidates,
    test_resolve_placebo_only_trial,
    test_resolve_diagnostic_only_trial,
    test_resolve_therapeutic_plus_placebo_plus_diagnostic_tracer,
    test_resolve_approved_background_alongside_investigational_candidate,
    test_resolve_trx0237_dose_variants_resolve_to_one_candidate,
    test_resolve_ar1001_dose_variants_resolve_to_one_candidate,
    test_resolve_avp786_variants_are_not_truncated_and_remain_distinct,
    test_resolve_wujia_yizhi_granules_plus_placebo,
    test_integration_multi_trial_dataframe,
    test_rollup_placebo_only_trial_produces_no_drug_row,
    test_rollup_diagnostic_only_trial_produces_no_drug_row,
    test_rollup_device_only_trial_produces_no_drug_row,
    test_rollup_ar1001_confirmed_trial_produces_one_row,
    test_rollup_drug_with_only_na_phase_trial_does_not_crash,
    test_rollup_drug_with_only_phase4_trial_does_not_crash,
    test_rollup_drug_with_only_early_phase1_trial_does_not_crash,
    test_rollup_combined_dual_phase_values_do_not_crash,
    test_rollup_phase4_outranks_na_for_the_same_drug,
    test_rollup_wujia_unverified_trial_produces_one_row,
    test_rollup_same_drug_across_two_trials_collapses_to_one_row,
    test_rollup_case_variant_duplicate_names_collapse,
    test_rollup_one_confirmed_plus_one_unverified_produces_one_mixed_row,
    test_rollup_multiple_unresolved_candidates_excluded,
    test_unresolved_trials_csv_includes_ambiguous_and_uncertain_trials,
    test_build_target_phase_counts_basic_crosstab,
    test_build_target_phase_counts_empty_dataframe,
    test_build_target_phase_counts_sum_equals_eligible_drug_count,
    test_build_resolved_drug_trial_links_df_explodes_nct_ids,
    test_build_resolved_drug_trial_links_df_handles_blank_nct_ids,
    test_build_resolved_drug_trial_links_df_row_count_matches_trial_count_sum,
    test_build_drug_date_rollup_basic,
    test_build_drug_date_rollup_missing_dates_become_nat_not_fabricated,
    test_build_drug_date_rollup_empty_links_returns_empty_with_columns,
    test_scope_dietary_supplement_investigational_excluded_from_therapeutic,
    test_scope_dietary_supplement_curcumin_c3_complex_excluded,
    test_scope_behavioral_excluded_from_therapeutic,
    test_scope_device_excluded_from_therapeutic,
    test_scope_blood_test_is_not_a_drug,
    test_scope_csf_biomarkers_is_not_a_drug,
    test_scope_cbti_with_application_is_not_a_drug,
    test_scope_generic_diagnostic_description_wins_over_generic_non_drug_net,
    test_scope_placebo_is_placebo_or_comparator_not_a_drug,
    test_scope_comparator_background_therapy_is_placebo_or_comparator,
    test_scope_diagnostic_imaging_agent_is_diagnostic_agent,
    test_scope_radiation_with_imaging_wording_is_diagnostic_agent,
    test_scope_radiation_without_imaging_wording_is_non_drug_and_flagged,
    test_scope_generic_diagnostic_test_type_still_gets_diagnostic_agent,
    test_scope_combination_product_with_known_compound_stays_therapeutic,
    test_scope_combination_product_with_no_recognized_compound_needs_review,
    test_scope_genetic_testing_is_not_classified_as_gene_therapy,
    test_scope_genetic_gene_therapy_product_detected,
    test_scope_genetic_ambiguous_type_needs_review,
    test_scope_drug_type_candidate_with_no_disqualifying_evidence_stays_therapeutic,
    test_scope_uncertain_drug_like_name_with_no_type_evidence_is_needs_review,
    test_scope_curated_override_can_promote_a_record,
    test_scope_curated_override_can_correct_a_record_to_excluded,
    test_scope_all_results_use_a_documented_label,
    test_load_scope_overrides_missing_file_returns_empty_dict,
    test_load_scope_overrides_parses_real_columns,
    test_load_scope_overrides_missing_required_column_raises,
    test_the_real_intervention_scope_overrides_csv_loads_without_error,
    test_build_interventions_dataframe_attaches_pipeline_scope_columns,
    test_build_interventions_dataframe_applies_scope_overrides,
    test_rollup_excludes_records_whose_scope_is_excluded,
    test_rollup_excludes_records_whose_scope_is_placebo_or_comparator,
    test_rollup_excludes_non_therapeutic_scopes_entirely,
    test_rollup_default_fixture_rows_stay_therapeutic_drug_scope,
    test_rollup_non_therapeutic_sibling_trial_does_not_block_the_eligible_one,
    test_resolve_developed_drug_carries_scope_from_winning_intervention,
    test_build_scope_audit_dataframe_basic_structure,
    test_build_scope_audit_dataframe_empty_input,
    test_build_scope_audit_dataframe_dashboard_eligible_matches_therapeutic_scope,
    test_isotope_labeled_name_matches_bracketed_18f,
    test_isotope_labeled_name_matches_bracketed_11c,
    test_isotope_labeled_name_matches_prefix_without_brackets,
    test_isotope_labeled_name_matches_fused_no_separator,
    test_isotope_labeled_name_false_positive_guard_embedded_digits,
    test_isotope_labeled_name_false_positive_guard_ordinary_names,
    test_scope_isotope_name_with_diagnostic_primary_purpose_is_diagnostic_agent,
    test_scope_isotope_name_with_pet_wording_in_summary_is_diagnostic_agent_even_without_diagnostic_purpose,
    test_scope_isotope_name_with_biodistribution_wording_is_diagnostic_agent,
    test_scope_isotope_name_without_any_diagnostic_context_is_not_reclassified,
    test_scope_ssr180711c_stays_therapeutic_despite_embedded_11c_substring,
    test_scope_sponsor_developed_therapeutic_never_overridden_by_isotope_check,
    test_diagnostic_subtype_amyloid,
    test_diagnostic_subtype_tau,
    test_diagnostic_subtype_tspo,
    test_diagnostic_subtype_generic_fallback,
    test_diagnostic_subtype_prioritizes_own_name_over_shared_trial_summary,
    test_build_diagnostic_agent_audit_dataframe_empty_input,
    test_build_diagnostic_agent_audit_dataframe_flags_newly_caught_as_leaked,
    test_build_diagnostic_agent_audit_dataframe_pre_existing_diagnostic_agent_not_flagged_as_leaked,
    test_build_diagnostic_agent_audit_dataframe_uncertain_case_flagged_low_confidence,
    test_build_diagnostic_agent_audit_dataframe_non_suspect_row_excluded,
    test_build_diagnostic_agent_audit_dataframe_aggregates_nct_ids_across_trials,
    test_resolved_drugs_df_eligible_scopes_is_therapeutic_drug_only,
    test_extended_net_catches_neuromodulation_device,
    test_extended_net_catches_digital_app_intervention,
    test_extended_net_catches_exercise_variants,
    test_extended_net_catches_educational_intervention,
    test_extended_net_catches_observational_monitoring,
    test_extended_net_does_not_match_ordinary_drug_names,
    test_extended_net_does_not_use_bare_therapy_token,
    test_classify_intervention_vagus_nerve_stimulation_is_behavioral_non_drug,
    test_classify_intervention_cognitive_stimulation_therapy_is_behavioral_non_drug,
    test_scope_extended_net_match_becomes_non_drug_intervention,
    test_has_known_therapeutic_evidence_true_for_curated_compound_substring,
    test_has_known_therapeutic_evidence_false_for_unrelated_text,
    test_classify_intervention_real_drug_combined_with_procedure_word_stays_therapeutic,
    test_classify_intervention_aln_app_is_not_misread_as_an_application,
    test_exclusion_audit_includes_scope_level_exclusion,
    test_exclusion_audit_includes_extended_net_classification_level_exclusion,
    test_exclusion_audit_excludes_unambiguous_placebo_arms,
    test_exclusion_audit_empty_input,
    test_rollup_excludes_extended_net_matches_from_resolved_drugs_df,
    test_isotope_labeled_name_matches_123i_notation,
    test_diagnostic_challenge_probe_pramlintide_challenge_test,
    test_diagnostic_challenge_probe_scopolamine_eeg_diagnostic_tool,
    test_diagnostic_challenge_probe_pet_imaging_with_challenge_in_title,
    test_diagnostic_challenge_probe_contrast_imaging_agent,
    test_diagnostic_challenge_probe_requires_diagnostic_purpose_not_just_a_token_match,
    test_diagnostic_challenge_probe_name_alone_is_sufficient_without_context,
    test_diagnostic_challenge_probe_protects_real_investigational_drug_phase1_studies,
    test_diagnostic_challenge_probe_protects_real_drug_administered_as_treatment_within_diagnostic_study,
    test_deprescribing_name_token_detected,
    test_deprescribing_phrases_detected,
    test_procedural_support_sedation_phrases_detected,
    test_deprescribing_does_not_match_ordinary_drug_names,
    test_scope_pramlintide_challenge_test_becomes_diagnostic_agent,
    test_scope_deprescribing_becomes_non_drug_intervention,
    test_scope_sponsor_developed_therapeutic_never_overridden_by_diagnostic_purpose_check,
    test_flyer_excluded_via_extended_net,
    test_classify_intervention_flyer_is_behavioral_non_drug,
    test_blood_withdrawal_is_a_procedure,
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
