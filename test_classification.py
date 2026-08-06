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
                    title="Test Trial", interventions="DRUG: X", classification_reason="test reason"):
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
