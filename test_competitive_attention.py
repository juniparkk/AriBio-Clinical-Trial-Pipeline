# ============================================================
# TESTS for competitive_attention.py + competitive_attention_viz.py
# (AriBio competitive-priority scoring, "Needs Attention" +
# "Upcoming Competitive Milestones")
#
# Plain-Python tests (no pytest install needed, matching this
# project's other test files) — run with:
#     .venv/bin/python test_competitive_attention.py
# ============================================================

import pandas as pd

import aribio_watchlist
import competitive_attention as ca
import competitive_attention_viz as cav

WATCHLIST = aribio_watchlist.load_watchlist()  # falls back to DEFAULT_WATCHLIST if the file is absent
TODAY = "2026-08-14"


def _change_row(entity_type="trial", nct_id="NCT00000001", canonical_drug_name="TestDrug",
                 sponsor_or_company="Acme Pharma", change_type="status_change",
                 old_value="Recruiting", new_value="Completed", importance="Medium",
                 needs_review=False, source="TEST"):
    return {
        "detected_date": TODAY, "entity_type": entity_type, "nct_id": nct_id,
        "canonical_drug_name": canonical_drug_name, "sponsor_or_company": sponsor_or_company,
        "change_type": change_type, "old_value": old_value, "new_value": new_value,
        "importance": importance, "source": source, "needs_review": needs_review,
    }


def _changes_df(rows):
    return pd.DataFrame(rows)


def _drug_row(display_name, phase_reached="Phase 2", sponsor="Acme Pharma", modality="Small Molecule",
              target_pathways="Amyloid", pipeline_scope="Therapeutic Drug",
              aribio_relevance_score=0, classification_confidence="high", status_summary="Active"):
    return {
        "display_name": display_name, "phase_reached": phase_reached, "sponsor": sponsor,
        "modality": modality, "target_pathways": target_pathways, "pipeline_scope": pipeline_scope,
        "aribio_relevance_score": aribio_relevance_score, "classification_confidence": classification_confidence,
        "status_summary": status_summary,
    }


def _drugs_df(rows):
    return pd.DataFrame(rows)


def _trial_row(nct_number="NCT00000001", sponsor="Acme Pharma", status="RECRUITING",
                primary_completion="2026-09-01", completion="2027-01-01", sex="ALL",
                age="ADULT, OLDER_ADULT", primary_outcomes="", secondary_outcomes=""):
    return {
        "NCT Number": nct_number, "Sponsor": sponsor, "Study Status": status,
        "Primary Completion Date": primary_completion, "Completion Date": completion,
        "Sex": sex, "Age": age,
        "Primary Outcome Measures": primary_outcomes, "Secondary Outcome Measures": secondary_outcomes,
        "Other Outcome Measures": "",
    }


def _trials_df(rows):
    return pd.DataFrame(rows)


def _annotated_row(nct_id="NCT00000001", pipeline_scope="Therapeutic Drug", verification_status="no_match",
                    classification_confidence="high", phase_clean="Phase 2", status_clean="Recruiting"):
    return {
        "nct_id": nct_id, "pipeline_scope": pipeline_scope, "verification_status": verification_status,
        "classification_confidence": classification_confidence, "phase_clean": phase_clean,
        "status_clean": status_clean,
    }


def _annotated_df(rows):
    return pd.DataFrame(rows)


# ------------------------------------------------------------
# Phase 3 ranks above Phase 1
# ------------------------------------------------------------

def test_phase_3_scores_higher_than_phase_1_for_otherwise_identical_changes():
    changes = _changes_df([
        _change_row(nct_id="NCT00000001", canonical_drug_name="DrugA"),
        _change_row(nct_id="NCT00000002", canonical_drug_name="DrugB"),
    ])
    drugs = _drugs_df([
        _drug_row("DrugA", phase_reached="Phase 3"),
        _drug_row("DrugB", phase_reached="Phase 1"),
    ])
    trials = _trials_df([
        _trial_row("NCT00000001", primary_completion="", completion=""),
        _trial_row("NCT00000002", primary_completion="", completion=""),
    ])
    annotated = _annotated_df([
        _annotated_row("NCT00000001", phase_clean="Phase 3"),
        _annotated_row("NCT00000002", phase_clean="Phase 1"),
    ])
    result = ca.compute_attention(changes, drugs, annotated, trials, WATCHLIST, today=TODAY)
    score_a = result[result["canonical_drug_name"] == "DrugA"]["relevance_score"].iloc[0]
    score_b = result[result["canonical_drug_name"] == "DrugB"]["relevance_score"].iloc[0]
    assert score_a > score_b


# ------------------------------------------------------------
# new Phase 3 trial ranks highly
# ------------------------------------------------------------

def test_new_phase_3_trial_ranks_critical_or_high():
    changes = _changes_df([
        _change_row(nct_id="NCT00000001", canonical_drug_name="DrugA", change_type="new_trial",
                    old_value="", new_value="Phase 3", needs_review=True),
    ])
    drugs = _drugs_df([_drug_row("DrugA", phase_reached="Phase 3", aribio_relevance_score=60)])
    trials = _trials_df([_trial_row("NCT00000001", status="RECRUITING")])
    annotated = _annotated_df([_annotated_row("NCT00000001", phase_clean="Phase 3")])
    result = ca.compute_attention(changes, drugs, annotated, trials, WATCHLIST, today=TODAY)
    assert result.iloc[0]["priority_level"] in ("Critical", "High")
    assert result.iloc[0]["relevance_score"] >= WATCHLIST["alert_thresholds"]["high_score"]


# ------------------------------------------------------------
# results newly posted ranks highly
# ------------------------------------------------------------

def test_results_newly_posted_ranks_critical_or_high():
    changes = _changes_df([
        _change_row(change_type="results_posted", old_value="NO", new_value="YES", needs_review=True),
    ])
    drugs = _drugs_df([_drug_row("TestDrug", phase_reached="Phase 2")])
    trials = _trials_df([_trial_row()])
    annotated = _annotated_df([_annotated_row()])
    result = ca.compute_attention(changes, drugs, annotated, trials, WATCHLIST, today=TODAY)
    assert result.iloc[0]["priority_level"] in ("Critical", "High")


# ------------------------------------------------------------
# minor metadata changes rank low
# ------------------------------------------------------------

def test_minor_enrollment_change_on_early_phase_drug_ranks_low():
    changes = _changes_df([
        _change_row(change_type="enrollment_change", old_value="100", new_value="102", importance="Low"),
    ])
    drugs = _drugs_df([_drug_row("TestDrug", phase_reached="Phase 1", aribio_relevance_score=0)])
    trials = _trials_df([_trial_row(primary_completion="", completion="")])
    annotated = _annotated_df([_annotated_row(phase_clean="Phase 1")])
    result = ca.compute_attention(changes, drugs, annotated, trials, WATCHLIST, today=TODAY)
    assert result.iloc[0]["priority_level"] == "Low"


# ------------------------------------------------------------
# non-therapeutic records excluded
# ------------------------------------------------------------

def test_non_therapeutic_drug_level_row_is_excluded():
    changes = _changes_df([
        _change_row(entity_type="drug", nct_id="", canonical_drug_name="DeviceThing",
                    change_type="new_drug", old_value="", new_value="Phase 2"),
    ])
    drugs = _drugs_df([_drug_row("DeviceThing", pipeline_scope="Exclude")])
    result = ca.compute_attention(changes, drugs, None, None, WATCHLIST, today=TODAY)
    assert len(result) == 0


def test_non_therapeutic_trial_level_row_is_excluded():
    changes = _changes_df([
        _change_row(nct_id="NCT00000001", canonical_drug_name="", change_type="status_change"),
    ])
    trials = _trials_df([_trial_row("NCT00000001")])
    annotated = _annotated_df([_annotated_row("NCT00000001", pipeline_scope="Exclude")])
    result = ca.compute_attention(changes, None, annotated, trials, WATCHLIST, today=TODAY)
    assert len(result) == 0


def test_therapeutic_trial_level_row_is_kept():
    changes = _changes_df([
        _change_row(nct_id="NCT00000001", canonical_drug_name="TestDrug", change_type="status_change"),
    ])
    trials = _trials_df([_trial_row("NCT00000001")])
    annotated = _annotated_df([_annotated_row("NCT00000001", pipeline_scope="Therapeutic Drug")])
    result = ca.compute_attention(changes, None, annotated, trials, WATCHLIST, today=TODAY)
    assert len(result) == 1


# ------------------------------------------------------------
# AR1001-similar features increase score only when evidence exists
# ------------------------------------------------------------

def test_aribio_relevance_score_contributes_points_when_present():
    changes = _changes_df([_change_row(canonical_drug_name="Similar", change_type="status_change")])
    drugs = _drugs_df([_drug_row("Similar", aribio_relevance_score=80)])
    trials = _trials_df([_trial_row(primary_completion="", completion="")])
    annotated = _annotated_df([_annotated_row()])
    result = ca.compute_attention(changes, drugs, annotated, trials, WATCHLIST, today=TODAY)
    assert "similar profile to AR1001" in result.iloc[0]["relevance_factors"]


def test_zero_aribio_relevance_score_contributes_no_similarity_points():
    changes = _changes_df([_change_row(canonical_drug_name="Unrelated", change_type="status_change")])
    drugs = _drugs_df([_drug_row("Unrelated", aribio_relevance_score=0)])
    trials = _trials_df([_trial_row(primary_completion="", completion="")])
    annotated = _annotated_df([_annotated_row()])
    result = ca.compute_attention(changes, drugs, annotated, trials, WATCHLIST, today=TODAY)
    assert "similar profile to AR1001" not in result.iloc[0]["relevance_factors"]


def test_population_similarity_awarded_only_on_exact_sex_and_age_match():
    primary_asset = aribio_watchlist.get_primary_asset(WATCHLIST)
    changes = _changes_df([
        _change_row(nct_id="NCT00000001", canonical_drug_name="", change_type="status_change"),
        _change_row(nct_id="NCT00000002", canonical_drug_name="", change_type="status_change"),
    ])
    trials = _trials_df([
        _trial_row("NCT00000001", sex=primary_asset["reference_sex"], age=primary_asset["reference_age"],
                    primary_completion="", completion=""),
        _trial_row("NCT00000002", sex="MALE", age="CHILD", primary_completion="", completion=""),
    ])
    annotated = _annotated_df([_annotated_row("NCT00000001"), _annotated_row("NCT00000002")])
    result = ca.compute_attention(changes, None, annotated, trials, WATCHLIST, today=TODAY)
    match_row = result[result["nct_id"] == "NCT00000001"].iloc[0]
    mismatch_row = result[result["nct_id"] == "NCT00000002"].iloc[0]
    assert "similar eligibility population" in match_row["relevance_factors"]
    assert "similar eligibility population" not in mismatch_row["relevance_factors"]


# ------------------------------------------------------------
# missing data does not receive similarity points
# ------------------------------------------------------------

def test_missing_drug_lookup_receives_no_similarity_points():
    # canonical_drug_name doesn't match any row in drugs_df -- drug_info
    # is None, so no aribio_relevance_score/modality points can be
    # awarded, but the row is still scored (not excluded, since it's
    # still tied to a therapeutic trial via annotated_df).
    changes = _changes_df([
        _change_row(nct_id="NCT00000001", canonical_drug_name="NotInDrugsCsv", change_type="status_change"),
    ])
    drugs = _drugs_df([_drug_row("SomeOtherDrug", aribio_relevance_score=90)])
    trials = _trials_df([_trial_row("NCT00000001", primary_completion="", completion="")])
    annotated = _annotated_df([_annotated_row("NCT00000001")])
    result = ca.compute_attention(changes, drugs, annotated, trials, WATCHLIST, today=TODAY)
    assert "similar profile to AR1001" not in result.iloc[0]["relevance_factors"]
    assert "priority modality" not in result.iloc[0]["relevance_factors"]


def test_missing_outcome_text_receives_no_endpoint_or_biomarker_points():
    watchlist = {**WATCHLIST, "priority_endpoints": ["CDR-SB"], "priority_biomarkers": ["p-tau217"]}
    changes = _changes_df([_change_row(nct_id="NCT00000001", canonical_drug_name="", change_type="status_change")])
    trials = _trials_df([_trial_row("NCT00000001", primary_completion="", completion="",
                                     primary_outcomes="", secondary_outcomes="")])
    annotated = _annotated_df([_annotated_row("NCT00000001")])
    result = ca.compute_attention(changes, None, annotated, trials, watchlist, today=TODAY)
    assert "overlapping endpoint" not in result.iloc[0]["relevance_factors"]
    assert "overlapping biomarker" not in result.iloc[0]["relevance_factors"]


def test_configured_endpoint_present_in_outcomes_text_awards_points():
    watchlist = {**WATCHLIST, "priority_endpoints": ["CDR-SB"]}
    changes = _changes_df([_change_row(nct_id="NCT00000001", canonical_drug_name="", change_type="status_change")])
    trials = _trials_df([_trial_row("NCT00000001", primary_completion="", completion="",
                                     primary_outcomes="Change in CDR-SB from baseline")])
    annotated = _annotated_df([_annotated_row("NCT00000001")])
    result = ca.compute_attention(changes, None, annotated, trials, watchlist, today=TODAY)
    assert "overlapping endpoint" in result.iloc[0]["relevance_factors"]


# ------------------------------------------------------------
# deterministic scoring
# ------------------------------------------------------------

def test_scoring_is_deterministic_across_repeated_runs():
    changes = _changes_df([
        _change_row(nct_id="NCT00000001", canonical_drug_name="DrugA", change_type="phase_change",
                    old_value="Phase 1", new_value="Phase 2"),
        _change_row(entity_type="drug", nct_id="", canonical_drug_name="DrugB",
                    change_type="new_drug", old_value="", new_value="Phase 1"),
    ])
    drugs = _drugs_df([_drug_row("DrugA", aribio_relevance_score=45), _drug_row("DrugB", phase_reached="Phase 1")])
    trials = _trials_df([_trial_row("NCT00000001")])
    annotated = _annotated_df([_annotated_row("NCT00000001")])

    result1 = ca.compute_attention(changes, drugs, annotated, trials, WATCHLIST, today=TODAY)
    result2 = ca.compute_attention(changes, drugs, annotated, trials, WATCHLIST, today=TODAY)
    pd.testing.assert_frame_equal(result1, result2)


def test_low_confidence_penalty_is_deterministic_and_reduces_score():
    changes = _changes_df([_change_row(canonical_drug_name="LowConf", change_type="status_change")])
    drugs_high = _drugs_df([_drug_row("LowConf", classification_confidence="high")])
    drugs_low = _drugs_df([_drug_row("LowConf", classification_confidence="low")])
    trials = _trials_df([_trial_row(primary_completion="", completion="")])
    annotated = _annotated_df([_annotated_row()])

    result_high = ca.compute_attention(changes, drugs_high, annotated, trials, WATCHLIST, today=TODAY)
    result_low = ca.compute_attention(changes, drugs_low, annotated, trials, WATCHLIST, today=TODAY)
    assert result_low.iloc[0]["relevance_score"] < result_high.iloc[0]["relevance_score"]
    assert result_low.iloc[0]["needs_human_review"] == True


# ------------------------------------------------------------
# ranking / priority_rank / schema
# ------------------------------------------------------------

def test_priority_rank_is_assigned_in_score_descending_order():
    changes = _changes_df([
        _change_row(nct_id="NCT00000001", canonical_drug_name="DrugLow", change_type="enrollment_change",
                    old_value="100", new_value="101", importance="Low"),
        _change_row(nct_id="NCT00000002", canonical_drug_name="DrugHigh", change_type="results_posted",
                    old_value="NO", new_value="YES"),
    ])
    drugs = _drugs_df([_drug_row("DrugLow", phase_reached="Phase 1"), _drug_row("DrugHigh", phase_reached="Phase 3")])
    trials = _trials_df([
        _trial_row("NCT00000001", primary_completion="", completion=""),
        _trial_row("NCT00000002", primary_completion="", completion=""),
    ])
    annotated = _annotated_df([_annotated_row("NCT00000001"), _annotated_row("NCT00000002")])
    result = ca.compute_attention(changes, drugs, annotated, trials, WATCHLIST, today=TODAY)
    assert list(result["priority_rank"]) == [1, 2]
    assert result.iloc[0]["canonical_drug_name"] == "DrugHigh"


def test_output_columns_match_required_schema():
    changes = _changes_df([_change_row()])
    drugs = _drugs_df([_drug_row("TestDrug")])
    trials = _trials_df([_trial_row()])
    annotated = _annotated_df([_annotated_row()])
    result = ca.compute_attention(changes, drugs, annotated, trials, WATCHLIST, today=TODAY)
    assert list(result.columns) == [
        "priority_rank", "priority_level", "relevance_score", "canonical_drug_name",
        "nct_id", "company_or_sponsor", "change_type", "old_value", "new_value",
        "highest_phase", "modality", "target_pathways", "trial_status",
        "primary_completion_date", "completion_date", "why_it_matters",
        "relevance_factors", "source", "needs_human_review",
    ]


def test_empty_changes_produces_empty_result_with_correct_columns():
    result = ca.compute_attention(_changes_df([]), None, None, None, WATCHLIST, today=TODAY)
    assert len(result) == 0
    assert list(result.columns) == ca.ATTENTION_COLUMNS


# ------------------------------------------------------------
# milestone bucketing
# ------------------------------------------------------------

def test_milestone_next_30_days_bucket():
    trials = _trials_df([_trial_row("NCT00000001", status="RECRUITING", primary_completion="2026-08-20", completion="2026-09-01")])
    annotated = _annotated_df([_annotated_row("NCT00000001", status_clean="Recruiting")])
    m = ca.build_milestones(annotated, trials, None, WATCHLIST, today=TODAY)
    assert len(m["next_30_days"]) == 1
    assert len(m["next_90_days"]) == 0


def test_milestone_next_90_days_bucket_excludes_next_30_days_items():
    trials = _trials_df([_trial_row("NCT00000001", status="RECRUITING", primary_completion="2026-10-01", completion="2027-01-01")])
    annotated = _annotated_df([_annotated_row("NCT00000001", status_clean="Recruiting")])
    m = ca.build_milestones(annotated, trials, None, WATCHLIST, today=TODAY)
    assert len(m["next_30_days"]) == 0
    assert len(m["next_90_days"]) == 1


def test_milestone_recently_completed_bucket():
    trials = _trials_df([_trial_row("NCT00000001", status="COMPLETED", primary_completion="2026-07-20", completion="2026-07-20")])
    annotated = _annotated_df([_annotated_row("NCT00000001", status_clean="Completed")])
    m = ca.build_milestones(annotated, trials, None, WATCHLIST, today=TODAY)
    assert len(m["recently_completed"]) == 1


def test_milestone_discontinued_trial_excluded_from_upcoming_buckets():
    trials = _trials_df([_trial_row("NCT00000001", status="TERMINATED", primary_completion="2026-08-20", completion="2026-08-20")])
    annotated = _annotated_df([_annotated_row("NCT00000001", status_clean="Discontinued")])
    m = ca.build_milestones(annotated, trials, None, WATCHLIST, today=TODAY)
    assert len(m["next_30_days"]) == 0
    assert len(m["next_90_days"]) == 0


def test_milestone_non_therapeutic_trial_excluded():
    trials = _trials_df([_trial_row("NCT00000001", primary_completion="2026-08-20", completion="2026-08-20")])
    annotated = _annotated_df([_annotated_row("NCT00000001", pipeline_scope="Exclude")])
    m = ca.build_milestones(annotated, trials, None, WATCHLIST, today=TODAY)
    assert len(m["next_30_days"]) == 0


def test_milestone_materially_delayed_bucket():
    changes = _changes_df([
        _change_row(nct_id="NCT00000001", change_type="primary_completion_date_change",
                    old_value="2026-01-01", new_value="2026-06-01", importance="Medium"),
    ])
    annotated = _annotated_df([_annotated_row("NCT00000001")])
    m = ca.build_milestones(annotated, None, changes, WATCHLIST, today=TODAY)
    assert len(m["materially_delayed"]) == 1
    assert m["materially_delayed"][0]["days_delayed"] >= WATCHLIST["alert_thresholds"]["major_delay_days"]


def test_milestone_earlier_date_shift_not_counted_as_delay():
    changes = _changes_df([
        _change_row(nct_id="NCT00000001", change_type="primary_completion_date_change",
                    old_value="2026-06-01", new_value="2026-01-01", importance="Medium"),
    ])
    annotated = _annotated_df([_annotated_row("NCT00000001")])
    m = ca.build_milestones(annotated, None, changes, WATCHLIST, today=TODAY)
    assert len(m["materially_delayed"]) == 0


# ------------------------------------------------------------
# no unsupported efficacy/FDA language, careful milestone wording
# ------------------------------------------------------------

_FORBIDDEN_PHRASES = [
    "results expected", "likely to succeed", "likely to fail", "will succeed", "will fail",
    "fda approval", "fda status", "proven effective", "proven safe", "is effective",
    "is ineffective", "better than", "worse than", "superior to", "inferior to",
]


def test_why_it_matters_never_uses_forbidden_efficacy_or_fda_language():
    changes = _changes_df([
        _change_row(change_type="status_change", old_value="Recruiting", new_value="Terminated"),
        _change_row(change_type="results_posted", old_value="NO", new_value="YES"),
        _change_row(change_type="new_trial", old_value="", new_value="Phase 3"),
    ])
    drugs = _drugs_df([_drug_row("TestDrug")])
    trials = _trials_df([_trial_row()])
    annotated = _annotated_df([_annotated_row()])
    result = ca.compute_attention(changes, drugs, annotated, trials, WATCHLIST, today=TODAY)
    for text in result["why_it_matters"]:
        lowered = text.lower()
        for phrase in _FORBIDDEN_PHRASES:
            assert phrase not in lowered, f"forbidden phrase '{phrase}' found in: {text}"


def test_terminated_status_change_wording_does_not_imply_failure():
    changes = _changes_df([_change_row(change_type="status_change", old_value="Recruiting", new_value="Discontinued")])
    drugs = _drugs_df([_drug_row("TestDrug")])
    trials = _trials_df([_trial_row()])
    annotated = _annotated_df([_annotated_row()])
    result = ca.compute_attention(changes, drugs, annotated, trials, WATCHLIST, today=TODAY)
    text = result.iloc[0]["why_it_matters"].lower()
    assert "fail" not in text and "unsuccessful" not in text


def test_milestone_notes_use_careful_non_promissory_wording():
    notes = " ".join(cav._milestone_note(k) for k in
                      ("next_30_days", "next_90_days", "recently_completed", "materially_delayed"))
    lowered = notes.lower()
    assert "results expected" not in lowered
    assert "may warrant monitoring" in lowered or "no outcome is implied" in lowered


# ------------------------------------------------------------
# dashboard HTML rendering sanity
# ------------------------------------------------------------

def test_render_needs_attention_section_handles_empty_dataframe():
    html = cav.render_needs_attention_section(pd.DataFrame(columns=ca.ATTENTION_COLUMNS))
    assert "Needs Attention" in html
    assert "No pipeline changes currently need attention" in html


def test_render_needs_attention_section_respects_top_n():
    rows = [
        {**{c: "" for c in ca.ATTENTION_COLUMNS}, "relevance_score": 90 - i, "priority_level": "High",
         "canonical_drug_name": f"Drug{i}", "nct_id": f"NCT{i:08d}", "relevance_factors": "", "why_it_matters": ""}
        for i in range(12)
    ]
    df = pd.DataFrame(rows, columns=ca.ATTENTION_COLUMNS)
    html = cav.render_needs_attention_section(df, top_n=5)
    assert "showing top 5 of 12" in html


def test_render_competitive_sections_includes_placeholder_replaceable_content():
    empty_milestones = {"next_30_days": [], "next_90_days": [], "recently_completed": [], "materially_delayed": []}
    html = cav.render_competitive_sections(pd.DataFrame(columns=ca.ATTENTION_COLUMNS), empty_milestones)
    assert "Needs Attention" in html
    assert "Upcoming Competitive Milestones" in html
    assert cav.PLACEHOLDER not in html  # the rendered section itself must not contain the raw token


ALL_TESTS = [
    test_phase_3_scores_higher_than_phase_1_for_otherwise_identical_changes,
    test_new_phase_3_trial_ranks_critical_or_high,
    test_results_newly_posted_ranks_critical_or_high,
    test_minor_enrollment_change_on_early_phase_drug_ranks_low,
    test_non_therapeutic_drug_level_row_is_excluded,
    test_non_therapeutic_trial_level_row_is_excluded,
    test_therapeutic_trial_level_row_is_kept,
    test_aribio_relevance_score_contributes_points_when_present,
    test_zero_aribio_relevance_score_contributes_no_similarity_points,
    test_population_similarity_awarded_only_on_exact_sex_and_age_match,
    test_missing_drug_lookup_receives_no_similarity_points,
    test_missing_outcome_text_receives_no_endpoint_or_biomarker_points,
    test_configured_endpoint_present_in_outcomes_text_awards_points,
    test_scoring_is_deterministic_across_repeated_runs,
    test_low_confidence_penalty_is_deterministic_and_reduces_score,
    test_priority_rank_is_assigned_in_score_descending_order,
    test_output_columns_match_required_schema,
    test_empty_changes_produces_empty_result_with_correct_columns,
    test_milestone_next_30_days_bucket,
    test_milestone_next_90_days_bucket_excludes_next_30_days_items,
    test_milestone_recently_completed_bucket,
    test_milestone_discontinued_trial_excluded_from_upcoming_buckets,
    test_milestone_non_therapeutic_trial_excluded,
    test_milestone_materially_delayed_bucket,
    test_milestone_earlier_date_shift_not_counted_as_delay,
    test_why_it_matters_never_uses_forbidden_efficacy_or_fda_language,
    test_terminated_status_change_wording_does_not_imply_failure,
    test_milestone_notes_use_careful_non_promissory_wording,
    test_render_needs_attention_section_handles_empty_dataframe,
    test_render_needs_attention_section_respects_top_n,
    test_render_competitive_sections_includes_placeholder_replaceable_content,
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
