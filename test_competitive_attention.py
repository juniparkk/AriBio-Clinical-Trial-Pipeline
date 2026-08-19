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
                    classification_confidence="high", phase_clean="Phase 2", status_clean="Recruiting",
                    developed_drug="TestDrug"):
    return {
        "nct_id": nct_id, "pipeline_scope": pipeline_scope, "verification_status": verification_status,
        "classification_confidence": classification_confidence, "phase_clean": phase_clean,
        "status_clean": status_clean, "developed_drug": developed_drug,
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
# named-competitor watchlist (config/aribio_watchlist.yaml's
# competitor_companies) -- POINTS_NAMED_COMPETITOR + is_named_competitor
# ------------------------------------------------------------

def test_named_competitor_sponsor_awards_points_and_factor():
    # "Eli Lilly" (the watchlist entry) must match ct.gov's real sponsor
    # string "Eli Lilly and Company" via word-overlap, not an exact
    # string match.
    watchlist = {**WATCHLIST, "competitor_companies": ["Eli Lilly"]}
    changes = _changes_df([_change_row(nct_id="NCT00000001", canonical_drug_name="Donanemab", change_type="status_change")])
    trials = _trials_df([_trial_row("NCT00000001", sponsor="Eli Lilly and Company",
                                     primary_completion="", completion="")])
    annotated = _annotated_df([_annotated_row("NCT00000001", developed_drug="Donanemab")])
    result = ca.compute_attention(changes, None, annotated, trials, watchlist, today=TODAY)
    assert "named competitor on watchlist (Eli Lilly)" in result.iloc[0]["relevance_factors"]
    assert result.iloc[0]["is_named_competitor"] == True  # noqa: E712 (want the actual bool, not just truthiness)


def test_named_competitor_word_overlap_avoids_false_positive():
    # A naive substring match on "Roche" would wrongly catch "University
    # of Rochester" -- the real word-overlap matcher must not.
    watchlist = {**WATCHLIST, "competitor_companies": ["Roche"]}
    changes = _changes_df([_change_row(nct_id="NCT00000001", canonical_drug_name="TestDrug", change_type="status_change")])
    trials = _trials_df([_trial_row("NCT00000001", sponsor="University of Rochester",
                                     primary_completion="", completion="")])
    annotated = _annotated_df([_annotated_row("NCT00000001")])
    result = ca.compute_attention(changes, None, annotated, trials, watchlist, today=TODAY)
    assert "named competitor" not in result.iloc[0]["relevance_factors"]
    assert result.iloc[0]["is_named_competitor"] == False  # noqa: E712


def test_named_competitor_no_match_when_list_empty():
    # Default/empty competitor_companies must never award points --
    # matches this project's existing behavior before the watchlist
    # was populated.
    watchlist = {**WATCHLIST, "competitor_companies": []}
    changes = _changes_df([_change_row(nct_id="NCT00000001", canonical_drug_name="Donanemab", change_type="status_change")])
    trials = _trials_df([_trial_row("NCT00000001", sponsor="Eli Lilly and Company",
                                     primary_completion="", completion="")])
    annotated = _annotated_df([_annotated_row("NCT00000001", developed_drug="Donanemab")])
    result = ca.compute_attention(changes, None, annotated, trials, watchlist, today=TODAY)
    assert "named competitor" not in result.iloc[0]["relevance_factors"]


def test_named_competitor_matches_via_drug_level_sponsor():
    # Drug-level changes (entity_type="drug") have no trial_info at all
    # -- the match must fall back to the drug's own (semicolon-joinable)
    # sponsor field.
    watchlist = {**WATCHLIST, "competitor_companies": ["Biogen"]}
    changes = _changes_df([_change_row(entity_type="drug", nct_id="", canonical_drug_name="TestBiogenDrug",
                                        change_type="new_drug", old_value="", new_value="Phase 2")])
    drugs = _drugs_df([_drug_row("TestBiogenDrug", sponsor="Biogen; Some Academic Hospital")])
    result = ca.compute_attention(changes, drugs, None, None, watchlist, today=TODAY)
    assert "named competitor on watchlist (Biogen)" in result.iloc[0]["relevance_factors"]


def test_needs_attention_always_shows_named_competitor_beyond_top_n():
    # A named-competitor row ranked BELOW the natural top_n cutoff must
    # still appear -- guaranteed display, not just a higher rank (see
    # render_needs_attention_section()).
    base = {c: "" for c in ca.ATTENTION_COLUMNS}
    top_rows = [
        {**base, "relevance_score": 90 - i, "priority_level": "High",
         "canonical_drug_name": f"HighDrug{i}", "nct_id": f"NCT0000000{i}",
         "is_named_competitor": False}
        for i in range(8)
    ]
    low_named_row = {**base, "relevance_score": 5, "priority_level": "Low",
                      "canonical_drug_name": "LowRankedCompetitorDrug", "nct_id": "NCT99999999",
                      "is_named_competitor": True, "relevance_factors": "named competitor on watchlist (Eisai) (+20)"}
    attention_df = pd.DataFrame(top_rows + [low_named_row], columns=ca.ATTENTION_COLUMNS)

    html = cav.render_needs_attention_section(attention_df, top_n=8)
    assert "LowRankedCompetitorDrug" in html
    assert "watchlist-company change" in html


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
        "priority_rank", "priority_level", "relevance_score", "aribio_relevance_score",
        "canonical_drug_name", "nct_id", "company_or_sponsor", "change_type", "old_value", "new_value",
        "highest_phase", "modality", "target_pathways", "trial_status",
        "primary_completion_date", "completion_date", "why_it_matters",
        "relevance_factors", "source", "needs_human_review", "is_named_competitor",
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


def test_milestone_item_carries_phase_from_annotated_df():
    trials = _trials_df([_trial_row("NCT00000001", status="RECRUITING", primary_completion="2026-08-20", completion="2026-09-01")])
    annotated = _annotated_df([_annotated_row("NCT00000001", status_clean="Recruiting", phase_clean="Phase 3")])
    m = ca.build_milestones(annotated, trials, None, WATCHLIST, today=TODAY)
    assert m["next_30_days"][0]["phase"] == "Phase 3"


def test_milestone_item_carries_drug_type_when_drugs_df_given():
    trials = _trials_df([_trial_row("NCT00000001", status="RECRUITING", primary_completion="2026-08-20", completion="2026-09-01")])
    annotated = _annotated_df([_annotated_row("NCT00000001", status_clean="Recruiting", developed_drug="TestDrug")])
    drugs = pd.DataFrame([{"display_name": "TestDrug", "drug_type": "Disease-Targeted Small Molecule"}])
    m = ca.build_milestones(annotated, trials, None, WATCHLIST, today=TODAY, drugs_df=drugs)
    assert m["next_30_days"][0]["drug_type"] == "Disease-Targeted Small Molecule"


def test_milestone_item_drug_type_blank_when_drugs_df_not_given():
    # drugs_df is optional -- must degrade gracefully, not error, when
    # the caller doesn't have it on hand.
    trials = _trials_df([_trial_row("NCT00000001", status="RECRUITING", primary_completion="2026-08-20", completion="2026-09-01")])
    annotated = _annotated_df([_annotated_row("NCT00000001", status_clean="Recruiting")])
    m = ca.build_milestones(annotated, trials, None, WATCHLIST, today=TODAY)
    assert m["next_30_days"][0]["drug_type"] == ""


def test_materially_delayed_milestone_item_carries_phase_and_drug_type():
    trials = _trials_df([_trial_row("NCT00000001")])
    annotated = _annotated_df([_annotated_row("NCT00000001", phase_clean="Phase 2", developed_drug="TestDrug")])
    drugs = pd.DataFrame([{"display_name": "TestDrug", "drug_type": "Biologic"}])
    changes = _changes_df([
        _change_row(nct_id="NCT00000001", change_type="primary_completion_date_change",
                    old_value="2026-01-01", new_value="2026-12-31", importance="Medium"),
    ])
    m = ca.build_milestones(annotated, trials, changes, WATCHLIST, today=TODAY, drugs_df=drugs)
    assert m["materially_delayed"][0]["phase"] == "Phase 2"
    assert m["materially_delayed"][0]["drug_type"] == "Biologic"


def test_render_milestones_section_shows_phase_and_drug_type():
    trials = _trials_df([_trial_row("NCT00000001", status="RECRUITING", primary_completion="2026-08-20", completion="2026-09-01")])
    annotated = _annotated_df([_annotated_row("NCT00000001", status_clean="Recruiting", phase_clean="Phase 3", developed_drug="TestDrug")])
    drugs = pd.DataFrame([{"display_name": "TestDrug", "drug_type": "Disease-Targeted Small Molecule"}])
    m = ca.build_milestones(annotated, trials, None, WATCHLIST, today=TODAY, drugs_df=drugs)
    html = cav.render_milestones_section(m)
    assert "Phase 3" in html
    assert "Disease-Targeted Small Molecule" in html
    assert "milestone-meta" in html


def test_render_milestones_section_omits_meta_line_when_both_blank():
    trials = _trials_df([_trial_row("NCT00000001", status="RECRUITING", primary_completion="2026-08-20", completion="2026-09-01")])
    annotated = _annotated_df([_annotated_row("NCT00000001", status_clean="Recruiting", phase_clean="")])
    m = ca.build_milestones(annotated, trials, None, WATCHLIST, today=TODAY)
    html = cav.render_milestones_section(m)
    assert '<span class="milestone-meta">' not in html


def test_milestone_item_carries_resolved_drug_name():
    trials = _trials_df([_trial_row("NCT00000001", status="RECRUITING", primary_completion="2026-08-20", completion="2026-09-01")])
    annotated = _annotated_df([_annotated_row("NCT00000001", status_clean="Recruiting", developed_drug="Wonderdrug")])
    m = ca.build_milestones(annotated, trials, None, WATCHLIST, today=TODAY)
    assert m["next_30_days"][0]["drug_name"] == "Wonderdrug"


def test_milestone_item_falls_back_to_nct_id_when_drug_unresolved():
    trials = _trials_df([_trial_row("NCT00000001", status="RECRUITING", primary_completion="2026-08-20", completion="2026-09-01")])
    annotated = _annotated_df([_annotated_row("NCT00000001", status_clean="Recruiting", developed_drug="")])
    m = ca.build_milestones(annotated, trials, None, WATCHLIST, today=TODAY)
    assert m["next_30_days"][0]["drug_name"] == "NCT00000001"


def test_milestone_rendering_shows_drug_name_not_nct_id_as_link_text():
    trials = _trials_df([_trial_row("NCT00000001", status="RECRUITING", primary_completion="2026-08-20", completion="2026-09-01")])
    annotated = _annotated_df([_annotated_row("NCT00000001", status_clean="Recruiting", developed_drug="Wonderdrug")])
    m = ca.build_milestones(annotated, trials, None, WATCHLIST, today=TODAY)
    html = cav.render_milestones_section(m)
    assert ">Wonderdrug</a>" in html
    assert ">NCT00000001</a>" not in html
    assert 'title="NCT00000001"' in html  # NCT ID preserved as a hover reference


def test_milestone_materially_delayed_uses_canonical_drug_name_from_changes_row():
    changes = _changes_df([
        _change_row(nct_id="NCT00000001", canonical_drug_name="Wonderdrug", change_type="primary_completion_date_change",
                    old_value="2026-01-01", new_value="2026-06-01", importance="Medium"),
    ])
    annotated = _annotated_df([_annotated_row("NCT00000001", developed_drug="Wonderdrug")])
    m = ca.build_milestones(annotated, None, changes, WATCHLIST, today=TODAY)
    assert m["materially_delayed"][0]["drug_name"] == "Wonderdrug"


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
    assert "No drug-related pipeline changes currently need attention" in html


def test_render_needs_attention_section_respects_top_n():
    rows = [
        {**{c: "" for c in ca.ATTENTION_COLUMNS}, "relevance_score": 90 - i, "priority_level": "High",
         "canonical_drug_name": f"Drug{i}", "nct_id": f"NCT{i:08d}", "relevance_factors": "", "why_it_matters": ""}
        for i in range(12)
    ]
    df = pd.DataFrame(rows, columns=ca.ATTENTION_COLUMNS)
    html = cav.render_needs_attention_section(df, top_n=5)
    assert "showing top 5 of 12" in html


def test_render_needs_attention_section_filters_out_unresolved_drug_items():
    # Needs Attention is drug-only: an item whose drug couldn't be
    # resolved to a name is dropped from the feed entirely rather than
    # shown labeled by NCT ID or "(unresolved)".
    unresolved = {**{c: "" for c in ca.ATTENTION_COLUMNS}, "relevance_score": 63, "priority_level": "High",
                  "canonical_drug_name": "", "nct_id": "NCT00663026"}
    resolved = {**{c: "" for c in ca.ATTENTION_COLUMNS}, "relevance_score": 50, "priority_level": "Medium",
                "canonical_drug_name": "bapineuzumab", "nct_id": "NCT00000002"}
    df = pd.DataFrame([unresolved, resolved], columns=ca.ATTENTION_COLUMNS)
    html = cav.render_needs_attention_section(df)
    assert "(unresolved)" not in html
    assert "NCT00663026" not in html
    assert "bapineuzumab" in html
    assert "showing top 1 of 1" in html


def test_render_needs_attention_section_handles_nan_fields_without_literal_nan():
    # A blank cell that round-tripped through CSV reads back as pandas
    # float NaN (truthy in Python) -- confirms a row that DOES have a
    # resolved drug name (and so survives the drug-only filter) still
    # never renders the literal word "nan" for any other blank field.
    row = {**{c: float("nan") for c in ca.ATTENTION_COLUMNS}, "relevance_score": 63, "priority_level": "High",
           "canonical_drug_name": "bapineuzumab", "nct_id": "NCT00663026"}
    df = pd.DataFrame([row], columns=ca.ATTENTION_COLUMNS)
    html = cav.render_needs_attention_section(df)
    assert "nan" not in html.lower()
    assert "(unresolved)" not in html
    assert "NCT00663026" in html
    assert "bapineuzumab" in html


def test_render_needs_attention_section_no_longer_shows_ar1001_relevance_inline():
    # The Needs Attention card itself must stay a pure, auditable
    # point-sum display, with no per-card AR1001 relevance line.
    row = {**{c: "" for c in ca.ATTENTION_COLUMNS}, "relevance_score": 63, "priority_level": "High",
           "aribio_relevance_score": 60, "canonical_drug_name": "bapineuzumab", "nct_id": "NCT00663026"}
    df = pd.DataFrame([row], columns=ca.ATTENTION_COLUMNS)
    html = cav.render_needs_attention_section(df)
    assert "attention-ar1001" not in html
    assert "AR1001 Relevance" not in html


# ------------------------------------------------------------
# Recent Changes (raw, unscored feed)
# ------------------------------------------------------------

def test_describe_change_produces_factual_text_without_scored_factors():
    row = _change_row(change_type="status_change", old_value="Recruiting", new_value="Completed")
    text = ca.describe_change(row)
    assert "Recruiting" in text and "Completed" in text
    # no "supporting factors" clause should ever be appended -- this
    # is the unscored description, factors=[] by construction
    for phrase in ("similar profile to ar1001", "reached phase"):
        assert phrase not in text.lower()


def test_describe_change_handles_nan_drug_name_without_rendering_literal_nan():
    # A blank cell that round-tripped through CSV (e.g. via
    # update_changes_history()'s accumulated history file) reads back
    # as pandas' float NaN, which is truthy in Python -- a naive
    # `value or ""` fallback would keep NaN and str() it into the
    # literal text "nan" showing up in the rendered description.
    row = _change_row(change_type="new_trial", canonical_drug_name=float("nan"))
    text = ca.describe_change(row)
    assert "nan" not in text.lower()
    assert text == "A new trial was registered."


def test_prepare_recent_changes_adds_description_column():
    changes = _changes_df([_change_row(change_type="results_posted", old_value="NO", new_value="YES")])
    prepared = ca.prepare_recent_changes(changes, today=TODAY)
    assert "description" in prepared.columns
    assert len(prepared.iloc[0]["description"]) > 0


def test_prepare_recent_changes_sorts_by_date_most_recent_first():
    changes = _changes_df([
        _change_row(nct_id="NCT00000001", change_type="enrollment_change", importance="High") | {"detected_date": "2026-08-01"},
        _change_row(nct_id="NCT00000002", change_type="results_posted", importance="Low") | {"detected_date": "2026-08-10"},
        _change_row(nct_id="NCT00000003", change_type="status_change", importance="Medium") | {"detected_date": "2026-08-05"},
    ])
    prepared = ca.prepare_recent_changes(changes, today=TODAY)
    # Purely chronological -- the Low-importance change from 8/10 comes
    # first because it's the most recent, ahead of the High-importance
    # change from 8/1.
    assert list(prepared["nct_id"]) == ["NCT00000002", "NCT00000003", "NCT00000001"]


def test_prepare_recent_changes_same_date_rows_keep_deterministic_order():
    changes = _changes_df([
        _change_row(nct_id="NCT00000001", change_type="status_change"),
        _change_row(nct_id="NCT00000002", change_type="enrollment_change"),
    ])  # both default to the same detected_date (TODAY)
    prepared1 = ca.prepare_recent_changes(changes, today=TODAY)
    prepared2 = ca.prepare_recent_changes(changes, today=TODAY)
    assert list(prepared1["nct_id"]) == list(prepared2["nct_id"])


def test_prepare_recent_changes_empty_input():
    prepared = ca.prepare_recent_changes(None, today=TODAY)
    assert len(prepared) == 0
    assert "description" in prepared.columns


def test_prepare_recent_changes_excludes_changes_older_than_30_days():
    changes = _changes_df([
        _change_row(nct_id="NCT00000001", change_type="status_change", importance="Medium")
        | {"detected_date": "2026-07-01"},  # 44 days before TODAY -- outside the 30-day window
        _change_row(nct_id="NCT00000002", change_type="results_posted", importance="Medium")
        | {"detected_date": "2026-08-01"},  # 13 days before TODAY -- inside the window
    ])
    prepared = ca.prepare_recent_changes(changes, today=TODAY)
    assert list(prepared["nct_id"]) == ["NCT00000002"]


def test_prepare_recent_changes_empty_when_nothing_in_window():
    changes = _changes_df([
        _change_row(nct_id="NCT00000001", change_type="status_change") | {"detected_date": "2026-01-01"},
    ])
    prepared = ca.prepare_recent_changes(changes, today=TODAY)
    assert len(prepared) == 0
    assert "description" in prepared.columns


def test_build_drug_to_nct_lookup_maps_developed_drug_to_nct_id():
    annotated = pd.DataFrame([
        {"nct_id": "NCT00000001", "developed_drug": "DrugA"},
        {"nct_id": "NCT00000002", "developed_drug": "DrugB"},
        {"nct_id": "NCT00000003", "developed_drug": ""},  # blank -- skipped
        {"nct_id": "NCT00000004", "developed_drug": float("nan")},  # NaN -- skipped
    ])
    lookup = ca.build_drug_to_nct_lookup(annotated)
    assert lookup == {"DrugA": "NCT00000001", "DrugB": "NCT00000002"}


def test_build_drug_to_nct_lookup_keeps_first_trial_when_drug_has_several():
    annotated = pd.DataFrame([
        {"nct_id": "NCT00000001", "developed_drug": "DrugA"},
        {"nct_id": "NCT00000002", "developed_drug": "DrugA"},
    ])
    lookup = ca.build_drug_to_nct_lookup(annotated)
    assert lookup["DrugA"] == "NCT00000001"


def test_prepare_recent_changes_backfills_nct_id_for_drug_level_changes():
    # Drug-level changes (e.g. new_drug) carry nct_id == "" by
    # construction -- confirms a real trial link is backfilled from
    # drug_nct_lookup instead of staying blank ("no linked trial")
    # when the drug is demonstrably associated with a real trial.
    changes = _changes_df([
        _change_row(entity_type="drug", nct_id="", canonical_drug_name="DNL921", change_type="new_drug"),
    ])
    lookup = {"DNL921": "NCT07758595"}
    prepared = ca.prepare_recent_changes(changes, today=TODAY, drug_nct_lookup=lookup)
    assert prepared.iloc[0]["nct_id"] == "NCT07758595"


def test_prepare_recent_changes_never_overwrites_an_existing_nct_id():
    changes = _changes_df([
        _change_row(nct_id="NCT00000009", canonical_drug_name="DrugA"),
    ])
    lookup = {"DrugA": "NCT99999999"}  # deliberately different -- must not override a real nct_id
    prepared = ca.prepare_recent_changes(changes, today=TODAY, drug_nct_lookup=lookup)
    assert prepared.iloc[0]["nct_id"] == "NCT00000009"


def test_prepare_recent_changes_leaves_nct_id_blank_when_drug_not_in_lookup():
    changes = _changes_df([
        _change_row(entity_type="drug", nct_id="", canonical_drug_name="UnmappedDrug", change_type="new_drug"),
    ])
    prepared = ca.prepare_recent_changes(changes, today=TODAY, drug_nct_lookup={"OtherDrug": "NCT00000001"})
    assert not prepared.iloc[0]["nct_id"]


def test_recent_changes_card_shows_backfilled_trial_link_not_no_linked_trial():
    changes = _changes_df([
        _change_row(entity_type="drug", nct_id="", canonical_drug_name="DNL921", change_type="new_drug"),
    ])
    prepared = ca.prepare_recent_changes(changes, today=TODAY, drug_nct_lookup={"DNL921": "NCT07758595"})
    html = cav.render_recent_changes_section(prepared)
    assert "no linked trial" not in html
    assert 'href="https://clinicaltrials.gov/study/NCT07758595"' in html


def test_recent_changes_card_shows_detected_date():
    changes = _changes_df([
        _change_row(nct_id="NCT00000001", canonical_drug_name="Drug1") | {"detected_date": "2026-08-01"},
    ])
    prepared = ca.prepare_recent_changes(changes, today=TODAY)
    html = cav.render_recent_changes_section(prepared)
    assert "2026-08-01" in html


def test_update_changes_history_appends_and_prunes_old_rows():
    import os
    import tempfile
    history_path = os.path.join(tempfile.mkdtemp(), "pipeline_changes_history.csv")
    try:
        # First "run": one change 44 days before TODAY -- should be pruned
        # away once a later run's window no longer covers it.
        old_run = _changes_df([
            _change_row(nct_id="NCT00000001", change_type="status_change") | {"detected_date": "2026-07-01"}
        ])
        combined1 = ca.update_changes_history(old_run, history_path, today="2026-07-01")
        assert len(combined1) == 1

        # Second "run", 44 days later: the old row falls outside the
        # trailing 30-day window and must be pruned from disk, while the
        # new row is appended.
        new_run = _changes_df([
            _change_row(nct_id="NCT00000002", change_type="results_posted") | {"detected_date": TODAY}
        ])
        combined2 = ca.update_changes_history(new_run, history_path, today=TODAY)
        assert list(combined2["nct_id"]) == ["NCT00000002"]

        # Confirm the prune was actually persisted to disk, not just
        # returned in memory.
        on_disk = pd.read_csv(history_path)
        assert list(on_disk["nct_id"]) == ["NCT00000002"]
    finally:
        import shutil
        shutil.rmtree(os.path.dirname(history_path), ignore_errors=True)


def test_update_changes_history_deduplicates_same_day_reruns():
    import os
    import tempfile
    history_path = os.path.join(tempfile.mkdtemp(), "pipeline_changes_history.csv")
    try:
        run = _changes_df([
            _change_row(nct_id="NCT00000001", change_type="status_change", old_value="Recruiting", new_value="Completed")
        ])
        ca.update_changes_history(run, history_path, today=TODAY)
        # Re-running the pipeline again the same day with an identical
        # detected change must not double the row count.
        combined = ca.update_changes_history(run, history_path, today=TODAY)
        assert len(combined) == 1
    finally:
        import shutil
        shutil.rmtree(os.path.dirname(history_path), ignore_errors=True)


def test_render_recent_changes_section_handles_empty_dataframe():
    html = cav.render_recent_changes_section(pd.DataFrame(columns=["description"]))
    assert "Recent Changes" in html
    assert "No drug-related pipeline changes were detected" in html


def test_render_recent_changes_section_shows_change_and_drug():
    changes = _changes_df([
        _change_row(change_type="results_posted", old_value="NO", new_value="YES", canonical_drug_name="TestDrug"),
    ])
    prepared = ca.prepare_recent_changes(changes, today=TODAY)
    html = cav.render_recent_changes_section(prepared)
    assert "TestDrug" in html
    assert "results posted" in html.lower()


def test_render_recent_changes_section_filters_out_unresolved_drug_changes():
    # Recent Changes is drug-only: a change whose drug couldn't be
    # resolved to a name is dropped from the feed entirely rather than
    # shown labeled by NCT ID or "(unresolved)".
    changes = _changes_df([
        _change_row(change_type="new_trial", nct_id="NCT00663026", canonical_drug_name=""),
        _change_row(change_type="new_trial", nct_id="NCT00000099", canonical_drug_name="TestDrug"),
    ])
    prepared = ca.prepare_recent_changes(changes, today=TODAY)
    html = cav.render_recent_changes_section(prepared)
    assert "(unresolved)" not in html
    assert "NCT00663026" not in html
    assert "TestDrug" in html
    assert "showing 1 of 1" in html


def test_render_recent_changes_section_handles_nan_fields_without_literal_nan():
    # Simulates a row that round-tripped through CSV (blank cells read
    # back as pandas float NaN, not "") -- the real failure mode this
    # session hit via outputs/pipeline_changes_history.csv. The drug
    # name itself is resolved here (so the row survives the drug-only
    # filter) -- this confirms no OTHER blank field renders "nan".
    df = pd.DataFrame([{
        "nct_id": "NCT07756294", "canonical_drug_name": "DNL921",
        "sponsor_or_company": float("nan"), "change_type": "new_trial",
        "description": "A new trial was registered for DNL921.",
    }])
    html = cav.render_recent_changes_section(df)
    assert "nan" not in html.lower()
    assert "(unresolved)" not in html
    assert "NCT07756294" in html
    assert 'href="https://clinicaltrials.gov/study/NCT07756294"' in html


def test_render_recent_changes_section_respects_top_n():
    rows = [
        _change_row(nct_id=f"NCT{i:08d}", canonical_drug_name=f"Drug{i}", change_type="status_change")
        for i in range(20)
    ]
    prepared = ca.prepare_recent_changes(_changes_df(rows), today=TODAY)
    html = cav.render_recent_changes_section(prepared, top_n=5)
    assert "showing 5 of 20" in html


def test_shown_recent_change_nct_ids_matches_the_rendered_top_n():
    changes = _changes_df([
        _change_row(nct_id="NCT00000001", canonical_drug_name="Drug1", importance="High"),
        _change_row(nct_id="NCT00000002", canonical_drug_name="Drug2", importance="Medium"),
        _change_row(nct_id="NCT00000003", canonical_drug_name="", importance="High"),  # unresolved -- excluded
    ])
    prepared = ca.prepare_recent_changes(changes, today=TODAY)
    shown = cav.shown_recent_change_nct_ids(prepared, top_n=1)
    # Only the single highest-importance, drug-resolved row is "shown".
    assert shown == {"NCT00000001"}


def test_needs_attention_deduplicates_items_already_shown_in_recent_changes():
    changes = _changes_df([
        _change_row(nct_id="NCT00000001", canonical_drug_name="DupDrug", importance="High"),
    ])
    recent_changes_df = ca.prepare_recent_changes(changes, today=TODAY)

    dup_row = {**{c: "" for c in ca.ATTENTION_COLUMNS}, "relevance_score": 90, "priority_level": "High",
               "canonical_drug_name": "DupDrug", "nct_id": "NCT00000001"}
    unique_row = {**{c: "" for c in ca.ATTENTION_COLUMNS}, "relevance_score": 80, "priority_level": "High",
                  "canonical_drug_name": "UniqueDrug", "nct_id": "NCT00000002"}
    attention_df = pd.DataFrame([dup_row, unique_row], columns=ca.ATTENTION_COLUMNS)

    html = cav.render_competitive_sections(recent_changes_df, attention_df)
    assert html.count("DupDrug") == 1  # shown in Recent Changes only, not repeated in Needs Attention
    assert html.count("UniqueDrug") == 1  # not a duplicate, shown normally in Needs Attention


def test_needs_attention_shows_explanatory_empty_state_when_everything_is_a_duplicate():
    changes = _changes_df([
        _change_row(nct_id="NCT00000001", canonical_drug_name="DupDrug", importance="High"),
    ])
    recent_changes_df = ca.prepare_recent_changes(changes, today=TODAY)
    dup_row = {**{c: "" for c in ca.ATTENTION_COLUMNS}, "relevance_score": 90, "priority_level": "High",
               "canonical_drug_name": "DupDrug", "nct_id": "NCT00000001"}
    attention_df = pd.DataFrame([dup_row], columns=ca.ATTENTION_COLUMNS)

    html = cav.render_competitive_sections(recent_changes_df, attention_df)
    assert "already shown in Recent Changes above" in html


def test_render_competitive_sections_includes_placeholder_replaceable_content():
    # Milestones is deliberately NOT part of this bundle anymore — it
    # renders separately (see test_render_milestones_section_* below)
    # into its own placeholder further down the page, beneath "AR1001
    # Competitive Landscape".
    html = cav.render_competitive_sections(
        pd.DataFrame(columns=ca.ATTENTION_COLUMNS),
        pd.DataFrame(columns=ca.ATTENTION_COLUMNS),
    )
    assert "AR1001 Relevance" not in html  # removed section must not reappear
    assert "Recent Changes" in html
    assert "Needs Attention" in html
    assert "Upcoming Competitive Milestones" not in html
    assert cav.PLACEHOLDER not in html  # the rendered section itself must not contain the raw token


def test_needs_attention_section_has_notes_edit_button():
    html = cav.render_needs_attention_section(pd.DataFrame(columns=ca.ATTENTION_COLUMNS))
    assert 'id="attentionNotesEditBtn"' in html
    assert "openAttentionNotesModal()" in html
    assert 'id="attentionNotesModalOverlay"' in html
    assert 'id="attentionNotesTextarea"' in html
    assert "saveAttentionNotes()" in html
    assert "cancelAttentionNotesEdit()" in html
    assert 'id="attentionNotesList"' in html


def test_notes_modal_has_separate_drug_name_and_general_notes_sections():
    html = cav.render_needs_attention_section(pd.DataFrame(columns=ca.ATTENTION_COLUMNS))
    assert 'id="attentionNotesDrugInput"' in html
    assert "Drug name" in html
    assert "General notes" in html
    # Two distinct labeled fields, not one combined field.
    assert html.count('class="attention-notes-field"') == 2


def test_recent_changes_section_has_no_notes_edit_button():
    # The notes feature was requested for Needs Attention specifically.
    html = cav.render_recent_changes_section(pd.DataFrame(columns=["description"]))
    assert "attentionNotesEditBtn" not in html


def test_attention_notes_script_included_exactly_once():
    html = cav.render_competitive_sections(
        pd.DataFrame(columns=ca.ATTENTION_COLUMNS),
        pd.DataFrame(columns=ca.ATTENTION_COLUMNS),
    )
    assert html.count("<script>") == 1
    assert html.count("saveAttentionNotes = function") == 1


def test_attention_notes_persist_via_localstorage_not_page_markup():
    # This is the specific failure mode that destroyed the original
    # version of this feature: notes must never be baked into the
    # page's own HTML, since the page is fully regenerated from
    # scratch on every pipeline refresh (see run_pipeline.py) --
    # anything living only in the HTML would be silently discarded the
    # next time the page is rebuilt. localStorage lives in the
    # browser, independent of the file's content on disk.
    assert "localStorage.setItem" in cav.ATTENTION_NOTES_SCRIPT


def test_attention_notes_are_a_dated_deletable_list_not_a_single_blob():
    # Matches the described interaction: Save APPENDS a new dated entry
    # (not overwrite), and each entry is individually deletable.
    assert "JSON.stringify" in cav.ATTENTION_NOTES_SCRIPT
    assert "JSON.parse" in cav.ATTENTION_NOTES_SCRIPT
    assert "notes.push(" in cav.ATTENTION_NOTES_SCRIPT
    assert "date:" in cav.ATTENTION_NOTES_SCRIPT
    assert "deleteAttentionNote" in cav.ATTENTION_NOTES_SCRIPT
    assert "attention-note-delete-btn" in cav.ATTENTION_NOTES_SCRIPT


def test_attention_notes_capture_drug_name_alongside_general_notes():
    assert "drug:" in cav.ATTENTION_NOTES_SCRIPT
    assert "attentionNotesDrugInput" in cav.ATTENTION_NOTES_SCRIPT
    assert "attention-note-drug" in cav.ATTENTION_NOTES_SCRIPT
    # Drug name is optional -- only the notes textarea gates whether
    # Save actually appends a new entry.
    assert 'if (text) {' in cav.ATTENTION_NOTES_SCRIPT
    assert "localStorage.getItem" in cav.ATTENTION_NOTES_SCRIPT


def test_deleting_a_note_requires_confirmation():
    delete_fn = cav.ATTENTION_NOTES_SCRIPT.split("window.deleteAttentionNote = function")[1].split("};")[0]
    assert "window.confirm(" in delete_fn
    # Confirmation must gate the actual deletion -- the filter/save call
    # has to appear AFTER the confirm() check, not before it.
    assert delete_fn.index("window.confirm(") < delete_fn.index("setNotes(")


def test_render_competitive_sections_stacks_needs_attention_below_recent_changes():
    # Recent Changes renders first (on top), Needs Attention second
    # (below it) -- both full-width, single-column, not side by side.
    html = cav.render_competitive_sections(
        pd.DataFrame(columns=ca.ATTENTION_COLUMNS),
        pd.DataFrame(columns=ca.ATTENTION_COLUMNS),
    )
    row_start = html.index('<div class="attention-row">')
    assert row_start < html.index("Recent Changes") < html.index("Needs Attention")


def test_attention_row_css_is_single_column():
    assert "grid-template-columns: 1fr" in cav.COMPETITIVE_ATTENTION_CSS
    assert "grid-template-columns: repeat(2, 1fr)" not in cav.COMPETITIVE_ATTENTION_CSS.split(".attention-cards-grid")[0]


def test_recent_changes_cards_render_in_two_column_grid():
    changes = _changes_df([
        _change_row(nct_id="NCT00000001", canonical_drug_name="Drug1"),
        _change_row(nct_id="NCT00000002", canonical_drug_name="Drug2"),
    ])
    prepared = ca.prepare_recent_changes(changes, today=TODAY)
    html = cav.render_recent_changes_section(prepared)
    assert '<div class="attention-cards-grid">' in html
    # Both cards must live inside the same grid wrapper, not two separate ones.
    assert html.count('<div class="attention-cards-grid">') == 1
    assert html.count('<div class="attention-card js-drug-row">') == 2


def test_needs_attention_cards_are_not_wrapped_in_two_column_grid():
    # The two-column layout was requested for Recent Changes only --
    # Needs Attention keeps its existing single-column list.
    row = {**{c: "" for c in ca.ATTENTION_COLUMNS}, "relevance_score": 63, "priority_level": "High",
           "canonical_drug_name": "bapineuzumab", "nct_id": "NCT00663026"}
    df = pd.DataFrame([row], columns=ca.ATTENTION_COLUMNS)
    html = cav.render_needs_attention_section(df)
    assert "attention-cards-grid" not in html


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
    test_named_competitor_sponsor_awards_points_and_factor,
    test_named_competitor_word_overlap_avoids_false_positive,
    test_named_competitor_no_match_when_list_empty,
    test_named_competitor_matches_via_drug_level_sponsor,
    test_needs_attention_always_shows_named_competitor_beyond_top_n,
    test_scoring_is_deterministic_across_repeated_runs,
    test_low_confidence_penalty_is_deterministic_and_reduces_score,
    test_priority_rank_is_assigned_in_score_descending_order,
    test_output_columns_match_required_schema,
    test_empty_changes_produces_empty_result_with_correct_columns,
    test_milestone_next_30_days_bucket,
    test_milestone_item_carries_phase_from_annotated_df,
    test_milestone_item_carries_drug_type_when_drugs_df_given,
    test_milestone_item_drug_type_blank_when_drugs_df_not_given,
    test_materially_delayed_milestone_item_carries_phase_and_drug_type,
    test_render_milestones_section_shows_phase_and_drug_type,
    test_render_milestones_section_omits_meta_line_when_both_blank,
    test_milestone_item_carries_resolved_drug_name,
    test_milestone_item_falls_back_to_nct_id_when_drug_unresolved,
    test_milestone_rendering_shows_drug_name_not_nct_id_as_link_text,
    test_milestone_materially_delayed_uses_canonical_drug_name_from_changes_row,
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
    test_render_needs_attention_section_no_longer_shows_ar1001_relevance_inline,
    test_render_needs_attention_section_filters_out_unresolved_drug_items,
    test_render_needs_attention_section_handles_nan_fields_without_literal_nan,
    test_describe_change_produces_factual_text_without_scored_factors,
    test_describe_change_handles_nan_drug_name_without_rendering_literal_nan,
    test_prepare_recent_changes_adds_description_column,
    test_prepare_recent_changes_sorts_by_date_most_recent_first,
    test_prepare_recent_changes_same_date_rows_keep_deterministic_order,
    test_prepare_recent_changes_empty_input,
    test_prepare_recent_changes_excludes_changes_older_than_30_days,
    test_prepare_recent_changes_empty_when_nothing_in_window,
    test_build_drug_to_nct_lookup_maps_developed_drug_to_nct_id,
    test_build_drug_to_nct_lookup_keeps_first_trial_when_drug_has_several,
    test_prepare_recent_changes_backfills_nct_id_for_drug_level_changes,
    test_prepare_recent_changes_never_overwrites_an_existing_nct_id,
    test_prepare_recent_changes_leaves_nct_id_blank_when_drug_not_in_lookup,
    test_recent_changes_card_shows_backfilled_trial_link_not_no_linked_trial,
    test_recent_changes_card_shows_detected_date,
    test_update_changes_history_appends_and_prunes_old_rows,
    test_update_changes_history_deduplicates_same_day_reruns,
    test_render_recent_changes_section_handles_empty_dataframe,
    test_render_recent_changes_section_shows_change_and_drug,
    test_render_recent_changes_section_filters_out_unresolved_drug_changes,
    test_render_recent_changes_section_handles_nan_fields_without_literal_nan,
    test_render_recent_changes_section_respects_top_n,
    test_shown_recent_change_nct_ids_matches_the_rendered_top_n,
    test_needs_attention_deduplicates_items_already_shown_in_recent_changes,
    test_needs_attention_shows_explanatory_empty_state_when_everything_is_a_duplicate,
    test_render_competitive_sections_includes_placeholder_replaceable_content,
    test_needs_attention_section_has_notes_edit_button,
    test_notes_modal_has_separate_drug_name_and_general_notes_sections,
    test_recent_changes_section_has_no_notes_edit_button,
    test_attention_notes_script_included_exactly_once,
    test_attention_notes_persist_via_localstorage_not_page_markup,
    test_attention_notes_are_a_dated_deletable_list_not_a_single_blob,
    test_attention_notes_capture_drug_name_alongside_general_notes,
    test_deleting_a_note_requires_confirmation,
    test_render_competitive_sections_stacks_needs_attention_below_recent_changes,
    test_attention_row_css_is_single_column,
    test_recent_changes_cards_render_in_two_column_grid,
    test_needs_attention_cards_are_not_wrapped_in_two_column_grid,
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
