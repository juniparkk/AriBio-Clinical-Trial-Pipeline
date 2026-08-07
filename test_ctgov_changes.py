# ============================================================
# TESTS for ctgov_changes.py (snapshot-based pipeline change detection)
#
# One synthetic test per major change_type the module is required to
# detect, plus formatting-only-change suppression and importance/
# needs_review grading. Plain-Python tests (no pytest install needed,
# matching this project's other test files) — run with:
#     .venv/bin/python test_ctgov_changes.py
# ============================================================

import pandas as pd

import ctgov_changes as cc

DETECTED_DATE = "2026-08-14"
SOURCE = "ClinicalTrials.gov API v2 (snapshot test)"


def _trial_row(nct_id, status="RECRUITING", phases="PHASE2", enrollment=100,
                sponsor="Acme Pharma", primary_completion="2027-01", completion="2027-06",
                results="NO"):
    return {
        "NCT Number": nct_id,
        "Study Status": status,
        "Phases": phases,
        "Enrollment": enrollment,
        "Sponsor": sponsor,
        "Primary Completion Date": primary_completion,
        "Completion Date": completion,
        "Study Results": results,
    }


def _trials_df(rows):
    return pd.DataFrame(rows)


def _drug_row(display_name, phase_reached="Phase 2", sponsor="Acme Pharma"):
    return {"display_name": display_name, "phase_reached": phase_reached, "sponsor": sponsor}


def _drugs_df(rows):
    return pd.DataFrame(rows)


def _annotated_df(mapping):
    return pd.DataFrame([{"nct_id": k, "developed_drug": v} for k, v in mapping.items()])


def _by_type(df, change_type):
    return df[df["change_type"] == change_type]


# ------------------------------------------------------------
# new_trial
# ------------------------------------------------------------

def test_new_trial_phase_2_or_3_is_high_importance():
    old_df = _trials_df([_trial_row("NCT00000001")])
    new_df = _trials_df([_trial_row("NCT00000001"), _trial_row("NCT00000002", phases="PHASE3")])
    changes = cc.detect_changes(old_df, new_df, None, None, None, None, DETECTED_DATE, SOURCE)
    rows = _by_type(changes, "new_trial")
    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["nct_id"] == "NCT00000002"
    assert row["importance"] == "High"
    assert row["needs_review"] == True
    assert row["new_value"] == "Phase 3"


def test_new_trial_phase_1_is_medium_importance():
    old_df = _trials_df([_trial_row("NCT00000001")])
    new_df = _trials_df([_trial_row("NCT00000001"), _trial_row("NCT00000002", phases="PHASE1")])
    changes = cc.detect_changes(old_df, new_df, None, None, None, None, DETECTED_DATE, SOURCE)
    row = _by_type(changes, "new_trial").iloc[0]
    assert row["importance"] == "Medium"
    assert row["needs_review"] == False


def test_new_trial_carries_canonical_drug_name_from_annotated_lookup():
    old_df = _trials_df([_trial_row("NCT00000001")])
    new_df = _trials_df([_trial_row("NCT00000001"), _trial_row("NCT00000002", phases="PHASE2")])
    new_annotated = _annotated_df({"NCT00000002": "wonderdrug"})
    changes = cc.detect_changes(old_df, new_df, None, None, None, new_annotated, DETECTED_DATE, SOURCE)
    row = _by_type(changes, "new_trial").iloc[0]
    assert row["canonical_drug_name"] == "wonderdrug"


# ------------------------------------------------------------
# trial_disappeared
# ------------------------------------------------------------

def test_trial_disappeared_when_nct_id_missing_from_new_snapshot():
    old_df = _trials_df([_trial_row("NCT00000001"), _trial_row("NCT00000002")])
    new_df = _trials_df([_trial_row("NCT00000001")])
    changes = cc.detect_changes(old_df, new_df, None, None, None, None, DETECTED_DATE, SOURCE)
    rows = _by_type(changes, "trial_disappeared")
    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["nct_id"] == "NCT00000002"
    assert row["needs_review"] == True
    assert row["importance"] == "Medium"


# ------------------------------------------------------------
# status_change
# ------------------------------------------------------------

def test_status_change_detected_and_medium_importance():
    old_df = _trials_df([_trial_row("NCT00000001", status="RECRUITING")])
    new_df = _trials_df([_trial_row("NCT00000001", status="COMPLETED")])
    changes = cc.detect_changes(old_df, new_df, None, None, None, None, DETECTED_DATE, SOURCE)
    row = _by_type(changes, "status_change").iloc[0]
    assert row["old_value"] == "Recruiting"
    assert row["new_value"] == "Completed"
    assert row["importance"] == "Medium"


def test_status_change_to_terminated_does_not_imply_failure_stays_medium():
    # Explicit project rule: never infer failure from termination alone —
    # a TERMINATED status change gets no special elevated importance or
    # editorialized value, just the same factual Medium status_change.
    old_df = _trials_df([_trial_row("NCT00000001", status="RECRUITING")])
    new_df = _trials_df([_trial_row("NCT00000001", status="TERMINATED")])
    changes = cc.detect_changes(old_df, new_df, None, None, None, None, DETECTED_DATE, SOURCE)
    row = _by_type(changes, "status_change").iloc[0]
    assert row["new_value"] == "Discontinued"
    assert row["importance"] == "Medium"


def test_status_change_formatting_only_is_ignored():
    old_df = _trials_df([_trial_row("NCT00000001", status="recruiting")])
    new_df = _trials_df([_trial_row("NCT00000001", status="RECRUITING")])
    changes = cc.detect_changes(old_df, new_df, None, None, None, None, DETECTED_DATE, SOURCE)
    assert len(_by_type(changes, "status_change")) == 0


# ------------------------------------------------------------
# phase_change
# ------------------------------------------------------------

def test_phase_change_advancement_is_high_importance():
    old_df = _trials_df([_trial_row("NCT00000001", phases="PHASE1")])
    new_df = _trials_df([_trial_row("NCT00000001", phases="PHASE2")])
    changes = cc.detect_changes(old_df, new_df, None, None, None, None, DETECTED_DATE, SOURCE)
    row = _by_type(changes, "phase_change").iloc[0]
    assert row["old_value"] == "Phase 1"
    assert row["new_value"] == "Phase 2"
    assert row["importance"] == "High"
    assert row["needs_review"] == True


def test_phase_change_regression_is_medium_not_high():
    old_df = _trials_df([_trial_row("NCT00000001", phases="PHASE3")])
    new_df = _trials_df([_trial_row("NCT00000001", phases="PHASE2")])
    changes = cc.detect_changes(old_df, new_df, None, None, None, None, DETECTED_DATE, SOURCE)
    row = _by_type(changes, "phase_change").iloc[0]
    assert row["importance"] == "Medium"
    assert row["needs_review"] == False


# ------------------------------------------------------------
# enrollment_change
# ------------------------------------------------------------

def test_enrollment_change_substantial_is_medium():
    old_df = _trials_df([_trial_row("NCT00000001", enrollment=100)])
    new_df = _trials_df([_trial_row("NCT00000001", enrollment=150)])  # 33% relative change
    changes = cc.detect_changes(old_df, new_df, None, None, None, None, DETECTED_DATE, SOURCE)
    row = _by_type(changes, "enrollment_change").iloc[0]
    assert row["old_value"] == "100"
    assert row["new_value"] == "150"
    assert row["importance"] == "Medium"


def test_enrollment_change_small_is_low():
    old_df = _trials_df([_trial_row("NCT00000001", enrollment=100)])
    new_df = _trials_df([_trial_row("NCT00000001", enrollment=102)])  # 2% relative change
    changes = cc.detect_changes(old_df, new_df, None, None, None, None, DETECTED_DATE, SOURCE)
    row = _by_type(changes, "enrollment_change").iloc[0]
    assert row["importance"] == "Low"


def test_enrollment_unchanged_produces_no_row():
    old_df = _trials_df([_trial_row("NCT00000001", enrollment=100)])
    new_df = _trials_df([_trial_row("NCT00000001", enrollment=100)])
    changes = cc.detect_changes(old_df, new_df, None, None, None, None, DETECTED_DATE, SOURCE)
    assert len(_by_type(changes, "enrollment_change")) == 0


def test_enrollment_int_vs_float_formatting_is_not_a_spurious_change():
    old_df = _trials_df([_trial_row("NCT00000001", enrollment=100)])
    new_df = _trials_df([_trial_row("NCT00000001", enrollment=100.0)])
    changes = cc.detect_changes(old_df, new_df, None, None, None, None, DETECTED_DATE, SOURCE)
    assert len(_by_type(changes, "enrollment_change")) == 0


# ------------------------------------------------------------
# primary_completion_date_change / completion_date_change
# ------------------------------------------------------------

def test_primary_completion_date_substantial_shift_is_medium():
    old_df = _trials_df([_trial_row("NCT00000001", primary_completion="2027-01-01")])
    new_df = _trials_df([_trial_row("NCT00000001", primary_completion="2027-06-01")])
    changes = cc.detect_changes(old_df, new_df, None, None, None, None, DETECTED_DATE, SOURCE)
    row = _by_type(changes, "primary_completion_date_change").iloc[0]
    assert row["importance"] == "Medium"


def test_primary_completion_date_small_shift_is_low():
    old_df = _trials_df([_trial_row("NCT00000001", primary_completion="2027-01-01")])
    new_df = _trials_df([_trial_row("NCT00000001", primary_completion="2027-01-05")])
    changes = cc.detect_changes(old_df, new_df, None, None, None, None, DETECTED_DATE, SOURCE)
    row = _by_type(changes, "primary_completion_date_change").iloc[0]
    assert row["importance"] == "Low"


def test_study_completion_date_change_detected_independently():
    old_df = _trials_df([_trial_row("NCT00000001", completion="2027-06-01")])
    new_df = _trials_df([_trial_row("NCT00000001", completion="2028-06-01")])
    changes = cc.detect_changes(old_df, new_df, None, None, None, None, DETECTED_DATE, SOURCE)
    rows = _by_type(changes, "completion_date_change")
    assert len(rows) == 1
    assert rows.iloc[0]["importance"] == "Medium"


# ------------------------------------------------------------
# sponsor_change
# ------------------------------------------------------------

def test_sponsor_change_detected_medium_and_needs_review():
    old_df = _trials_df([_trial_row("NCT00000001", sponsor="Old Sponsor Inc.")])
    new_df = _trials_df([_trial_row("NCT00000001", sponsor="New Sponsor LLC")])
    changes = cc.detect_changes(old_df, new_df, None, None, None, None, DETECTED_DATE, SOURCE)
    row = _by_type(changes, "sponsor_change").iloc[0]
    assert row["importance"] == "Medium"
    assert row["needs_review"] == True


def test_sponsor_change_case_and_whitespace_only_is_ignored():
    old_df = _trials_df([_trial_row("NCT00000001", sponsor="  Acme  Pharma ")])
    new_df = _trials_df([_trial_row("NCT00000001", sponsor="ACME PHARMA")])
    changes = cc.detect_changes(old_df, new_df, None, None, None, None, DETECTED_DATE, SOURCE)
    assert len(_by_type(changes, "sponsor_change")) == 0


# ------------------------------------------------------------
# results_posted
# ------------------------------------------------------------

def test_results_newly_posted_is_high_importance():
    old_df = _trials_df([_trial_row("NCT00000001", results="NO")])
    new_df = _trials_df([_trial_row("NCT00000001", results="YES")])
    changes = cc.detect_changes(old_df, new_df, None, None, None, None, DETECTED_DATE, SOURCE)
    row = _by_type(changes, "results_posted").iloc[0]
    assert row["importance"] == "High"
    assert row["needs_review"] == True


def test_results_reversal_yes_to_no_is_not_flagged():
    # Only the NEWLY-posted direction is detected, per spec.
    old_df = _trials_df([_trial_row("NCT00000001", results="YES")])
    new_df = _trials_df([_trial_row("NCT00000001", results="NO")])
    changes = cc.detect_changes(old_df, new_df, None, None, None, None, DETECTED_DATE, SOURCE)
    assert len(_by_type(changes, "results_posted")) == 0


# ------------------------------------------------------------
# new_drug / highest_drug_phase_change (drug-level, from pipeline_drugs.csv)
# ------------------------------------------------------------

def test_new_drug_detected_medium_importance():
    old_drugs = _drugs_df([_drug_row("KnownDrug")])
    new_drugs = _drugs_df([_drug_row("KnownDrug"), _drug_row("BrandNewDrug", phase_reached="Phase 1")])
    changes = cc.detect_changes(None, None, old_drugs, new_drugs, None, None, DETECTED_DATE, SOURCE)
    rows = _by_type(changes, "new_drug")
    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["canonical_drug_name"] == "BrandNewDrug"
    assert row["new_value"] == "Phase 1"
    assert row["importance"] == "Medium"


def test_highest_drug_phase_advancement_is_high():
    old_drugs = _drugs_df([_drug_row("KnownDrug", phase_reached="Phase 1")])
    new_drugs = _drugs_df([_drug_row("KnownDrug", phase_reached="Phase 3")])
    changes = cc.detect_changes(None, None, old_drugs, new_drugs, None, None, DETECTED_DATE, SOURCE)
    row = _by_type(changes, "highest_drug_phase_change").iloc[0]
    assert row["old_value"] == "Phase 1"
    assert row["new_value"] == "Phase 3"
    assert row["importance"] == "High"
    assert row["needs_review"] == True


def test_highest_drug_phase_regression_is_medium():
    old_drugs = _drugs_df([_drug_row("KnownDrug", phase_reached="Phase 3")])
    new_drugs = _drugs_df([_drug_row("KnownDrug", phase_reached="Phase 2")])
    changes = cc.detect_changes(None, None, old_drugs, new_drugs, None, None, DETECTED_DATE, SOURCE)
    row = _by_type(changes, "highest_drug_phase_change").iloc[0]
    assert row["importance"] == "Medium"
    assert row["needs_review"] == False


def test_drug_unchanged_produces_no_row():
    old_drugs = _drugs_df([_drug_row("KnownDrug", phase_reached="Phase 2")])
    new_drugs = _drugs_df([_drug_row("KnownDrug", phase_reached="Phase 2")])
    changes = cc.detect_changes(None, None, old_drugs, new_drugs, None, None, DETECTED_DATE, SOURCE)
    assert len(changes) == 0


# ------------------------------------------------------------
# graceful handling of missing "previous" state (first-ever refresh)
# ------------------------------------------------------------

def test_no_previous_trials_df_produces_no_trial_level_changes():
    new_df = _trials_df([_trial_row("NCT00000001")])
    changes = cc.detect_changes(None, new_df, None, None, None, None, DETECTED_DATE, SOURCE)
    assert len(changes) == 0


def test_empty_previous_trials_df_produces_no_trial_level_changes():
    old_df = _trials_df([])
    new_df = _trials_df([_trial_row("NCT00000001")])
    changes = cc.detect_changes(old_df, new_df, None, None, None, None, DETECTED_DATE, SOURCE)
    assert len(changes) == 0


def test_no_previous_drugs_df_produces_no_drug_level_changes():
    new_drugs = _drugs_df([_drug_row("BrandNewDrug")])
    changes = cc.detect_changes(None, None, None, new_drugs, None, None, DETECTED_DATE, SOURCE)
    assert len(changes) == 0


# ------------------------------------------------------------
# schema / general shape
# ------------------------------------------------------------

def test_output_columns_match_required_schema():
    old_df = _trials_df([_trial_row("NCT00000001", status="RECRUITING")])
    new_df = _trials_df([_trial_row("NCT00000001", status="COMPLETED")])
    changes = cc.detect_changes(old_df, new_df, None, None, None, None, DETECTED_DATE, SOURCE)
    assert list(changes.columns) == [
        "detected_date", "entity_type", "nct_id", "canonical_drug_name",
        "sponsor_or_company", "change_type", "old_value", "new_value",
        "importance", "source", "needs_review",
    ]


def test_trial_level_rows_carry_entity_type_trial_and_drug_level_carry_drug():
    old_df = _trials_df([_trial_row("NCT00000001", status="RECRUITING")])
    new_df = _trials_df([_trial_row("NCT00000001", status="COMPLETED")])
    old_drugs = _drugs_df([_drug_row("KnownDrug", phase_reached="Phase 1")])
    new_drugs = _drugs_df([_drug_row("KnownDrug", phase_reached="Phase 2")])
    changes = cc.detect_changes(old_df, new_df, old_drugs, new_drugs, None, None, DETECTED_DATE, SOURCE)
    assert set(_by_type(changes, "status_change")["entity_type"]) == {"trial"}
    assert set(_by_type(changes, "highest_drug_phase_change")["entity_type"]) == {"drug"}


def test_detected_date_and_source_are_stamped_on_every_row():
    old_df = _trials_df([_trial_row("NCT00000001", status="RECRUITING")])
    new_df = _trials_df([_trial_row("NCT00000001", status="COMPLETED")])
    changes = cc.detect_changes(old_df, new_df, None, None, None, None, DETECTED_DATE, SOURCE)
    assert (changes["detected_date"] == DETECTED_DATE).all()
    assert (changes["source"] == SOURCE).all()


def test_multiple_changes_on_same_trial_each_get_their_own_row():
    old_df = _trials_df([_trial_row("NCT00000001", status="RECRUITING", phases="PHASE1", enrollment=100)])
    new_df = _trials_df([_trial_row("NCT00000001", status="COMPLETED", phases="PHASE2", enrollment=180)])
    changes = cc.detect_changes(old_df, new_df, None, None, None, None, DETECTED_DATE, SOURCE)
    types = set(changes["change_type"])
    assert {"status_change", "phase_change", "enrollment_change"} <= types
    assert (changes["nct_id"] == "NCT00000001").all()


ALL_TESTS = [
    test_new_trial_phase_2_or_3_is_high_importance,
    test_new_trial_phase_1_is_medium_importance,
    test_new_trial_carries_canonical_drug_name_from_annotated_lookup,
    test_trial_disappeared_when_nct_id_missing_from_new_snapshot,
    test_status_change_detected_and_medium_importance,
    test_status_change_to_terminated_does_not_imply_failure_stays_medium,
    test_status_change_formatting_only_is_ignored,
    test_phase_change_advancement_is_high_importance,
    test_phase_change_regression_is_medium_not_high,
    test_enrollment_change_substantial_is_medium,
    test_enrollment_change_small_is_low,
    test_enrollment_unchanged_produces_no_row,
    test_enrollment_int_vs_float_formatting_is_not_a_spurious_change,
    test_primary_completion_date_substantial_shift_is_medium,
    test_primary_completion_date_small_shift_is_low,
    test_study_completion_date_change_detected_independently,
    test_sponsor_change_detected_medium_and_needs_review,
    test_sponsor_change_case_and_whitespace_only_is_ignored,
    test_results_newly_posted_is_high_importance,
    test_results_reversal_yes_to_no_is_not_flagged,
    test_new_drug_detected_medium_importance,
    test_highest_drug_phase_advancement_is_high,
    test_highest_drug_phase_regression_is_medium,
    test_drug_unchanged_produces_no_row,
    test_no_previous_trials_df_produces_no_trial_level_changes,
    test_empty_previous_trials_df_produces_no_trial_level_changes,
    test_no_previous_drugs_df_produces_no_drug_level_changes,
    test_output_columns_match_required_schema,
    test_trial_level_rows_carry_entity_type_trial_and_drug_level_carry_drug,
    test_detected_date_and_source_are_stamped_on_every_row,
    test_multiple_changes_on_same_trial_each_get_their_own_row,
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
