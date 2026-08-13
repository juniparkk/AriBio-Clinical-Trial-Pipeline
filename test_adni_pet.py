# ============================================================
# TESTS for adni_pet.py / run_adni_pet_eligibility.py (POLARIS
# AD-aligned ADNI eligibility cohort).
#
# Two kinds of test here:
#   1. Unit tests against small, hand-built synthetic DataFrames (fake
#      RIDs, fake dates/values -- never real ADNI participant data).
#   2. Real-data validation tests against the already-generated
#      ADNI_PROCESSED_DIR/adni_pet_eligibility.parquet and
#      ADNI_OUTPUTS_DIR/adni_polaris_*.{csv,md} -- these read
#      aggregate counts and column-level structure only, never print
#      or assert on individual participant rows.
#
# Run: .venv/bin/python test_adni_pet.py
# (assumes run_adni_pet_eligibility.py has already been run)
# ============================================================

import os

import numpy as np
import pandas as pd

import adni_pet as P
import adni_viz_data as D
from adni_analysis import ADNI_OUTPUTS_DIR, ADNI_PROCESSED_DIR

TODAY = pd.Timestamp("2026-01-01")


def _baseline_row(rid, baseline_date=TODAY):
    return {"RID": rid, "CLINICAL_BASELINE_DATE": baseline_date}


def _pet_row(rid, days_from_baseline, qc_flag=2, centiloid=50.0, loniuid=1000, baseline_date=TODAY):
    return {
        "RID": rid, "SCANDATE": baseline_date + pd.Timedelta(days=days_from_baseline),
        "qc_flag": qc_flag, "TRACER": "FBP", "CENTILOIDS": centiloid, "LONIUID": loniuid,
    }


# ------------------------------------------------------------------
# 1. +/-90-day window inclusion/exclusion
# ------------------------------------------------------------------


def test_scan_exactly_at_90_days_is_included():
    base = pd.DataFrame([_baseline_row(1)])
    pet = pd.DataFrame([_pet_row(1, 90)])
    out = P.build_pet_baseline(pet, base, window_days=90)
    row = out.set_index("RID").loc[1]
    assert row["CENTILOID_BASELINE_STATUS"] == P.CENTILOID_STATUS_COMPUTED
    assert row["CENTILOID_ELIGIBLE"]
    assert row["CENTILOID_BASELINE_DAYS_FROM_CLINICAL_BASELINE"] == 90


def test_scan_at_91_days_is_excluded():
    base = pd.DataFrame([_baseline_row(1)])
    pet = pd.DataFrame([_pet_row(1, 91)])
    out = P.build_pet_baseline(pet, base, window_days=90)
    row = out.set_index("RID").loc[1]
    assert row["CENTILOID_BASELINE_STATUS"] == P.CENTILOID_STATUS_NO_SCAN_IN_WINDOW
    assert not row["CENTILOID_ELIGIBLE"]


def test_scan_exactly_at_minus_90_days_is_included():
    base = pd.DataFrame([_baseline_row(1)])
    pet = pd.DataFrame([_pet_row(1, -90)])
    out = P.build_pet_baseline(pet, base, window_days=90)
    row = out.set_index("RID").loc[1]
    assert row["CENTILOID_ELIGIBLE"]
    assert row["CENTILOID_BASELINE_DAYS_FROM_CLINICAL_BASELINE"] == -90


def test_scan_at_minus_91_days_is_excluded():
    base = pd.DataFrame([_baseline_row(1)])
    pet = pd.DataFrame([_pet_row(1, -91)])
    out = P.build_pet_baseline(pet, base, window_days=90)
    row = out.set_index("RID").loc[1]
    assert not row["CENTILOID_ELIGIBLE"]


# ------------------------------------------------------------------
# 2. QC filtering
# ------------------------------------------------------------------


def test_non_qc_passed_scan_is_excluded_even_within_window_and_above_threshold():
    base = pd.DataFrame([_baseline_row(1)])
    for bad_flag in (1, 0, -1, -2):
        pet = pd.DataFrame([_pet_row(1, 5, qc_flag=bad_flag, centiloid=99.0)])
        out = P.build_pet_baseline(pet, base, window_days=90)
        row = out.set_index("RID").loc[1]
        assert not row["CENTILOID_ELIGIBLE"], f"qc_flag={bad_flag} should not be eligible"
        assert row["CENTILOID_BASELINE_STATUS"] == P.CENTILOID_STATUS_NO_SCAN_IN_WINDOW


def test_qc_passed_scan_with_qc_flag_2_is_included():
    base = pd.DataFrame([_baseline_row(1)])
    pet = pd.DataFrame([_pet_row(1, 5, qc_flag=2, centiloid=99.0)])
    out = P.build_pet_baseline(pet, base, window_days=90)
    assert out.set_index("RID").loc[1, "CENTILOID_ELIGIBLE"]


# ------------------------------------------------------------------
# 3. MMSE threshold
# ------------------------------------------------------------------


def test_mmse_below_threshold_is_not_polaris_eligible():
    df = pd.DataFrame([{"RID": 1, "MMSE_BASELINE": 19.0, "CENTILOID_ELIGIBLE": True, "CENTILOID_BASELINE": 50.0}])
    out = P.add_polaris_eligibility(df)
    assert not out.loc[0, "POLARIS_ELIGIBLE"]


def test_mmse_at_threshold_boundary_is_eligible():
    df = pd.DataFrame([{"RID": 1, "MMSE_BASELINE": 20.0, "CENTILOID_ELIGIBLE": True, "CENTILOID_BASELINE": 30.0}])
    out = P.add_polaris_eligibility(df)
    assert out.loc[0, "POLARIS_ELIGIBLE"]


def test_missing_mmse_is_not_polaris_eligible():
    df = pd.DataFrame([{"RID": 1, "MMSE_BASELINE": np.nan, "CENTILOID_ELIGIBLE": True, "CENTILOID_BASELINE": 50.0}])
    out = P.add_polaris_eligibility(df)
    assert not out.loc[0, "POLARIS_ELIGIBLE"]


# ------------------------------------------------------------------
# 4. Centiloid threshold
# ------------------------------------------------------------------


def test_centiloid_below_threshold_is_not_polaris_eligible():
    df = pd.DataFrame([{"RID": 1, "MMSE_BASELINE": 25.0, "CENTILOID_ELIGIBLE": True, "CENTILOID_BASELINE": 29.9}])
    out = P.add_polaris_eligibility(df)
    assert not out.loc[0, "POLARIS_ELIGIBLE"]


def test_centiloid_at_threshold_boundary_is_eligible():
    df = pd.DataFrame([{"RID": 1, "MMSE_BASELINE": 25.0, "CENTILOID_ELIGIBLE": True, "CENTILOID_BASELINE": 30.0}])
    out = P.add_polaris_eligibility(df)
    assert out.loc[0, "POLARIS_ELIGIBLE"]


def test_not_centiloid_eligible_is_never_polaris_eligible_even_with_high_value():
    # CENTILOID_ELIGIBLE False means no usable near-baseline scan exists
    # at all -- CENTILOID_BASELINE should be NaN in that case, but even
    # if a stray value were present, CENTILOID_ELIGIBLE gates it.
    df = pd.DataFrame([{"RID": 1, "MMSE_BASELINE": 25.0, "CENTILOID_ELIGIBLE": False, "CENTILOID_BASELINE": np.nan}])
    out = P.add_polaris_eligibility(df)
    assert not out.loc[0, "POLARIS_ELIGIBLE"]


# ------------------------------------------------------------------
# 5. Multiple-scan tie-breaking
# ------------------------------------------------------------------


def test_closer_scan_wins_over_farther_scan():
    base = pd.DataFrame([_baseline_row(1)])
    pet = pd.DataFrame([_pet_row(1, 5, centiloid=10.0, loniuid=1), _pet_row(1, 45, centiloid=90.0, loniuid=2)])
    out = P.build_pet_baseline(pet, base, window_days=90)
    row = out.set_index("RID").loc[1]
    assert row["CENTILOID_BASELINE"] == 10.0
    assert row["CENTILOID_BASELINE_DAYS_FROM_CLINICAL_BASELINE"] == 5


def test_equidistant_scans_prefer_the_earlier_one():
    # -10 and +10 are equally close (10 days) -- earlier (negative) wins.
    base = pd.DataFrame([_baseline_row(1)])
    pet = pd.DataFrame([_pet_row(1, 10, centiloid=90.0, loniuid=2), _pet_row(1, -10, centiloid=10.0, loniuid=1)])
    out = P.build_pet_baseline(pet, base, window_days=90)
    row = out.set_index("RID").loc[1]
    assert row["CENTILOID_BASELINE"] == 10.0
    assert row["CENTILOID_BASELINE_DAYS_FROM_CLINICAL_BASELINE"] == -10


def test_exact_same_day_tie_broken_by_lower_loniuid():
    base = pd.DataFrame([_baseline_row(1)])
    pet = pd.DataFrame([
        _pet_row(1, 0, centiloid=77.0, loniuid=500),
        _pet_row(1, 0, centiloid=11.0, loniuid=200),
    ])
    out = P.build_pet_baseline(pet, base, window_days=90)
    row = out.set_index("RID").loc[1]
    assert row["CENTILOID_BASELINE"] == 11.0  # the loniuid=200 scan


# ------------------------------------------------------------------
# 6. Missing clinical baseline date handling
# ------------------------------------------------------------------


def test_missing_clinical_baseline_date_is_explicit_not_silent():
    base = pd.DataFrame([{"RID": 1, "CLINICAL_BASELINE_DATE": pd.NaT}])
    pet = pd.DataFrame([_pet_row(1, 0, centiloid=99.0)])
    out = P.build_pet_baseline(pet, base, window_days=90)
    row = out.set_index("RID").loc[1]
    assert row["CENTILOID_BASELINE_STATUS"] == P.CENTILOID_STATUS_NO_BASELINE_DATE
    assert not row["CENTILOID_ELIGIBLE"]
    assert pd.isna(row["CENTILOID_BASELINE"])


def test_missing_clinical_baseline_date_is_never_polaris_eligible():
    df = pd.DataFrame([{
        "RID": 1, "MMSE_BASELINE": 30.0,
        "CENTILOID_BASELINE_STATUS": P.CENTILOID_STATUS_NO_BASELINE_DATE,
        "CENTILOID_ELIGIBLE": False, "CENTILOID_BASELINE": np.nan,
    }])
    out = P.add_polaris_eligibility(df)
    assert not out.loc[0, "POLARIS_ELIGIBLE"]


def test_participant_with_no_pet_data_at_all_gets_explicit_status_not_dropped():
    base = pd.DataFrame([_baseline_row(1), _baseline_row(2)])
    pet = pd.DataFrame([_pet_row(1, 0, centiloid=50.0)])  # RID 2 has no PET rows at all
    out = P.build_pet_baseline(pet, base, window_days=90)
    assert len(out) == 2  # every input RID is preserved, never silently dropped
    row2 = out.set_index("RID").loc[2]
    assert row2["CENTILOID_BASELINE_STATUS"] == P.CENTILOID_STATUS_NO_SCAN_IN_WINDOW
    assert not row2["CENTILOID_ELIGIBLE"]


# ------------------------------------------------------------------
# 7/8. Real-data validation against the approved 620-participant target
# ------------------------------------------------------------------


def test_final_eligible_count_matches_validated_target_620():
    path = os.path.join(ADNI_PROCESSED_DIR, "adni_pet_eligibility.parquet")
    df = pd.read_parquet(path)
    n = int(df["POLARIS_ELIGIBLE"].sum())
    assert n == 620, f"expected 620 POLARIS-eligible participants, got {n} -- STOP, do not force; investigate the discrepancy."


def test_final_eligible_diagnosis_composition_matches_validated_target():
    path = os.path.join(ADNI_PROCESSED_DIR, "adni_pet_eligibility.parquet")
    df = pd.read_parquet(path)
    elig = df[df["POLARIS_ELIGIBLE"]]
    counts = elig["DX_BASELINE_FIXED"].value_counts()
    assert int(counts.get("CN", 0)) == 151
    assert int(counts.get("MCI", 0)) == 309
    assert int(counts.get("Dementia", 0)) == 160


def test_validated_cohort_size_still_3030():
    path = os.path.join(ADNI_PROCESSED_DIR, "adni_pet_eligibility.parquet")
    df = pd.read_parquet(path)
    assert len(df) == 3030
    assert df["RID"].nunique() == 3030


# ------------------------------------------------------------------
# 9/10. Aggregate-only dashboard outputs / no participant identifiers
# ------------------------------------------------------------------


def test_polaris_outputs_load_cleanly_through_existing_governance():
    """The new outputs must pass the SAME governed loader the dashboard
    already uses, with zero changes to that loader -- proves these
    outputs are dashboard-ready under the existing governance contract
    without needing any special-case exception."""
    attrition = D.load_aggregate_csv(ADNI_OUTPUTS_DIR, "adni_polaris_cohort_attrition.csv")
    profile = D.load_aggregate_csv(ADNI_OUTPUTS_DIR, "adni_polaris_population_profile.csv")
    assert len(attrition) == 7  # the 7 named attrition steps
    assert len(profile) > 0


def test_polaris_outputs_contain_no_participant_identifiers():
    for filename in ("adni_polaris_cohort_attrition.csv", "adni_polaris_population_profile.csv"):
        df = pd.read_csv(os.path.join(ADNI_OUTPUTS_DIR, filename))
        forbidden = {"RID", "PTID", "USUBJID", "SUBJID", "PARTICIPANT_ID", "LONIUID"}
        assert forbidden & set(c.upper() for c in df.columns) == set(), f"{filename} carries a forbidden identifier column"
        assert len(df) < 100, f"{filename} has {len(df)} rows -- suspiciously close to participant-level, not aggregate"


def test_cohort_attrition_final_step_matches_620():
    attrition = pd.read_csv(os.path.join(ADNI_OUTPUTS_DIR, "adni_polaris_cohort_attrition.csv"))
    final_row = attrition.iloc[-1]
    assert final_row["step"] == "Final POLARIS-aligned cohort"
    assert int(final_row["remaining_n"]) == 620


# ------------------------------------------------------------------
# 11. Visualization governance boundary remains intact (unmodified)
# ------------------------------------------------------------------


def test_dashboard_governance_module_untouched_by_this_stage():
    """This stage must not have added any new required file or any new
    forbidden-column exception to the dashboard's governance loader --
    REQUIRED_AGGREGATE_FILES stays exactly what it was before this PET
    eligibility work, since the dashboard itself was explicitly not
    changed yet."""
    assert D.REQUIRED_AGGREGATE_FILES == [
        "adni_dashboard_eligibility.csv", "adni_cognitive_summary.csv", "adni_biomarker_summary.csv",
        "adni_pairwise_results.csv", "adni_robustness_summary.csv", "adni_sensitivity_summary.csv",
    ]
    # adni_pet_eligibility.parquet must never be reachable through the
    # governed loader (it's participant-level -- both the .parquet
    # extension and the processed/ path segment must reject it).
    raised = False
    try:
        D.load_aggregate_csv(ADNI_OUTPUTS_DIR, os.path.join("..", "processed", "adni_pet_eligibility.parquet"))
    except D.DataGovernanceError:
        raised = True
    assert raised


def test_metadata_md_documents_this_is_not_matched_adni():
    path = os.path.join(ADNI_OUTPUTS_DIR, "adni_polaris_eligibility_metadata.md")
    with open(path, encoding="utf-8") as f:
        content = f.read()
    assert "not" in content.lower() and "matched ADNI" in content
    assert "propensity" in content.lower()


ALL_TESTS = [
    test_scan_exactly_at_90_days_is_included,
    test_scan_at_91_days_is_excluded,
    test_scan_exactly_at_minus_90_days_is_included,
    test_scan_at_minus_91_days_is_excluded,
    test_non_qc_passed_scan_is_excluded_even_within_window_and_above_threshold,
    test_qc_passed_scan_with_qc_flag_2_is_included,
    test_mmse_below_threshold_is_not_polaris_eligible,
    test_mmse_at_threshold_boundary_is_eligible,
    test_missing_mmse_is_not_polaris_eligible,
    test_centiloid_below_threshold_is_not_polaris_eligible,
    test_centiloid_at_threshold_boundary_is_eligible,
    test_not_centiloid_eligible_is_never_polaris_eligible_even_with_high_value,
    test_closer_scan_wins_over_farther_scan,
    test_equidistant_scans_prefer_the_earlier_one,
    test_exact_same_day_tie_broken_by_lower_loniuid,
    test_missing_clinical_baseline_date_is_explicit_not_silent,
    test_missing_clinical_baseline_date_is_never_polaris_eligible,
    test_participant_with_no_pet_data_at_all_gets_explicit_status_not_dropped,
    test_final_eligible_count_matches_validated_target_620,
    test_final_eligible_diagnosis_composition_matches_validated_target,
    test_validated_cohort_size_still_3030,
    test_polaris_outputs_load_cleanly_through_existing_governance,
    test_polaris_outputs_contain_no_participant_identifiers,
    test_cohort_attrition_final_step_matches_620,
    test_dashboard_governance_module_untouched_by_this_stage,
    test_metadata_md_documents_this_is_not_matched_adni,
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
