# ============================================================
# TESTS for run_adni_polaris_trajectories.py (POLARIS AD-Aligned
# longitudinal cognitive/biomarker trajectory outputs).
#
# Two kinds of test here:
#   1. Unit tests against small, hand-built synthetic inputs (never
#      real ADNI participant data).
#   2. Real-data validation tests against the already-generated
#      outputs/adni_polaris_{cognitive,biomarker}_trajectories.csv and
#      outputs/adni_polaris_trajectory_status.csv -- aggregate counts
#      and column-level structure only, never a participant-level row.
#
# Run: .venv/bin/python test_adni_polaris_trajectories.py
# (assumes run_adni_polaris_trajectories.py has already been run)
# ============================================================

import os

import pandas as pd

import adni_stats as S
import adni_viz_data as D
import run_adni_polaris_trajectories as P
from adni_analysis import ADNI_OUTPUTS_DIR

COGNITIVE_PATH = os.path.join(ADNI_OUTPUTS_DIR, "adni_polaris_cognitive_trajectories.csv")
BIOMARKER_PATH = os.path.join(ADNI_OUTPUTS_DIR, "adni_polaris_biomarker_trajectories.csv")
STATUS_PATH = os.path.join(ADNI_OUTPUTS_DIR, "adni_polaris_trajectory_status.csv")

OVERALL_COGNITIVE_PATH = os.path.join(ADNI_OUTPUTS_DIR, "adni_cognitive_summary.csv")
OVERALL_BIOMARKER_PATH = os.path.join(ADNI_OUTPUTS_DIR, "adni_biomarker_summary.csv")
OVERALL_ELIGIBILITY_PATH = os.path.join(ADNI_OUTPUTS_DIR, "adni_dashboard_eligibility.csv")
OVERALL_ROBUSTNESS_PATH = os.path.join(ADNI_OUTPUTS_DIR, "adni_robustness_summary.csv")


# ------------------------------------------------------------------
# Unit tests: population restriction / status flattening
# ------------------------------------------------------------------


def test_restrict_to_polaris_keeps_only_given_rids():
    df = pd.DataFrame({"RID": [1, 2, 3, 4], "value": ["a", "b", "c", "d"]})
    out = P.restrict_to_polaris(df, {2, 4})
    assert sorted(out["RID"].tolist()) == [2, 4]
    assert len(df) == 4, "original DataFrame must not be mutated"


def test_flatten_status_rows_no_fit_produces_single_na_row():
    elig = {
        "endpoint_or_biomarker": "MMSE", "assay_platform": "", "analysis_type": "primary",
        "month": 18, "classification": D.CLASS_NOT_AVAILABLE, "reason": "No supported visit data.",
    }
    out = P.flatten_status_rows([(elig, [])])
    assert len(out) == 1
    row = out.iloc[0]
    assert row["robustness_check"] == "N/A (no model fit)"
    assert row["classification"] == D.CLASS_NOT_AVAILABLE
    assert pd.isna(row["alternative_estimate"])


def test_flatten_status_rows_with_fit_produces_one_row_per_robustness_check():
    elig = {
        "endpoint_or_biomarker": "MMSE", "assay_platform": "", "analysis_type": "primary",
        "month": 12, "classification": D.CLASS_ADJUSTED, "reason": "ANCOVA fitted.",
    }
    rob_rows = [
        {"level": "overall_group_test", "group_or_comparison": "", "robustness_check": "HC3",
         "conventional_estimate": 1.0, "conventional_se": 0.1, "conventional_ci_lower": 0.8,
         "conventional_ci_upper": 1.2, "conventional_p": 0.01,
         "alternative_estimate": 1.0, "alternative_se": 0.12, "alternative_ci_lower": 0.76,
         "alternative_ci_upper": 1.24, "alternative_p": 0.02, "sensitive_flag": False},
        {"level": "overall_group_test", "group_or_comparison": "", "robustness_check": "Influence_exclusion (n_excluded=1)",
         "conventional_estimate": 1.0, "conventional_se": 0.1, "conventional_ci_lower": 0.8,
         "conventional_ci_upper": 1.2, "conventional_p": 0.01,
         "alternative_estimate": 1.05, "alternative_se": 0.11, "alternative_ci_lower": 0.83,
         "alternative_ci_upper": 1.27, "alternative_p": 0.015, "sensitive_flag": False},
    ]
    out = P.flatten_status_rows([(elig, rob_rows)])
    assert len(out) == 2
    assert set(out["robustness_check"]) == {"HC3", "Influence_exclusion (n_excluded=1)"}
    assert (out["classification"] == D.CLASS_ADJUSTED).all()


def test_flatten_status_rows_output_matches_declared_column_schema():
    elig = {"endpoint_or_biomarker": "GFAP", "assay_platform": "Quanterix", "analysis_type": "primary", "month": 0, "classification": D.CLASS_DESCRIPTIVE, "reason": "Baseline."}
    out = P.flatten_status_rows([(elig, [])])
    assert list(out.columns) == P.STATUS_COLS


# ------------------------------------------------------------------
# Real-data: cohort membership fixed at the validated 620-person set
# ------------------------------------------------------------------


def test_polaris_rid_count_matches_validated_620():
    rids = P.load_polaris_rids()
    assert len(rids) == P.EXPECTED_POLARIS_N == 620


def test_polaris_cohort_diagnosis_composition_matches_validated_target():
    import run_adni_statistics as R
    rids = P.load_polaris_rids()
    tables = R.load_processed_tables()
    clinical_p = P.restrict_to_polaris(tables["clinical_long"], rids)
    dx = clinical_p.drop_duplicates("RID")["DX_BASELINE_FIXED"].value_counts().to_dict()
    assert dx.get("CN") == 151
    assert dx.get("MCI") == 309
    assert dx.get("Dementia") == 160


# ------------------------------------------------------------------
# Real-data: endpoint/month aggregation
# ------------------------------------------------------------------


def test_cognitive_trajectories_cover_both_endpoints():
    df = pd.read_csv(COGNITIVE_PATH)
    assert set(df["endpoint"]) == {"ADAS_COG13", "MMSE"}
    assert set(df["analysis_type"]) >= {"primary", "sensitivity_interval_excl"}


def test_cognitive_trajectories_only_use_canonical_months():
    df = pd.read_csv(COGNITIVE_PATH)
    assert set(df["month"].unique()) <= set(S.TARGET_MONTHS)


def test_biomarker_trajectories_cover_all_five_biomarkers():
    df = pd.read_csv(BIOMARKER_PATH)
    assert set(df["biomarker"]) == {"pTau181", "pTau217", "Abeta42_40_ratio", "GFAP", "NfL"}


def test_biomarker_trajectories_preserve_assay_platform_separation():
    """GFAP/NfL must appear on both Quanterix (primary) and Fujirebio
    (sensitivity_fujirebio) as SEPARATE rows, never averaged/combined
    into one platform-agnostic row -- same discipline as Overall ADNI."""
    df = pd.read_csv(BIOMARKER_PATH)
    for biomarker in ["GFAP", "NfL"]:
        sub = df[df["biomarker"] == biomarker]
        platforms = set(zip(sub["assay_platform"], sub["analysis_type"]))
        assert ("Quanterix", "primary") in platforms
        assert ("Fujirebio", "sensitivity_fujirebio") in platforms
    pt217 = df[df["biomarker"] == "pTau217"]
    assert set(pt217["analysis_type"]) == {"primary", "sensitivity_incl_lot_bias"}
    assert (pt217["assay_platform"] == "Fujirebio_Lumipulse").all()


def test_status_covers_full_seven_month_grid_for_every_primary_cell():
    df = pd.read_csv(STATUS_PATH)
    primary = df[df["analysis_type"] == "primary"].drop_duplicates(
        ["endpoint_or_biomarker", "assay_platform", "analysis_type", "month"]
    )
    for (entity, platform), group in primary.groupby(["endpoint_or_biomarker", "assay_platform"]):
        assert sorted(group["month"].tolist()) == S.TARGET_MONTHS, f"{entity}/{platform} missing a month from the canonical grid"


# ------------------------------------------------------------------
# Real-data: baseline consistency (cross-check vs. population profile)
# ------------------------------------------------------------------


def test_baseline_values_match_polaris_population_profile_exactly():
    import run_adni_statistics as R
    rids = P.load_polaris_rids()
    tables = R.load_processed_tables()
    clinical_p = P.restrict_to_polaris(tables["clinical_long"], rids).drop_duplicates("RID")
    profile = pd.read_csv(os.path.join(ADNI_OUTPUTS_DIR, "adni_polaris_population_profile.csv"))

    for baseline_col, profile_var in [("MMSE_BASELINE", "Baseline MMSE"), ("ADAS_COG13_BASELINE", "Baseline ADAS-Cog13")]:
        this_n = int(clinical_p[baseline_col].notna().sum())
        this_mean = float(clinical_p[baseline_col].mean())
        prof_row = profile[(profile["variable"] == profile_var) & (profile["population"] == "POLARIS-aligned ADNI")].iloc[0]
        assert this_n == int(prof_row["n"]) == 620
        assert abs(this_mean - float(prof_row["mean"])) < 1e-9


def test_baseline_month_is_never_inferentially_tested():
    """Month 0 is structurally descriptive (change is zero by
    construction) in both cognitive and biomarker trajectories --
    never classified Adjusted/Sensitivity-concern."""
    status = pd.read_csv(STATUS_PATH)
    baseline_rows = status[status["month"] == 0].drop_duplicates(
        ["endpoint_or_biomarker", "assay_platform", "analysis_type"]
    )
    assert (baseline_rows["classification"] == D.CLASS_DESCRIPTIVE).all()


# ------------------------------------------------------------------
# Real-data: small-cell rules
# ------------------------------------------------------------------


def test_fitted_cognitive_rows_never_fall_below_min_group_n():
    df = pd.read_csv(COGNITIVE_PATH)
    fitted = df[df["inferential_status"] == "Fitted"]
    assert len(fitted) > 0, "expected at least one fitted cognitive cell in the POLARIS cohort"
    assert (fitted["n"] >= S.MIN_GROUP_N).all()


def test_fitted_biomarker_rows_never_fall_below_min_group_n():
    df = pd.read_csv(BIOMARKER_PATH)
    fitted = df[df["inferential_status"] == "Fitted"]
    assert len(fitted) > 0, "expected at least one fitted biomarker cell in the POLARIS cohort"
    assert (fitted["n"] >= S.MIN_GROUP_N).all()


def test_suppressed_rows_report_the_limiting_group_and_n():
    df = pd.read_csv(COGNITIVE_PATH)
    suppressed = df[df["inferential_status"].astype(str).str.startswith("Suppressed")]
    assert len(suppressed) > 0
    assert suppressed["inferential_status"].str.contains("limiting group=").all()


# ------------------------------------------------------------------
# Real-data: status assignment (does not force a fit where the same
# rule already found no support in Overall ADNI)
# ------------------------------------------------------------------


def test_status_never_forces_a_fit_for_a_known_sparse_platform_month():
    """GFAP/NfL on Fujirebio have zero POLARIS records at months 6/18/36/48
    (confirmed directly against the restricted plasma tables) -- these
    must land as Not-available, never Adjusted, since nothing was forced."""
    status = pd.read_csv(STATUS_PATH)
    for biomarker in ["GFAP", "NfL"]:
        sub = status[
            (status["endpoint_or_biomarker"] == biomarker)
            & (status["assay_platform"] == "Fujirebio")
            & (status["month"].isin([6, 18, 36, 48]))
        ].drop_duplicates(["month"])
        assert (sub["classification"] == D.CLASS_NOT_AVAILABLE).all(), f"{biomarker}/Fujirebio should be Not-available at sparse months"


def test_status_classification_is_always_one_of_the_four_categories():
    status = pd.read_csv(STATUS_PATH)
    allowed = {D.CLASS_ADJUSTED, D.CLASS_DESCRIPTIVE, D.CLASS_SENSITIVITY_CONCERN, D.CLASS_NOT_AVAILABLE}
    assert set(status["classification"].unique()) <= allowed


# ------------------------------------------------------------------
# Governance: no participant identifiers, governed-loader compatible
# ------------------------------------------------------------------


def test_no_participant_identifiers_in_any_polaris_trajectory_output():
    forbidden = {"RID", "PTID", "USUBJID", "SUBJID", "PARTICIPANT_ID", "LONIUID"}
    for path in [COGNITIVE_PATH, BIOMARKER_PATH, STATUS_PATH]:
        df = pd.read_csv(path)
        assert forbidden & {c.upper() for c in df.columns} == set(), f"{path} carries a forbidden identifier column"


def test_polaris_trajectory_outputs_load_through_governed_visualization_loader():
    for filename in [
        "adni_polaris_cognitive_trajectories.csv",
        "adni_polaris_biomarker_trajectories.csv",
        "adni_polaris_trajectory_status.csv",
    ]:
        df = D.load_aggregate_csv(ADNI_OUTPUTS_DIR, filename)
        assert len(df) > 0


def test_polaris_trajectory_outputs_reuse_overall_adni_column_schema():
    cognitive_overall = pd.read_csv(OVERALL_COGNITIVE_PATH)
    cognitive_polaris = pd.read_csv(COGNITIVE_PATH)
    assert list(cognitive_polaris.columns) == list(cognitive_overall.columns)

    biomarker_overall = pd.read_csv(OVERALL_BIOMARKER_PATH)
    biomarker_polaris = pd.read_csv(BIOMARKER_PATH)
    assert list(biomarker_polaris.columns) == list(biomarker_overall.columns)


# ------------------------------------------------------------------
# No regressions to Overall ADNI results
# ------------------------------------------------------------------


def test_overall_adni_outputs_untouched_by_this_stage():
    """This script must never write to any Overall-ADNI-scoped output
    file -- a static guard against a future edit accidentally reusing
    an Overall-ADNI filename as a POLARIS write target."""
    with open("run_adni_polaris_trajectories.py", encoding="utf-8") as f:
        source = f.read()
    for forbidden_write in [
        '"adni_cognitive_summary.csv"', '"adni_biomarker_summary.csv"',
        '"adni_dashboard_eligibility.csv"', '"adni_robustness_summary.csv"',
        '"adni_pairwise_results.csv"', '"adni_sensitivity_summary.csv"',
    ]:
        assert forbidden_write not in source


def test_overall_adni_cognitive_summary_still_reflects_the_full_cohort():
    """Sanity check that the (unmodified) Overall-ADNI cognitive summary
    still describes the full ~3,030-participant cohort, not the smaller
    620-participant POLARIS cohort -- confirms this stage did not
    overwrite it with POLARIS-sized numbers."""
    overall = pd.read_csv(OVERALL_COGNITIVE_PATH)
    overall_baseline_n = overall[(overall["endpoint"] == "MMSE") & (overall["analysis_type"] == "primary") & (overall["month"] == 0)]["n"].sum()
    polaris_baseline_n = pd.read_csv(COGNITIVE_PATH)
    polaris_baseline_n = polaris_baseline_n[(polaris_baseline_n["endpoint"] == "MMSE") & (polaris_baseline_n["analysis_type"] == "primary") & (polaris_baseline_n["month"] == 0)]["n"].sum()
    assert overall_baseline_n > 2000
    assert polaris_baseline_n < overall_baseline_n
    assert polaris_baseline_n < 600


def test_overall_adni_dashboard_eligibility_and_robustness_files_unchanged_shape():
    """The Overall-ADNI eligibility/robustness files must still exist
    with their original (endpoint_or_biomarker, assay_platform,
    analysis_type, month) key coverage -- a coarse but real regression
    guard that this stage did not truncate or overwrite them."""
    elig = pd.read_csv(OVERALL_ELIGIBILITY_PATH)
    assert len(elig) > 0
    assert set(elig["endpoint_or_biomarker"]) >= {"ADAS_COG13", "MMSE", "pTau181", "pTau217", "GFAP", "NfL"}
    rob = pd.read_csv(OVERALL_ROBUSTNESS_PATH)
    assert len(rob) > 0
    assert "HC3" in set(rob["robustness_check"])


ALL_TESTS = [
    test_restrict_to_polaris_keeps_only_given_rids,
    test_flatten_status_rows_no_fit_produces_single_na_row,
    test_flatten_status_rows_with_fit_produces_one_row_per_robustness_check,
    test_flatten_status_rows_output_matches_declared_column_schema,
    test_polaris_rid_count_matches_validated_620,
    test_polaris_cohort_diagnosis_composition_matches_validated_target,
    test_cognitive_trajectories_cover_both_endpoints,
    test_cognitive_trajectories_only_use_canonical_months,
    test_biomarker_trajectories_cover_all_five_biomarkers,
    test_biomarker_trajectories_preserve_assay_platform_separation,
    test_status_covers_full_seven_month_grid_for_every_primary_cell,
    test_baseline_values_match_polaris_population_profile_exactly,
    test_baseline_month_is_never_inferentially_tested,
    test_fitted_cognitive_rows_never_fall_below_min_group_n,
    test_fitted_biomarker_rows_never_fall_below_min_group_n,
    test_suppressed_rows_report_the_limiting_group_and_n,
    test_status_never_forces_a_fit_for_a_known_sparse_platform_month,
    test_status_classification_is_always_one_of_the_four_categories,
    test_no_participant_identifiers_in_any_polaris_trajectory_output,
    test_polaris_trajectory_outputs_load_through_governed_visualization_loader,
    test_polaris_trajectory_outputs_reuse_overall_adni_column_schema,
    test_overall_adni_outputs_untouched_by_this_stage,
    test_overall_adni_cognitive_summary_still_reflects_the_full_cohort,
    test_overall_adni_dashboard_eligibility_and_robustness_files_unchanged_shape,
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
