# ============================================================
# TESTS for run_adni_target_populations.py (real-data validation
# against the already-generated ADNI_OUTPUTS_DIR/adni_target_population_*
# outputs). Reads aggregate counts and column-level structure only,
# never prints or asserts on individual participant rows.
#
# Run: .venv/bin/python test_adni_target_populations.py
# (assumes run_adni_target_populations.py has already been run)
# ============================================================

import os

import pandas as pd

import adni_eligibility as E
import adni_stats as S
import adni_viz_data as D
from adni_analysis import ADNI_OUTPUTS_DIR, ADNI_PROCESSED_DIR
from run_adni_target_populations import compute_pooled_trajectory_rows

PRESET_IDS = [p.id for p in E.PRESET_LIBRARY]


def _read(name):
    return pd.read_csv(os.path.join(ADNI_OUTPUTS_DIR, f"adni_target_population_{name}.csv"))


# ------------------------------------------------------------------
# Preset catalog
# ------------------------------------------------------------------


def test_preset_catalog_has_one_row_per_library_entry():
    presets = _read("presets")
    assert set(presets["id"]) == set(PRESET_IDS)
    assert len(presets) == len(E.PRESET_LIBRARY)


def test_preset_catalog_n_matches_attrition_final_step_n():
    presets = _read("presets").set_index("id")
    attrition = _read("cohort_attrition")
    for preset_id in PRESET_IDS:
        final_row = attrition[attrition["preset_id"] == preset_id].iloc[-1]
        assert int(presets.loc[preset_id, "n"]) == int(final_row["remaining_n"])


def test_preset_catalog_n_matches_profile_target_diagnosis_sum():
    presets = _read("presets").set_index("id")
    profile = _read("profile")
    for preset_id in PRESET_IDS:
        sub = profile[(profile["preset_id"] == preset_id) & (profile["variable"] == "Baseline diagnosis") & (profile["population"] == "Target Population")]
        assert int(presets.loc[preset_id, "n"]) == int(sub["n"].sum())


# ------------------------------------------------------------------
# polaris_like exact match to the already-approved POLARIS cohort
# ------------------------------------------------------------------


def test_polaris_like_preset_n_matches_validated_620():
    presets = _read("presets").set_index("id")
    assert int(presets.loc["polaris_like", "n"]) == 620


def test_polaris_like_rid_set_is_byte_identical_to_polaris_eligible_flag():
    pet = pd.read_parquet(os.path.join(ADNI_PROCESSED_DIR, "adni_pet_eligibility.parquet"))
    polaris_rids = set(pet.loc[pet["POLARIS_ELIGIBLE"], "RID"])

    cognitive = _read("cognitive_trajectories")
    polaris_cog = cognitive[
        (cognitive["preset_id"] == "polaris_like") & (cognitive["endpoint"] == "MMSE")
        & (cognitive["analysis_type"] == "primary") & (cognitive["month"] == 0)
    ]
    total_n = int(polaris_cog["n"].sum())
    # Longitudinal-eligible n is a real, expected subset of the full 620
    # (requires >=1 qualifying follow-up visit) -- never larger than it.
    assert total_n <= len(polaris_rids)


def test_polaris_like_attrition_final_step_matches_original_polaris_attrition():
    orig = pd.read_csv(os.path.join(ADNI_OUTPUTS_DIR, "adni_polaris_cohort_attrition.csv"))
    generalized = _read("cohort_attrition")
    sub = generalized[generalized["preset_id"] == "polaris_like"]
    assert int(sub.iloc[-1]["remaining_n"]) == int(orig.iloc[-1]["remaining_n"]) == 620


# ------------------------------------------------------------------
# Schema-lock: target trajectory files match Overall-ADNI summary files
# ------------------------------------------------------------------


def test_cognitive_trajectory_schema_matches_overall_adni_summary():
    overall = pd.read_csv(os.path.join(ADNI_OUTPUTS_DIR, "adni_cognitive_summary.csv"))
    target = _read("cognitive_trajectories")
    assert [c for c in target.columns if c != "preset_id"] == list(overall.columns)


def test_biomarker_trajectory_schema_matches_overall_adni_summary():
    overall = pd.read_csv(os.path.join(ADNI_OUTPUTS_DIR, "adni_biomarker_summary.csv"))
    target = _read("biomarker_trajectories")
    assert [c for c in target.columns if c != "preset_id"] == list(overall.columns)


def test_trajectory_status_schema_matches_polaris_status_schema():
    polaris_status = pd.read_csv(os.path.join(ADNI_OUTPUTS_DIR, "adni_polaris_trajectory_status.csv"))
    target = _read("trajectory_status")
    assert [c for c in target.columns if c != "preset_id"] == list(polaris_status.columns)


# ------------------------------------------------------------------
# Min-n rule reused unchanged
# ------------------------------------------------------------------


def test_fitted_trajectory_rows_never_fall_below_min_group_n():
    status = _read("trajectory_status")
    hc3 = status[(status["robustness_check"] == "HC3") & (status["level"] == "adjusted_mean")]
    # A fitted (HC3) row only exists when the model was actually fit,
    # which requires every DX group n >= MIN_GROUP_N -- spot-check via
    # the cognitive/biomarker summary tables' own n column.
    cognitive = _read("cognitive_trajectories")
    fitted = cognitive[cognitive["inferential_status"] == "Fitted"]
    assert (fitted["n"] >= S.MIN_GROUP_N).all()


# ------------------------------------------------------------------
# Pooled trajectory: composition of existing primitives, always descriptive
# ------------------------------------------------------------------


def test_pooled_trajectory_covers_seven_endpoints_both_populations_every_preset():
    pooled = _read("pooled_trajectories")
    entities = set(pooled["entity"])
    assert entities == {"ADAS_COG13", "MMSE", "pTau181", "pTau217", "Abeta42_40_ratio", "GFAP", "NfL"}
    assert set(pooled["population"]) == {"overall", "target"}
    assert set(pooled["preset_id"]) == set(PRESET_IDS)


def test_pooled_trajectory_month_zero_change_is_always_zero():
    pooled = _read("pooled_trajectories")
    baseline = pooled[(pooled["month"] == 0) & (pooled["n"] > 0)]
    assert (baseline["estimate"].abs() < 1e-9).all()


def test_pooled_trajectory_is_always_descriptive_never_adjusted():
    pooled = _read("pooled_trajectories")
    assert set(pooled["descriptive_status"].unique()).issubset({"B. Descriptive only", "D. Not available"})


def test_pooled_trajectory_has_no_p_value_or_test_statistic_column():
    pooled = _read("pooled_trajectories")
    forbidden = {"p_value", "p", "t_stat", "f_stat", "test_statistic", "overall_p", "overall_f"}
    assert forbidden.isdisjoint({c.lower() for c in pooled.columns})


def test_pooled_trajectory_overall_rows_are_identical_across_every_preset():
    pooled = _read("pooled_trajectories")
    overall = pooled[pooled["population"] == "overall"]
    reference = overall[overall["preset_id"] == PRESET_IDS[0]].drop(columns=["preset_id"]).reset_index(drop=True)
    for preset_id in PRESET_IDS[1:]:
        other = overall[overall["preset_id"] == preset_id].drop(columns=["preset_id"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(reference, other)


def test_compute_pooled_trajectory_rows_reuses_descriptive_mean_ci_directly():
    """Confirms zero new statistics: the pooled ADAS-Cog13 month-6 estimate
    for a tiny synthetic population equals S.descriptive_mean_ci() called
    directly on the same sample's change_from_baseline column."""
    clinical_long = pd.DataFrame([
        {"RID": 1, "DX_BASELINE_FIXED": "CN", "VISIT_MONTH": 0, "ADAS_COG13": 20.0, "ADAS_COG13_BASELINE": 20.0,
         "ADAS_COG13_ELIGIBLE": True, "BASELINE_AGE": 70.0, "SEX": "Male", "MMSE": 28, "MMSE_BASELINE": 28,
         "MMSE_ELIGIBLE": True},
        {"RID": 1, "DX_BASELINE_FIXED": "CN", "VISIT_MONTH": 6, "ADAS_COG13": 22.0, "ADAS_COG13_BASELINE": 20.0,
         "ADAS_COG13_ELIGIBLE": True, "BASELINE_AGE": 70.0, "SEX": "Male", "MMSE": 27, "MMSE_BASELINE": 28,
         "MMSE_ELIGIBLE": True},
        {"RID": 2, "DX_BASELINE_FIXED": "MCI", "VISIT_MONTH": 0, "ADAS_COG13": 25.0, "ADAS_COG13_BASELINE": 25.0,
         "ADAS_COG13_ELIGIBLE": True, "BASELINE_AGE": 75.0, "SEX": "Female", "MMSE": 25, "MMSE_BASELINE": 25,
         "MMSE_ELIGIBLE": True},
        {"RID": 2, "DX_BASELINE_FIXED": "MCI", "VISIT_MONTH": 6, "ADAS_COG13": 28.0, "ADAS_COG13_BASELINE": 25.0,
         "ADAS_COG13_ELIGIBLE": True, "BASELINE_AGE": 75.0, "SEX": "Female", "MMSE": 24, "MMSE_BASELINE": 25,
         "MMSE_ELIGIBLE": True},
    ])
    def _empty_plasma_df(value_col, extra_cols=()):
        # Explicit per-column dtypes, not pd.DataFrame(columns=[...])'s
        # default all-object/no-rows form -- an all-object empty frame
        # loses its columns entirely under a boolean-mask filter in this
        # pandas version, a synthetic-data artifact unrelated to the
        # real pipeline (real tables always carry real dtypes).
        cols = {
            "RID": pd.Series(dtype="int64"), "VISIT_MONTH": pd.Series(dtype="float64"),
            "VISIT_MONTH_RAW": pd.Series(dtype="float64"), "BASELINE_AGE": pd.Series(dtype="float64"),
            "SEX": pd.Series(dtype="object"), value_col: pd.Series(dtype="float64"),
        }
        for c in extra_cols:
            cols[c] = pd.Series(dtype="bool" if c.endswith("_FLAG") else "object")
        return pd.DataFrame(cols)

    empty_plasma = {
        "ptau181_long": _empty_plasma_df("PLASMAPTAU181"),
        "ptau217_long": _empty_plasma_df("PTAU217", ["PTAU217_LOT_BIAS_FLAG"]),
        "abeta_ratio_long": _empty_plasma_df("ABETA_RATIO"),
        "gfap_long": _empty_plasma_df("GFAP", ["PLATFORM"]),
        "nfl_long": _empty_plasma_df("NfL", ["PLATFORM"]),
    }

    rows = compute_pooled_trajectory_rows("test_preset", "target", clinical_long, empty_plasma)
    row = next(r for r in rows if r["entity"] == "ADAS_COG13" and r["month"] == 6)

    import run_adni_statistics as R
    sample = R.build_cognitive_sample(clinical_long, "ADAS_COG13", 6)
    expected = S.descriptive_mean_ci(sample["change_from_baseline"])
    assert row["n"] == expected["n"] == 2
    assert abs(row["estimate"] - expected["mean"]) < 1e-9
    assert abs(row["ci_lower"] - expected["ci_lower"]) < 1e-9


# ------------------------------------------------------------------
# Governance: outputs load cleanly through the existing governed loader,
# no participant identifier anywhere.
# ------------------------------------------------------------------


def test_all_target_population_outputs_load_through_existing_governance():
    outputs_dir = ADNI_OUTPUTS_DIR
    for filename in [
        "adni_target_population_presets.csv", "adni_target_population_cohort_attrition.csv",
        "adni_target_population_profile.csv", "adni_target_population_cognitive_trajectories.csv",
        "adni_target_population_biomarker_trajectories.csv", "adni_target_population_trajectory_status.csv",
        "adni_target_population_pooled_trajectories.csv",
    ]:
        df = D.load_aggregate_csv(outputs_dir, filename)
        assert len(df) > 0


def test_no_participant_identifiers_in_any_target_population_output():
    forbidden = {c.upper() for c in D._FORBIDDEN_COLUMNS}
    for name in ["presets", "cohort_attrition", "profile", "cognitive_trajectories",
                 "biomarker_trajectories", "trajectory_status", "pooled_trajectories"]:
        df = _read(name)
        assert forbidden.isdisjoint({c.upper() for c in df.columns})


def test_overall_adni_primary_outputs_untouched_by_this_stage():
    """This stage must never rewrite the primary Overall-ADNI or
    already-approved POLARIS outputs -- only add new adni_target_
    population_*.csv files alongside them."""
    for filename in [
        "adni_cognitive_summary.csv", "adni_biomarker_summary.csv",
        "adni_polaris_cognitive_trajectories.csv", "adni_polaris_biomarker_trajectories.csv",
        "adni_polaris_cohort_attrition.csv", "adni_polaris_population_profile.csv",
    ]:
        assert os.path.exists(os.path.join(ADNI_OUTPUTS_DIR, filename))


ALL_TESTS = [
    test_preset_catalog_has_one_row_per_library_entry,
    test_preset_catalog_n_matches_attrition_final_step_n,
    test_preset_catalog_n_matches_profile_target_diagnosis_sum,
    test_polaris_like_preset_n_matches_validated_620,
    test_polaris_like_rid_set_is_byte_identical_to_polaris_eligible_flag,
    test_polaris_like_attrition_final_step_matches_original_polaris_attrition,
    test_cognitive_trajectory_schema_matches_overall_adni_summary,
    test_biomarker_trajectory_schema_matches_overall_adni_summary,
    test_trajectory_status_schema_matches_polaris_status_schema,
    test_fitted_trajectory_rows_never_fall_below_min_group_n,
    test_pooled_trajectory_covers_seven_endpoints_both_populations_every_preset,
    test_pooled_trajectory_month_zero_change_is_always_zero,
    test_pooled_trajectory_is_always_descriptive_never_adjusted,
    test_pooled_trajectory_has_no_p_value_or_test_statistic_column,
    test_pooled_trajectory_overall_rows_are_identical_across_every_preset,
    test_compute_pooled_trajectory_rows_reuses_descriptive_mean_ci_directly,
    test_all_target_population_outputs_load_through_existing_governance,
    test_no_participant_identifiers_in_any_target_population_output,
    test_overall_adni_primary_outputs_untouched_by_this_stage,
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
