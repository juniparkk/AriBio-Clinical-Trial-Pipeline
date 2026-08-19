# ============================================================
# TESTS for adni_eligibility.py (generalized target-population
# eligibility engine).
#
# Unit tests only, against small hand-built synthetic DataFrames (fake
# RIDs, fake values -- never real ADNI participant data). Real-data
# validation of the full pipeline (including the polaris_like preset's
# byte-identical match to the already-approved POLARIS cohort) lives in
# test_adni_target_populations.py.
#
# Run: .venv/bin/python test_adni_eligibility.py
# ============================================================

import numpy as np
import pandas as pd

import adni_eligibility as E

# ------------------------------------------------------------------
# Synthetic master eligibility table
# ------------------------------------------------------------------


def _master_row(rid, dx, mmse, centiloid, centiloid_eligible=True, age=70.0, sex="Male", apoe4=0.0,
                 has_ptau217=False, has_abeta_ratio=False):
    return {
        "RID": rid, "DX_BASELINE_FIXED": dx, "BASELINE_AGE": age, "SEX": sex, "APOE4_CARRIER": apoe4,
        "MMSE_BASELINE": mmse, "ADAS_COG13_BASELINE": 20.0,
        "CENTILOID_BASELINE": centiloid, "CENTILOID_ELIGIBLE": centiloid_eligible,
        "POLARIS_ELIGIBLE": bool(centiloid_eligible and centiloid is not None and centiloid >= 30 and mmse is not None and mmse >= 20),
        "HAS_PTAU217": has_ptau217, "HAS_ABETA_RATIO": has_abeta_ratio,
        "HAS_PTAU181": False, "HAS_GFAP": False, "HAS_NFL": False,
    }


def _synthetic_master():
    return pd.DataFrame([
        _master_row(1, "CN", 29, 10, centiloid_eligible=True),
        _master_row(2, "MCI", 25, 40, centiloid_eligible=True, has_ptau217=True, has_abeta_ratio=True),
        _master_row(3, "MCI", 18, 50, centiloid_eligible=True),
        _master_row(4, "Dementia", 22, 35, centiloid_eligible=True, age=90, has_ptau217=True),
        _master_row(5, "Dementia", np.nan, 60, centiloid_eligible=True),
        _master_row(6, "MCI", 24, np.nan, centiloid_eligible=False),
        _master_row(7, "CN", 28, 5, centiloid_eligible=True),
    ])


# ------------------------------------------------------------------
# build_master_eligibility_table
# ------------------------------------------------------------------


def test_build_master_eligibility_table_adds_biomarker_columns_without_touching_existing():
    pet = pd.DataFrame([
        {"RID": 1, "DX_BASELINE_FIXED": "CN", "POLARIS_ELIGIBLE": False},
        {"RID": 2, "DX_BASELINE_FIXED": "MCI", "POLARIS_ELIGIBLE": True},
    ])
    out = E.build_master_eligibility_table(pet, {"HAS_PTAU217": {2}})
    assert out.set_index("RID").loc[1, "HAS_PTAU217"] == False
    assert out.set_index("RID").loc[2, "HAS_PTAU217"] == True
    assert out.set_index("RID").loc[2, "POLARIS_ELIGIBLE"] == True  # untouched passthrough


def test_build_master_eligibility_table_defaults_missing_biomarker_column_to_all_false():
    pet = pd.DataFrame([{"RID": 1, "DX_BASELINE_FIXED": "CN", "POLARIS_ELIGIBLE": False}])
    out = E.build_master_eligibility_table(pet, {})
    for col in E.BIOMARKER_COLUMNS:
        assert out.iloc[0][col] == False


# ------------------------------------------------------------------
# evaluate_preset
# ------------------------------------------------------------------


def test_evaluate_preset_diagnosis_filter():
    master = _synthetic_master()
    preset = E.PresetSpec(id="p", label="p", description="", diagnosis=("MCI",))
    mask = E.evaluate_preset(master, preset)
    assert set(master.loc[mask, "RID"]) == {2, 3, 6}


def test_evaluate_preset_mmse_range():
    master = _synthetic_master()
    preset = E.PresetSpec(id="p", label="p", description="", mmse_min=20, mmse_max=26)
    mask = E.evaluate_preset(master, preset)
    # RID 5 has NaN MMSE -> excluded, not silently included
    assert set(master.loc[mask, "RID"]) == {2, 4, 6}


def test_evaluate_preset_missing_mmse_never_treated_as_meeting_threshold():
    master = _synthetic_master()
    preset = E.PresetSpec(id="p", label="p", description="", mmse_min=0)
    mask = E.evaluate_preset(master, preset)
    assert 5 not in set(master.loc[mask, "RID"])  # RID 5's MMSE is NaN


def test_evaluate_preset_centiloid_threshold_requires_data_availability_too():
    master = _synthetic_master()
    preset = E.PresetSpec(id="p", label="p", description="", centiloid_min=30)
    mask = E.evaluate_preset(master, preset)
    # RID 6 has centiloid_eligible=False despite no NaN value -- must be excluded
    assert 6 not in set(master.loc[mask, "RID"])
    assert set(master.loc[mask, "RID"]) == {2, 3, 4, 5}


def test_evaluate_preset_age_range():
    master = _synthetic_master()
    preset = E.PresetSpec(id="p", label="p", description="", age_min=80)
    mask = E.evaluate_preset(master, preset)
    assert set(master.loc[mask, "RID"]) == {4}


def test_evaluate_preset_require_biomarkers_all_must_be_true():
    master = _synthetic_master()
    preset = E.PresetSpec(id="p", label="p", description="", require_biomarkers=("HAS_PTAU217", "HAS_ABETA_RATIO"))
    mask = E.evaluate_preset(master, preset)
    # RID 2 has both; RID 4 has only HAS_PTAU217
    assert set(master.loc[mask, "RID"]) == {2}


def test_evaluate_preset_combines_all_active_criteria_with_and():
    master = _synthetic_master()
    preset = E.PresetSpec(id="p", label="p", description="", diagnosis=("MCI", "Dementia"), centiloid_min=30, mmse_min=20)
    mask = E.evaluate_preset(master, preset)
    assert set(master.loc[mask, "RID"]) == {2, 4}


def test_evaluate_preset_no_criteria_selects_everyone():
    master = _synthetic_master()
    preset = E.PresetSpec(id="p", label="p", description="")
    mask = E.evaluate_preset(master, preset)
    assert mask.all()


# ------------------------------------------------------------------
# build_preset_attrition
# ------------------------------------------------------------------


def test_attrition_first_and_last_step_bracket_cohort_and_final_membership():
    master = _synthetic_master()
    preset = E.PresetSpec(id="p", label="Preset: Test", description="", diagnosis=("MCI",), mmse_min=20)
    attrition = E.build_preset_attrition(master, preset)
    assert attrition.iloc[0]["step"] == "Validated ADNI cohort"
    assert attrition.iloc[0]["remaining_n"] == len(master)
    assert attrition.iloc[-1]["remaining_n"] == int(E.evaluate_preset(master, preset).sum())


def test_attrition_is_self_consistent_sequential_narrowing():
    master = _synthetic_master()
    preset = E.PresetSpec(id="p", label="Preset: Test", description="", diagnosis=("MCI", "Dementia"), centiloid_min=30, age_max=85)
    attrition = E.build_preset_attrition(master, preset)
    for i in range(1, len(attrition)):
        assert attrition.iloc[i]["starting_n"] == attrition.iloc[i - 1]["remaining_n"]
        assert attrition.iloc[i]["remaining_n"] + attrition.iloc[i]["excluded_n"] == attrition.iloc[i]["starting_n"]


def test_attrition_distinguishes_data_availability_step_from_threshold_step():
    master = _synthetic_master()
    preset = E.PresetSpec(id="p", label="Preset: Test", description="", centiloid_min=30)
    attrition = E.build_preset_attrition(master, preset)
    steps = attrition["step"].tolist()
    assert any("available" in s.lower() for s in steps)
    assert any("Centiloid" in s and "available" not in s.lower() for s in steps)
    # RID 6 is excluded at the "available" step (not evaluable), not the threshold step
    avail_idx = next(i for i, s in enumerate(steps) if "available" in s.lower())
    assert attrition.iloc[avail_idx]["excluded_n"] >= 1


def test_attrition_no_criteria_produces_only_cohort_and_final_rows():
    master = _synthetic_master()
    preset = E.PresetSpec(id="p", label="Preset: Everyone", description="")
    attrition = E.build_preset_attrition(master, preset)
    assert len(attrition) == 2
    assert attrition.iloc[-1]["remaining_n"] == len(master)


# ------------------------------------------------------------------
# build_preset_profile
# ------------------------------------------------------------------


def test_profile_numeric_rows_present_for_both_populations():
    master = _synthetic_master()
    mask = master["RID"].isin([2, 4])
    profile = E.build_preset_profile(master, mask, overall_label="Overall ADNI", target_label="Target Population")
    age_rows = profile[profile["variable"] == "Baseline age (years)"]
    assert set(age_rows["population"]) == {"Overall ADNI", "Target Population"}
    overall_row = age_rows[age_rows["population"] == "Overall ADNI"].iloc[0]
    target_row = age_rows[age_rows["population"] == "Target Population"].iloc[0]
    assert overall_row["n"] == len(master)
    assert target_row["n"] == 2


def test_profile_has_no_p_value_or_test_statistic_column():
    master = _synthetic_master()
    mask = master["RID"].isin([2, 4])
    profile = E.build_preset_profile(master, mask)
    forbidden = {"p_value", "p", "t_stat", "f_stat", "test_statistic"}
    assert forbidden.isdisjoint({c.lower() for c in profile.columns})


def test_profile_categorical_rows_sum_to_population_n():
    master = _synthetic_master()
    mask = master["RID"].isin([2, 3, 4])
    profile = E.build_preset_profile(master, mask, target_label="Target Population")
    dx_rows = profile[(profile["variable"] == "Baseline diagnosis") & (profile["population"] == "Target Population")]
    assert dx_rows["n"].sum() == 3


def test_profile_includes_biomarker_availability_rows():
    master = _synthetic_master()
    mask = master["RID"].isin([2, 4])
    profile = E.build_preset_profile(master, mask)
    assert any("pTau217 available" in v for v in profile["variable"].unique())


# ------------------------------------------------------------------
# PRESET_LIBRARY sanity
# ------------------------------------------------------------------


def test_preset_library_ids_are_unique():
    ids = [p.id for p in E.PRESET_LIBRARY]
    assert len(ids) == len(set(ids))


def test_exactly_one_polaris_equivalent_preset():
    equivalents = [p for p in E.PRESET_LIBRARY if p.is_polaris_equivalent]
    assert len(equivalents) == 1
    assert equivalents[0].id == "polaris_like"


def test_preset_library_never_uses_a_diagnosis_stage_finer_than_cn_mci_dementia():
    allowed = {"CN", "MCI", "Dementia"}
    for preset in E.PRESET_LIBRARY:
        assert set(preset.diagnosis).issubset(allowed)


def test_preset_by_id_lookup_matches_library():
    for preset in E.PRESET_LIBRARY:
        assert E.PRESET_BY_ID[preset.id] is preset


ALL_TESTS = [
    test_build_master_eligibility_table_adds_biomarker_columns_without_touching_existing,
    test_build_master_eligibility_table_defaults_missing_biomarker_column_to_all_false,
    test_evaluate_preset_diagnosis_filter,
    test_evaluate_preset_mmse_range,
    test_evaluate_preset_missing_mmse_never_treated_as_meeting_threshold,
    test_evaluate_preset_centiloid_threshold_requires_data_availability_too,
    test_evaluate_preset_age_range,
    test_evaluate_preset_require_biomarkers_all_must_be_true,
    test_evaluate_preset_combines_all_active_criteria_with_and,
    test_evaluate_preset_no_criteria_selects_everyone,
    test_attrition_first_and_last_step_bracket_cohort_and_final_membership,
    test_attrition_is_self_consistent_sequential_narrowing,
    test_attrition_distinguishes_data_availability_step_from_threshold_step,
    test_attrition_no_criteria_produces_only_cohort_and_final_rows,
    test_profile_numeric_rows_present_for_both_populations,
    test_profile_has_no_p_value_or_test_statistic_column,
    test_profile_categorical_rows_sum_to_population_n,
    test_profile_includes_biomarker_availability_rows,
    test_preset_library_ids_are_unique,
    test_exactly_one_polaris_equivalent_preset,
    test_preset_library_never_uses_a_diagnosis_stage_finer_than_cn_mci_dementia,
    test_preset_by_id_lookup_matches_library,
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
