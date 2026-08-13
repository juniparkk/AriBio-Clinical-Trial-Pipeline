# ============================================================
# TESTS for adni_cohort.py / adni_plasma.py (ADNI preprocessing stage).
#
# Every fixture here is hand-built synthetic data (fake RIDs, fake
# dates, fake values) -- never real ADNI participant data -- so these
# tests are fast, deterministic, and safe to run/commit anywhere,
# consistent with the "participant-level ADNI data must remain local"
# rule the preprocessing modules themselves follow.
#
# Run: .venv/bin/python test_adni_preprocessing.py
# ============================================================

import pandas as pd

import adni_cohort as C
import adni_plasma as P


# ------------------------------------------------------------------
# Shared fixtures
# ------------------------------------------------------------------


def make_visits_df():
    rows = [
        ("ADNI1", 1, "sc", "Screening", 1),
        ("ADNI1", 2, "bl", "Baseline", 2),
        ("ADNI1", 3, "m06", "Month 6", 3),
        ("ADNI1", 4, "m12", "Month 12", 4),
        ("ADNIGO", 1, "sc", "Screening", 1),
        ("ADNIGO", 2, "bl", "Baseline", 2),
        ("ADNI2", 1, "v01", "Screening - New Pt", 1),
        ("ADNI2", 2, "v02", "Screening MRI - New Pt", 2),
        ("ADNI2", 3, "v03", "Baseline - New Pt", 3),
        ("ADNI2", 4, "v11", "ADNI2 Year 1 Visit", 4),
        ("ADNI2", 5, "v21", "ADNI2 Year 2 Visit", 5),
        ("ADNI2", 6, "init", "ADNI2 Initial Visit - Continuing Pt", 6),
        ("ADNI3", 1, "sc", "Screening - New Pt", 1),
        ("ADNI3", 2, "bl", "Baseline - New Pt", 2),
        ("ADNI3", 3, "init", "ADNI3 Initial Visit - Continuing Pt", 3),
        ("ADNI4", 1, "4_sc", "Screening - New Pt", 1),
        ("ADNI4", 2, "4_bl", "Baseline - New Pt", 2),
        ("ADNI4", 3, "4_init", "ADNI4 Initial Visit - Continuing Pt", 4),
    ]
    return pd.DataFrame(rows, columns=["PHASE", "ID", "VISCODE", "VISNAME", "VISORDER"])


VISITS_DF = make_visits_df()


# ------------------------------------------------------------------
# 1. Baseline diagnosis extraction
# ------------------------------------------------------------------


def test_baseline_viscode_map_excludes_rollover_continuing_pt_visits():
    bmap = C.build_baseline_viscode_map(VISITS_DF)
    assert bmap == {
        "ADNI1": "bl",
        "ADNIGO": "bl",
        "ADNI2": "v03",
        "ADNI3": "bl",
        "ADNI4": "4_bl",
    }
    # "init"/"4_init" (rollover continuing-pt visits) must never be picked.
    assert "init" not in bmap.values()
    assert "4_init" not in bmap.values()


def test_build_baseline_diagnosis_extracts_only_original_protocol_baseline_record():
    dxsum = pd.DataFrame(
        [
            {"RID": 1, "ORIGPROT": "ADNI1", "COLPROT": "ADNI1", "VISCODE": "bl", "EXAMDATE": "2010-01-01", "DIAGNOSIS": "CN"},
            # a follow-up-visit diagnosis for RID 1 must NOT be picked as baseline
            {"RID": 1, "ORIGPROT": "ADNI1", "COLPROT": "ADNI1", "VISCODE": "m06", "EXAMDATE": "2010-07-01", "DIAGNOSIS": "MCI"},
            {"RID": 3, "ORIGPROT": "ADNI2", "COLPROT": "ADNI2", "VISCODE": "v03", "EXAMDATE": "2011-01-01", "DIAGNOSIS": "MCI"},
        ]
    )
    enrollment = pd.DataFrame({"RID": [1, 3], "ORIGPROT": ["ADNI1", "ADNI2"], "ENRLDT": ["2010-01-01", "2011-01-01"]})
    bmap = C.build_baseline_viscode_map(VISITS_DF)

    result, qc = C.build_baseline_diagnosis(dxsum, enrollment, bmap)

    assert set(result["RID"]) == {1, 3}
    assert result.loc[result["RID"] == 1, "DX_BASELINE"].item() == "CN"
    assert result.loc[result["RID"] == 3, "DX_BASELINE"].item() == "MCI"
    assert qc["n_verified_baseline_dx"] == 2


# ------------------------------------------------------------------
# 2. Screen-failure handling
# ------------------------------------------------------------------


def test_build_enrollment_table_drops_visit_not_conducted():
    bmap = C.build_baseline_viscode_map(VISITS_DF)
    registry = pd.DataFrame(
        [
            {"RID": 1, "ORIGPROT": "ADNI1", "COLPROT": "ADNI1", "VISCODE": "bl", "RGCONDCT": "Yes", "VISTYPE": None, "EXAMDATE": "2010-01-01"},
            # RGCONDCT == "No" -> visit not conducted -> not enrolled
            {"RID": 2, "ORIGPROT": "ADNI1", "COLPROT": "ADNI1", "VISCODE": "bl", "RGCONDCT": "No", "VISTYPE": None, "EXAMDATE": "2010-01-01"},
            {"RID": 3, "ORIGPROT": "ADNI2", "COLPROT": "ADNI2", "VISCODE": "v03", "RGCONDCT": None, "VISTYPE": "Standard", "EXAMDATE": "2011-01-01"},
            # VISTYPE == "Not done" -> visit not conducted -> not enrolled
            {"RID": 4, "ORIGPROT": "ADNI2", "COLPROT": "ADNI2", "VISCODE": "v03", "RGCONDCT": None, "VISTYPE": "Not done", "EXAMDATE": "2011-01-01"},
        ]
    )
    enrollment = C.build_enrollment_table(registry, bmap)
    assert set(enrollment["RID"]) == {1, 3}


def test_screen_failure_trap_excludes_baseline_dx_with_no_enrollment_record():
    """
    A participant with a DXSUM record at the baseline VISCODE, but no
    matching enrollment record (a screen failure -- e.g. the visit was
    never actually conducted), must NOT appear in
    build_baseline_diagnosis()'s output.
    """
    bmap = C.build_baseline_viscode_map(VISITS_DF)
    dxsum = pd.DataFrame(
        [
            {"RID": 1, "ORIGPROT": "ADNI1", "COLPROT": "ADNI1", "VISCODE": "bl", "EXAMDATE": "2010-01-01", "DIAGNOSIS": "CN"},
            {"RID": 2, "ORIGPROT": "ADNI1", "COLPROT": "ADNI1", "VISCODE": "bl", "EXAMDATE": "2010-01-05", "DIAGNOSIS": "CN"},
        ]
    )
    # Only RID 1 has a qualifying enrollment record; RID 2 is a screen failure.
    enrollment = pd.DataFrame({"RID": [1], "ORIGPROT": ["ADNI1"], "ENRLDT": ["2010-01-01"]})

    result, qc = C.build_baseline_diagnosis(dxsum, enrollment, bmap)

    assert list(result["RID"]) == [1]
    assert qc["n_screen_failure_records_excluded"] == 1
    assert qc["n_candidate_baseline_dx_records"] == 2


# ------------------------------------------------------------------
# 3. Visit-code mapping
# ------------------------------------------------------------------


def test_visit_month_map_tiers_and_confidence():
    m = C.build_visit_month_map(VISITS_DF)
    m = m.set_index(["PHASE", "VISCODE"])

    assert m.loc[("ADNI1", "bl"), "canonical_month"] == 0
    assert m.loc[("ADNI1", "bl"), "mapping_method"] == "viscode_baseline"
    assert m.loc[("ADNI1", "bl"), "mapping_confidence"] == "high"

    assert m.loc[("ADNI1", "m12"), "canonical_month"] == 12
    assert m.loc[("ADNI1", "m12"), "mapping_method"] == "viscode_month_label"
    assert m.loc[("ADNI1", "m12"), "mapping_confidence"] == "high"

    assert m.loc[("ADNI2", "v11"), "canonical_month"] == 12
    assert m.loc[("ADNI2", "v11"), "mapping_method"] == "viscode_year_label_inferred"
    assert m.loc[("ADNI2", "v11"), "mapping_confidence"] == "medium"

    assert pd.isna(m.loc[("ADNI1", "sc"), "canonical_month"])
    assert m.loc[("ADNI1", "sc"), "mapping_method"] == "unmapped"
    assert m.loc[("ADNI1", "sc"), "mapping_confidence"] == "none"

    # Rollover "Continuing Pt" visits must never be treated as baseline.
    assert m.loc[("ADNI2", "init"), "mapping_method"] == "unmapped"


def test_snap_to_target_month_respects_documented_tolerance():
    assert C.snap_to_target_month(0) == 0
    assert C.snap_to_target_month(13) == 12  # within tolerance of 2
    assert pd.isna(C.snap_to_target_month(15))  # 3 away from both 12 and 18 -- outside tolerance
    assert pd.isna(C.snap_to_target_month(float("nan")))


# ------------------------------------------------------------------
# 4. Duplicate resolution
# ------------------------------------------------------------------


def test_resolve_cognitive_duplicates_keeps_latest_and_flags_conflicts():
    df = pd.DataFrame(
        [
            {"RID": 1, "COLPROT": "ADNI1", "VISCODE": "bl", "EXAMDATE": pd.Timestamp("2010-01-01"), "SCORE": 20},
            # same key, later date, DIFFERENT value -> conflict, latest kept
            {"RID": 1, "COLPROT": "ADNI1", "VISCODE": "bl", "EXAMDATE": pd.Timestamp("2010-01-10"), "SCORE": 22},
            # different key entirely -> untouched
            {"RID": 2, "COLPROT": "ADNI1", "VISCODE": "bl", "EXAMDATE": pd.Timestamp("2010-02-01"), "SCORE": 25},
        ]
    )
    deduped, qc = C.resolve_cognitive_duplicates(
        df, key_cols=["RID", "COLPROT", "VISCODE"], value_col="SCORE", date_col="EXAMDATE"
    )
    assert len(deduped) == 2
    kept = deduped.loc[deduped["RID"] == 1, "SCORE"].item()
    assert kept == 22  # latest EXAMDATE kept
    assert qc["n_records_removed"] == 1
    assert qc["n_duplicate_groups"] == 1
    assert qc["n_conflicting_groups"] == 1


def test_resolve_cognitive_duplicates_no_conflict_when_values_agree():
    df = pd.DataFrame(
        [
            {"RID": 1, "COLPROT": "ADNI1", "VISCODE": "bl", "EXAMDATE": pd.Timestamp("2010-01-01"), "SCORE": 20},
            {"RID": 1, "COLPROT": "ADNI1", "VISCODE": "bl", "EXAMDATE": pd.Timestamp("2010-01-10"), "SCORE": 20},
        ]
    )
    _, qc = C.resolve_cognitive_duplicates(
        df, key_cols=["RID", "COLPROT", "VISCODE"], value_col="SCORE", date_col="EXAMDATE"
    )
    assert qc["n_records_removed"] == 1
    assert qc["n_conflicting_groups"] == 0


# ------------------------------------------------------------------
# Plasma fixtures
# ------------------------------------------------------------------


def make_fuji_base_like_df():
    """
    A hand-built stand-in for what _clean_fuji_base() produces, used
    directly by tests that only need to exercise the downstream
    builders (build_platform_long / build_ptau217_long /
    build_abeta_ratio_long) without depending on _clean_fuji_base's
    own VISITS-merge behavior (that's covered separately).
    """
    rows = [
        # RID 1: Quanterix baseline + follow-up (eligible on Quanterix);
        # Fujirebio has a baseline only, no follow-up (NOT eligible on Fujirebio).
        {"RID": 1, "PHASE": "ADNI4", "VISCODE": "4_bl", "EXAMDATE": pd.Timestamp("2024-01-01"), "VISIT_MONTH_RAW": 0, "VISIT_MONTH": 0, "mapping_source": "viscode2", "mapping_confidence": "high", "GFAP_Q": 100.0, "GFAP_Q_QC_REASON": None, "GFAP_F": 50.0, "GFAP_F_QC_REASON": None},
        {"RID": 1, "PHASE": "ADNI4", "VISCODE": "4_m12", "EXAMDATE": pd.Timestamp("2025-01-01"), "VISIT_MONTH_RAW": 12, "VISIT_MONTH": 12, "mapping_method": "viscode_month_label", "mapping_confidence": "high", "GFAP_Q": 110.0, "GFAP_Q_QC_REASON": None, "GFAP_F": None, "GFAP_F_QC_REASON": None},
    ]
    return pd.DataFrame(rows)


def make_demog_and_enrollment():
    demog = pd.DataFrame({"RID": [1, 2], "DOB": [pd.Timestamp("1950-01-15"), pd.Timestamp("1945-06-15")], "SEX": ["Male", "Female"]})
    enrollment = pd.DataFrame({"RID": [1, 2], "ORIGPROT": ["ADNI4", "ADNI4"], "ENRLDT": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01")]})
    return demog, enrollment


# ------------------------------------------------------------------
# 5. Assay-platform separation
# ------------------------------------------------------------------


def test_build_platform_long_keeps_quanterix_and_fujirebio_as_separate_rows():
    fuji_base = make_fuji_base_like_df()
    demog, enrollment = make_demog_and_enrollment()
    long_df = P.build_platform_long(fuji_base, demog, enrollment, analyte="GFAP", quanterix_col="GFAP_Q", fujirebio_col="GFAP_F")

    assert set(long_df["PLATFORM"].unique()) == {"Quanterix", "Fujirebio"}
    q_baseline = long_df[(long_df["PLATFORM"] == "Quanterix") & (long_df["VISCODE"] == "4_bl")]["GFAP"].item()
    f_baseline = long_df[(long_df["PLATFORM"] == "Fujirebio") & (long_df["VISCODE"] == "4_bl")]["GFAP"].item()
    assert q_baseline == 100.0
    assert f_baseline == 50.0  # never averaged/mixed with the Quanterix value


# ------------------------------------------------------------------
# 6. pTau217 lot-bias flagging
# ------------------------------------------------------------------


def test_ptau217_lot_bias_flag_matches_batch3_comment_field_only():
    fuji_df = pd.DataFrame(
        [
            {"PHASE": "ADNI4", "PTID": "x", "RID": 1, "VISCODE": "4_bl", "VISCODE2": "bl", "EXAMDATE": "2025-01-01",
             "Primary": "BLD", "Additive": "EDT", "pT217_F": 0.2, "AB42_F": 20.0, "AB40_F": 200.0, "AB42_AB40_F": 0.1,
             "pT217_AB42_F": 0.01, "NfL_Q": 10.0, "GFAP_Q": 100.0, "NfL_F": None, "GFAP_F": None,
             "Comment": "Batch 3: QC drift noted; results validated. Refer to Methods Special Note.", "update_stamp": None},
            {"PHASE": "ADNI4", "PTID": "y", "RID": 2, "VISCODE": "4_bl", "VISCODE2": "bl", "EXAMDATE": "2025-01-01",
             "Primary": "BLD", "Additive": "EDT", "pT217_F": 0.3, "AB42_F": 22.0, "AB40_F": 210.0, "AB42_AB40_F": 0.11,
             "pT217_AB42_F": 0.014, "NfL_Q": 11.0, "GFAP_Q": 105.0, "NfL_F": None, "GFAP_F": None,
             "Comment": None, "update_stamp": None},
        ]
    )
    _, enrollment = make_demog_and_enrollment()
    cleaned, _ = P._clean_fuji_base(fuji_df, VISITS_DF, enrollment)
    flag_by_rid = cleaned.set_index("RID")["PTAU217_LOT_BIAS_FLAG"]
    assert flag_by_rid.loc[1] == True  # noqa: E712
    assert flag_by_rid.loc[2] == False  # noqa: E712


# ------------------------------------------------------------------
# 7. Biomarker baseline selection
# ------------------------------------------------------------------


def test_select_biomarker_baseline_never_substitutes_a_non_baseline_visit():
    df = pd.DataFrame(
        [
            {"RID": 1, "VISIT_MONTH_RAW": 0, "EXAMDATE": pd.Timestamp("2024-01-01"), "VALUE": 5.0},
            {"RID": 1, "VISIT_MONTH_RAW": 12, "EXAMDATE": pd.Timestamp("2025-01-01"), "VALUE": 6.0},
            # RID 2 has no month-0 record at all -- must get NO baseline, not the nearest one.
            {"RID": 2, "VISIT_MONTH_RAW": 12, "EXAMDATE": pd.Timestamp("2025-01-01"), "VALUE": 9.0},
        ]
    )
    baseline = P.select_biomarker_baseline(df, rid_col="RID", value_col="VALUE", month_col="VISIT_MONTH_RAW", date_col="EXAMDATE")
    assert set(baseline["RID"]) == {1}
    assert baseline.loc[baseline["RID"] == 1, "BASELINE_VALUE"].item() == 5.0


# ------------------------------------------------------------------
# 8. Same-platform longitudinal requirement
# ------------------------------------------------------------------


def test_biomarker_eligibility_is_computed_within_platform_not_mixed():
    fuji_base = make_fuji_base_like_df()
    demog, enrollment = make_demog_and_enrollment()
    long_df = P.build_platform_long(fuji_base, demog, enrollment, analyte="GFAP", quanterix_col="GFAP_Q", fujirebio_col="GFAP_F")

    q_eligible = long_df[long_df["PLATFORM"] == "Quanterix"]["BIOMARKER_ELIGIBLE"].iloc[0]
    f_eligible = long_df[long_df["PLATFORM"] == "Fujirebio"]["BIOMARKER_ELIGIBLE"].iloc[0]
    # Quanterix has both a baseline and a follow-up value for RID 1 -> eligible.
    assert q_eligible == True  # noqa: E712
    # Fujirebio has only a baseline value for RID 1 (no follow-up) -> NOT eligible,
    # even though Quanterix's follow-up exists for the same participant/visit.
    assert f_eligible == False  # noqa: E712


# ------------------------------------------------------------------
# 9. Positive-value requirement for log analysis
# ------------------------------------------------------------------


def test_biomarker_eligibility_requires_strictly_positive_baseline():
    df = pd.DataFrame(
        [
            {"RID": 1, "VISIT_MONTH_RAW": 0, "VALUE": -1.0, "BASELINE_AGE": 70.0, "SEX": "Male"},
            {"RID": 1, "VISIT_MONTH_RAW": 12, "VALUE": 5.0, "BASELINE_AGE": 70.0, "SEX": "Male"},
            {"RID": 2, "VISIT_MONTH_RAW": 0, "VALUE": 3.0, "BASELINE_AGE": 65.0, "SEX": "Female"},
            {"RID": 2, "VISIT_MONTH_RAW": 12, "VALUE": 4.0, "BASELINE_AGE": 65.0, "SEX": "Female"},
        ]
    )
    baseline = P.select_biomarker_baseline(df, rid_col="RID", value_col="VALUE", month_col="VISIT_MONTH_RAW", date_col="VISIT_MONTH_RAW")
    df = df.merge(baseline[["RID", "BASELINE_VALUE"]], on="RID", how="left")
    df = P.add_biomarker_eligibility(df, rid_col="RID", value_col="VALUE", baseline_value_col="BASELINE_VALUE", month_col="VISIT_MONTH_RAW", group_cols=["RID"])

    assert df.loc[df["RID"] == 1, "BIOMARKER_ELIGIBLE"].iloc[0] == False  # noqa: E712 -- negative baseline
    assert df.loc[df["RID"] == 2, "BIOMARKER_ELIGIBLE"].iloc[0] == True  # noqa: E712 -- positive baseline


# ------------------------------------------------------------------
# 10. Abeta42/Abeta40 validated-field selection
# ------------------------------------------------------------------


def test_abeta_ratio_same_sample_verification_flags_mismatch():
    fuji_base = pd.DataFrame(
        [
            # both present, ratio present -> consistent
            {"RID": 1, "PHASE": "ADNI4", "VISCODE": "4_bl", "EXAMDATE": pd.Timestamp("2024-01-01"), "VISIT_MONTH_RAW": 0, "VISIT_MONTH": 0, "mapping_source": "viscode2", "mapping_confidence": "high", "AB42_F": 20.0, "AB40_F": 200.0, "AB42_AB40_F": 0.1},
            # both present, ratio MISSING -> inconsistent / same-sample verification failure
            {"RID": 2, "PHASE": "ADNI4", "VISCODE": "4_bl", "EXAMDATE": pd.Timestamp("2024-01-01"), "VISIT_MONTH_RAW": 0, "VISIT_MONTH": 0, "mapping_source": "viscode2", "mapping_confidence": "high", "AB42_F": 22.0, "AB40_F": 210.0, "AB42_AB40_F": None},
        ]
    )
    demog = pd.DataFrame({"RID": [1, 2], "DOB": [pd.Timestamp("1950-01-15"), pd.Timestamp("1950-01-15")], "SEX": ["Male", "Female"]})
    enrollment = pd.DataFrame({"RID": [1, 2], "ORIGPROT": ["ADNI4", "ADNI4"], "ENRLDT": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01")]})

    result, qc = P.build_abeta_ratio_long(fuji_base, demog, enrollment)
    assert result.loc[result["RID"] == 1, "SAME_SAMPLE_VERIFIED"].item() == True  # noqa: E712
    assert result.loc[result["RID"] == 2, "SAME_SAMPLE_VERIFIED"].item() == False  # noqa: E712
    assert qc["n_same_sample_mismatches"] == 1
    # The pre-computed ratio must be used as-is, never recalculated from AB42_F/AB40_F.
    assert result.loc[result["RID"] == 1, "ABETA_RATIO"].item() == 0.1


# ------------------------------------------------------------------
# 11. Analysis eligibility (cognitive)
# ------------------------------------------------------------------


def _base_clinical_row(**overrides):
    row = {
        "RID": 1,
        "DX_BASELINE_FIXED": "CN",
        "ADAS_COG13_BASELINE": 10.0,
        "BASELINE_AGE": 70.0,
        "SEX": "Male",
        "VISIT_MONTH": 0,
        "ADAS_COG13": 10.0,
        "MMSE": 28.0,
        "MMSE_BASELINE": 28.0,
    }
    row.update(overrides)
    return row


def test_cognitive_eligibility_requires_all_five_conditions():
    rows = [
        # RID 1: fully eligible (baseline row + a follow-up row)
        _base_clinical_row(RID=1, VISIT_MONTH=0),
        _base_clinical_row(RID=1, VISIT_MONTH=12, ADAS_COG13=12.0, MMSE=27.0),
        # RID 2: missing baseline diagnosis -> not eligible
        _base_clinical_row(RID=2, DX_BASELINE_FIXED=None, VISIT_MONTH=0),
        _base_clinical_row(RID=2, DX_BASELINE_FIXED=None, VISIT_MONTH=12, ADAS_COG13=12.0, MMSE=27.0),
        # RID 3: no follow-up record at all -> not eligible
        _base_clinical_row(RID=3, VISIT_MONTH=0),
        # RID 4: missing sex -> not eligible
        _base_clinical_row(RID=4, SEX=None, VISIT_MONTH=0),
        _base_clinical_row(RID=4, SEX=None, VISIT_MONTH=12, ADAS_COG13=12.0, MMSE=27.0),
    ]
    df = pd.DataFrame(rows)
    df = C.add_cognitive_eligibility(df)

    eligible_by_rid = df.groupby("RID")["ADAS_COG13_ELIGIBLE"].any()
    assert eligible_by_rid.loc[1] == True  # noqa: E712
    assert eligible_by_rid.loc[2] == False  # noqa: E712
    assert eligible_by_rid.loc[3] == False  # noqa: E712
    assert eligible_by_rid.loc[4] == False  # noqa: E712


# ------------------------------------------------------------------
# 12. Baseline diagnosis validation against ADSL.DX
# ------------------------------------------------------------------


def test_resolve_baseline_diagnosis_with_validation_categorizes_correctly():
    dxsum_baseline_df = pd.DataFrame(
        [
            {"RID": 1, "DX_BASELINE": "CN"},       # agrees with ADSL -> unchanged
            {"RID": 2, "DX_BASELINE": "MCI"},       # disagrees with ADSL -> changed
            {"RID": 4, "DX_BASELINE": "CN"},        # not in ADSL enrolled set -> dxsum-only
        ]
    )
    adsl_df = pd.DataFrame(
        {
            "SUBJID": [1, 2, 3, 5],
            "ENRLFL": ["Y", "Y", "Y", "Y"],
            # ADSL uses the short "DEM" code -- must be normalized to "Dementia".
            "DX": ["CN", "Dementia", "DEM", None],
        }
    )
    result, qc = C.resolve_baseline_diagnosis_with_validation(dxsum_baseline_df, adsl_df)

    assert qc["n_unchanged"] == 1  # RID 1
    assert qc["n_changed"] == 1  # RID 2
    assert qc["n_newly_assigned"] == 1  # RID 3 (only in ADSL)
    assert qc["n_dxsum_only_not_in_adsl"] == 1  # RID 4
    assert qc["n_unresolved"] == 1  # RID 5 (ADSL DX is None)

    result_by_rid = result.set_index("RID")
    assert result_by_rid.loc[1, "DX_BASELINE"] == "CN"
    # ADSL wins on disagreement -- adopted as the validated primary source.
    assert result_by_rid.loc[2, "DX_BASELINE"] == "Dementia"
    assert result_by_rid.loc[2, "DX_BASELINE_SOURCE"] == "adsl_official"
    assert result_by_rid.loc[3, "DX_BASELINE"] == "DEM".replace("DEM", "Dementia")
    # RID 4 keeps this module's own value since ADSL has nothing for it.
    assert result_by_rid.loc[4, "DX_BASELINE"] == "CN"
    assert result_by_rid.loc[4, "DX_BASELINE_SOURCE"] == "dxsum_baseline_viscode_only"
    assert 5 not in result_by_rid.index


# ------------------------------------------------------------------
# 3 (validation pass). VISCODE2-first visit-code mapping
# ------------------------------------------------------------------


def test_parse_viscode2_month_only_matches_bl_and_mNN():
    assert C.parse_viscode2_month("bl") == (0.0, True)
    assert C.parse_viscode2_month("m12") == (12.0, True)
    assert C.parse_viscode2_month("m06") == (6.0, True)
    assert C.parse_viscode2_month("sc") == (None, False)
    assert C.parse_viscode2_month(None) == (None, False)


def test_map_canonical_month_prefers_viscode2_over_crosswalk_and_dates():
    crosswalk_df = C.build_visit_month_map(VISITS_DF)
    df = pd.DataFrame(
        [
            # Tier 1: VISCODE2 resolves directly ("m60" -- exactly the
            # previously-anomalous ADNI2 case) -- must win even though the
            # VISITS crosswalk has no entry for (ADNI2, "m60") at all.
            {"RID": 1, "PHASE": "ADNI2", "VISCODE": "m60", "VISCODE2": "m60", "EXAMDATE": pd.Timestamp("2015-01-01"), "ENRLDT": pd.Timestamp("2010-01-01")},
            # Tier 2: VISCODE2 missing, but (PHASE, VISCODE) resolves via the
            # VISNAME-text crosswalk (ADNI2's "v11" -> Year 1 Visit -> 12mo).
            {"RID": 2, "PHASE": "ADNI2", "VISCODE": "v11", "VISCODE2": None, "EXAMDATE": pd.Timestamp("2015-01-01"), "ENRLDT": pd.Timestamp("2010-01-01")},
            # Tier 3: neither resolves, but EXAMDATE - ENRLDT gives ~6 months.
            {"RID": 3, "PHASE": "ADNI2", "VISCODE": "nv", "VISCODE2": None, "EXAMDATE": pd.Timestamp("2010-07-02"), "ENRLDT": pd.Timestamp("2010-01-01")},
            # Tier 4: nothing resolves -- must stay unmapped, not guessed.
            {"RID": 4, "PHASE": "ADNI2", "VISCODE": "nv", "VISCODE2": None, "EXAMDATE": None, "ENRLDT": None},
            # Negative gap (pre-enrollment/screening record): must NOT be
            # forced into the positive target-month framework via tier 3.
            {"RID": 5, "PHASE": "ADNI2", "VISCODE": "nv", "VISCODE2": None, "EXAMDATE": pd.Timestamp("2009-12-01"), "ENRLDT": pd.Timestamp("2010-01-01")},
        ]
    )
    mapped = C.map_canonical_month(df, crosswalk_df)
    result = mapped.set_index("RID")

    assert result.loc[1, "canonical_month"] == 60
    assert result.loc[1, "mapping_source"] == "viscode2"
    assert result.loc[1, "mapping_confidence"] == "high"

    assert result.loc[2, "canonical_month"] == 12
    assert result.loc[2, "mapping_source"].startswith("viscode_crosswalk_fallback:")
    assert result.loc[2, "mapping_confidence"] == "medium"

    assert result.loc[3, "canonical_month"] == 6
    assert result.loc[3, "mapping_source"] == "date_elapsed_from_enrollment"
    assert result.loc[3, "mapping_confidence"] == "medium"

    assert pd.isna(result.loc[4, "canonical_month"])
    assert result.loc[4, "mapping_source"] == "unmapped"
    assert result.loc[4, "mapping_confidence"] == "none"

    assert pd.isna(result.loc[5, "canonical_month"])
    assert result.loc[5, "mapping_source"] == "unmapped"


# ------------------------------------------------------------------
# 4 (validation pass). MMSE baseline validation against ADSL.MMSCORE
# ------------------------------------------------------------------


def test_validate_mmse_baseline_matches_adsl_and_flags_long_intervals():
    mmse_baseline_df = pd.DataFrame(
        [
            {"RID": 1, "MMSE_BASELINE": 28.0, "MMSE_BASELINE_METHOD": "screening_viscode_fallback", "MMSE_BASELINE_DATE": pd.Timestamp("2010-01-01")},
            {"RID": 2, "MMSE_BASELINE": 27.0, "MMSE_BASELINE_METHOD": "screening_viscode_fallback", "MMSE_BASELINE_DATE": pd.Timestamp("2010-01-01")},
            {"RID": 3, "MMSE_BASELINE": 25.0, "MMSE_BASELINE_METHOD": "baseline_viscode", "MMSE_BASELINE_DATE": pd.Timestamp("2010-03-01")},
        ]
    )
    adsl_df = pd.DataFrame(
        {"SUBJID": [1, 2, 3], "ENRLFL": ["Y", "Y", "Y"], "MMSCORE": [28.0, 27.0, 25.0]}
    )
    enrollment_df = pd.DataFrame(
        {
            "RID": [1, 2, 3],
            "ENRLDT": [
                pd.Timestamp("2010-01-20"),  # 19-day gap -- within the 90-day window
                pd.Timestamp("2010-05-01"),  # ~120-day gap -- flagged as long
                pd.Timestamp("2010-03-01"),
            ],
        }
    )
    interval_df, qc = C.validate_mmse_baseline(mmse_baseline_df, adsl_df, enrollment_df)

    assert qc["n_match"] == 3
    assert qc["n_mismatch"] == 0
    assert qc["n_screening_sourced"] == 2
    assert qc["n_baseline_visit_sourced"] == 1
    assert qc["n_long_interval_flagged"] == 1

    by_rid = interval_df.set_index("RID")
    assert bool(by_rid.loc[1, "LONG_INTERVAL_FLAG"]) is False
    assert bool(by_rid.loc[2, "LONG_INTERVAL_FLAG"]) is True
    assert 3 not in by_rid.index  # baseline-visit-sourced participant has no "interval" concept


ALL_TESTS = [
    test_baseline_viscode_map_excludes_rollover_continuing_pt_visits,
    test_build_baseline_diagnosis_extracts_only_original_protocol_baseline_record,
    test_build_enrollment_table_drops_visit_not_conducted,
    test_screen_failure_trap_excludes_baseline_dx_with_no_enrollment_record,
    test_visit_month_map_tiers_and_confidence,
    test_snap_to_target_month_respects_documented_tolerance,
    test_resolve_cognitive_duplicates_keeps_latest_and_flags_conflicts,
    test_resolve_cognitive_duplicates_no_conflict_when_values_agree,
    test_build_platform_long_keeps_quanterix_and_fujirebio_as_separate_rows,
    test_ptau217_lot_bias_flag_matches_batch3_comment_field_only,
    test_select_biomarker_baseline_never_substitutes_a_non_baseline_visit,
    test_biomarker_eligibility_is_computed_within_platform_not_mixed,
    test_biomarker_eligibility_requires_strictly_positive_baseline,
    test_abeta_ratio_same_sample_verification_flags_mismatch,
    test_cognitive_eligibility_requires_all_five_conditions,
    test_resolve_baseline_diagnosis_with_validation_categorizes_correctly,
    test_parse_viscode2_month_only_matches_bl_and_mNN,
    test_map_canonical_month_prefers_viscode2_over_crosswalk_and_dates,
    test_validate_mmse_baseline_matches_adsl_and_flags_long_intervals,
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
