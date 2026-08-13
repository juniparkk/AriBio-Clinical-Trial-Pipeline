# ============================================================
# ADNI_PLASMA -- plasma biomarker preprocessing, kept in four
# strictly separate assay families per the audit's platform/assay
# findings (adni_merge_risk_report.csv R6/R7):
#
#   A. Gothenburg plasma pTau181   (UGOTPTAU181, ADNI1/GO/2 only)
#   B. UPenn pTau217 / Abeta42 / Abeta40 / Abeta42:Abeta40 (Fujirebio)
#   C. GFAP, by assay platform (Quanterix vs Fujirebio -- NOT pooled)
#   D. NfL,  by assay platform (Quanterix vs Fujirebio -- NOT pooled)
#
# pTau181 and pTau217 are never combined into one "p-tau" series (they
# are chemically distinct phosphorylation-site epitopes on different
# platforms). GFAP/NfL Quanterix and Fujirebio values are never
# mathematically harmonized here -- no validated cross-platform
# conversion was found in the reviewed methods documentation, and
# inventing one is explicitly out of scope.
#
# Same file I/O and participant-level-data discipline as adni_cohort.py
# (see that module's header docstring) -- every function here is pure,
# no I/O at import time, and no function returns anything intended for
# outputs/ that still contains a participant-level row.
# ============================================================

import numpy as np
import pandas as pd

from adni_cohort import (
    build_visit_month_map,
    compute_age_years,
    map_canonical_month,
    parse_viscode2_month,
    snap_to_target_month,
)

# ------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------

_SENTINEL_CODES = {
    -1: "not_available",
    -4: "insufficient_sample",
    -5: "sample_quantity_not_sufficient",
}


def _clean_sentinels(series):
    """
    Replace ADNI's numeric sentinel missing codes with NaN, returning
    (cleaned_series, reason_series) so the *reason* a value is missing
    (assay-reported "insufficient sample" vs. simply never measured)
    is preserved as an explicit QC field rather than collapsed into
    ordinary missingness.
    """
    reason = series.map(_SENTINEL_CODES)
    cleaned = series.where(~series.isin(_SENTINEL_CODES.keys()), np.nan)
    return cleaned, reason


def resolve_visit_duplicates(df, key_cols, date_col):
    """
    Same deterministic rule as adni_cohort.resolve_cognitive_duplicates:
    within each key_cols group, keep the record with the latest
    date_col (ties broken by original row order). Returns
    (deduped_df, qc_dict).
    """
    df = df.copy()
    df["_dupe_group"] = df.groupby(key_cols, dropna=False).ngroup()
    group_sizes = df.groupby("_dupe_group")["_dupe_group"].transform("size")
    n_duplicate_groups = df.loc[group_sizes > 1, "_dupe_group"].nunique()

    df_sorted = df.sort_values(date_col, na_position="first")
    deduped = df_sorted.drop_duplicates(key_cols, keep="last").drop(columns=["_dupe_group"])

    qc = {
        "n_input_records": len(df),
        "n_duplicate_groups": int(n_duplicate_groups),
        "n_records_removed": len(df) - len(deduped),
    }
    return deduped.reset_index(drop=True), qc


def _join_demographics_and_age(df, demog_df, enrollment_df, date_col):
    df = df.merge(demog_df[["RID", "DOB", "SEX"]], on="RID", how="left")
    df = df.merge(enrollment_df[["RID", "ENRLDT"]], on="RID", how="left")
    df["AGE_AT_VISIT"] = df.apply(lambda r: compute_age_years(r["DOB"], r[date_col]), axis=1)
    df["BASELINE_AGE"] = df.apply(lambda r: compute_age_years(r["DOB"], r["ENRLDT"]), axis=1)
    return df


def select_biomarker_baseline(df, rid_col, value_col, month_col, date_col):
    """
    Participant-specific biomarker baseline, per requirement 7: the
    assay's OWN baseline-visit record (canonical VISIT_MONTH == 0),
    not assumed to be the same row as the clinical baseline. If a
    participant has no assay measurement at month 0 (assay-specific
    availability can differ from the clinical visit schedule), their
    biomarker baseline is left missing rather than substituted from a
    nearby visit.

    Returns a per-participant DataFrame with columns:
    <rid_col>, BASELINE_VALUE, BASELINE_VISIT_MONTH, BASELINE_DATE,
    BASELINE_SELECTION_METHOD (constant "visit_month_0" for all
    resolved rows -- kept as an explicit column so a future baseline
    rule change is visible in the data, not just in code).
    """
    at_baseline = df[(df[month_col] == 0) & df[value_col].notna()].copy()
    at_baseline = at_baseline.sort_values(date_col).drop_duplicates(rid_col, keep="first")
    out = at_baseline[[rid_col, value_col, date_col]].rename(
        columns={value_col: "BASELINE_VALUE", date_col: "BASELINE_DATE"}
    )
    out["BASELINE_VISIT_MONTH"] = 0
    out["BASELINE_SELECTION_METHOD"] = "visit_month_0"
    return out.reset_index(drop=True)


def add_biomarker_eligibility(df, rid_col, value_col, baseline_value_col, month_col, group_cols):
    """
    Generic biomarker-endpoint eligibility flag, requiring (per
    participant, within `group_cols` -- e.g. RID alone, or RID+
    PLATFORM for the platform-split families so a participant is never
    judged eligible by mixing platforms): a valid baseline value, at
    least one valid follow-up (VISIT_MONTH > 0) value, baseline age,
    sex, and a strictly positive baseline value (required for any
    later log-transformed analysis). QC failures (sentinel-derived
    missingness) are already NaN by this point, so "valid" here always
    means "not NaN, not sentinel-derived".
    """
    df = df.copy()
    has_baseline = df[baseline_value_col].notna() & (df[baseline_value_col] > 0)
    has_baseline_age = df["BASELINE_AGE"].notna()
    has_sex = df["SEX"].notna()

    df["_is_valid_followup"] = (
        (df[month_col] > 0) & df[value_col].notna() & (df[value_col] > 0)
    )
    followup_flag = (
        df.groupby(group_cols)["_is_valid_followup"]
        .any()
        .rename("_has_followup")
        .reset_index()
    )
    df = df.drop(columns=["_is_valid_followup"]).merge(followup_flag, on=group_cols, how="left")

    df["BIOMARKER_ELIGIBLE"] = has_baseline & has_baseline_age & has_sex & df["_has_followup"].fillna(False)
    df = df.drop(columns=["_has_followup"])
    return df


# ------------------------------------------------------------------
# A. Gothenburg plasma pTau181 (ADNI1/GO/2 only, no phase column)
# ------------------------------------------------------------------


def build_ptau181_long(ugot_df, demog_df, enrollment_df):
    """
    Clean, deduplicated, visit-month-mapped, baseline-resolved,
    eligibility-flagged pTau181 long table.

    No ORIGPROT/COLPROT/PHASE column exists in this file (Risk R1), so
    the phase-aware VISITS crosswalk used elsewhere doesn't apply here.
    Visit-month mapping instead uses VISCODE2 directly (confirmed
    populated on 100% of records in this file, and, per
    parse_viscode2_month()'s docstring, phase-independent by
    construction) -- "high" confidence -- falling back to date-elapsed-
    from-enrollment for the rare record where VISCODE2 is absent.
    """
    df = ugot_df.copy()
    df["EXAMDATE"] = pd.to_datetime(df["EXAMDATE"], errors="coerce")
    df["BELOW_LLOQ"] = df["COMMENT"] == "<LLOQ"

    deduped, dedup_qc = resolve_visit_duplicates(df, key_cols=["RID", "VISCODE"], date_col="EXAMDATE")
    deduped = deduped.merge(enrollment_df[["RID", "ENRLDT"]], on="RID", how="left")

    parsed = deduped["VISCODE2"].apply(parse_viscode2_month)
    v2_month = parsed.apply(lambda t: t[0])
    v2_ok = parsed.apply(lambda t: t[1])

    month = pd.Series(np.nan, index=deduped.index, dtype="float64")
    source = pd.Series("unmapped", index=deduped.index, dtype="object")
    confidence = pd.Series("none", index=deduped.index, dtype="object")

    month[v2_ok] = v2_month[v2_ok]
    source[v2_ok] = "viscode2"
    confidence[v2_ok] = "high"

    examdate = pd.to_datetime(deduped["EXAMDATE"], errors="coerce")
    enrldt = pd.to_datetime(deduped["ENRLDT"], errors="coerce")
    elapsed_days = (examdate - enrldt).dt.days
    # elapsed_days >= 0 only -- see map_canonical_month()'s matching guard
    # in adni_cohort.py: a negative gap predates enrollment and must not be
    # forced into the positive target-month framework.
    date_ok = (~v2_ok) & elapsed_days.notna() & (elapsed_days >= 0)
    month[date_ok] = (elapsed_days[date_ok] / 30.4375).round()
    source[date_ok] = "date_elapsed_from_enrollment"
    confidence[date_ok] = "medium"

    deduped["VISIT_MONTH_RAW"] = month
    deduped["VISIT_MAPPING_SOURCE"] = source
    deduped["VISIT_MAPPING_CONFIDENCE"] = confidence
    deduped["VISIT_MONTH"] = deduped["VISIT_MONTH_RAW"].apply(snap_to_target_month)
    deduped = deduped.drop(columns=["ENRLDT"])

    deduped = _join_demographics_and_age(deduped, demog_df, enrollment_df, date_col="EXAMDATE")

    baseline = select_biomarker_baseline(
        deduped, rid_col="RID", value_col="PLASMAPTAU181", month_col="VISIT_MONTH_RAW", date_col="EXAMDATE"
    )
    baseline = baseline.rename(
        columns={
            "BASELINE_VALUE": "PTAU181_BASELINE",
            "BASELINE_DATE": "PTAU181_BASELINE_DATE",
            "BASELINE_VISIT_MONTH": "PTAU181_BASELINE_VISIT_MONTH",
        }
    )
    deduped = deduped.merge(baseline, on="RID", how="left")

    deduped = add_biomarker_eligibility(
        deduped,
        rid_col="RID",
        value_col="PLASMAPTAU181",
        baseline_value_col="PTAU181_BASELINE",
        month_col="VISIT_MONTH_RAW",
        group_cols=["RID"],
    )

    final_cols = [
        "RID",
        "VISCODE",
        "EXAMDATE",
        "VISIT_MONTH_RAW",
        "VISIT_MONTH",
        "VISIT_MAPPING_SOURCE",
        "VISIT_MAPPING_CONFIDENCE",
        "PLASMAPTAU181",
        "BELOW_LLOQ",
        "PTAU181_BASELINE",
        "PTAU181_BASELINE_DATE",
        "PTAU181_BASELINE_VISIT_MONTH",
        "BASELINE_SELECTION_METHOD",
        "BASELINE_AGE",
        "AGE_AT_VISIT",
        "SEX",
        "BIOMARKER_ELIGIBLE",
    ]
    result = deduped[final_cols].sort_values(["RID", "VISCODE"]).reset_index(drop=True)

    qc = dict(dedup_qc)
    qc.update(
        {
            "n_below_lloq": int(deduped["BELOW_LLOQ"].sum()),
            "n_unique_participants": result["RID"].nunique(),
            "n_eligible_participants": result.loc[result["BIOMARKER_ELIGIBLE"], "RID"].nunique(),
        }
    )
    return result, qc


# ------------------------------------------------------------------
# B/C/D. UPenn Fujirebio/Quanterix panel: pTau217, Abeta, GFAP, NfL
# ------------------------------------------------------------------


def _clean_fuji_base(fuji_df, visits_df, enrollment_df):
    """
    Shared cleaning pass for the UPenn plasma file: sentinel codes to
    NaN (with a reason column per analyte), deduplication on
    (RID, PHASE, VISCODE), and visit-month mapping via
    map_canonical_month() (VISCODE2 first, VISITS-text crosswalk second,
    date-elapsed-from-enrollment third). Returns the single cleaned/
    deduped base DataFrame every family B/C/D table is built from, so
    all four stay mutually consistent on visit mapping and
    deduplication.

    The previously-anomalous PHASE=ADNI2, VISCODE="m60" record
    (Risk R13 in the original audit) is resolved by this: its VISCODE2
    is "m60", a valid translated code, so it maps to month 60 at "high"
    confidence via tier 1 -- it was never bad data, just a VISCODE that
    doesn't match ADNI2's usual native vocabulary (likely a rollover
    participant's visit recorded under a different phase's naming).
    """
    df = fuji_df.copy()
    df["EXAMDATE"] = pd.to_datetime(df["EXAMDATE"], errors="coerce")

    for col in ["pT217_F", "AB42_F", "AB40_F", "AB42_AB40_F", "pT217_AB42_F", "NfL_Q", "GFAP_Q", "NfL_F", "GFAP_F"]:
        cleaned, reason = _clean_sentinels(df[col])
        df[col] = cleaned
        df[f"{col}_QC_REASON"] = reason

    deduped, dedup_qc = resolve_visit_duplicates(
        df, key_cols=["RID", "PHASE", "VISCODE"], date_col="EXAMDATE"
    )
    deduped = deduped.merge(enrollment_df[["RID", "ENRLDT"]], on="RID", how="left")

    crosswalk_df = build_visit_month_map(visits_df)
    deduped = map_canonical_month(
        deduped,
        crosswalk_df,
        viscode2_col="VISCODE2",
        viscode_col="VISCODE",
        phase_col="PHASE",
        examdate_col="EXAMDATE",
        enrldt_col="ENRLDT",
    )
    deduped = deduped.rename(columns={"canonical_month": "VISIT_MONTH_RAW"})
    deduped["VISIT_MONTH"] = deduped["VISIT_MONTH_RAW"].apply(snap_to_target_month)
    deduped = deduped.drop(columns=["ENRLDT"])

    deduped["PTAU217_LOT_BIAS_FLAG"] = deduped["Comment"].astype(str).str.contains(
        "Batch 3", case=False, na=False
    )

    return deduped, dedup_qc


def build_ptau217_long(fuji_base, demog_df, enrollment_df):
    """
    pTau217 long table with the documented ADNI4 Batch #3 reagent-lot
    QC-drift episode flagged per record from the raw file's own
    `Comment` field ("Batch 3: QC drift noted; results validated.
    Refer to Methods Special Note.") -- not a date-window guess. See
    adni_data_audit.md / adni_merge_risk_report.csv R8 for the
    documented episode this flag corresponds to.

    Recommendation implemented here (per the source methods PDF's own
    "results validated" note, and per instructions to default to the
    conservative option rather than trust an unreviewed correction):
    flagged records are EXCLUDED from `ptau217_primary_analysis_eligible`
    but are kept, not deleted, for an explicit sensitivity analysis
    that includes them. No corrected/adjusted value is invented -- the
    reported pT217_F value is used as-is either way.
    """
    df = _join_demographics_and_age(fuji_base, demog_df, enrollment_df, date_col="EXAMDATE")

    baseline = select_biomarker_baseline(
        df, rid_col="RID", value_col="pT217_F", month_col="VISIT_MONTH_RAW", date_col="EXAMDATE"
    )
    baseline = baseline.rename(
        columns={
            "BASELINE_VALUE": "PTAU217_BASELINE",
            "BASELINE_DATE": "PTAU217_BASELINE_DATE",
            "BASELINE_VISIT_MONTH": "PTAU217_BASELINE_VISIT_MONTH",
        }
    )
    df = df.merge(baseline, on="RID", how="left")

    df = add_biomarker_eligibility(
        df,
        rid_col="RID",
        value_col="pT217_F",
        baseline_value_col="PTAU217_BASELINE",
        month_col="VISIT_MONTH_RAW",
        group_cols=["RID"],
    )

    # Primary-analysis eligibility is recomputed treating every
    # lot-bias-flagged record's value as unavailable (not just
    # checking the flag on the same row that happened to define
    # eligibility) -- a participant whose only baseline record is
    # flagged is NOT primary-analysis-eligible even if an unrelated,
    # unflagged follow-up row exists, and vice versa. This is the
    # correct way to implement "primary analysis excluding flagged
    # records": exclude the records, then ask whether baseline +
    # follow-up requirements are still met from what's left.
    primary_view = df.copy()
    primary_view.loc[primary_view["PTAU217_LOT_BIAS_FLAG"], "pT217_F"] = pd.NA
    primary_baseline = select_biomarker_baseline(
        primary_view, rid_col="RID", value_col="pT217_F", month_col="VISIT_MONTH_RAW", date_col="EXAMDATE"
    )[["RID", "BASELINE_VALUE"]].rename(columns={"BASELINE_VALUE": "_PRIMARY_BASELINE"})
    primary_view = primary_view.merge(primary_baseline, on="RID", how="left")
    primary_view = add_biomarker_eligibility(
        primary_view,
        rid_col="RID",
        value_col="pT217_F",
        baseline_value_col="_PRIMARY_BASELINE",
        month_col="VISIT_MONTH_RAW",
        group_cols=["RID"],
    )
    df["PTAU217_PRIMARY_ANALYSIS_ELIGIBLE"] = primary_view["BIOMARKER_ELIGIBLE"]

    final_cols = [
        "RID",
        "PHASE",
        "VISCODE",
        "EXAMDATE",
        "VISIT_MONTH_RAW",
        "VISIT_MONTH",
        "mapping_source",
        "mapping_confidence",
        "pT217_F",
        "pT217_F_QC_REASON",
        "PTAU217_LOT_BIAS_FLAG",
        "PTAU217_BASELINE",
        "PTAU217_BASELINE_DATE",
        "PTAU217_BASELINE_VISIT_MONTH",
        "BASELINE_SELECTION_METHOD",
        "BASELINE_AGE",
        "AGE_AT_VISIT",
        "SEX",
        "BIOMARKER_ELIGIBLE",
        "PTAU217_PRIMARY_ANALYSIS_ELIGIBLE",
    ]
    result = df[final_cols].rename(
        columns={
            "pT217_F": "PTAU217",
            "pT217_F_QC_REASON": "PTAU217_QC_REASON",
            "mapping_source": "VISIT_MAPPING_SOURCE",
            "mapping_confidence": "VISIT_MAPPING_CONFIDENCE",
        }
    )
    result = result.sort_values(["RID", "PHASE", "VISCODE"]).reset_index(drop=True)

    qc = {
        "n_records": len(result),
        "n_unique_participants": result["RID"].nunique(),
        "n_lot_bias_flagged_records": int(result["PTAU217_LOT_BIAS_FLAG"].sum()),
        "n_lot_bias_flagged_participants": result.loc[
            result["PTAU217_LOT_BIAS_FLAG"], "RID"
        ].nunique(),
        "n_eligible_participants_including_flagged": result.loc[
            result["BIOMARKER_ELIGIBLE"], "RID"
        ].nunique(),
        "n_eligible_participants_primary_analysis": result.loc[
            result["PTAU217_PRIMARY_ANALYSIS_ELIGIBLE"], "RID"
        ].nunique(),
    }
    return result, qc


def build_abeta_ratio_long(fuji_base, demog_df, enrollment_df):
    """
    Abeta42/Abeta40 long table. Uses the pre-computed AB42_AB40_F
    field as the validated ratio (confirmed during the audit to be
    non-missing in exactly the records where both AB42_F and AB40_F
    are non-missing, with zero discrepancies) -- the ratio is NOT
    recalculated here, only re-verified on this cleaned/deduped base.
    """
    df = _join_demographics_and_age(fuji_base, demog_df, enrollment_df, date_col="EXAMDATE")

    both_present = df["AB42_F"].notna() & df["AB40_F"].notna()
    ratio_present = df["AB42_AB40_F"].notna()
    df["SAME_SAMPLE_VERIFIED"] = both_present == ratio_present
    n_mismatches = int((both_present != ratio_present).sum())

    baseline = select_biomarker_baseline(
        df, rid_col="RID", value_col="AB42_AB40_F", month_col="VISIT_MONTH_RAW", date_col="EXAMDATE"
    )
    baseline = baseline.rename(
        columns={
            "BASELINE_VALUE": "ABETA_RATIO_BASELINE",
            "BASELINE_DATE": "ABETA_RATIO_BASELINE_DATE",
            "BASELINE_VISIT_MONTH": "ABETA_RATIO_BASELINE_VISIT_MONTH",
        }
    )
    df = df.merge(baseline, on="RID", how="left")

    df = add_biomarker_eligibility(
        df,
        rid_col="RID",
        value_col="AB42_AB40_F",
        baseline_value_col="ABETA_RATIO_BASELINE",
        month_col="VISIT_MONTH_RAW",
        group_cols=["RID"],
    )

    final_cols = [
        "RID",
        "PHASE",
        "VISCODE",
        "EXAMDATE",
        "VISIT_MONTH_RAW",
        "VISIT_MONTH",
        "mapping_source",
        "mapping_confidence",
        "AB42_F",
        "AB40_F",
        "AB42_AB40_F",
        "SAME_SAMPLE_VERIFIED",
        "ABETA_RATIO_BASELINE",
        "ABETA_RATIO_BASELINE_DATE",
        "ABETA_RATIO_BASELINE_VISIT_MONTH",
        "BASELINE_SELECTION_METHOD",
        "BASELINE_AGE",
        "AGE_AT_VISIT",
        "SEX",
        "BIOMARKER_ELIGIBLE",
    ]
    result = df[final_cols].rename(
        columns={
            "AB42_AB40_F": "ABETA_RATIO",
            "mapping_source": "VISIT_MAPPING_SOURCE",
            "mapping_confidence": "VISIT_MAPPING_CONFIDENCE",
        }
    )
    result = result.sort_values(["RID", "PHASE", "VISCODE"]).reset_index(drop=True)

    qc = {
        "n_records": len(result),
        "n_unique_participants": result["RID"].nunique(),
        "n_same_sample_mismatches": n_mismatches,
        "n_eligible_participants": result.loc[result["BIOMARKER_ELIGIBLE"], "RID"].nunique(),
    }
    return result, qc


def build_platform_long(fuji_base, demog_df, enrollment_df, analyte, quanterix_col, fujirebio_col):
    """
    Generic builder for GFAP/NfL: reshapes the two platform-specific
    columns into a long table with one row per (RID, PHASE, VISCODE,
    PLATFORM), keeping Quanterix and Fujirebio values in separate rows
    rather than separate columns so eligibility can never accidentally
    mix them within a participant (see add_biomarker_eligibility's
    group_cols=["RID","PLATFORM"] usage below).
    """
    base_cols = [
        "RID", "PHASE", "VISCODE", "EXAMDATE", "VISIT_MONTH_RAW", "VISIT_MONTH",
        "mapping_source", "mapping_confidence",
    ]
    quanterix = fuji_base[base_cols + [quanterix_col, f"{quanterix_col}_QC_REASON"]].copy()
    quanterix["PLATFORM"] = "Quanterix"
    quanterix = quanterix.rename(columns={quanterix_col: analyte, f"{quanterix_col}_QC_REASON": "QC_REASON"})

    fujirebio = fuji_base[base_cols + [fujirebio_col, f"{fujirebio_col}_QC_REASON"]].copy()
    fujirebio["PLATFORM"] = "Fujirebio"
    fujirebio = fujirebio.rename(columns={fujirebio_col: analyte, f"{fujirebio_col}_QC_REASON": "QC_REASON"})

    long_df = pd.concat([quanterix, fujirebio], ignore_index=True)
    long_df = long_df.rename(
        columns={"mapping_source": "VISIT_MAPPING_SOURCE", "mapping_confidence": "VISIT_MAPPING_CONFIDENCE"}
    )
    long_df = _join_demographics_and_age(long_df, demog_df, enrollment_df, date_col="EXAMDATE")

    baseline = long_df[long_df["VISIT_MONTH_RAW"] == 0].dropna(subset=[analyte]).copy()
    baseline = baseline.sort_values("EXAMDATE").drop_duplicates(["RID", "PLATFORM"], keep="first")
    baseline = baseline[["RID", "PLATFORM", analyte, "EXAMDATE"]].rename(
        columns={analyte: f"{analyte}_BASELINE", "EXAMDATE": f"{analyte}_BASELINE_DATE"}
    )
    baseline[f"{analyte}_BASELINE_VISIT_MONTH"] = 0
    baseline["BASELINE_SELECTION_METHOD"] = "visit_month_0"
    long_df = long_df.merge(baseline, on=["RID", "PLATFORM"], how="left")

    long_df = add_biomarker_eligibility(
        long_df,
        rid_col="RID",
        value_col=analyte,
        baseline_value_col=f"{analyte}_BASELINE",
        month_col="VISIT_MONTH_RAW",
        group_cols=["RID", "PLATFORM"],
    )

    long_df = long_df.sort_values(["RID", "PLATFORM", "PHASE", "VISCODE"]).reset_index(drop=True)
    return long_df


def recommend_primary_platform(platform_long_df, analyte):
    """
    Compares Quanterix vs Fujirebio on LONGITUDINAL coverage (number of
    participants with 2+ non-missing visits on that platform, i.e.
    usable for a within-participant trajectory, not just total
    non-missing record count) and returns
    {"primary_platform": ..., "rationale": ..., "coverage": {...}}.
    Does not alter any data -- purely a reporting/recommendation
    helper for assay_platform_summary.csv.
    """
    coverage = {}
    for platform in ["Quanterix", "Fujirebio"]:
        sub = platform_long_df[
            (platform_long_df["PLATFORM"] == platform) & platform_long_df[analyte].notna()
        ]
        visits_per_rid = sub.groupby("RID").size()
        coverage[platform] = {
            "n_participants_any_measurement": int(sub["RID"].nunique()),
            "n_records": int(len(sub)),
            "n_participants_2plus_visits": int((visits_per_rid >= 2).sum()),
        }
    primary = max(coverage, key=lambda p: coverage[p]["n_participants_2plus_visits"])
    other = [p for p in coverage if p != primary][0]
    rationale = (
        f"{primary} has {coverage[primary]['n_participants_2plus_visits']} participants with "
        f"2+ non-missing {analyte} visits vs {coverage[other]['n_participants_2plus_visits']} for "
        f"{other} -- recommended as the primary platform for longitudinal {analyte} analysis. "
        f"{other} is preserved in the same output table for sensitivity analysis, not discarded."
    )
    return {"primary_platform": primary, "rationale": rationale, "coverage": coverage}
