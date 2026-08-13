# ============================================================
# ADNI_PET -- amyloid-PET (Centiloid) preprocessing, kept strictly
# separate from adni_cohort.py's clinical/cognitive preprocessing and
# adni_plasma.py's plasma-biomarker preprocessing, per the same
# "distinct assay/modality families are never silently combined"
# discipline used throughout this project.
#
# Source: UCBERKELEY_AMY_6MM (UC Berkeley's amyloid-PET quantification
# pipeline for ADNI, the field-standard source for Centiloid values --
# see raw/clinical/ADNIMERGE2/man/UCBERKELEY_AMY_6MM.Rd). Centiloids
# are, by design, harmonized across the three amyloid tracers present
# in ADNI (FBB/FBP/NAV) -- this module relies on that already-published
# harmonization rather than re-deriving or re-validating it.
#
# This module implements ONLY the validated near-baseline PET matching
# rule (see outputs/ investigative report preceding this module):
#   - QC gate: qc_flag == 2 ("Pass") only. "Partial pass" (1) and
#     "Not assessed" (-1) are excluded, not silently guessed at.
#   - Window: the QC-passed scan must fall within +/- window_days
#     (default 90) of the participant's OWN already-validated clinical
#     baseline date (VISIT_MONTH == 0's EXAMDATE in
#     adni_clinical_long.parquet) -- never a literal VISCODE2 == "bl"
#     match alone, since that label was found to span up to ~1 year
#     from the true clinical baseline for a real minority of scans.
#   - Deterministic tie-break when more than one QC-passed scan falls
#     inside the window: (1) smallest |days from clinical baseline|,
#     (2) on a tie, the EARLIER scan (more negative day difference),
#     (3) on a further tie, the lower LONIUID (PET image ID) --
#     arbitrary but fully reproducible.
#
# Same participant-level-data discipline as adni_cohort.py /
# adni_plasma.py: every function here is pure, no I/O at import time,
# and no function returns anything intended for outputs/ that still
# contains a participant-level row (RID, LONIUID, etc.) -- aggregation
# for outputs/ happens exclusively in run_adni_pet_eligibility.py.
# ============================================================

import numpy as np
import pandas as pd

PET_QC_PASS = 2

CENTILOID_STATUS_COMPUTED = "Computed"
CENTILOID_STATUS_NO_BASELINE_DATE = "No clinical baseline date"
CENTILOID_STATUS_NO_SCAN_IN_WINDOW = "No QC-passed scan within window"

MMSE_THRESHOLD = 20
CENTILOID_THRESHOLD = 30
PET_WINDOW_DAYS = 90


def build_pet_baseline(pet_df, clinical_baseline_df, window_days=PET_WINDOW_DAYS):
    """
    For every participant in `clinical_baseline_df` (one row per RID,
    must carry RID and CLINICAL_BASELINE_DATE -- CLINICAL_BASELINE_DATE
    may be NaT/missing, handled explicitly below, never silently),
    selects the single QC-passed (qc_flag == PET_QC_PASS) Centiloid PET
    scan from `pet_df` (must carry RID, SCANDATE, qc_flag, CENTILOIDS,
    LONIUID) closest in calendar time to that participant's clinical
    baseline date, restricted to +/- window_days.

    Returns one row per input RID (never drops a participant, so
    downstream attrition counting stays exact), with:
      - CENTILOID_BASELINE: the selected scan's Centiloid value, or NaN
      - CENTILOID_BASELINE_DAYS_FROM_CLINICAL_BASELINE: signed day
        difference (scan date minus clinical baseline date; negative =
        scan before baseline), or NaN
      - CENTILOID_BASELINE_STATUS: CENTILOID_STATUS_COMPUTED,
        CENTILOID_STATUS_NO_BASELINE_DATE (participant has no valid
        clinical baseline date to measure against -- ELIGIBLE FOR
        NEITHER inclusion NOR exclusion is never implicit; this status
        makes the reason explicit), or CENTILOID_STATUS_NO_SCAN_IN_WINDOW
      - CENTILOID_ELIGIBLE: True iff CENTILOID_BASELINE_STATUS ==
        "Computed" (i.e. a usable near-baseline value exists) -- this
        is a DATA-AVAILABILITY flag, matching the meaning of
        MMSE_ELIGIBLE/ADAS_COG13_ELIGIBLE elsewhere in this pipeline
        (data is present and valid), NOT a claim that the value meets
        the >= 30 Centiloid threshold -- that threshold check belongs
        to POLARIS eligibility, computed separately in
        add_polaris_eligibility().
    """
    base = clinical_baseline_df[["RID", "CLINICAL_BASELINE_DATE"]].drop_duplicates("RID").copy()

    qc = pet_df[pet_df["qc_flag"] == PET_QC_PASS].copy()
    qc["SCANDATE"] = pd.to_datetime(qc["SCANDATE"], errors="coerce")
    qc = qc.dropna(subset=["SCANDATE"])

    merged = qc.merge(base, on="RID", how="inner")
    has_baseline_date = merged["CLINICAL_BASELINE_DATE"].notna()
    merged = merged[has_baseline_date].copy()
    merged["DAYS_DIFF"] = (merged["SCANDATE"] - merged["CLINICAL_BASELINE_DATE"]).dt.days
    merged["ABS_DAYS"] = merged["DAYS_DIFF"].abs()
    in_window = merged[merged["ABS_DAYS"] <= window_days].copy()

    # Deterministic tie-break: smallest |days|, then earlier scan
    # (smaller/more negative signed DAYS_DIFF), then lower LONIUID.
    in_window = in_window.sort_values(["RID", "ABS_DAYS", "DAYS_DIFF", "LONIUID"])
    selected = in_window.drop_duplicates("RID", keep="first")[
        ["RID", "CENTILOIDS", "DAYS_DIFF"]
    ].rename(columns={"CENTILOIDS": "CENTILOID_BASELINE", "DAYS_DIFF": "CENTILOID_BASELINE_DAYS_FROM_CLINICAL_BASELINE"})

    out = base.merge(selected, on="RID", how="left")

    def _status(row):
        if pd.isna(row["CLINICAL_BASELINE_DATE"]):
            return CENTILOID_STATUS_NO_BASELINE_DATE
        if pd.notna(row["CENTILOID_BASELINE"]):
            return CENTILOID_STATUS_COMPUTED
        return CENTILOID_STATUS_NO_SCAN_IN_WINDOW

    out["CENTILOID_BASELINE_STATUS"] = out.apply(_status, axis=1)
    out["CENTILOID_ELIGIBLE"] = out["CENTILOID_BASELINE_STATUS"] == CENTILOID_STATUS_COMPUTED
    return out.drop(columns=["CLINICAL_BASELINE_DATE"])


def add_polaris_eligibility(cohort_df, mmse_threshold=MMSE_THRESHOLD, centiloid_threshold=CENTILOID_THRESHOLD):
    """
    Adds POLARIS_ELIGIBLE to `cohort_df` (must already carry
    MMSE_BASELINE, CENTILOID_BASELINE, CENTILOID_ELIGIBLE -- e.g. the
    output of build_pet_baseline() merged with the clinical baseline
    table). A participant is POLARIS_ELIGIBLE iff:
        MMSE_BASELINE is present and >= mmse_threshold
        AND CENTILOID_ELIGIBLE is True (a usable near-baseline PET
            value exists at all -- see build_pet_baseline())
        AND CENTILOID_BASELINE >= centiloid_threshold

    Never silently treats a missing MMSE or missing/out-of-window PET
    value as either eligible or ineligible by accident -- both simply
    evaluate to False through the explicit conditions below (NaN
    comparisons are False in pandas, and CENTILOID_ELIGIBLE is already
    an explicit, reasoned boolean from build_pet_baseline()), so a
    "why not eligible" reason is always reconstructable from the
    intermediate columns already present, never hidden.
    """
    out = cohort_df.copy()
    mmse_ok = out["MMSE_BASELINE"].notna() & (out["MMSE_BASELINE"] >= mmse_threshold)
    centiloid_ok = out["CENTILOID_ELIGIBLE"] & (out["CENTILOID_BASELINE"] >= centiloid_threshold)
    out["POLARIS_ELIGIBLE"] = mmse_ok & centiloid_ok
    return out
