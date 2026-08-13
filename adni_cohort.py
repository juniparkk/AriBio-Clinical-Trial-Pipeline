# ============================================================
# ADNI_COHORT -- clinical cohort construction, cognitive source
# validation, and visit harmonization (preprocessing only; see
# run_adni_preprocessing.py for orchestration and PREPROCESSING.md-
# equivalent scope notes in the module docstrings below).
#
# NOT in scope here, on purpose: ANCOVA, change-from-baseline, any
# inferential statistic, or dashboard rendering. This module only
# builds clean, analysis-ready longitudinal tables and aggregate QC
# reports -- see adni_analysis.py's compute_ancova_results() etc. for
# where that work belongs once explicitly requested.
#
# Every function here is pure (DataFrame in, DataFrame/dict out) and
# does no file I/O at import time, same convention as
# drug_classification.py / competitive_attention.py. run_adni_
# preprocessing.py is the only place that reads/writes files.
#
# Participant-level data discipline: every DataFrame here keeps RID
# and other participant-level fields for as long as needed to build
# the processed/ tables, but no function in this module ever prints,
# logs, or returns anything containing RID/participant rows to a
# caller whose output is destined for outputs/ (aggregate-only) or the
# dashboard -- callers are responsible for only ever writing the
# *_long DataFrames to ADNI_PROCESSED_DIR (local, gitignored, never
# git-tracked) and only ever writing the QC/summary DataFrames (built
# from .agg()/.value_counts()/groupby-count patterns, never raw rows)
# to ADNI_OUTPUTS_DIR.
# ============================================================

import re

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

ADNI_PHASES = ["ADNI1", "ADNIGO", "ADNI2", "ADNI3", "ADNI4"]

# Visit codes that represent a genuine baseline visit vs. a rollover
# participant's first visit in a later phase (which looks similar but
# is NOT baseline -- it is a continuation of an already-enrolled
# participant's follow-up, see Risk R3 in adni_merge_risk_report.csv).
# VISNAME text distinguishes them cleanly: "Baseline..." vs
# "...Continuing Pt" -- see build_baseline_viscode_map().
_BASELINE_VISNAME_PATTERN = re.compile(r"^Baseline")

# Canonical target months this project reports sample sizes at.
TARGET_MONTHS = [0, 6, 12, 18, 24, 36, 48]

# Tolerance (in months) for snapping an off-schedule visit to a target
# month. Half the shortest scheduled inter-visit gap (6 months) minus
# a small buffer, so two adjacent target months can never both claim
# the same visit. Documented here rather than left as a magic number
# anywhere it's used.
MONTH_SNAP_TOLERANCE = 2


# ------------------------------------------------------------------
# 3. Visit harmonization
# ------------------------------------------------------------------


def build_baseline_viscode_map(visits_df):
    """
    Phase -> baseline VISCODE, derived directly from VISITS.VISNAME
    (rows whose name starts with "Baseline"), not hardcoded. This is
    the "documented ADNIMERGE2 baseline-diagnosis logic" the audit
    identified (get_baseline_vistcode() in the R package returns the
    same three codes for ADNI1/ADNIGO/ADNI3 vs ADNI2 vs ADNI4, but
    deriving it from VISNAME text here keeps the mapping self-
    verifying against the actual visit-schedule table rather than a
    second, independently-typed hardcoded list that could drift out
    of sync with it).

    Deliberately excludes VISNAMEs like "ADNI3 Initial Visit -
    Continuing Pt" / "ADNI4 Initial Visit - Continuing Pt" (VISCODE
    init/4_init) -- those are a rollover participant's first visit in
    a later phase, not a true baseline (see MONTH_SNAP_TOLERANCE
    docstring and Risk R3).
    """
    baseline_rows = visits_df[
        visits_df["VISNAME"].astype(str).str.match(_BASELINE_VISNAME_PATTERN)
    ]
    result = dict(zip(baseline_rows["PHASE"], baseline_rows["VISCODE"]))
    missing = set(ADNI_PHASES) - set(result)
    if missing:
        raise ValueError(
            f"build_baseline_viscode_map: no VISNAME starting with 'Baseline' "
            f"found for phase(s) {sorted(missing)} -- VISITS.csv may have changed."
        )
    return result


_SCREENING_VISNAME_PATTERN = re.compile(r"^Screening(?!\s*MRI)")


def build_screening_viscode_map(visits_df):
    """
    Phase -> screening VISCODE, derived from VISNAME the same way
    build_baseline_viscode_map() derives baseline codes (rows starting
    with "Screening", excluding "Screening MRI..." sub-visits). Needed
    because MMSE turns out to be recorded at the screening visit, not
    the baseline visit, in every ADNI phase (see
    select_cognitive_baseline()'s docstring) -- MMSE is a study
    eligibility criterion checked at screening, before baseline.
    """
    screen_rows = visits_df[
        visits_df["VISNAME"].astype(str).str.match(_SCREENING_VISNAME_PATTERN)
    ]
    return dict(zip(screen_rows["PHASE"], screen_rows["VISCODE"]))


def build_visit_month_map(visits_df):
    """
    (PHASE, VISCODE) -> (canonical_month, mapping_method, mapping_confidence).

    Three tiers, in order of confidence:
      - "high" / "viscode_baseline": VISNAME starts with "Baseline" -> month 0.
      - "high" / "viscode_month_label": VISNAME contains "Month N" -> month N
        (e.g. ADNI1's "m12"/"Month 12").
      - "medium" / "viscode_year_label_inferred": VISNAME matches "...Year N
        Visit" or "...Year N TelCheck" (ADNI2's v11/v21/... codes, which are
        not literally month-labeled) -> inferred as 12*N months. Flagged
        medium confidence specifically because it is inferred from the
        "Year N" text, not a literal month label, and has not been
        cross-validated against actual elapsed time in this pass.
      - Everything else (screening, registration, disposition, unscheduled,
        autopsy, telcheck-without-a-year-label, rollover "Continuing Pt"
        visits) is left unmapped: canonical_month = NaN, method =
        "unmapped", confidence = "none". These visits are never forced
        into a canonical month by this function.
    """
    df = visits_df.copy()
    df["_visname"] = df["VISNAME"].astype(str)

    month = pd.Series(np.nan, index=df.index, dtype="float64")
    method = pd.Series("unmapped", index=df.index, dtype="object")
    confidence = pd.Series("none", index=df.index, dtype="object")

    is_baseline = df["_visname"].str.match(_BASELINE_VISNAME_PATTERN)
    month[is_baseline] = 0
    method[is_baseline] = "viscode_baseline"
    confidence[is_baseline] = "high"

    month_label = df["_visname"].str.extract(r"[Mm]onth\s*(\d+)", expand=False)
    has_month_label = month_label.notna() & ~is_baseline
    month[has_month_label] = month_label[has_month_label].astype(float)
    method[has_month_label] = "viscode_month_label"
    confidence[has_month_label] = "high"

    year_label = df["_visname"].str.extract(r"Year\s*(\d+)\s*(?:Visit|TelCheck)", expand=False)
    has_year_label = year_label.notna() & ~is_baseline & ~has_month_label
    month[has_year_label] = year_label[has_year_label].astype(float) * 12
    method[has_year_label] = "viscode_year_label_inferred"
    confidence[has_year_label] = "medium"

    out = pd.DataFrame(
        {
            "PHASE": df["PHASE"],
            "VISCODE": df["VISCODE"],
            "canonical_month": month,
            "mapping_method": method,
            "mapping_confidence": confidence,
        }
    )
    return out


_VISCODE2_MONTH_PATTERN = re.compile(r"^m(\d+)$")


def parse_viscode2_month(viscode2):
    """
    ADNI's VISCODE2 ("translated visit code") is confirmed empirically to be
    a single, phase-independent, literal month-relative vocabulary --
    "bl", "m06", "m12", ... up to "m240" -- populated on ~99%+ of records in
    every raw eCRF table AND both live plasma files (100% in both), unlike
    VISCODE (which is phase-specific and, for the plasma panel, mixes five
    different phases' native vocabularies in one column -- see Risk R2).
    Returns (month, True) on a direct "bl"/"mNN" match, (None, False) for
    anything else (sc, scmri, f, uns1, nv, missing, ...) -- never guesses at
    a non-standard VISCODE2 value.
    """
    if pd.isna(viscode2):
        return None, False
    v = str(viscode2)
    if v == "bl":
        return 0.0, True
    m = _VISCODE2_MONTH_PATTERN.match(v)
    if m:
        return float(m.group(1)), True
    return None, False


def map_canonical_month(
    df,
    crosswalk_df,
    viscode2_col="VISCODE2",
    viscode_col="VISCODE",
    phase_col="PHASE",
    examdate_col="EXAMDATE",
    enrldt_col="ENRLDT",
):
    """
    Adds canonical_month, mapping_source, mapping_confidence to df, in three
    preference tiers (VISCODE2 first, per instructions):

      1. "viscode2" / "high": VISCODE2 parses directly via
         parse_viscode2_month() -- ADNI's own phase-independent translated
         visit code. Preferred over everything else whenever present.
      2. "viscode_crosswalk_fallback:<inner method>" / "medium": VISCODE2
         didn't resolve, but (phase_col, viscode_col) is found in
         crosswalk_df (this module's build_visit_month_map() output, which
         reads VISNAME's own "Baseline"/"Month N"/"Year N" text) -- kept as
         a fallback for the handful of records missing VISCODE2 entirely,
         downgraded to "medium" confidence relative to tier 1 regardless of
         the crosswalk's own internal tier, since it's only reached when the
         preferred, validated translated code is unavailable.
      3. "date_elapsed_from_enrollment" / "medium": still unresolved, but
         both examdate_col and enrldt_col (the participant's own baseline/
         enrollment date, from build_enrollment_table()) are present --
         computed as round((EXAMDATE - ENRLDT).days / 30.4375). Only used as
         a last resort, exactly as instructed ("use dates only when
         necessary and justified").
      4. "unmapped" / "none": none of the above -- canonical_month stays NaN
         and the record is never forced into a target month.

    Snapping the resulting canonical_month to one of TARGET_MONTHS (with
    MONTH_SNAP_TOLERANCE) is a separate step (snap_to_target_month) -- this
    function only establishes the best-supported *continuous* month value
    and how it was obtained.
    """
    df = df.copy()
    parsed = df[viscode2_col].apply(parse_viscode2_month)
    v2_month = parsed.apply(lambda t: t[0])
    v2_ok = parsed.apply(lambda t: t[1])

    cw = crosswalk_df.rename(
        columns={
            "PHASE": "_cw_phase",
            "VISCODE": "_cw_viscode",
            "canonical_month": "_cw_month",
            "mapping_method": "_cw_method",
        }
    )[["_cw_phase", "_cw_viscode", "_cw_month", "_cw_method"]]
    df = df.merge(
        cw, left_on=[phase_col, viscode_col], right_on=["_cw_phase", "_cw_viscode"], how="left"
    )

    month = pd.Series(np.nan, index=df.index, dtype="float64")
    source = pd.Series("unmapped", index=df.index, dtype="object")
    confidence = pd.Series("none", index=df.index, dtype="object")

    month[v2_ok] = v2_month[v2_ok]
    source[v2_ok] = "viscode2"
    confidence[v2_ok] = "high"

    cw_ok = (~v2_ok) & df["_cw_month"].notna()
    month[cw_ok] = df.loc[cw_ok, "_cw_month"]
    source[cw_ok] = "viscode_crosswalk_fallback:" + df.loc[cw_ok, "_cw_method"].astype(str)
    confidence[cw_ok] = "medium"

    if examdate_col in df.columns and enrldt_col in df.columns:
        examdate = pd.to_datetime(df[examdate_col], errors="coerce")
        enrldt = pd.to_datetime(df[enrldt_col], errors="coerce")
        elapsed_days = (examdate - enrldt).dt.days
        # elapsed_days >= 0 only -- a negative gap means this record predates
        # enrollment (most often a screening-visit record with no month-label
        # VISCODE2, e.g. MMSE), which has no meaningful "month since baseline"
        # in the positive target-month framework and must not be forced into
        # one (see MONTH_SNAP_TOLERANCE docstring: never force ambiguous
        # visits). Left unmapped instead.
        date_ok = (~v2_ok) & (~cw_ok) & elapsed_days.notna() & (elapsed_days >= 0)
        month[date_ok] = (elapsed_days[date_ok] / 30.4375).round()
        source[date_ok] = "date_elapsed_from_enrollment"
        confidence[date_ok] = "medium"

    df["canonical_month"] = month
    df["mapping_source"] = source
    df["mapping_confidence"] = confidence
    return df.drop(columns=["_cw_phase", "_cw_viscode", "_cw_month", "_cw_method"])


def snap_to_target_month(month_value, tolerance=MONTH_SNAP_TOLERANCE):
    """
    Nearest value in TARGET_MONTHS within `tolerance`, else NaN. Never
    forces an off-schedule value into a canonical month silently -- a
    value outside tolerance of every target month returns NaN, and
    callers are expected to keep (not drop) the underlying record with
    that NaN canonical month rather than mis-bucket it.
    """
    if pd.isna(month_value):
        return np.nan
    diffs = [abs(month_value - t) for t in TARGET_MONTHS]
    best = min(diffs)
    if best <= tolerance:
        return TARGET_MONTHS[diffs.index(best)]
    return np.nan


# ------------------------------------------------------------------
# 1. Clinical cohort construction -- enrollment & baseline diagnosis
# ------------------------------------------------------------------


def build_enrollment_table(registry_df, baseline_viscode_map):
    """
    RID -> ENRLDT (enrollment/baseline exam date), ORIGPROT.

    A participant counts as enrolled only if REGISTRY has a record at
    their *original* protocol (ORIGPROT == COLPROT -- enrollment
    happens exactly once, at a participant's first phase) whose
    VISCODE is that phase's documented baseline code, the visit was
    actually conducted, and EXAMDATE is present.

    "Visit actually conducted" uses the field REGISTRY actually
    populates per phase (confirmed empirically -- RGCONDCT is only
    ever non-missing for ADNI1; every other phase instead populates
    VISTYPE, coding "Not done" as one of its values): ADNI1 uses
    RGCONDCT == "Yes"; every other phase uses VISTYPE not missing and
    not "Not done". This mirrors the intent of ADNIMERGE2's own
    get_adni_enrollment() (REGISTRY-based, ORIGPROT==COLPROT, baseline
    VISCODE, visit conducted, phase-specific conducted-signal) without
    reproducing its full PTTYPE/adni_study_track branching --
    documented here as a deliberately conservative, fully-inspectable
    reimplementation rather than a byte-for-byte port.
    """
    df = registry_df.copy()
    df["_expected_baseline_viscode"] = df["COLPROT"].map(baseline_viscode_map)
    is_adni1 = df["COLPROT"] == "ADNI1"
    visit_conducted = (is_adni1 & (df["RGCONDCT"] == "Yes")) | (
        ~is_adni1 & df["VISTYPE"].notna() & (df["VISTYPE"] != "Not done")
    )
    is_enrollment_record = (
        (df["ORIGPROT"] == df["COLPROT"])
        & (df["VISCODE"] == df["_expected_baseline_viscode"])
        & visit_conducted
        & df["EXAMDATE"].notna()
    )
    enrolled = df[is_enrollment_record].copy()
    enrolled = enrolled.sort_values("EXAMDATE").drop_duplicates("RID", keep="first")
    enrolled = enrolled.rename(columns={"EXAMDATE": "ENRLDT"})
    return enrolled[["RID", "ORIGPROT", "ENRLDT"]].reset_index(drop=True)


def build_baseline_diagnosis(dxsum_df, enrollment_df, baseline_viscode_map):
    """
    RID -> baseline DIAGNOSIS + baseline EXAMDATE, with the
    screen-failure trap explicitly closed: a DXSUM record at the
    phase's baseline VISCODE is only accepted as *the* baseline
    diagnosis if that RID also has a qualifying enrollment record
    (build_enrollment_table) at the same original protocol. A
    participant who was scored at what looks like a baseline visit but
    never actually enrolled (a screening failure) has no row in
    enrollment_df and is therefore dropped here, not silently kept.

    Deliberately conservative: unlike ADNIMERGE2's own
    get_adni_blscreen_dxsum(), this does NOT fall back to a screening-
    visit diagnosis when a baseline-visit diagnosis is missing -- that
    fallback logic (adjust_scbl_record() in the R package) was not
    independently verified against real data during the audit, so it
    is intentionally left out rather than guessed at. This means
    baseline-diagnosis coverage here may be narrower than the
    package's own ADSL.DX derivation, which is the documented,
    expected trade-off for not inventing unverified logic.
    """
    df = dxsum_df.copy()
    df["_expected_baseline_viscode"] = df["COLPROT"].map(baseline_viscode_map)
    candidate = df[
        (df["ORIGPROT"] == df["COLPROT"])
        & (df["VISCODE"] == df["_expected_baseline_viscode"])
        & df["DIAGNOSIS"].notna()
    ].copy()
    candidate = candidate.sort_values("EXAMDATE").drop_duplicates("RID", keep="first")
    candidate = candidate.rename(
        columns={"DIAGNOSIS": "DX_BASELINE", "EXAMDATE": "DX_BASELINE_DATE"}
    )

    screen_failure_rids = set(candidate["RID"]) - set(enrollment_df["RID"])

    verified = candidate[candidate["RID"].isin(enrollment_df["RID"])].copy()
    result = verified[["RID", "ORIGPROT", "DX_BASELINE", "DX_BASELINE_DATE"]].reset_index(
        drop=True
    )

    qc = {
        "n_candidate_baseline_dx_records": len(candidate),
        "n_screen_failure_records_excluded": len(screen_failure_rids),
        "n_verified_baseline_dx": len(result),
    }
    return result, qc


def build_longitudinal_diagnosis(dxsum_df):
    """
    Full per-visit diagnosis history: RID, COLPROT, VISCODE, EXAMDATE,
    DIAGNOSIS for every DXSUM record with a non-missing diagnosis (not
    just baseline). This is the only longitudinal-diagnosis source
    identified during the audit (open question: whether ADQS carries
    an equivalent per-visit DX PARAMCD was not resolved and is not
    assumed here).
    """
    df = dxsum_df[dxsum_df["DIAGNOSIS"].notna()].copy()
    return df[["RID", "ORIGPROT", "COLPROT", "VISCODE", "EXAMDATE", "DIAGNOSIS"]].reset_index(
        drop=True
    )


# ------------------------------------------------------------------
# Age, sex, APOE4
# ------------------------------------------------------------------


def compute_dob(ptdob_str):
    """
    Parse ADNI's MM/YYYY PTDOB format into a date, using day=15
    (mid-month) since ADNI only discloses birth month/year for
    privacy, never the exact day. This introduces at most ~15 days of
    error in any computed age -- negligible for age-in-years use.
    Returns NaT for anything that doesn't match MM/YYYY.
    """
    if pd.isna(ptdob_str):
        return pd.NaT
    m = re.match(r"^(\d{1,2})/(\d{4})$", str(ptdob_str))
    if not m:
        return pd.NaT
    month, year = int(m.group(1)), int(m.group(2))
    if not (1 <= month <= 12):
        return pd.NaT
    return pd.Timestamp(year=year, month=month, day=15)


def build_demographics(ptdemog_df):
    """
    RID -> DOB, SEX, using each participant's earliest record at their
    original protocol (ORIGPROT == COLPROT), matching how ADNIMERGE2's
    own DM derivation selects a single demographic record per subject.
    """
    df = ptdemog_df[ptdemog_df["ORIGPROT"] == ptdemog_df["COLPROT"]].copy()
    df["VISDATE"] = pd.to_datetime(df["VISDATE"], errors="coerce")
    df = df.sort_values("VISDATE").drop_duplicates("RID", keep="first")
    df["DOB"] = df["PTDOB"].apply(compute_dob)
    return df[["RID", "DOB", "PTGENDER"]].rename(columns={"PTGENDER": "SEX"}).reset_index(
        drop=True
    )


def compute_age_years(dob, on_date):
    if pd.isna(dob) or pd.isna(on_date):
        return np.nan
    return round((pd.Timestamp(on_date) - pd.Timestamp(dob)).days / 365.25, 1)


_VALID_GENOTYPES = {"2/2", "2/3", "2/4", "3/3", "3/4", "4/4"}


def build_apoe4(apoeres_df):
    """
    RID -> GENOTYPE, APOE4_CARRIER (bool), APOE4_ALLELE_COUNT (0/1/2).
    No column literally named APOE4 exists anywhere in ADNI -- this is
    the derivation the audit flagged as necessary. Genotyped once per
    participant; unknown/unrecognized genotype strings are left as
    missing (APOE4_CARRIER/ALLELE_COUNT = NaN) rather than guessed at.
    """
    df = apoeres_df[["RID", "GENOTYPE"]].dropna(subset=["GENOTYPE"]).copy()
    df = df.drop_duplicates("RID", keep="first")
    is_valid = df["GENOTYPE"].isin(_VALID_GENOTYPES)
    df["APOE4_CARRIER"] = np.where(is_valid, df["GENOTYPE"].str.contains("4"), np.nan)
    df["APOE4_ALLELE_COUNT"] = np.where(
        is_valid, df["GENOTYPE"].str.count("4"), np.nan
    )
    return df.reset_index(drop=True)


# ------------------------------------------------------------------
# 2. Cognitive source validation
# ------------------------------------------------------------------
#
# ADQS (the derived long-format analysis table) is NOT used here.
# Reasons, both found during the audit (adni_merge_risk_report.csv
# R10/R11) and treated as disqualifying for this stage:
#   1. 33,784 duplicate (USUBJID, PARAMCD, AVISITN) key combinations,
#      concentrated in exactly the kind of composite/sub-scores that
#      would silently double-count if joined naively.
#   2. AVISITN (meant to be an elapsed-day count) ranges up to 35,075
#      (~96 years) -- clearly contains unvalidated/placeholder values
#      that would corrupt any visit-month mapping built on it.
# The raw ADAS and MMSE eCRF tables are used instead: both were
# confirmed during the audit to have zero duplicates on their natural
# key (RID, COLPROT, VISCODE), and their dates/visit codes come
# straight from REGISTRY-anchored eCRF collection, not a derived
# pipeline with its own unresolved issues. The trade-off is that
# ADQS's pre-computed BASE/CHG/PCHG/ABLFL fields are not available --
# acceptable here since this task explicitly excludes change-from-
# baseline computation.


def resolve_cognitive_duplicates(df, key_cols, value_col, date_col):
    """
    Deterministic duplicate-resolution rule for a raw cognitive table:
    group by key_cols, and within each group keep the record with the
    latest `date_col`; if `date_col` ties, keep the first row
    (pandas's stable sort preserves original row order for ties, so
    this is deterministic given a fixed input row order). Any group
    where the *kept* record's value_col disagrees with at least one
    *dropped* record's non-missing value_col is reported as a
    conflict (kept, not silently discarded) so a QC report can surface
    it.

    Returns (deduped_df, qc_dict) where qc_dict has n_input_records,
    n_duplicate_groups, n_records_removed, n_conflicting_groups.
    """
    df = df.copy()
    df["_dupe_group"] = df.groupby(key_cols, dropna=False).ngroup()
    group_sizes = df.groupby("_dupe_group")["_dupe_group"].transform("size")
    dup_groups = df[group_sizes > 1]
    n_duplicate_groups = dup_groups["_dupe_group"].nunique()

    def _values_conflict(g):
        vals = g[value_col].dropna().unique()
        return len(vals) > 1

    conflicting_group_ids = set(
        gid for gid, g in dup_groups.groupby("_dupe_group") if _values_conflict(g)
    )

    df_sorted = df.sort_values(date_col, na_position="first")
    deduped = df_sorted.drop_duplicates(key_cols, keep="last").drop(columns=["_dupe_group"])

    qc = {
        "n_input_records": len(df),
        "n_duplicate_groups": int(n_duplicate_groups),
        "n_records_removed": len(df) - len(deduped),
        "n_conflicting_groups": len(conflicting_group_ids),
    }
    return deduped.reset_index(drop=True), qc


def build_cognitive_source(
    raw_df,
    value_col,
    out_value_name,
    crosswalk_df,
    enrollment_df,
    key_cols=("RID", "COLPROT", "VISCODE"),
):
    """
    One raw cognitive table (ADAS or MMSE) -> a clean, deduplicated,
    visit-month-mapped long table with columns:
    RID, ORIGPROT, COLPROT, VISCODE, EXAMDATE, <out_value_name>,
    canonical_month, mapping_source, mapping_confidence.

    Visit-month mapping uses map_canonical_month() (VISCODE2 first, VISITS-
    text crosswalk second, date-elapsed-from-enrollment third -- see that
    function's docstring). Implausible visit-month values (anything that
    fails snap_to_target_month's documented tolerance) are kept in the
    output with target_month = NaN rather than dropped -- they remain part
    of the full longitudinal record, just excluded from any target-month-
    bucketed sample-size reporting. Returns (clean_df, qc_dict).
    """
    date_col = "EXAMDATE" if "EXAMDATE" in raw_df.columns else "VISDATE"
    df = raw_df[list(key_cols) + ["ORIGPROT", "VISCODE2", date_col, value_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.rename(columns={date_col: "EXAMDATE", value_col: out_value_name})

    n_before_value_filter = len(df)
    df = df[df[out_value_name].notna()]
    n_missing_value = n_before_value_filter - len(df)

    deduped, dedup_qc = resolve_cognitive_duplicates(
        df, key_cols=list(key_cols), value_col=out_value_name, date_col="EXAMDATE"
    )

    deduped = deduped.merge(enrollment_df[["RID", "ENRLDT"]], on="RID", how="left")
    mapped = map_canonical_month(
        deduped,
        crosswalk_df,
        viscode2_col="VISCODE2",
        viscode_col="VISCODE",
        phase_col="COLPROT",
        examdate_col="EXAMDATE",
        enrldt_col="ENRLDT",
    )

    mapped["target_month"] = mapped["canonical_month"].apply(snap_to_target_month)
    n_mapped = mapped["canonical_month"].notna().sum()
    n_unmapped = mapped["canonical_month"].isna().sum()
    n_within_tolerance = mapped["target_month"].notna().sum()
    n_off_schedule = mapped["canonical_month"].notna().sum() - n_within_tolerance
    source_counts = mapped["mapping_source"].value_counts().to_dict()

    qc = dict(dedup_qc)
    qc.update(
        {
            "n_missing_value_excluded": int(n_missing_value),
            "n_visits_mapped_to_a_canonical_month_label": int(n_mapped),
            "n_visits_unmapped": int(n_unmapped),
            "n_visits_within_target_month_tolerance": int(n_within_tolerance),
            "n_visits_mapped_but_off_schedule": int(n_off_schedule),
            "mapping_source_counts": {str(k): int(v) for k, v in source_counts.items()},
        }
    )
    return mapped.drop(columns=["ENRLDT"]).reset_index(drop=True), qc


def select_cognitive_baseline(clean_df, value_col, baseline_viscodes, screening_viscodes):
    """
    Participant-specific baseline for one cognitive measure (ADAS-
    Cog13 or MMSE), preferring the phase's documented baseline VISCODE
    and falling back to the phase's documented screening VISCODE only
    for participants with no valid baseline-VISCODE record for this
    measure. Never falls back to "earliest record of any kind" (that
    would be the naive rule this task explicitly rejects) -- only
    these two documented, phase-derived visit codes are ever
    considered, in that fixed preference order.

    This exists because MMSE, empirically, has zero records under any
    phase's baseline VISCODE in this dataset -- MMSE is a study
    eligibility criterion checked at the screening visit, before
    baseline, in every ADNI phase. ADAS-Cog13 does not have this
    issue (it has real baseline-VISCODE records in every phase), so
    for ADAS the screening fallback tier is expected to contribute
    ~0 participants -- exercised here anyway so both measures go
    through one shared, documented, testable rule rather than two
    different hand-tuned code paths.
    """
    candidates = clean_df[clean_df[value_col].notna()].copy()

    def _pick(viscode_pool, method_label):
        pool = candidates[candidates["VISCODE"].isin(viscode_pool)]
        pool = pool.sort_values("EXAMDATE").drop_duplicates("RID", keep="first")
        out = pool[["RID", value_col, "VISCODE", "EXAMDATE"]].copy()
        out["_method"] = method_label
        return out

    at_baseline = _pick(set(baseline_viscodes.values()), "baseline_viscode")
    at_screen_fallback_pool = candidates[~candidates["RID"].isin(at_baseline["RID"])]
    at_screen = _pick(set(screening_viscodes.values()), "screening_viscode_fallback")
    at_screen = at_screen[at_screen["RID"].isin(at_screen_fallback_pool["RID"])]

    result = pd.concat([at_baseline, at_screen], ignore_index=True)
    result = result.rename(
        columns={
            value_col: f"{value_col}_BASELINE",
            "VISCODE": f"{value_col}_BASELINE_VISCODE",
            "EXAMDATE": f"{value_col}_BASELINE_DATE",
            "_method": f"{value_col}_BASELINE_METHOD",
        }
    )
    qc = {
        "n_from_baseline_viscode": len(at_baseline),
        "n_from_screening_fallback": len(at_screen),
    }
    return result, qc


def _rename_month_map_for_merge(month_map_df):
    return month_map_df.rename(columns={"PHASE": "COLPROT"})


# ------------------------------------------------------------------
# Baseline-diagnosis validation against ADSL.DX
# ------------------------------------------------------------------

# ADSL uses CDISC/ADaM-style short codes; DXSUM (and this module's own
# derivation) uses the full word. Confirmed via ADNI's own DATADIC
# MAPPING_NOTES that this is a label-format difference only, not a
# semantic one: DXSUM.DIAGNOSIS is the field ADSL.DX is itself built from
# (DXCURREN for ADNI1 and DXCHANGE for ADNIGO/ADNI2 were both mapped into
# DXSUM.DIAGNOSIS -- and then excluded as separate columns -- during the
# ADNIMERGE2 package build; DIAGNOSIS is used directly for ADNI2/3/4).
_ADSL_DX_LABEL_MAP = {"DEM": "Dementia"}


def resolve_baseline_diagnosis_with_validation(dxsum_baseline_df, adsl_df):
    """
    Adopts ADSL.DX -- ADNIMERGE2's own official baseline-diagnosis
    derivation -- as the primary source, cross-validated against this
    module's independent, more conservative DXSUM-based re-derivation
    (build_baseline_diagnosis(): strict baseline-VISCODE match only, no
    enrollment-window fallback). ADSL.DX is preferred because it already
    implements the ADNI-documented enrollment-window baseline-record
    fallback (see derive_blfl_adni() in the package's own R source: a
    record collected within 90 days after enrollment, closest to the
    baseline visit, is accepted as baseline when no exact-baseline-visit
    record exists) that this module's stricter rule deliberately omitted
    rather than reimplement unverified.

    Empirically (see baseline_diagnosis_validation.csv for the actual
    run's numbers), ADSL.DX and this module's independent re-derivation
    agree on every participant both resolve, and ADSL.DX never omits a
    participant the stricter rule resolved -- it only adds participants
    the stricter rule's narrower window missed. That agreement is the
    validation: it confirms DXSUM.DIAGNOSIS (this module's source field)
    and ADSL.DX (built from the same field via the documented window rule)
    are semantically consistent per phase, and that adopting ADSL.DX only
    gains coverage, never silently overrides a disagreeing value.

    Returns (result_df, qc_dict). result_df has RID, DX_BASELINE,
    DX_BASELINE_SOURCE ("adsl_official" | "dxsum_baseline_viscode_only").
    A participant unresolved by both sources simply has no row.
    """
    adsl = adsl_df.copy()
    adsl["RID"] = adsl["SUBJID"].astype(int)
    adsl_enrolled = adsl[adsl["ENRLFL"] == "Y"][["RID", "DX"]].rename(columns={"DX": "DX_ADSL"})
    adsl_enrolled["DX_ADSL"] = adsl_enrolled["DX_ADSL"].replace(_ADSL_DX_LABEL_MAP)

    mine = dxsum_baseline_df[["RID", "DX_BASELINE"]].rename(columns={"DX_BASELINE": "DX_MINE"})
    cmp = mine.merge(adsl_enrolled, on="RID", how="outer")

    unchanged = cmp[cmp["DX_MINE"].notna() & (cmp["DX_MINE"] == cmp["DX_ADSL"])]
    changed = cmp[cmp["DX_MINE"].notna() & cmp["DX_ADSL"].notna() & (cmp["DX_MINE"] != cmp["DX_ADSL"])]
    newly_assigned = cmp[cmp["DX_MINE"].isna() & cmp["DX_ADSL"].notna()]
    mine_only = cmp[cmp["DX_MINE"].notna() & cmp["DX_ADSL"].isna()]
    unresolved = cmp[cmp["DX_MINE"].isna() & cmp["DX_ADSL"].isna()]

    counts_before = {str(k): int(v) for k, v in mine["DX_MINE"].value_counts().items()}

    cmp["DX_BASELINE"] = cmp["DX_ADSL"].combine_first(cmp["DX_MINE"])
    cmp["DX_BASELINE_SOURCE"] = np.where(
        cmp["DX_ADSL"].notna(),
        "adsl_official",
        np.where(cmp["DX_MINE"].notna(), "dxsum_baseline_viscode_only", None),
    )
    result = (
        cmp.dropna(subset=["DX_BASELINE"])[["RID", "DX_BASELINE", "DX_BASELINE_SOURCE"]]
        .reset_index(drop=True)
    )
    counts_after = {str(k): int(v) for k, v in result["DX_BASELINE"].value_counts().items()}

    qc = {
        "n_unchanged": len(unchanged),
        "n_changed": len(changed),
        "n_newly_assigned": len(newly_assigned),
        "n_dxsum_only_not_in_adsl": len(mine_only),
        "n_unresolved": len(unresolved),
        "counts_before": counts_before,
        "counts_after": counts_after,
    }
    return result, qc


# ------------------------------------------------------------------
# Assembling the clinical long table
# ------------------------------------------------------------------


_MMSE_BASELINE_SOURCE_LABELS = {
    "baseline_viscode": "baseline_visit",
    "screening_viscode_fallback": "screening",
}


def build_clinical_long(
    registry_df, ptdemog_df, dxsum_df, adas_df, mmse_df, apoeres_df, visits_df, adsl_df
):
    """
    Top-level assembly: returns (clinical_long_df, qc_dict, artifacts_dict).
    clinical_long_df has one row per (RID, COLPROT, VISCODE) cognitive
    assessment record (ADAS and MMSE outer-merged on the same visit
    key, since a participant may have one, both, or neither at a given
    visit) with:

      RID, ORIGPROT, COLPROT, VISCODE, VISIT_MONTH, EXAMDATE,
      DX_BASELINE_FIXED, DX_BASELINE_SOURCE, DX_AT_VISIT, BASELINE_AGE,
      AGE_AT_VISIT, SEX, APOE4_CARRIER, ADAS_COG13, MMSE,
      ADAS_COG13_BASELINE, MMSE_BASELINE, MMSE_BASELINE_SOURCE,
      VISIT_MAPPING_SOURCE, VISIT_MAPPING_CONFIDENCE.

    "Fixed baseline diagnosis group" (DX_BASELINE_FIXED) is each
    participant's single validated baseline diagnosis (see
    resolve_baseline_diagnosis_with_validation) broadcast onto every one
    of their visit rows -- it does not change over time by construction,
    as requested. DX_AT_VISIT is the actual per-visit diagnosis from
    build_longitudinal_diagnosis and DOES vary over time.
    """
    crosswalk_df = build_visit_month_map(visits_df)
    baseline_viscode_map = build_baseline_viscode_map(visits_df)
    screening_viscode_map = build_screening_viscode_map(visits_df)

    enrollment_df = build_enrollment_table(registry_df, baseline_viscode_map)
    dxsum_baseline_df, dxsum_baseline_qc = build_baseline_diagnosis(
        dxsum_df, enrollment_df, baseline_viscode_map
    )
    baseline_dx_df, baseline_dx_validation_qc = resolve_baseline_diagnosis_with_validation(
        dxsum_baseline_df, adsl_df
    )
    longitudinal_dx_df = build_longitudinal_diagnosis(dxsum_df)
    demog_df = build_demographics(ptdemog_df)
    apoe_df = build_apoe4(apoeres_df)

    adas_clean, adas_qc = build_cognitive_source(
        adas_df, "TOTAL13", "ADAS_COG13", crosswalk_df, enrollment_df
    )
    mmse_clean, mmse_qc = build_cognitive_source(
        mmse_df, "MMSCORE", "MMSE", crosswalk_df, enrollment_df
    )

    cog_key = ["RID", "ORIGPROT", "COLPROT", "VISCODE"]
    adas_side = adas_clean.drop(columns=["target_month"]).rename(
        columns={
            "canonical_month": "canonical_month_adas",
            "mapping_source": "mapping_source_adas",
            "mapping_confidence": "mapping_confidence_adas",
            "EXAMDATE": "EXAMDATE_adas",
        }
    )
    mmse_side = mmse_clean.drop(columns=["target_month"]).rename(
        columns={
            "canonical_month": "canonical_month_mmse",
            "mapping_source": "mapping_source_mmse",
            "mapping_confidence": "mapping_confidence_mmse",
            "EXAMDATE": "EXAMDATE_mmse",
        }
    )

    merged = adas_side.merge(mmse_side, on=cog_key, how="outer")

    merged["EXAMDATE"] = merged["EXAMDATE_adas"].combine_first(merged["EXAMDATE_mmse"])
    merged["VISIT_MONTH"] = merged["canonical_month_adas"].combine_first(
        merged["canonical_month_mmse"]
    )
    merged["VISIT_MAPPING_SOURCE"] = merged["mapping_source_adas"].combine_first(
        merged["mapping_source_mmse"]
    )
    merged["VISIT_MAPPING_CONFIDENCE"] = merged["mapping_confidence_adas"].combine_first(
        merged["mapping_confidence_mmse"]
    )

    merged = merged.merge(
        baseline_dx_df[["RID", "DX_BASELINE", "DX_BASELINE_SOURCE"]], on="RID", how="left"
    )
    merged = merged.merge(
        longitudinal_dx_df[["RID", "COLPROT", "VISCODE", "DIAGNOSIS"]],
        on=["RID", "COLPROT", "VISCODE"],
        how="left",
    )
    merged = merged.merge(demog_df, on="RID", how="left")
    merged = merged.merge(
        apoe_df[["RID", "APOE4_CARRIER", "APOE4_ALLELE_COUNT"]], on="RID", how="left"
    )
    merged = merged.merge(enrollment_df[["RID", "ENRLDT"]], on="RID", how="left")

    merged["AGE_AT_VISIT"] = merged.apply(
        lambda r: compute_age_years(r["DOB"], r["EXAMDATE"]), axis=1
    )
    merged["BASELINE_AGE"] = merged.apply(
        lambda r: compute_age_years(r["DOB"], r["ENRLDT"]), axis=1
    )

    adas_baseline, adas_baseline_qc = select_cognitive_baseline(
        adas_clean, "ADAS_COG13", baseline_viscode_map, screening_viscode_map
    )
    mmse_baseline, mmse_baseline_qc = select_cognitive_baseline(
        mmse_clean, "MMSE", baseline_viscode_map, screening_viscode_map
    )
    mmse_baseline["MMSE_BASELINE_SOURCE"] = mmse_baseline["MMSE_BASELINE_METHOD"].map(
        _MMSE_BASELINE_SOURCE_LABELS
    )
    merged = merged.merge(
        adas_baseline[["RID", "ADAS_COG13_BASELINE"]], on="RID", how="left"
    )
    merged = merged.merge(
        mmse_baseline[["RID", "MMSE_BASELINE", "MMSE_BASELINE_SOURCE"]], on="RID", how="left"
    )

    merged = merged.rename(columns={"DX_BASELINE": "DX_BASELINE_FIXED", "DIAGNOSIS": "DX_AT_VISIT"})

    final_cols = [
        "RID",
        "ORIGPROT",
        "COLPROT",
        "VISCODE",
        "VISIT_MONTH",
        "EXAMDATE",
        "DX_BASELINE_FIXED",
        "DX_BASELINE_SOURCE",
        "DX_AT_VISIT",
        "BASELINE_AGE",
        "AGE_AT_VISIT",
        "SEX",
        "APOE4_CARRIER",
        "APOE4_ALLELE_COUNT",
        "ADAS_COG13",
        "MMSE",
        "ADAS_COG13_BASELINE",
        "MMSE_BASELINE",
        "MMSE_BASELINE_SOURCE",
        "VISIT_MAPPING_SOURCE",
        "VISIT_MAPPING_CONFIDENCE",
    ]
    clinical_long = merged[final_cols].sort_values(["RID", "COLPROT", "VISCODE"]).reset_index(
        drop=True
    )

    qc = {
        "baseline_diagnosis_dxsum_only": dxsum_baseline_qc,
        "baseline_diagnosis_validation": baseline_dx_validation_qc,
        "adas_cog13_source": adas_qc,
        "mmse_source": mmse_qc,
        "adas_cog13_baseline_selection": adas_baseline_qc,
        "mmse_baseline_selection": mmse_baseline_qc,
        "n_participants_with_verified_baseline_dx": baseline_dx_df["RID"].nunique(),
        "n_participants_in_clinical_long": clinical_long["RID"].nunique(),
        "n_rows_clinical_long": len(clinical_long),
    }
    artifacts = {"mmse_baseline_df": mmse_baseline, "enrollment_df": enrollment_df}
    return clinical_long, qc, artifacts


# ------------------------------------------------------------------
# MMSE baseline validation
# ------------------------------------------------------------------


def validate_mmse_baseline(mmse_baseline_df, adsl_df, enrollment_df, long_interval_days=90):
    """
    Cross-validates this module's screening-fallback MMSE baseline
    (select_cognitive_baseline()'s MMSE output) against ADSL.MMSCORE
    (ADNIMERGE2's own official derivation), and quantifies the gap
    between each screening-sourced participant's MMSE date and their
    enrollment/baseline date (ENRLDT).

    `long_interval_days` defaults to 90 -- not an arbitrary threshold,
    but ADNI's own documented enrollment-window length (see
    derive_blfl_adni() / resolve_baseline_diagnosis_with_validation()'s
    docstring: a record is only accepted as a baseline substitute if
    collected within 90 days after enrollment). A screening MMSE
    further from baseline than the window ADNI itself uses to accept a
    substitute baseline record is flagged, not silently accepted.

    Returns (interval_df, qc_dict). interval_df is participant-level
    (RID, MMSE_BASELINE_DATE, ENRLDT, interval_days,
    LONG_INTERVAL_FLAG) for screening-sourced participants only -- kept
    local/never written to outputs/ as-is; callers merge it into
    processed/ tables or reduce it to aggregates before reporting.
    """
    adsl = adsl_df.copy()
    adsl["RID"] = adsl["SUBJID"].astype(int)
    adsl_enrolled = adsl[adsl["ENRLFL"] == "Y"][["RID", "MMSCORE"]].rename(
        columns={"MMSCORE": "MMSE_ADSL"}
    )
    mine = mmse_baseline_df[
        ["RID", "MMSE_BASELINE", "MMSE_BASELINE_METHOD", "MMSE_BASELINE_DATE"]
    ].copy()
    cmp = mine.merge(adsl_enrolled, on="RID", how="outer")
    both = cmp.dropna(subset=["MMSE_BASELINE", "MMSE_ADSL"])
    n_match = int((both["MMSE_BASELINE"] == both["MMSE_ADSL"]).sum())

    screen_source = mine[mine["MMSE_BASELINE_METHOD"] == "screening_viscode_fallback"][
        ["RID", "MMSE_BASELINE_DATE"]
    ]
    interval_df = screen_source.merge(enrollment_df[["RID", "ENRLDT"]], on="RID", how="inner")
    interval_df["MMSE_BASELINE_DATE"] = pd.to_datetime(interval_df["MMSE_BASELINE_DATE"])
    interval_df["ENRLDT"] = pd.to_datetime(interval_df["ENRLDT"])
    interval_df["interval_days"] = (
        interval_df["ENRLDT"] - interval_df["MMSE_BASELINE_DATE"]
    ).dt.days
    interval_df["LONG_INTERVAL_FLAG"] = interval_df["interval_days"] > long_interval_days

    desc = interval_df["interval_days"].describe()
    qc = {
        "n_compared_against_adsl": len(both),
        "n_match": n_match,
        "n_mismatch": len(both) - n_match,
        "n_baseline_visit_sourced": int(
            (mine["MMSE_BASELINE_METHOD"] == "baseline_viscode").sum()
        ),
        "n_screening_sourced": len(screen_source),
        "n_interval_computable": len(interval_df),
        "interval_days_mean": round(float(desc["mean"]), 1) if len(interval_df) else None,
        "interval_days_median": round(float(interval_df["interval_days"].median()), 1)
        if len(interval_df)
        else None,
        "interval_days_min": int(desc["min"]) if len(interval_df) else None,
        "interval_days_max": int(desc["max"]) if len(interval_df) else None,
        "n_negative_interval": int((interval_df["interval_days"] < 0).sum()),
        "n_long_interval_flagged": int(interval_df["LONG_INTERVAL_FLAG"].sum()),
        "long_interval_threshold_days": long_interval_days,
    }
    return interval_df, qc


# ------------------------------------------------------------------
# Longitudinal diagnosis source description (descriptive/sensitivity
# use only -- does NOT replace the fixed-baseline-group strategy)
# ------------------------------------------------------------------


def describe_longitudinal_diagnosis_sources(dxsum_longitudinal_df, adrs_df):
    """
    Confirms and describes the available per-visit (longitudinal)
    diagnosis source for descriptive/sensitivity use. DXSUM.DIAGNOSIS
    (this module's build_longitudinal_diagnosis() output) is the
    primary source used; ADRS (PARAMCD == "DX") is ADNIMERGE2's own
    ADaM-style long-format equivalent, built from the same underlying
    field. Cross-checked here at the aggregate level only -- not
    row-by-row -- since this task explicitly does not require replacing
    the fixed-baseline-group primary analysis strategy with a
    longitudinal one.
    """
    adrs_dx_all = adrs_df[adrs_df["PARAMCD"] == "DX"].copy()
    adrs_dx_all["AVALC"] = adrs_dx_all["AVALC"].replace({"DEM": "Dementia"})
    # dxsum_longitudinal_df (build_longitudinal_diagnosis()'s output) is
    # already filtered to non-missing DIAGNOSIS by design; filter ADRS's
    # AVALC the same way before comparing record counts, so the comparison
    # is apples-to-apples rather than a filtered count vs. an unfiltered one
    # (ADRS additionally carries ~45 placeholder rows with a missing AVALC
    # that DXSUM.DIAGNOSIS never included in the first place).
    adrs_dx = adrs_dx_all[adrs_dx_all["AVALC"].notna()]

    dxsum_counts = {
        str(k): int(v) for k, v in dxsum_longitudinal_df["DIAGNOSIS"].value_counts().items()
    }
    adrs_counts = {str(k): int(v) for k, v in adrs_dx["AVALC"].value_counts().items()}

    return {
        "recommended_source": "DXSUM.DIAGNOSIS (this module's build_longitudinal_diagnosis())",
        "alternative_source": 'ADRS (PARAMCD == "DX", AVALC/AVAL, ABLFL for baseline flag)',
        "dxsum_n_records": len(dxsum_longitudinal_df),
        "adrs_dx_n_records": len(adrs_dx),
        "adrs_dx_n_records_incl_missing_diagnosis": len(adrs_dx_all),
        "dxsum_diagnosis_counts": dxsum_counts,
        "adrs_diagnosis_counts": adrs_counts,
        "n_adrs_baseline_flagged": int((adrs_dx_all["ABLFL"] == "Y").sum()),
        "note": (
            "ADRS.DX record count and diagnosis-category counts match DXSUM.DIAGNOSIS exactly, "
            "once both are restricted to non-missing diagnosis values (ADRS additionally carries "
            "~45 placeholder rows with a missing AVALC that DXSUM.DIAGNOSIS never included) and "
            "after normalizing ADRS's 'DEM' label to 'Dementia' -- both are built from the same "
            "underlying harmonized field, confirmed via ADNI's own DATADIC MAPPING_NOTES (DXCURREN "
            "for ADNI1 and DXCHANGE for ADNIGO/ADNI2 were both mapped into DIAGNOSIS during the "
            "package build; DIAGNOSIS is used directly for ADNI2/3/4). ADRS additionally carries its "
            "own ABLFL baseline flag (the same enrollment-window logic used to build ADSL.DX) and "
            "unifies diagnosis with death records in one domain -- useful for a future descriptive/"
            "sensitivity view, not adopted as the primary fixed-baseline-group source here."
        ),
    }


# ------------------------------------------------------------------
# 8. Analysis eligibility -- cognitive
# ------------------------------------------------------------------


def add_cognitive_eligibility(clinical_long):
    """
    Adds two eligibility flag columns for cognitive-endpoint analysis
    readiness, per participant (broadcast onto every row for that
    RID), requiring: a valid (verified, non-screen-failure) baseline
    diagnosis, a valid baseline score, at least one valid follow-up
    (VISIT_MONTH > 0) score, baseline age, and sex. Computed
    separately for ADAS-Cog13 and MMSE since a participant may be
    eligible for one endpoint but not the other.
    """
    df = clinical_long.copy()

    def _eligibility_for(value_col, baseline_col):
        has_baseline_dx = df["DX_BASELINE_FIXED"].notna()
        has_baseline_score = df[baseline_col].notna()
        has_baseline_age = df["BASELINE_AGE"].notna()
        has_sex = df["SEX"].notna()

        is_valid_followup = (df["VISIT_MONTH"] > 0) & df[value_col].notna()
        followup_flag = (
            pd.DataFrame({"RID": df["RID"], "_v": is_valid_followup})
            .groupby("RID")["_v"]
            .any()
        )
        has_followup_per_row = df["RID"].map(followup_flag).fillna(False)

        return (
            has_baseline_dx
            & has_baseline_score
            & has_baseline_age
            & has_sex
            & has_followup_per_row
        )

    df["ADAS_COG13_ELIGIBLE"] = _eligibility_for("ADAS_COG13", "ADAS_COG13_BASELINE")
    df["MMSE_ELIGIBLE"] = _eligibility_for("MMSE", "MMSE_BASELINE")
    return df
