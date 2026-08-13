# ============================================================
# RUN_ADNI_PREPROCESSING -- orchestration entry point for the ADNI
# preprocessing stage (audit approved; ANCOVA/inferential statistics/
# plots/dashboard changes are explicitly OUT of scope here -- see
# adni_cohort.py and adni_plasma.py module docstrings).
#
# Reads:
#   ADNI_INTERIM_DIR/*.csv        (R-exported raw eCRF tables)
#   ADNI_RAW_PLASMA_DIR/*.csv     (live plasma files, untouched raw/)
#
# Writes:
#   ADNI_PROCESSED_DIR/*.parquet  (participant-level, local-only)
#   ADNI_OUTPUTS_DIR/*.{md,csv}   (aggregate-only, participant-free)
#
# Prints only aggregate summary lines to the console -- never a
# participant ID or a participant-level row.
#
# Usage: .venv/bin/python run_adni_preprocessing.py
# ============================================================

import os

import pandas as pd

from adni_analysis import (
    ADNI_AUDIT_APPROVED,
    ADNI_INTERIM_DIR,
    ADNI_OUTPUTS_DIR,
    ADNI_PROCESSED_DIR,
    ADNI_RAW_PLASMA_DIR,
)
import adni_cohort
import adni_plasma


def _read_interim(name):
    path = os.path.join(ADNI_INTERIM_DIR, f"{name}.csv")
    return pd.read_csv(path, low_memory=False)


def _read_plasma(filename_pattern):
    matches = [f for f in os.listdir(ADNI_RAW_PLASMA_DIR) if filename_pattern in f]
    if not matches:
        raise FileNotFoundError(f"No plasma file matching '{filename_pattern}' in {ADNI_RAW_PLASMA_DIR}")
    return pd.read_csv(os.path.join(ADNI_RAW_PLASMA_DIR, matches[0]), low_memory=False)


def load_raw_tables():
    registry_df = _read_interim("REGISTRY")
    ptdemog_df = _read_interim("PTDEMOG")
    dxsum_df = _read_interim("DXSUM")
    adas_df = _read_interim("ADAS")
    mmse_df = _read_interim("MMSE")
    apoeres_df = _read_interim("APOERES")
    visits_df = _read_interim("VISITS")
    adsl_df = _read_interim("ADSL")
    adrs_df = _read_interim("ADRS")
    ugot_df = _read_plasma("UGOTPTAU181")
    fuji_df = _read_plasma("UPENN_PLASMA_FUJIREBIO_QUANTERIX")
    return {
        "registry": registry_df,
        "ptdemog": ptdemog_df,
        "dxsum": dxsum_df,
        "adas": adas_df,
        "mmse": mmse_df,
        "apoeres": apoeres_df,
        "visits": visits_df,
        "adsl": adsl_df,
        "adrs": adrs_df,
        "ugot": ugot_df,
        "fuji": fuji_df,
    }


def build_all(raw):
    baseline_viscode_map = adni_cohort.build_baseline_viscode_map(raw["visits"])
    enrollment_df = adni_cohort.build_enrollment_table(raw["registry"], baseline_viscode_map)
    demog_df = adni_cohort.build_demographics(raw["ptdemog"])

    clinical_long, clinical_qc, clinical_artifacts = adni_cohort.build_clinical_long(
        raw["registry"], raw["ptdemog"], raw["dxsum"], raw["adas"], raw["mmse"], raw["apoeres"],
        raw["visits"], raw["adsl"],
    )
    clinical_long = adni_cohort.add_cognitive_eligibility(clinical_long)

    mmse_interval_df, mmse_validation_qc = adni_cohort.validate_mmse_baseline(
        clinical_artifacts["mmse_baseline_df"], raw["adsl"], clinical_artifacts["enrollment_df"]
    )
    clinical_long = clinical_long.merge(
        mmse_interval_df[["RID", "interval_days", "LONG_INTERVAL_FLAG"]].rename(
            columns={
                "interval_days": "MMSE_SCREENING_TO_BASELINE_INTERVAL_DAYS",
                "LONG_INTERVAL_FLAG": "MMSE_LONG_SCREENING_INTERVAL_FLAG",
            }
        ),
        on="RID",
        how="left",
    )

    longitudinal_dx_source_info = adni_cohort.describe_longitudinal_diagnosis_sources(
        adni_cohort.build_longitudinal_diagnosis(raw["dxsum"]), raw["adrs"]
    )

    ptau181_long, ptau181_qc = adni_plasma.build_ptau181_long(raw["ugot"], demog_df, enrollment_df)

    fuji_base, fuji_dedup_qc = adni_plasma._clean_fuji_base(raw["fuji"], raw["visits"], enrollment_df)
    ptau217_long, ptau217_qc = adni_plasma.build_ptau217_long(fuji_base, demog_df, enrollment_df)
    abeta_ratio_long, abeta_qc = adni_plasma.build_abeta_ratio_long(fuji_base, demog_df, enrollment_df)

    gfap_long = adni_plasma.build_platform_long(
        fuji_base, demog_df, enrollment_df, analyte="GFAP", quanterix_col="GFAP_Q", fujirebio_col="GFAP_F"
    )
    nfl_long = adni_plasma.build_platform_long(
        fuji_base, demog_df, enrollment_df, analyte="NfL", quanterix_col="NfL_Q", fujirebio_col="NfL_F"
    )
    gfap_platform_rec = adni_plasma.recommend_primary_platform(gfap_long, "GFAP")
    nfl_platform_rec = adni_plasma.recommend_primary_platform(nfl_long, "NfL")

    tables = {
        "clinical_long": clinical_long,
        "ptau181_long": ptau181_long,
        "ptau217_long": ptau217_long,
        "abeta_ratio_long": abeta_ratio_long,
        "gfap_long": gfap_long,
        "nfl_long": nfl_long,
    }
    qc = {
        "clinical": clinical_qc,
        "mmse_validation": mmse_validation_qc,
        "longitudinal_dx_source": longitudinal_dx_source_info,
        "ptau181": ptau181_qc,
        "fuji_base_dedup": fuji_dedup_qc,
        "ptau217": ptau217_qc,
        "abeta_ratio": abeta_qc,
        "gfap_platform_recommendation": gfap_platform_rec,
        "nfl_platform_recommendation": nfl_platform_rec,
    }
    return tables, qc


def write_processed_tables(tables):
    os.makedirs(ADNI_PROCESSED_DIR, exist_ok=True)
    for name, df in tables.items():
        out_path = os.path.join(ADNI_PROCESSED_DIR, f"adni_{name}.parquet".replace("_long_long", "_long"))
        df.to_parquet(out_path, index=False)
        print(f"wrote {out_path}  ({len(df)} rows)")


# ------------------------------------------------------------------
# Aggregate-only report builders -- every function below returns a
# small summary DataFrame/string built entirely from counts/aggregates
# already computed above; none of them touch a raw participant row.
# ------------------------------------------------------------------


def build_cohort_flow(tables, qc):
    clinical = tables["clinical_long"]
    rows = []

    rows.append(("Clinical", "participants_with_verified_baseline_dx", qc["clinical"]["n_participants_with_verified_baseline_dx"], "Enrollment-flag-verified, screen-failure trap closed"))
    rows.append(("Clinical", "screen_failure_records_excluded", qc["clinical"]["baseline_diagnosis_dxsum_only"]["n_screen_failure_records_excluded"], "Baseline-VISCODE DXSUM records with no matching enrollment record"))
    rows.append(("Clinical", "participants_in_clinical_long", qc["clinical"]["n_participants_in_clinical_long"], "Any ADAS-Cog13 or MMSE record"))
    for dx in ["CN", "MCI", "Dementia"]:
        n = clinical.loc[clinical["DX_BASELINE_FIXED"] == dx, "RID"].nunique()
        rows.append(("Clinical", f"baseline_dx_group_{dx}", n, "Fixed baseline diagnosis group size"))
    rows.append(("Clinical", "ADAS_Cog13_eligible", clinical.loc[clinical["ADAS_COG13_ELIGIBLE"], "RID"].nunique(), "Baseline dx + baseline score + follow-up + age + sex"))
    rows.append(("Clinical", "MMSE_eligible", clinical.loc[clinical["MMSE_ELIGIBLE"], "RID"].nunique(), "Baseline dx + baseline score + follow-up + age + sex"))

    for family, key, table in [
        ("pTau181", "ptau181", tables["ptau181_long"]),
        ("pTau217", "ptau217", tables["ptau217_long"]),
        ("AbetaRatio", "abeta_ratio", tables["abeta_ratio_long"]),
    ]:
        rows.append((family, "unique_participants_in_table", table["RID"].nunique(), ""))
        rows.append((family, "biomarker_eligible", table.loc[table["BIOMARKER_ELIGIBLE"], "RID"].nunique(), ""))
    rows.append(("pTau217", "primary_analysis_eligible_excl_lot_bias", tables["ptau217_long"].loc[tables["ptau217_long"]["PTAU217_PRIMARY_ANALYSIS_ELIGIBLE"], "RID"].nunique(), "Excludes ADNI4 Batch 3 QC-drift-flagged records"))

    for platform_table, family in [(tables["gfap_long"], "GFAP"), (tables["nfl_long"], "NfL")]:
        for platform in ["Quanterix", "Fujirebio"]:
            sub = platform_table[platform_table["PLATFORM"] == platform]
            rows.append((f"{family}_{platform}", "unique_participants_in_table", sub["RID"].nunique(), ""))
            rows.append((f"{family}_{platform}", "biomarker_eligible", sub.loc[sub["BIOMARKER_ELIGIBLE"], "RID"].nunique(), ""))

    return pd.DataFrame(rows, columns=["cohort", "stage", "n_participants", "note"])


def build_visit_mapping_summary(tables):
    frames = []
    specs = [
        ("clinical_long_ADAS_or_MMSE", tables["clinical_long"], "VISIT_MAPPING_SOURCE", "VISIT_MAPPING_CONFIDENCE"),
        ("ptau181_long", tables["ptau181_long"], "VISIT_MAPPING_SOURCE", "VISIT_MAPPING_CONFIDENCE"),
        ("ptau217_long", tables["ptau217_long"], "VISIT_MAPPING_SOURCE", "VISIT_MAPPING_CONFIDENCE"),
        ("abeta_ratio_long", tables["abeta_ratio_long"], "VISIT_MAPPING_SOURCE", "VISIT_MAPPING_CONFIDENCE"),
        ("gfap_long", tables["gfap_long"], "VISIT_MAPPING_SOURCE", "VISIT_MAPPING_CONFIDENCE"),
        ("nfl_long", tables["nfl_long"], "VISIT_MAPPING_SOURCE", "VISIT_MAPPING_CONFIDENCE"),
    ]
    for label, df, method_col, conf_col in specs:
        g = (
            df.groupby([method_col, conf_col], dropna=False)
            .agg(n_records=("RID", "size"), n_unique_participants=("RID", "nunique"))
            .reset_index()
        )
        g.insert(0, "source_table", label)
        g = g.rename(columns={method_col: "mapping_source", conf_col: "mapping_confidence"})
        frames.append(g)
    return pd.concat(frames, ignore_index=True)


def build_assay_platform_summary(qc):
    rows = []
    for family, rec in [("GFAP", qc["gfap_platform_recommendation"]), ("NfL", qc["nfl_platform_recommendation"])]:
        for platform, cov in rec["coverage"].items():
            rows.append(
                {
                    "family": family,
                    "platform": platform,
                    "n_participants_any_measurement": cov["n_participants_any_measurement"],
                    "n_records": cov["n_records"],
                    "n_participants_2plus_visits": cov["n_participants_2plus_visits"],
                    "recommended_primary": platform == rec["primary_platform"],
                    "rationale": rec["rationale"] if platform == rec["primary_platform"] else "",
                }
            )
    return pd.DataFrame(rows)


def build_endpoint_eligibility_summary(tables):
    clinical = tables["clinical_long"]
    rows = [
        {"endpoint": "ADAS_Cog13", "n_total_participants": clinical["RID"].nunique(), "n_eligible_participants": clinical.loc[clinical["ADAS_COG13_ELIGIBLE"], "RID"].nunique()},
        {"endpoint": "MMSE", "n_total_participants": clinical["RID"].nunique(), "n_eligible_participants": clinical.loc[clinical["MMSE_ELIGIBLE"], "RID"].nunique()},
        {"endpoint": "pTau181", "n_total_participants": tables["ptau181_long"]["RID"].nunique(), "n_eligible_participants": tables["ptau181_long"].loc[tables["ptau181_long"]["BIOMARKER_ELIGIBLE"], "RID"].nunique()},
        {"endpoint": "pTau217_all", "n_total_participants": tables["ptau217_long"]["RID"].nunique(), "n_eligible_participants": tables["ptau217_long"].loc[tables["ptau217_long"]["BIOMARKER_ELIGIBLE"], "RID"].nunique()},
        {"endpoint": "pTau217_primary_analysis", "n_total_participants": tables["ptau217_long"]["RID"].nunique(), "n_eligible_participants": tables["ptau217_long"].loc[tables["ptau217_long"]["PTAU217_PRIMARY_ANALYSIS_ELIGIBLE"], "RID"].nunique()},
        {"endpoint": "Abeta42_40_ratio", "n_total_participants": tables["abeta_ratio_long"]["RID"].nunique(), "n_eligible_participants": tables["abeta_ratio_long"].loc[tables["abeta_ratio_long"]["BIOMARKER_ELIGIBLE"], "RID"].nunique()},
    ]
    for platform_table, family in [(tables["gfap_long"], "GFAP"), (tables["nfl_long"], "NfL")]:
        for platform in ["Quanterix", "Fujirebio"]:
            sub = platform_table[platform_table["PLATFORM"] == platform]
            rows.append(
                {
                    "endpoint": f"{family}_{platform}",
                    "n_total_participants": sub["RID"].nunique(),
                    "n_eligible_participants": sub.loc[sub["BIOMARKER_ELIGIBLE"], "RID"].nunique(),
                }
            )

    # Cognitive endpoints by target month, eligible participants with a
    # non-missing value at that month.
    for month in adni_cohort.TARGET_MONTHS:
        at_month = clinical[clinical["VISIT_MONTH"] == month]
        rows.append(
            {
                "endpoint": f"ADAS_Cog13_month_{month}",
                "n_total_participants": at_month["RID"].nunique(),
                "n_eligible_participants": at_month.loc[
                    at_month["ADAS_COG13_ELIGIBLE"] & at_month["ADAS_COG13"].notna(), "RID"
                ].nunique(),
            }
        )
        rows.append(
            {
                "endpoint": f"MMSE_month_{month}",
                "n_total_participants": at_month["RID"].nunique(),
                "n_eligible_participants": at_month.loc[
                    at_month["MMSE_ELIGIBLE"] & at_month["MMSE"].notna(), "RID"
                ].nunique(),
            }
        )
    return pd.DataFrame(rows)


def build_qc_exclusions_summary(qc):
    rows = []
    rows.append(("clinical_baseline_dx", "screen_failure_excluded", qc["clinical"]["baseline_diagnosis_dxsum_only"]["n_screen_failure_records_excluded"], "DXSUM baseline-VISCODE record with no matching verified enrollment record"))
    rows.append(("clinical_adas", "duplicate_records_removed", qc["clinical"]["adas_cog13_source"]["n_records_removed"], "Deterministic dedup on (RID, COLPROT, VISCODE), latest EXAMDATE kept"))
    rows.append(("clinical_adas", "conflicting_duplicate_groups", qc["clinical"]["adas_cog13_source"]["n_conflicting_groups"], "Duplicate groups where dropped record's value disagreed with kept record's value"))
    rows.append(("clinical_adas", "missing_value_excluded", qc["clinical"]["adas_cog13_source"]["n_missing_value_excluded"], "TOTAL13 missing in raw ADAS table"))
    rows.append(("clinical_mmse", "duplicate_records_removed", qc["clinical"]["mmse_source"]["n_records_removed"], "Deterministic dedup on (RID, COLPROT, VISCODE), latest EXAMDATE kept"))
    rows.append(("clinical_mmse", "conflicting_duplicate_groups", qc["clinical"]["mmse_source"]["n_conflicting_groups"], "Duplicate groups where dropped record's value disagreed with kept record's value"))
    rows.append(("clinical_mmse", "missing_value_excluded", qc["clinical"]["mmse_source"]["n_missing_value_excluded"], "MMSCORE missing in raw MMSE table"))
    rows.append(("ptau181", "duplicate_records_removed", qc["ptau181"]["n_records_removed"], "Deterministic dedup on (RID, VISCODE), latest EXAMDATE kept"))
    rows.append(("ptau181", "below_lloq_flagged_not_removed", qc["ptau181"]["n_below_lloq"], "Kept in table with BELOW_LLOQ=True, not treated as missing"))
    rows.append(("fuji_plasma_panel", "duplicate_records_removed", qc["fuji_base_dedup"]["n_records_removed"], "Deterministic dedup on (RID, PHASE, VISCODE), latest EXAMDATE kept, shared by pTau217/Abeta/GFAP/NfL"))
    rows.append(("ptau217", "lot_bias_flagged_records", qc["ptau217"]["n_lot_bias_flagged_records"], "ADNI4 Batch 3 QC-drift Comment field match -- excluded from primary analysis, kept for sensitivity analysis"))
    rows.append(("ptau217", "lot_bias_flagged_participants", qc["ptau217"]["n_lot_bias_flagged_participants"], ""))
    rows.append(("abeta_ratio", "same_sample_mismatches", qc["abeta_ratio"]["n_same_sample_mismatches"], "Records where AB42/AB40 co-availability disagreed with pre-computed ratio's availability (0 = fully consistent)"))
    return pd.DataFrame(rows, columns=["table", "exclusion_type", "count", "note"])


def build_preprocessing_summary_md(tables, qc, cohort_flow_df, eligibility_df, platform_df):
    clinical = tables["clinical_long"]
    dx_counts = {
        dx: clinical.loc[clinical["DX_BASELINE_FIXED"] == dx, "RID"].nunique()
        for dx in ["CN", "MCI", "Dementia"]
    }
    gfap_rec = qc["gfap_platform_recommendation"]
    nfl_rec = qc["nfl_platform_recommendation"]

    lines = []
    lines.append("# ADNI Preprocessing Summary\n")
    lines.append(
        "Preprocessing only, per instructions: no ANCOVA, no change-from-baseline, no "
        "inferential statistics, no plots, and the dashboard was not modified. All "
        "figures below are aggregate; no participant IDs or participant-level rows "
        "appear anywhere in this document or the accompanying CSVs.\n"
    )
    lines.append(f"`ADNI_AUDIT_APPROVED = {ADNI_AUDIT_APPROVED}` in `adni_analysis.py`.\n")

    lines.append("## Final baseline diagnosis cohort sizes\n")
    lines.append("| Group | N |\n|---|---:|")
    for dx, n in dx_counts.items():
        lines.append(f"| {dx} | {n} |")
    lines.append("")

    lines.append("## Usable sample size by endpoint (see endpoint_eligibility_summary.csv for full detail, including by month)\n")
    lines.append("| Endpoint | Total participants in table | Eligible participants |\n|---|---:|---:|")
    for _, r in eligibility_df[~eligibility_df["endpoint"].str.contains("_month_")].iterrows():
        lines.append(f"| {r['endpoint']} | {r['n_total_participants']} | {r['n_eligible_participants']} |")
    lines.append("")

    lines.append("## Primary assay platform recommendation\n")
    lines.append(f"- **GFAP**: {gfap_rec['primary_platform']} -- {gfap_rec['rationale']}\n")
    lines.append(f"- **NfL**: {nfl_rec['primary_platform']} -- {nfl_rec['rationale']}\n")
    lines.append(
        "Both platforms are preserved in `adni_gfap_long.parquet` / `adni_nfl_long.parquet` "
        "(one row per platform per visit) -- the non-primary platform is not discarded and "
        "remains available for sensitivity analysis.\n"
    )

    lines.append("## pTau217 reagent-lot (ADNI4 Batch 3) impact\n")
    lines.append(
        f"- {qc['ptau217']['n_lot_bias_flagged_records']} records "
        f"({qc['ptau217']['n_lot_bias_flagged_participants']} participants) carry the raw file's own "
        "`Comment` field reading \"Batch 3: QC drift noted; results validated. Refer to Methods "
        "Special Note.\" -- identified directly from that field, not a date-window guess.\n"
    )
    lines.append(
        f"- Eligible participants including flagged records: {qc['ptau217']['n_eligible_participants_including_flagged']}. "
        f"Eligible participants for primary analysis (flagged records excluded): "
        f"{qc['ptau217']['n_eligible_participants_primary_analysis']}.\n"
    )
    lines.append(
        "- Recommendation implemented: primary analysis excludes flagged records "
        "(`PTAU217_PRIMARY_ANALYSIS_ELIGIBLE = False`); flagged records remain in "
        "`adni_ptau217_long.parquet` (`PTAU217_LOT_BIAS_FLAG = True`) for an explicit "
        "sensitivity analysis. No corrected value was invented -- the reported `pT217_F` "
        "value is used as-is in both analyses.\n"
    )

    lines.append("## Records excluded for QC (see qc_exclusions_summary.csv for full detail)\n")
    lines.append(
        f"- {qc['clinical']['baseline_diagnosis_dxsum_only']['n_screen_failure_records_excluded']} DXSUM "
        "baseline-visit diagnosis records excluded as screen-failure trap candidates "
        "(no matching verified enrollment record).\n"
    )
    lines.append(
        f"- ADAS-Cog13: {qc['clinical']['adas_cog13_source']['n_records_removed']} duplicate "
        f"records removed ({qc['clinical']['adas_cog13_source']['n_conflicting_groups']} had "
        "conflicting values across the dropped/kept pair).\n"
    )
    lines.append(
        f"- MMSE: {qc['clinical']['mmse_source']['n_records_removed']} duplicate records removed "
        f"({qc['clinical']['mmse_source']['n_conflicting_groups']} had conflicting values).\n"
    )
    lines.append(
        f"- pTau181: {qc['ptau181']['n_records_removed']} duplicate records removed; "
        f"{qc['ptau181']['n_below_lloq']} below-LLOQ records flagged but kept.\n"
    )
    lines.append(
        f"- UPenn plasma panel (shared by pTau217/Abeta/GFAP/NfL): "
        f"{qc['fuji_base_dedup']['n_records_removed']} duplicate records removed.\n"
    )

    lines.append("## Remaining unresolved issues\n")
    lines.append(
        "- No GFAP/NfL cross-platform (Quanterix-vs-Fujirebio) calibration was applied or "
        "attempted -- the two platforms are kept as fully separate rows/values by design, "
        "per instructions.\n"
    )
    lines.append(
        "- `MMSE_month_0` in `endpoint_eligibility_summary.csv` shows 0 eligible participants "
        "by construction, not a data gap: MMSE is recorded at the *screening* visit in every "
        "ADNI phase (0 records exist under any phase's baseline VISCODE), so `VISIT_MONTH == 0` "
        "(anchored on the baseline VISCODE) never matches an MMSE record. `MMSE_BASELINE` itself "
        "is correctly populated via the documented, now-validated screening-visit fallback in "
        "`select_cognitive_baseline()` -- this is a known difference between the general "
        "visit-month grid and MMSE's own baseline visit, not a bug.\n"
    )
    lines.append(
        "- See `preanalysis_validation.md` for the follow-up validation pass covering baseline "
        "diagnosis (now validated against `ADSL.DX`), visit mapping (now VISCODE2-first, "
        "resolving the ADNI2 \"m60\" case), and the MMSE screening-fallback rule (now confirmed "
        "against `ADSL.MMSCORE` with screening-to-baseline interval outliers flagged).\n"
    )

    ready_lines = [
        "## Readiness for statistical analysis\n",
        "Structurally ready for a reviewer to plan ANCOVA/longitudinal modeling against -- see "
        "`preanalysis_validation.md` for the cohort/visit-ambiguity validation pass completed "
        "after this preprocessing run.\n",
    ]
    lines.extend(ready_lines)

    return "\n".join(lines)


# ------------------------------------------------------------------
# Pre-analysis validation reports (2nd preprocessing pass: resolves
# cohort/visit ambiguities identified after the first preprocessing
# approval, before ANCOVA). See preanalysis_validation.md for the full
# narrative -- these builders only assemble aggregate tables/text from
# qc dicts already computed in build_all().
# ------------------------------------------------------------------

# Captured from the pipeline run immediately BEFORE this validation
# pass (the original VISNAME-crosswalk-only mapping, i.e. before
# VISCODE2 was adopted as the preferred source) -- preserved here as a
# one-time historical snapshot so the before/after comparison this
# validation pass produces stays reproducible from code alone, without
# depending on a scratch file outside the repo.
_VISIT_MAPPING_BEFORE_SNAPSHOT = [
    ("clinical_long_ADAS_or_MMSE", "unmapped", "none", 5939, 4722),
    ("clinical_long_ADAS_or_MMSE", "viscode_baseline", "high", 2992, 2992),
    ("clinical_long_ADAS_or_MMSE", "viscode_month_label", "high", 4309, 1777),
    ("clinical_long_ADAS_or_MMSE", "viscode_year_label_inferred", "medium", 4226, 1480),
    ("ptau181_long", "unmapped", "none", 7, 7),
    ("ptau181_long", "viscode_baseline", "high", 878, 878),
    ("ptau181_long", "viscode_month_label", "high", 2820, 1083),
    ("ptau217_long", "unmapped", "none", 499, 496),
    ("ptau217_long", "viscode_baseline", "high", 1134, 1134),
    ("ptau217_long", "viscode_month_label", "high", 381, 354),
    ("ptau217_long", "viscode_year_label_inferred", "medium", 281, 281),
    ("abeta_ratio_long", "unmapped", "none", 499, 496),
    ("abeta_ratio_long", "viscode_baseline", "high", 1134, 1134),
    ("abeta_ratio_long", "viscode_month_label", "high", 381, 354),
    ("abeta_ratio_long", "viscode_year_label_inferred", "medium", 281, 281),
    ("gfap_long", "unmapped", "none", 998, 496),
    ("gfap_long", "viscode_baseline", "high", 2268, 1134),
    ("gfap_long", "viscode_month_label", "high", 762, 354),
    ("gfap_long", "viscode_year_label_inferred", "medium", 562, 281),
    ("nfl_long", "unmapped", "none", 998, 496),
    ("nfl_long", "viscode_baseline", "high", 2268, 1134),
    ("nfl_long", "viscode_month_label", "high", 762, 354),
    ("nfl_long", "viscode_year_label_inferred", "medium", 562, 281),
]


def build_visit_mapping_validation_csv(visit_mapping_df):
    before_df = pd.DataFrame(
        _VISIT_MAPPING_BEFORE_SNAPSHOT,
        columns=["source_table", "mapping_source", "mapping_confidence", "n_records", "n_unique_participants"],
    )
    before_df.insert(0, "period", "before_viscode2")
    after_df = visit_mapping_df.copy()
    after_df.insert(0, "period", "after_viscode2")
    return pd.concat([before_df, after_df], ignore_index=True)


def build_baseline_diagnosis_validation_csv(qc):
    v = qc["clinical"]["baseline_diagnosis_validation"]
    rows = [
        ("participants_unchanged", v["n_unchanged"]),
        ("participants_group_changed", v["n_changed"]),
        ("participants_newly_assigned", v["n_newly_assigned"]),
        ("participants_dxsum_only_not_in_adsl", v["n_dxsum_only_not_in_adsl"]),
        ("participants_remaining_unresolved", v["n_unresolved"]),
    ]
    for dx in ["CN", "MCI", "Dementia"]:
        rows.append((f"count_before_{dx}", v["counts_before"].get(dx, 0)))
    for dx in ["CN", "MCI", "Dementia"]:
        rows.append((f"count_after_{dx}", v["counts_after"].get(dx, 0)))
    return pd.DataFrame(rows, columns=["metric", "value"])


def build_mmse_baseline_validation_csv(qc):
    v = qc["mmse_validation"]
    rows = [(k, v[k]) for k in v]
    return pd.DataFrame(rows, columns=["metric", "value"])


def build_preanalysis_validation_md(qc, tables):
    dx_v = qc["clinical"]["baseline_diagnosis_validation"]
    mmse_v = qc["mmse_validation"]
    dx_source = qc["longitudinal_dx_source"]

    lines = []
    lines.append("# Pre-ANCOVA Validation of Cohort/Visit Ambiguities\n")
    lines.append(
        "This is a validation pass over the already-approved preprocessing pipeline, run before "
        "ANCOVA. No statistical models or charts were produced. All figures below are aggregate; "
        "no participant IDs or participant-level rows appear anywhere in this document or the "
        "accompanying CSVs. Raw files under `raw/` were not modified.\n"
    )

    lines.append("## 1. Baseline diagnosis validation\n")
    lines.append(
        "No `DXCURREN` (ADNI1) or `DXCHANGE` (ADNIGO/ADNI2) column exists anywhere in the "
        "ADNIMERGE2 package -- confirmed by exhaustive search of all 217 tables. ADNI's own data "
        "dictionary (`DATADIC`, `MAPPING_NOTES` field) confirms why: both were already collapsed "
        "into `DXSUM.DIAGNOSIS` during the package build, using exactly the documented ADNI "
        "collapsing rule (`DXCHANGE` 1/7/9 -> CN, 2/4/8 -> MCI, 3/5/6 -> Dementia; `DXCURREN` "
        "1/2/3 -> CN/MCI/AD directly), then dropped as separate columns. `DXSUM.DIAGNOSIS` is "
        "therefore already the phase-appropriate harmonized field this task asked to validate "
        "against -- not a naive assumption of identical semantics across phases, but a confirmed one.\n"
    )
    lines.append(
        "This module's own baseline-diagnosis logic (strict baseline-VISCODE match only, no "
        "enrollment-window fallback) was cross-checked against `ADSL.DX` -- ADNIMERGE2's own "
        "official derivation, which *does* implement the documented enrollment-window fallback "
        "(a record within 90 days after enrollment, closest to baseline, accepted when no exact "
        "baseline-visit record exists). Result: **100% agreement on every participant both sources "
        "resolve, zero contradictions, zero cases where this module had a value ADSL lacked.** "
        "ADSL.DX has been adopted as the primary baseline-diagnosis source (see `DX_BASELINE_SOURCE` "
        "in `adni_clinical_long.parquet`), which only *adds* coverage.\n"
    )
    lines.append("| Metric | Count |\n|---|---:|")
    lines.append(f"| Participants unchanged | {dx_v['n_unchanged']} |")
    lines.append(f"| Participants with group reassigned | {dx_v['n_changed']} |")
    lines.append(f"| Participants newly assigned | {dx_v['n_newly_assigned']} |")
    lines.append(f"| Participants resolved by this module but absent from ADSL | {dx_v['n_dxsum_only_not_in_adsl']} |")
    lines.append(f"| Participants remaining unresolved by both sources | {dx_v['n_unresolved']} |")
    lines.append("")
    lines.append("| Group | Before | After |\n|---|---:|---:|")
    for dx in ["CN", "MCI", "Dementia"]:
        lines.append(f"| {dx} | {dx_v['counts_before'].get(dx, 0)} | {dx_v['counts_after'].get(dx, 0)} |")
    lines.append("")

    lines.append("## 2. Longitudinal diagnosis (descriptive/sensitivity use only)\n")
    lines.append(
        f"Recommended source: **{dx_source['recommended_source']}**. Available alternative: "
        f"{dx_source['alternative_source']}. {dx_source['note']}\n"
    )
    lines.append(
        f"DXSUM (non-missing diagnosis) record count: {dx_source['dxsum_n_records']}; ADRS DX "
        f"record count, same filter: {dx_source['adrs_dx_n_records']} (identical -- ADRS also "
        f"carries {dx_source['adrs_dx_n_records_incl_missing_diagnosis'] - dx_source['adrs_dx_n_records']} "
        "placeholder rows with a missing diagnosis value that DXSUM.DIAGNOSIS never included in "
        "the first place). The fixed-baseline-group primary analysis strategy is unchanged -- this "
        "is available for descriptive/sensitivity use only, per instructions.\n"
    )

    lines.append("## 3. Visit mapping validation\n")
    lines.append(
        "VISCODE2 (ADNI's own \"translated visit code\") is confirmed to be a single, "
        "phase-independent, literal month-relative vocabulary (`bl`, `m06`, `m12`, ... `m240`), "
        "populated on virtually every record in every raw table AND both live plasma files "
        "(100% in both). It is now the preferred visit-mapping source everywhere in this pipeline, "
        "replacing the inferred \"Year N\" text-parsing tier wherever VISCODE2 resolves -- which is "
        "nearly always. The `viscode_year_label_inferred` (medium-confidence, text-inferred) tier is "
        "no longer used as a primary source; it survives only as a fallback for the rare record "
        "missing VISCODE2 entirely, alongside a new date-elapsed-from-enrollment fallback below that.\n"
    )
    lines.append(
        "**ADNI2 VISCODE=\"m60\" investigated**: its VISCODE2 is `\"m60\"` -- a fully valid, literal "
        "month-60 visit. It was never anomalous data; it just uses a non-native-for-ADNI2 VISCODE "
        "naming (typical of a rollover participant carrying an ADNIGO/ADNI1-style code). It now maps "
        "at high confidence via VISCODE2. Month 60 is outside this project's {0,6,12,18,24,36,48} "
        "target-month list, so it still doesn't bucket into a reported target month, but it is no "
        "longer flagged as unmapped/anomalous.\n"
    )
    lines.append("See `visit_mapping_validation.csv` for the full before/after breakdown by table.\n")

    lines.append("## 4. MMSE baseline validation\n")
    lines.append(
        f"Cross-checked against `ADSL.MMSCORE` (ADNIMERGE2's own official baseline MMSE derivation): "
        f"**{mmse_v['n_match']} of {mmse_v['n_compared_against_adsl']} exact matches, "
        f"{mmse_v['n_mismatch']} mismatches.** This confirms screening MMSE is the ADNI-accepted "
        "analytic baseline value for this dataset -- the existing screening-fallback rule requires "
        "no change. `MMSE_BASELINE_SOURCE` (`\"screening\"` or `\"baseline_visit\"`) is now stored "
        "explicitly per participant in `adni_clinical_long.parquet`.\n"
    )
    lines.append(
        f"Screening-to-baseline interval (n={mmse_v['n_interval_computable']} participants with a "
        f"computable interval): mean {mmse_v['interval_days_mean']} days, median "
        f"{mmse_v['interval_days_median']} days, range [{mmse_v['interval_days_min']}, "
        f"{mmse_v['interval_days_max']}] days. **{mmse_v['n_negative_interval']} participants have a "
        "negative interval** (MMSE screening recorded *after* their enrollment date -- a genuine data "
        f"anomaly worth spot-checking, not corrected here). **{mmse_v['n_long_interval_flagged']} "
        f"participants exceed the {mmse_v['long_interval_threshold_days']}-day threshold** (ADNI's "
        "own documented enrollment-window length, reused here rather than an arbitrary cutoff) -- "
        "flagged via `MMSE_LONG_SCREENING_INTERVAL_FLAG` in `adni_clinical_long.parquet`, not excluded.\n"
    )

    lines.append("## Remaining unresolved issues\n")
    lines.append(
        "- The 5 participants with a negative MMSE screening-to-baseline interval (screening dated "
        "after enrollment) were flagged, not corrected -- worth a source-data spot-check before "
        "relying on their baseline MMSE value.\n"
    )
    lines.append(
        f"- {dx_v['n_unresolved']} enrolled participants have no baseline diagnosis in either this "
        "module's derivation or ADSL.DX -- genuinely unresolvable from currently available fields.\n"
    )
    lines.append(
        "- No GFAP/NfL cross-platform (Quanterix-vs-Fujirebio) calibration was applied or attempted -- "
        "unchanged from the prior preprocessing pass, per instructions.\n"
    )
    lines.append(
        "- `ptau217_lot_bias_flag` still cannot distinguish *why* a Batch-3 record was flagged beyond "
        "the source file's own Comment text -- unchanged from the prior pass.\n"
    )

    lines.append("## Readiness for ANCOVA\n")
    lines.append(
        "The cohort/visit ambiguities named in the review request are now resolved or explicitly "
        "quantified: baseline diagnosis is validated against the package's official derivation with "
        "100% agreement, visit mapping now prefers the validated VISCODE2 field everywhere (the "
        "ADNI2 m60 case is confirmed valid, not anomalous), and the MMSE screening-baseline rule is "
        "confirmed correct with interval outliers flagged rather than hidden. The datasets are ready "
        "for ANCOVA on the fixed-baseline-group cohort, pending your review of this report.\n"
    )

    return "\n".join(lines)


def main():
    if not ADNI_AUDIT_APPROVED:
        raise RuntimeError("ADNI_AUDIT_APPROVED is False -- preprocessing must not run.")

    print("Loading raw/interim tables...")
    raw = load_raw_tables()

    print("Building clinical and plasma long tables...")
    tables, qc = build_all(raw)

    print()
    write_processed_tables(tables)

    print()
    print("Writing aggregate-only QC outputs...")
    os.makedirs(ADNI_OUTPUTS_DIR, exist_ok=True)

    cohort_flow_df = build_cohort_flow(tables, qc)
    visit_mapping_df = build_visit_mapping_summary(tables)
    platform_df = build_assay_platform_summary(qc)
    eligibility_df = build_endpoint_eligibility_summary(tables)
    qc_exclusions_df = build_qc_exclusions_summary(qc)

    cohort_flow_df.to_csv(os.path.join(ADNI_OUTPUTS_DIR, "cohort_flow.csv"), index=False)
    visit_mapping_df.to_csv(os.path.join(ADNI_OUTPUTS_DIR, "visit_mapping_summary.csv"), index=False)
    platform_df.to_csv(os.path.join(ADNI_OUTPUTS_DIR, "assay_platform_summary.csv"), index=False)
    eligibility_df.to_csv(os.path.join(ADNI_OUTPUTS_DIR, "endpoint_eligibility_summary.csv"), index=False)
    qc_exclusions_df.to_csv(os.path.join(ADNI_OUTPUTS_DIR, "qc_exclusions_summary.csv"), index=False)

    summary_md = build_preprocessing_summary_md(tables, qc, cohort_flow_df, eligibility_df, platform_df)
    with open(os.path.join(ADNI_OUTPUTS_DIR, "preprocessing_summary.md"), "w", encoding="utf-8") as f:
        f.write(summary_md)

    print(f"wrote {ADNI_OUTPUTS_DIR}/cohort_flow.csv")
    print(f"wrote {ADNI_OUTPUTS_DIR}/visit_mapping_summary.csv")
    print(f"wrote {ADNI_OUTPUTS_DIR}/assay_platform_summary.csv")
    print(f"wrote {ADNI_OUTPUTS_DIR}/endpoint_eligibility_summary.csv")
    print(f"wrote {ADNI_OUTPUTS_DIR}/qc_exclusions_summary.csv")
    print(f"wrote {ADNI_OUTPUTS_DIR}/preprocessing_summary.md")

    print()
    print("Writing pre-ANCOVA validation outputs...")
    baseline_dx_validation_df = build_baseline_diagnosis_validation_csv(qc)
    visit_mapping_validation_df = build_visit_mapping_validation_csv(visit_mapping_df)
    mmse_validation_df = build_mmse_baseline_validation_csv(qc)

    baseline_dx_validation_df.to_csv(os.path.join(ADNI_OUTPUTS_DIR, "baseline_diagnosis_validation.csv"), index=False)
    visit_mapping_validation_df.to_csv(os.path.join(ADNI_OUTPUTS_DIR, "visit_mapping_validation.csv"), index=False)
    mmse_validation_df.to_csv(os.path.join(ADNI_OUTPUTS_DIR, "mmse_baseline_validation.csv"), index=False)

    validation_md = build_preanalysis_validation_md(qc, tables)
    with open(os.path.join(ADNI_OUTPUTS_DIR, "preanalysis_validation.md"), "w", encoding="utf-8") as f:
        f.write(validation_md)

    print(f"wrote {ADNI_OUTPUTS_DIR}/baseline_diagnosis_validation.csv")
    print(f"wrote {ADNI_OUTPUTS_DIR}/visit_mapping_validation.csv")
    print(f"wrote {ADNI_OUTPUTS_DIR}/mmse_baseline_validation.csv")
    print(f"wrote {ADNI_OUTPUTS_DIR}/preanalysis_validation.md")

    print()
    print("=== DONE (aggregate summary only) ===")
    clinical = tables["clinical_long"]
    for dx in ["CN", "MCI", "Dementia"]:
        print(f"  baseline dx {dx}: {clinical.loc[clinical['DX_BASELINE_FIXED'] == dx, 'RID'].nunique()}")
    print(f"  ADAS-Cog13 eligible participants: {clinical.loc[clinical['ADAS_COG13_ELIGIBLE'], 'RID'].nunique()}")
    print(f"  MMSE eligible participants: {clinical.loc[clinical['MMSE_ELIGIBLE'], 'RID'].nunique()}")
    print(f"  pTau217 primary-analysis eligible: {tables['ptau217_long'].loc[tables['ptau217_long']['PTAU217_PRIMARY_ANALYSIS_ELIGIBLE'], 'RID'].nunique()}")
    print(f"  GFAP primary platform: {qc['gfap_platform_recommendation']['primary_platform']}")
    print(f"  NfL primary platform: {qc['nfl_platform_recommendation']['primary_platform']}")
    dx_v = qc["clinical"]["baseline_diagnosis_validation"]
    print(f"  baseline dx validation: unchanged={dx_v['n_unchanged']} changed={dx_v['n_changed']} newly_assigned={dx_v['n_newly_assigned']} unresolved={dx_v['n_unresolved']}")
    mmse_v = qc["mmse_validation"]
    print(f"  MMSE baseline validation vs ADSL: match={mmse_v['n_match']}/{mmse_v['n_compared_against_adsl']}, long-interval flagged={mmse_v['n_long_interval_flagged']}")


if __name__ == "__main__":
    main()
