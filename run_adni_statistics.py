# ============================================================
# RUN_ADNI_STATISTICS -- orchestration entry point for the ANCOVA
# statistical-analysis stage. STATISTICAL ANALYSIS ONLY:
#   - does not modify raw/ or the preprocessing rules in
#     adni_cohort.py/adni_plasma.py
#   - does not build biomarker_dashboard.html
#   - writes aggregate-only CSVs/markdown to ADNI_OUTPUTS_DIR; no
#     participant identifier is ever written there
#   - results are exploratory group comparisons, not treatment effects
#     (baseline diagnosis is an observational grouping variable)
#
# Reads (locked, read-only inputs for this stage):
#   ADNI_PROCESSED_DIR/adni_clinical_long.parquet
#   ADNI_PROCESSED_DIR/adni_ptau181_long.parquet
#   ADNI_PROCESSED_DIR/adni_ptau217_long.parquet
#   ADNI_PROCESSED_DIR/adni_abeta_ratio_long.parquet
#   ADNI_PROCESSED_DIR/adni_gfap_long.parquet
#   ADNI_PROCESSED_DIR/adni_nfl_long.parquet
#
# Writes:
#   ADNI_OUTPUTS_DIR/adni_cognitive_summary.csv
#   ADNI_OUTPUTS_DIR/adni_biomarker_summary.csv
#   ADNI_OUTPUTS_DIR/adni_pairwise_results.csv
#   ADNI_OUTPUTS_DIR/adni_model_diagnostics.csv
#   ADNI_OUTPUTS_DIR/adni_sensitivity_summary.csv
#   ADNI_OUTPUTS_DIR/adni_analysis_methods.md
#
# Usage: .venv/bin/python run_adni_statistics.py
# ============================================================

import os

import numpy as np
import pandas as pd

from adni_analysis import ADNI_AUDIT_APPROVED, ADNI_OUTPUTS_DIR, ADNI_PROCESSED_DIR
import adni_stats

MULTIPLICITY_LABEL = "None (primary exploratory analysis)"


# ------------------------------------------------------------------
# I/O
# ------------------------------------------------------------------


def load_processed_tables():
    def _read(name):
        return pd.read_parquet(os.path.join(ADNI_PROCESSED_DIR, f"adni_{name}.parquet"))

    return {
        "clinical_long": _read("clinical_long"),
        "ptau181_long": _read("ptau181_long"),
        "ptau217_long": _read("ptau217_long"),
        "abeta_ratio_long": _read("abeta_ratio_long"),
        "gfap_long": _read("gfap_long"),
        "nfl_long": _read("nfl_long"),
    }


# ------------------------------------------------------------------
# Analysis-sample construction
# ------------------------------------------------------------------


def mmse_sensitivity_exclusion_rids(clinical_long_df):
    """RIDs to drop for the MMSE sensitivity analysis: negative or unusually
    long MMSE screening-to-baseline interval (see preanalysis_validation.md)."""
    df = clinical_long_df.drop_duplicates("RID")
    cond = (df["MMSE_SCREENING_TO_BASELINE_INTERVAL_DAYS"] < 0) | (
        df["MMSE_LONG_SCREENING_INTERVAL_FLAG"].fillna(False)
    )
    return set(df.loc[cond, "RID"])


def build_cognitive_sample(clinical_long_df, endpoint, month, exclude_rids=None):
    """
    One row per eligible participant with a valid endpoint value at
    `month` (or, for month 0, one row per eligible participant with a
    valid baseline value -- change is structurally 0). Columns:
    RID, DX_BASELINE_FIXED, BASELINE_VALUE_FOR_MODEL, BASELINE_AGE,
    SEX, change_from_baseline.
    """
    value_col = endpoint
    baseline_col = f"{endpoint}_BASELINE"
    eligible_col = f"{endpoint}_ELIGIBLE"

    df = clinical_long_df[clinical_long_df[eligible_col].fillna(False)].copy()
    if exclude_rids:
        df = df[~df["RID"].isin(exclude_rids)]

    if month == 0:
        base = df.drop_duplicates("RID").dropna(subset=[baseline_col, "BASELINE_AGE", "SEX", "DX_BASELINE_FIXED"])
        base = base[["RID", "DX_BASELINE_FIXED", baseline_col, "BASELINE_AGE", "SEX"]].rename(
            columns={baseline_col: "BASELINE_VALUE_FOR_MODEL"}
        )
        base["change_from_baseline"] = 0.0
        return base.reset_index(drop=True)

    at_month = df[(df["VISIT_MONTH"] == month) & df[value_col].notna() & df[baseline_col].notna()].copy()
    at_month = at_month.drop_duplicates("RID")
    at_month["change_from_baseline"] = at_month[value_col] - at_month[baseline_col]
    at_month = at_month.rename(columns={baseline_col: "BASELINE_VALUE_FOR_MODEL"})
    cols = ["RID", "DX_BASELINE_FIXED", "BASELINE_VALUE_FOR_MODEL", "BASELINE_AGE", "SEX", "change_from_baseline"]
    return at_month[cols].dropna(subset=["DX_BASELINE_FIXED", "BASELINE_AGE", "SEX"]).reset_index(drop=True)


def build_biomarker_sample(
    plasma_df, dx_df, value_col, month, lot_bias_col=None, exclude_flagged=False, platform=None
):
    """
    One row per participant with a strictly positive baseline value and
    a strictly positive follow-up value at `month` (or, for month 0,
    one row per participant with a positive baseline value -- log-change
    is structurally 0). Columns: RID, DX_BASELINE_FIXED,
    BASELINE_VALUE_FOR_MODEL (raw baseline, kept for reference),
    log_baseline, BASELINE_AGE, SEX, log_change.

    `lot_bias_col` + `exclude_flagged=True` drops any individual record
    (baseline or follow-up) flagged for the ADNI4 Batch 3 QC-drift
    episode before baseline/follow-up are (re)computed -- this
    guarantees no flagged value contributes to a specific month's
    primary-analysis model, not just a coarser participant-level gate.
    """
    df = plasma_df.copy()
    if platform is not None:
        df = df[df["PLATFORM"] == platform]
    if lot_bias_col is not None and exclude_flagged:
        df = df[~df[lot_bias_col].fillna(False)]
    df = df[df[value_col] > 0]

    baseline = (
        df[df["VISIT_MONTH_RAW"] == 0][["RID", value_col]]
        .rename(columns={value_col: "BASELINE_VALUE_FOR_MODEL"})
        .drop_duplicates("RID")
    )

    if month == 0:
        sample = baseline.merge(dx_df, on="RID", how="inner")
        demog = df[["RID", "BASELINE_AGE", "SEX"]].drop_duplicates("RID")
        sample = sample.merge(demog, on="RID", how="left")
        sample = sample.dropna(subset=["DX_BASELINE_FIXED", "BASELINE_AGE", "SEX"])
        sample["log_baseline"] = np.log(sample["BASELINE_VALUE_FOR_MODEL"])
        sample["log_change"] = 0.0
        return sample.reset_index(drop=True)

    followup = df[(df["VISIT_MONTH"] == month) & (df["VISIT_MONTH_RAW"] != 0)][
        ["RID", value_col, "BASELINE_AGE", "SEX"]
    ].rename(columns={value_col: "FOLLOWUP_VALUE"})
    merged = followup.merge(baseline, on="RID", how="inner").merge(dx_df, on="RID", how="inner")
    merged = merged.dropna(subset=["DX_BASELINE_FIXED", "BASELINE_AGE", "SEX"])
    merged["log_baseline"] = np.log(merged["BASELINE_VALUE_FOR_MODEL"])
    merged["log_change"] = np.log(merged["FOLLOWUP_VALUE"]) - np.log(merged["BASELINE_VALUE_FOR_MODEL"])
    cols = ["RID", "DX_BASELINE_FIXED", "BASELINE_VALUE_FOR_MODEL", "log_baseline", "BASELINE_AGE", "SEX", "log_change"]
    return merged[cols].reset_index(drop=True)


def build_biomarker_cross_sectional_sample(
    plasma_df, dx_df, value_col, month, lot_bias_col=None, exclude_flagged=False, platform=None
):
    """
    One row per participant with a strictly positive value AT `month`
    (any month, including 0) and a known diagnosis group -- deliberately
    NOT requiring that participant to also have a paired baseline value
    the way build_biomarker_sample() does. Columns: RID,
    DX_BASELINE_FIXED, <value_col>.

    Validated need: comparing this cross-sectional count against
    build_biomarker_sample()'s paired-sample count (both restricted to
    known DX) showed the paired sample undercounts real cross-sectional
    support at every non-baseline month across all five biomarker
    families -- from a ~5% shortfall (pTau181, month 12) up to a ~70%
    shortfall (GFAP/NfL Quanterix, month 48), and two (biomarker,
    platform, month) cells (GFAP and NfL, Fujirebio, months 36 and 48)
    had real cross-sectional data (43 and 26 participants respectively)
    but ZERO paired-sample support, so previously received no row at
    all. This function exists specifically to give the Absolute-view
    "what is the actual biomarker level at this month" question its own
    correct denominator, separate from the change-from-baseline ANCOVA
    sample (build_biomarker_sample(), unchanged, still the correct
    denominator for that different question).
    """
    df = plasma_df.copy()
    if platform is not None:
        df = df[df["PLATFORM"] == platform]
    if lot_bias_col is not None and exclude_flagged:
        df = df[~df[lot_bias_col].fillna(False)]
    df = df[(df["VISIT_MONTH"] == month) & (df[value_col] > 0)]
    df = df.drop_duplicates("RID")
    sample = df[["RID", value_col]].merge(dx_df, on="RID", how="inner")
    return sample.dropna(subset=["DX_BASELINE_FIXED"]).reset_index(drop=True)


def available_months(df, month_col="VISIT_MONTH"):
    """Months from adni_stats.TARGET_MONTHS that actually have >=1 record
    in df -- never fabricates a row for a month with zero supported visits."""
    present = set(df[month_col].dropna().unique())
    return [m for m in adni_stats.TARGET_MONTHS if m in present]


# ------------------------------------------------------------------
# Shared fit-or-suppress core
# ------------------------------------------------------------------


def _fit_or_suppress(sample, outcome_col, baseline_col, month):
    ok, group_ns, limiting_group, limiting_n = adni_stats.check_group_sizes(sample)
    raw = adni_stats.raw_change_stats(sample, outcome_col)

    if month == 0:
        return {
            "status": adni_stats.STATUS_BASELINE,
            "raw": raw,
            "group_ns": group_ns,
            "limiting_group": limiting_group,
            "limiting_n": limiting_n,
            "fit": None,
        }
    if not ok:
        return {
            "status": adni_stats.STATUS_SUPPRESSED,
            "raw": raw,
            "group_ns": group_ns,
            "limiting_group": limiting_group,
            "limiting_n": limiting_n,
            "fit": None,
        }
    fit = adni_stats.fit_ancova(
        sample,
        outcome_col=outcome_col,
        continuous_covariates=(baseline_col, "BASELINE_AGE"),
        categorical_covariates=("SEX",),
    )
    return {
        "status": adni_stats.STATUS_FITTED,
        "raw": raw,
        "group_ns": group_ns,
        "limiting_group": limiting_group,
        "limiting_n": limiting_n,
        "fit": fit,
    }


# ------------------------------------------------------------------
# Cognitive endpoint runner
# ------------------------------------------------------------------


def run_cognitive_endpoint(clinical_long_df, endpoint, analysis_type="primary", exclude_rids=None):
    summary_rows, pairwise_rows, diagnostics_rows = [], [], []
    months = available_months(
        clinical_long_df[clinical_long_df[f"{endpoint}_ELIGIBLE"].fillna(False)]
    )
    for month in months:
        sample = build_cognitive_sample(clinical_long_df, endpoint, month, exclude_rids)
        if sample.empty:
            continue
        result = _fit_or_suppress(sample, "change_from_baseline", "BASELINE_VALUE_FOR_MODEL", month)
        # Purely descriptive, no-model-fit absolute-score stats --
        # computed for EVERY cell (including suppressed ones), never
        # influencing and never influenced by the inferential
        # (ANCOVA) columns below. See Medical Affairs redesign:
        # "Absolute" trajectory view needs the real score level, not
        # just change-from-baseline.
        absolute_stats = adni_stats.compute_absolute_cognitive_stats(sample, month)

        for level in adni_stats.DX_LEVELS:
            r = result["raw"][level]
            abs_ = absolute_stats[level]
            row = {
                "endpoint": endpoint,
                "analysis_type": analysis_type,
                "month": month,
                "group": level,
                "n": r["n"],
                "raw_mean_change": r["mean"],
                "raw_sd": r["sd"],
                "adjusted_mean_change": np.nan,
                "adjusted_se": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "overall_F": np.nan,
                "overall_p": np.nan,
                "partial_eta_squared": np.nan,
                "inferential_status": result["status"],
                "raw_absolute_mean": abs_["raw_absolute_mean"],
                "raw_absolute_ci_lower": abs_["raw_absolute_ci_lower"],
                "raw_absolute_ci_upper": abs_["raw_absolute_ci_upper"],
                "descriptive_status": abs_["descriptive_status"],
            }
            if result["status"] == adni_stats.STATUS_SUPPRESSED:
                row["inferential_status"] = f"{result['status']} (limiting group={result['limiting_group']}, n={result['limiting_n']})"
            if result["fit"] is not None:
                am = result["fit"]["adjusted_means"][level]
                row.update(
                    adjusted_mean_change=am["mean"],
                    adjusted_se=am["se"],
                    ci_lower=am["ci_lower"],
                    ci_upper=am["ci_upper"],
                    overall_F=result["fit"]["anova"]["F"],
                    overall_p=result["fit"]["anova"]["p"],
                    partial_eta_squared=result["fit"]["anova"]["partial_eta_squared"],
                )
            summary_rows.append(row)

        if result["fit"] is not None:
            for pw in result["fit"]["pairwise"]:
                pairwise_rows.append(
                    {
                        "endpoint": endpoint,
                        "assay_platform": "",
                        "analysis_type": analysis_type,
                        "month": month,
                        "comparison": pw["comparison"],
                        "adjusted_difference": pw["estimate"],
                        "se": pw["se"],
                        "ci_lower": pw["ci_lower"],
                        "ci_upper": pw["ci_upper"],
                        "p_value": pw["p_value"],
                        "multiplicity_adjustment": MULTIPLICITY_LABEL,
                    }
                )
            diagnostics_rows.append(
                {
                    "endpoint_or_biomarker": endpoint,
                    "assay_platform": "",
                    "analysis_type": analysis_type,
                    "month": month,
                    **result["fit"]["diagnostics"],
                }
            )
    return summary_rows, pairwise_rows, diagnostics_rows


# ------------------------------------------------------------------
# Biomarker family runner
# ------------------------------------------------------------------


def run_biomarker_family(
    plasma_df, dx_df, biomarker, value_col, assay_platform, analysis_type,
    lot_bias_col=None, exclude_flagged=False, platform=None,
):
    summary_rows, pairwise_rows, diagnostics_rows = [], [], []
    scoped = plasma_df if platform is None else plasma_df[plasma_df["PLATFORM"] == platform]
    # Union of two independent month sets, per the two different
    # analytical questions the Absolute vs. change-from-baseline views
    # answer: paired-sample months support the change-from-baseline
    # ANCOVA (build_biomarker_sample requires a matched baseline+
    # follow-up pair); cross-sectional months only require a valid
    # value AT that month, with no baseline pairing. Some months (see
    # build_biomarker_cross_sectional_sample()'s docstring) have real
    # cross-sectional data but zero paired-sample support -- these must
    # still get a row so the Absolute view isn't silently blank there.
    eligible_months = available_months(scoped[scoped["BIOMARKER_ELIGIBLE"].fillna(False)])
    cross_sectional_months = available_months(scoped[scoped[value_col] > 0])
    months = sorted(set(eligible_months) | set(cross_sectional_months))

    for month in months:
        sample = build_biomarker_sample(
            plasma_df, dx_df, value_col, month,
            lot_bias_col=lot_bias_col, exclude_flagged=exclude_flagged, platform=platform,
        )
        cross_sample = build_biomarker_cross_sectional_sample(
            plasma_df, dx_df, value_col, month,
            lot_bias_col=lot_bias_col, exclude_flagged=exclude_flagged, platform=platform,
        )
        if sample.empty and cross_sample.empty:
            continue
        # _fit_or_suppress tolerates an empty (but correctly-shaped)
        # sample gracefully -- every group's n is 0, the month is
        # reported STATUS_SUPPRESSED/STATUS_BASELINE with no fit, never
        # a crash. This lets a month with cross-sectional-only support
        # (no paired sample at all) still get a real row instead of
        # being skipped outright, with the change-from-baseline columns
        # honestly reflecting "no paired data" rather than being absent.
        result = _fit_or_suppress(sample, "log_change", "log_baseline", month)
        # Purely descriptive, no-model-fit stats -- computed for EVERY
        # cell (including suppressed ones) from the same raw sample
        # _fit_or_suppress already summarized via raw_change_stats().
        # Never influences, and is never influenced by, the inferential
        # (ANCOVA/HC3) columns below.
        descriptive = adni_stats.compute_descriptive_biomarker_stats(sample, result["raw"], month)
        # Purely additive: a CI on the absolute level itself
        # (raw_geometric_mean above), never touching the existing,
        # already-validated pct-change descriptive stats. See Medical
        # Affairs redesign: "Absolute" trajectory view needs a
        # displayable CI on the level, not just the point estimate.
        level_ci = adni_stats.compute_absolute_biomarker_level_ci(sample, month)
        # The CORRECT denominator for "actual biomarker level at this
        # month" -- see compute_cross_sectional_biomarker_stats()'s
        # docstring for why this must be separate from `descriptive`/
        # `level_ci` above (both scoped to the paired change sample).
        cross_sectional = adni_stats.compute_cross_sectional_biomarker_stats(cross_sample, value_col)

        for level in adni_stats.DX_LEVELS:
            r = result["raw"][level]
            adjusted_log = np.nan
            geo_pct = np.nan
            ci_lo_pct = np.nan
            ci_hi_pct = np.nan
            F = p = eta = np.nan
            status = result["status"]
            if status == adni_stats.STATUS_SUPPRESSED:
                status = f"{status} (limiting group={result['limiting_group']}, n={result['limiting_n']})"
            if result["fit"] is not None:
                am = result["fit"]["adjusted_means"][level]
                adjusted_log = am["mean"]
                geo_pct = adni_stats.geometric_percent_change(am["mean"])
                ci_lo_pct = adni_stats.geometric_percent_change(am["ci_lower"])
                ci_hi_pct = adni_stats.geometric_percent_change(am["ci_upper"])
                F = result["fit"]["anova"]["F"]
                p = result["fit"]["anova"]["p"]
                eta = result["fit"]["anova"]["partial_eta_squared"]
            desc = descriptive[level]
            lci = level_ci[level]
            cs = cross_sectional[level]
            summary_rows.append(
                {
                    "biomarker": biomarker,
                    "assay_platform": assay_platform,
                    "analysis_type": analysis_type,
                    "month": month,
                    "group": level,
                    "n": r["n"],
                    "adjusted_log_change": adjusted_log,
                    "geometric_percent_change": geo_pct,
                    "ci_lower_percent": ci_lo_pct,
                    "ci_upper_percent": ci_hi_pct,
                    "overall_F": F,
                    "overall_p": p,
                    "partial_eta_squared": eta,
                    "inferential_status": status,
                    "raw_geometric_mean": desc["raw_geometric_mean"],
                    "raw_geometric_pct_change": desc["raw_geometric_pct_change"],
                    "raw_geometric_pct_change_ci_lower": desc["raw_geometric_pct_change_ci_lower"],
                    "raw_geometric_pct_change_ci_upper": desc["raw_geometric_pct_change_ci_upper"],
                    "descriptive_status": desc["descriptive_status"],
                    "raw_geometric_mean_ci_lower": lci["raw_geometric_mean_ci_lower"],
                    "raw_geometric_mean_ci_upper": lci["raw_geometric_mean_ci_upper"],
                    "n_cross_sectional": cs["n_cross_sectional"],
                    "raw_geometric_mean_cross_sectional": cs["raw_geometric_mean_cross_sectional"],
                    "raw_geometric_mean_ci_lower_cross_sectional": cs["raw_geometric_mean_ci_lower_cross_sectional"],
                    "raw_geometric_mean_ci_upper_cross_sectional": cs["raw_geometric_mean_ci_upper_cross_sectional"],
                }
            )

        if result["fit"] is not None:
            for pw in result["fit"]["pairwise"]:
                pairwise_rows.append(
                    {
                        "endpoint": biomarker,
                        "assay_platform": assay_platform,
                        "analysis_type": analysis_type,
                        "month": month,
                        "comparison": pw["comparison"],
                        "adjusted_difference": pw["estimate"],
                        "se": pw["se"],
                        "ci_lower": pw["ci_lower"],
                        "ci_upper": pw["ci_upper"],
                        "p_value": pw["p_value"],
                        "multiplicity_adjustment": MULTIPLICITY_LABEL,
                    }
                )
            diagnostics_rows.append(
                {
                    "endpoint_or_biomarker": biomarker,
                    "assay_platform": assay_platform,
                    "analysis_type": analysis_type,
                    "month": month,
                    **result["fit"]["diagnostics"],
                }
            )
    return summary_rows, pairwise_rows, diagnostics_rows


# ------------------------------------------------------------------
# Sensitivity comparison
# ------------------------------------------------------------------


def build_sensitivity_summary(cognitive_summary_df, biomarker_summary_df):
    rows = []

    def _compare(analysis_name, primary_df, sensitivity_df, key_cols, estimate_col, p_col):
        merged = primary_df.merge(
            sensitivity_df, on=key_cols, suffixes=("_primary", "_sensitivity"), how="outer"
        )
        for _, r in merged.iterrows():
            n_p, n_s = r.get("n_primary"), r.get("n_sensitivity")
            est_p, est_s = r.get(f"{estimate_col}_primary"), r.get(f"{estimate_col}_sensitivity")
            p_p, p_s = r.get(f"{p_col}_primary"), r.get(f"{p_col}_sensitivity")
            sig_p = pd.notna(p_p) and p_p < 0.05
            sig_s = pd.notna(p_s) and p_s < 0.05
            sign_flip = (
                pd.notna(est_p) and pd.notna(est_s) and np.sign(est_p) != np.sign(est_s) and est_p != 0 and est_s != 0
            )
            material = bool(sign_flip or (sig_p != sig_s))
            rows.append(
                {
                    "analysis": analysis_name,
                    "month": r.get("month"),
                    "group": r.get("group"),
                    "n_primary": n_p,
                    "n_sensitivity": n_s,
                    "adjusted_estimate_primary": est_p,
                    "adjusted_estimate_sensitivity": est_s,
                    "delta": (est_s - est_p) if pd.notna(est_p) and pd.notna(est_s) else np.nan,
                    "p_primary": p_p,
                    "p_sensitivity": p_s,
                    "material_difference_flag": material,
                }
            )

    # MMSE: primary vs screening-interval sensitivity
    mmse_primary = cognitive_summary_df[
        (cognitive_summary_df["endpoint"] == "MMSE") & (cognitive_summary_df["analysis_type"] == "primary")
    ]
    mmse_sens = cognitive_summary_df[
        (cognitive_summary_df["endpoint"] == "MMSE") & (cognitive_summary_df["analysis_type"] == "sensitivity_interval_excl")
    ]
    _compare("MMSE_screening_interval", mmse_primary, mmse_sens, ["month", "group"], "adjusted_mean_change", "overall_p")

    # pTau217: primary (lot-bias excluded) vs sensitivity (lot-bias included)
    pt_primary = biomarker_summary_df[
        (biomarker_summary_df["biomarker"] == "pTau217") & (biomarker_summary_df["analysis_type"] == "primary")
    ]
    pt_sens = biomarker_summary_df[
        (biomarker_summary_df["biomarker"] == "pTau217") & (biomarker_summary_df["analysis_type"] == "sensitivity_incl_lot_bias")
    ]
    _compare("pTau217_lot_bias", pt_primary, pt_sens, ["month", "group"], "adjusted_log_change", "overall_p")

    # GFAP / NfL: Quanterix (primary) vs Fujirebio (sensitivity)
    for biomarker in ["GFAP", "NfL"]:
        prim = biomarker_summary_df[
            (biomarker_summary_df["biomarker"] == biomarker) & (biomarker_summary_df["analysis_type"] == "primary")
        ]
        sens = biomarker_summary_df[
            (biomarker_summary_df["biomarker"] == biomarker) & (biomarker_summary_df["analysis_type"] == "sensitivity_fujirebio")
        ]
        _compare(f"{biomarker}_platform", prim, sens, ["month", "group"], "adjusted_log_change", "overall_p")

    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Methods document
# ------------------------------------------------------------------


def build_methods_md(counts):
    lines = []
    lines.append("# ADNI Statistical Analysis Methods\n")
    lines.append(
        "**Statistical analysis only.** Raw files under `raw/` were not modified, preprocessing "
        "rules in `adni_cohort.py`/`adni_plasma.py` were not changed, and `biomarker_dashboard.html` "
        "was not built or touched. All outputs under `outputs/` are aggregate-only; no participant "
        "identifier appears in any output file.\n"
    )
    lines.append(
        "**This is an exploratory, observational group comparison, not a treatment-effect estimate.** "
        "Baseline diagnosis (CN/MCI/Dementia) is an observational grouping variable, not a randomized "
        "arm -- nothing here should be read as evidence of a causal treatment effect. No multiplicity "
        "adjustment is applied anywhere; every pairwise result is labeled "
        f"`{MULTIPLICITY_LABEL}`.\n"
    )

    lines.append("## Analysis population\n")
    lines.append(
        "Validated fixed baseline diagnosis (`DX_BASELINE_FIXED` in `adni_clinical_long.parquet`): "
        "CN, MCI, Dementia. All samples are observed cases only -- no imputation anywhere in this "
        "analysis stage.\n"
    )
    lines.append(
        f"Cognitive months: {adni_stats.TARGET_MONTHS} (month 0 descriptive only). Biomarker months: "
        "only months with an actual supported visit in the corresponding processed table for that "
        "biomarker/platform -- never forced into the 7-month grid.\n"
    )

    lines.append("## Cognitive endpoints (ADAS-Cog13, MMSE)\n")
    lines.append(
        "`change_from_baseline = followup_value - baseline_value` for both endpoints (no sign flip "
        "applied). Directionality is a labeling convention only: for ADAS-Cog13 (higher = more "
        "impaired), positive change = worsening; for MMSE (higher = better), positive change = "
        "improvement.\n"
    )
    lines.append(
        "Model (fit with `statsmodels` OLS via `smf.ols`), at each follow-up month "
        "(6/12/18/24/36/48), fit separately per endpoint:\n\n"
        "```\n"
        "change_from_baseline ~ C(DX_BASELINE_FIXED, Treatment(reference='CN'))\n"
        "                       + BASELINE_VALUE_FOR_MODEL + BASELINE_AGE + C(SEX)\n"
        "```\n"
    )
    lines.append(
        "Adjusted group means use a G-computation reference grid: for each group level g, every "
        "participant's `DX_BASELINE_FIXED` is counterfactually set to g while keeping their own "
        "`BASELINE_VALUE_FOR_MODEL`/`BASELINE_AGE`/`SEX`, the full model design matrix is built under "
        "that counterfactual, and averaged column-wise -- for this linear, no-interaction model this "
        "is mathematically identical to evaluating the model at the sample mean of every covariate, "
        "giving CN/MCI/Dementia a genuinely common covariate-adjustment basis. SE/CI come from "
        "`get_prediction()` on that averaged design row.\n"
    )
    lines.append(
        "Overall group significance: Type-II ANOVA F-test on the `DX_BASELINE_FIXED` term "
        "(`statsmodels.stats.anova_lm`, `typ=2`); partial eta squared = SS_group / (SS_group + "
        "SS_residual). Pairwise contrasts (MCI-CN, Dementia-CN, Dementia-MCI) via `t_test()` on the "
        "fitted coefficients directly -- exact for this no-interaction linear model.\n"
    )

    lines.append("## Plasma biomarkers\n")
    lines.append(
        "Five families analyzed **separately**, never pooled: pTau181 (Gothenburg), pTau217 "
        "(Fujirebio), Aβ42/Aβ40 (Fujirebio, pre-computed validated ratio field, not recalculated), "
        "GFAP (Quanterix primary / Fujirebio sensitivity), NfL (Quanterix primary / Fujirebio "
        "sensitivity). pTau181 and pTau217 are never combined into one series.\n"
    )
    lines.append(
        "Only strictly positive baseline and follow-up values are used. "
        "`log_change = ln(followup_value) - ln(baseline_value)` -- natural log throughout, never "
        "log2. Model, at each supported follow-up month:\n\n"
        "```\n"
        "log_change ~ C(DX_BASELINE_FIXED, Treatment(reference='CN'))\n"
        "             + log_baseline + BASELINE_AGE + C(SEX)\n"
        "```\n"
    )
    lines.append(
        "Adjusted group log-change means use the identical G-computation reference-grid approach "
        "described above for cognitive endpoints. Back-transform: "
        "`geometric_percent_change = (exp(adjusted_log_change) - 1) * 100`, applied identically to "
        "both CI limits (not re-derived on the percent scale).\n"
    )
    lines.append(
        "pTau217 primary analysis excludes any individual record flagged for the documented ADNI4 "
        "Batch 3 QC-drift episode (`PTAU217_LOT_BIAS_FLAG`, sourced from the raw plasma file's own "
        "`Comment` field) from serving as either a baseline or follow-up value for that month's model "
        "-- record-level exclusion, not just the coarser participant-level "
        "`PTAU217_PRIMARY_ANALYSIS_ELIGIBLE` gate. Sensitivity analysis includes those records with "
        "no correction applied -- no correction factor was invented.\n"
    )

    lines.append("## Minimum sample size\n")
    lines.append(
        f"Before fitting each month-specific model, n is computed per diagnosis group. If any group "
        f"has n < {adni_stats.MIN_GROUP_N}, descriptive statistics (n, raw mean, raw SD) are still "
        "reported, but no ANCOVA is fit -- `inferential_status` is set to `Suppressed: small cell`, "
        "with the limiting group and its n recorded in that same field. No significance stars are "
        "generated. Sparse timepoints are never silently dropped -- they appear with suppressed status.\n"
    )

    lines.append("## Model diagnostics\n")
    lines.append(
        "For every model that was actually fit (not suppressed, not the baseline month): residual n, "
        "Shapiro-Wilk p-value, residual skewness/kurtosis, maximum Cook's distance, count of "
        "observations with Cook's D > 4/n, and a Breusch-Pagan heteroscedasticity p-value. "
        "**Cognitive endpoints are never automatically transformed** because Shapiro-Wilk p < 0.05 -- "
        "questionable models are flagged for human review in `adni_model_diagnostics.csv`, the model "
        "specification is never silently changed. Biomarkers remain log-transformed by design "
        "regardless of diagnostic results.\n"
    )

    lines.append("## Sensitivity analyses\n")
    lines.append(
        "Three sensitivity comparisons, each reported against its primary analysis in "
        "`adni_sensitivity_summary.csv` (n, adjusted estimate, p-value, delta, and a "
        "`material_difference_flag` -- true if the estimate's sign flips between primary and "
        "sensitivity, or if the p<0.05 significance conclusion flips; a documented heuristic, not a "
        "formal equivalence test): MMSE (primary screening-based baseline vs. excluding participants "
        "with a negative or long screening-to-baseline interval), pTau217 (primary lot-bias-excluded "
        "vs. sensitivity lot-bias-included), and GFAP/NfL (primary Quanterix vs. sensitivity "
        "Fujirebio). Primary results are not replaced by sensitivity results in these outputs "
        "regardless of the comparison's outcome.\n"
    )

    lines.append("## Output files\n")
    lines.append(
        "`adni_cognitive_summary.csv`, `adni_biomarker_summary.csv`, `adni_pairwise_results.csv`, "
        "`adni_model_diagnostics.csv`, `adni_sensitivity_summary.csv` -- all aggregate-only, no "
        "participant identifiers. Biomarker pairwise contrasts in `adni_pairwise_results.csv` are "
        "reported on the natural-log-change scale (not back-transformed to percent), consistent with "
        "the model's own estimation scale.\n"
    )

    lines.append("## Run counts\n")
    lines.append("| Metric | Count |\n|---|---:|")
    for k, v in counts.items():
        lines.append(f"| {k} | {v} |")

    return "\n".join(lines)


# ------------------------------------------------------------------
# Orchestration
# ------------------------------------------------------------------


def main():
    if not ADNI_AUDIT_APPROVED:
        raise RuntimeError("ADNI_AUDIT_APPROVED is False -- statistical analysis must not run.")

    print("Loading locked processed datasets...")
    tables = load_processed_tables()
    clinical_long = tables["clinical_long"]
    dx_df = clinical_long.drop_duplicates("RID")[["RID", "DX_BASELINE_FIXED"]]

    all_summary_cognitive = []
    all_summary_biomarker = []
    all_pairwise = []
    all_diagnostics = []

    print("Fitting cognitive endpoint models (ADAS-Cog13, MMSE)...")
    for endpoint in ["ADAS_COG13", "MMSE"]:
        s, p, d = run_cognitive_endpoint(clinical_long, endpoint, analysis_type="primary")
        all_summary_cognitive += s
        all_pairwise += p
        all_diagnostics += d

    mmse_excl = mmse_sensitivity_exclusion_rids(clinical_long)
    s, p, d = run_cognitive_endpoint(
        clinical_long, "MMSE", analysis_type="sensitivity_interval_excl", exclude_rids=mmse_excl
    )
    all_summary_cognitive += s
    all_pairwise += p
    all_diagnostics += d

    print("Fitting plasma biomarker models...")

    # A. pTau181 -- primary only
    s, p, d = run_biomarker_family(
        tables["ptau181_long"], dx_df, "pTau181", "PLASMAPTAU181", "Gothenburg_Simoa", "primary"
    )
    all_summary_biomarker += s
    all_pairwise += p
    all_diagnostics += d

    # B. pTau217 -- primary (lot-bias records excluded record-level) + sensitivity (included)
    s, p, d = run_biomarker_family(
        tables["ptau217_long"], dx_df, "pTau217", "PTAU217", "Fujirebio_Lumipulse", "primary",
        lot_bias_col="PTAU217_LOT_BIAS_FLAG", exclude_flagged=True,
    )
    all_summary_biomarker += s
    all_pairwise += p
    all_diagnostics += d
    s, p, d = run_biomarker_family(
        tables["ptau217_long"], dx_df, "pTau217", "PTAU217", "Fujirebio_Lumipulse", "sensitivity_incl_lot_bias",
        lot_bias_col="PTAU217_LOT_BIAS_FLAG", exclude_flagged=False,
    )
    all_summary_biomarker += s
    all_pairwise += p
    all_diagnostics += d

    # Abeta42/40 ratio -- primary only, pre-computed validated field
    s, p, d = run_biomarker_family(
        tables["abeta_ratio_long"], dx_df, "Abeta42_40_ratio", "ABETA_RATIO", "Fujirebio_Lumipulse", "primary"
    )
    all_summary_biomarker += s
    all_pairwise += p
    all_diagnostics += d

    # C/D. GFAP / NfL -- Quanterix primary, Fujirebio sensitivity
    for biomarker, value_col, table_key in [("GFAP", "GFAP", "gfap_long"), ("NfL", "NfL", "nfl_long")]:
        s, p, d = run_biomarker_family(
            tables[table_key], dx_df, biomarker, value_col, "Quanterix", "primary", platform="Quanterix"
        )
        all_summary_biomarker += s
        all_pairwise += p
        all_diagnostics += d
        s, p, d = run_biomarker_family(
            tables[table_key], dx_df, biomarker, value_col, "Fujirebio", "sensitivity_fujirebio", platform="Fujirebio"
        )
        all_summary_biomarker += s
        all_pairwise += p
        all_diagnostics += d

    cognitive_summary_df = pd.DataFrame(all_summary_cognitive)
    biomarker_summary_df = pd.DataFrame(all_summary_biomarker)
    pairwise_df = pd.DataFrame(all_pairwise)
    diagnostics_df = pd.DataFrame(all_diagnostics)

    print("Building sensitivity comparison...")
    sensitivity_df = build_sensitivity_summary(cognitive_summary_df, biomarker_summary_df)

    os.makedirs(ADNI_OUTPUTS_DIR, exist_ok=True)
    cognitive_summary_df.to_csv(os.path.join(ADNI_OUTPUTS_DIR, "adni_cognitive_summary.csv"), index=False)
    biomarker_summary_df.to_csv(os.path.join(ADNI_OUTPUTS_DIR, "adni_biomarker_summary.csv"), index=False)
    pairwise_df.to_csv(os.path.join(ADNI_OUTPUTS_DIR, "adni_pairwise_results.csv"), index=False)
    diagnostics_df.to_csv(os.path.join(ADNI_OUTPUTS_DIR, "adni_model_diagnostics.csv"), index=False)
    sensitivity_df.to_csv(os.path.join(ADNI_OUTPUTS_DIR, "adni_sensitivity_summary.csv"), index=False)

    n_fitted = int((diagnostics_df.shape[0])) if not diagnostics_df.empty else 0
    n_suppressed_cognitive = int(
        cognitive_summary_df["inferential_status"].astype(str).str.startswith("Suppressed").sum()
    ) if not cognitive_summary_df.empty else 0
    n_suppressed_biomarker = int(
        biomarker_summary_df["inferential_status"].astype(str).str.startswith("Suppressed").sum()
    ) if not biomarker_summary_df.empty else 0

    counts = {
        "Cognitive summary rows": len(cognitive_summary_df),
        "Biomarker summary rows": len(biomarker_summary_df),
        "Pairwise comparison rows": len(pairwise_df),
        "Fitted models (with diagnostics)": n_fitted,
        "Cognitive group-rows suppressed for small n": n_suppressed_cognitive,
        "Biomarker group-rows suppressed for small n": n_suppressed_biomarker,
    }
    methods_md = build_methods_md(counts)
    with open(os.path.join(ADNI_OUTPUTS_DIR, "adni_analysis_methods.md"), "w", encoding="utf-8") as f:
        f.write(methods_md)

    print(f"wrote {ADNI_OUTPUTS_DIR}/adni_cognitive_summary.csv")
    print(f"wrote {ADNI_OUTPUTS_DIR}/adni_biomarker_summary.csv")
    print(f"wrote {ADNI_OUTPUTS_DIR}/adni_pairwise_results.csv")
    print(f"wrote {ADNI_OUTPUTS_DIR}/adni_model_diagnostics.csv")
    print(f"wrote {ADNI_OUTPUTS_DIR}/adni_sensitivity_summary.csv")
    print(f"wrote {ADNI_OUTPUTS_DIR}/adni_analysis_methods.md")

    print()
    print("=== DONE (aggregate summary only) ===")
    for k, v in counts.items():
        print(f"  {k}: {v}")

    if not diagnostics_df.empty:
        flagged = diagnostics_df[
            (diagnostics_df["shapiro_wilk_p"] < 0.05)
            | (diagnostics_df["n_high_cooks_distance"] > 0)
            | (diagnostics_df["breusch_pagan_p"] < 0.05)
        ]
        print(f"  Models flagged for diagnostic review: {len(flagged)} of {len(diagnostics_df)}")

    material = sensitivity_df["material_difference_flag"].sum() if not sensitivity_df.empty else 0
    print(f"  Sensitivity comparisons with a material difference flag: {int(material)} of {len(sensitivity_df)}")


if __name__ == "__main__":
    main()
