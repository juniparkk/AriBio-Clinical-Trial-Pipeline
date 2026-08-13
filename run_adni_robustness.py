# ============================================================
# RUN_ADNI_ROBUSTNESS -- final robustness pass, layered strictly on
# top of the already-approved primary statistical-analysis stage.
#
# Does NOT touch (imported and reused, never modified):
#   - preprocessing (adni_cohort.py / adni_plasma.py)
#   - primary model formulas / sample-building (adni_stats.py /
#     run_adni_statistics.py -- imported, not edited)
#   - eligibility rules, small-cell suppression, cohorts, endpoints,
#     transformations, raw/ files
#   - the already-written adni_cognitive_summary.csv /
#     adni_biomarker_summary.csv / adni_pairwise_results.csv /
#     adni_model_diagnostics.csv / adni_sensitivity_summary.csv /
#     adni_analysis_methods.md from the prior stage (not re-written
#     here at all)
#
# Every (endpoint/biomarker, assay_platform, analysis_type, month)
# combination is re-derived HERE using run_adni_statistics.py's own
# build_cognitive_sample()/build_biomarker_sample()/_fit_or_suppress()
# functions (imported, called, never edited) so the sample and the
# conventional fit are guaranteed identical to the approved primary
# analysis -- suppressed models are read as suppressed and never
# refit, per instructions.
#
# Writes aggregate-only:
#   ADNI_OUTPUTS_DIR/adni_robustness_summary.csv
#   ADNI_OUTPUTS_DIR/adni_dashboard_eligibility.csv
#   ADNI_OUTPUTS_DIR/adni_robustness_methods.md
#
# Usage: .venv/bin/python run_adni_robustness.py
# ============================================================

import os

import pandas as pd

from adni_analysis import ADNI_AUDIT_APPROVED, ADNI_OUTPUTS_DIR
import adni_robustness as RB
import adni_stats as S
import run_adni_statistics as R


def _process_one(label_cols, sample, month):
    """
    Runs the identical conventional fit-or-suppress step used by the
    approved primary analysis (R._fit_or_suppress, unmodified), then
    layers HC3 + influence-sensitivity checks on top ONLY when a model
    was actually fit. Returns (robustness_rows, eligibility_row).
    """
    endpoint_or_biomarker, assay_platform, analysis_type = label_cols
    robustness_rows = []

    if sample.empty:
        _, reason = RB.classify_eligibility(None, False, False, has_any_data=False)
        eligibility_row = {
            "endpoint_or_biomarker": endpoint_or_biomarker,
            "assay_platform": assay_platform,
            "analysis_type": analysis_type,
            "month": month,
            "classification": RB.CLASS_NOT_AVAILABLE,
            "reason": reason,
        }
        return robustness_rows, eligibility_row

    outcome_col = "change_from_baseline" if "change_from_baseline" in sample.columns else "log_change"
    baseline_col = "BASELINE_VALUE_FOR_MODEL" if outcome_col == "change_from_baseline" else "log_baseline"
    result = R._fit_or_suppress(sample, outcome_col, baseline_col, month)

    hc3_sensitive = False
    influence_sensitive = False
    n_excluded = 0

    if result["fit"] is not None:
        fit = result["fit"]
        hc3_model = RB.fit_hc3(fit["model"])
        hc3_rows = RB.compare_conventional_vs_hc3(
            fit, hc3_model, fit["group_term"], fit["_group_levels"], fit["_reference_level"]
        )
        for r in hc3_rows:
            r2 = dict(r)
            r2.update(
                endpoint_or_biomarker=endpoint_or_biomarker,
                assay_platform=assay_platform,
                analysis_type=analysis_type,
                month=month,
                robustness_check="HC3",
            )
            robustness_rows.append(r2)
        hc3_sensitive = any(r["sensitive_flag"] for r in hc3_rows)

        influence_rows, n_excluded = RB.run_influence_sensitivity(
            fit, fit["group_term"], fit["_group_levels"], fit["_reference_level"]
        )
        for r in influence_rows:
            r2 = dict(r)
            r2.update(
                endpoint_or_biomarker=endpoint_or_biomarker,
                assay_platform=assay_platform,
                analysis_type=analysis_type,
                month=month,
                robustness_check=f"Influence_exclusion (n_excluded={n_excluded})",
            )
            robustness_rows.append(r2)
        influence_sensitive = any(r["sensitive_flag"] for r in influence_rows)

    classification, reason = RB.classify_eligibility(
        result["status"], hc3_sensitive, influence_sensitive, has_any_data=True
    )
    eligibility_row = {
        "endpoint_or_biomarker": endpoint_or_biomarker,
        "assay_platform": assay_platform,
        "analysis_type": analysis_type,
        "month": month,
        "classification": classification,
        "reason": reason,
    }
    return robustness_rows, eligibility_row


def run_cognitive(clinical_long, endpoint, analysis_type, exclude_rids=None):
    robustness_rows, eligibility_rows = [], []
    eligible_df = clinical_long[clinical_long[f"{endpoint}_ELIGIBLE"].fillna(False)]
    months = R.available_months(eligible_df) if not eligible_df.empty else []
    for month in S.TARGET_MONTHS:
        if month not in months:
            _, elig = _process_one((endpoint, "", analysis_type), pd.DataFrame(), month)
            eligibility_rows.append(elig)
            continue
        sample = R.build_cognitive_sample(clinical_long, endpoint, month, exclude_rids)
        rob, elig = _process_one((endpoint, "", analysis_type), sample, month)
        robustness_rows += rob
        eligibility_rows.append(elig)
    return robustness_rows, eligibility_rows


def run_biomarker(plasma_df, dx_df, biomarker, value_col, assay_platform, analysis_type,
                   lot_bias_col=None, exclude_flagged=False, platform=None):
    robustness_rows, eligibility_rows = [], []
    scoped = plasma_df if platform is None else plasma_df[plasma_df["PLATFORM"] == platform]
    eligible_scoped = scoped[scoped["BIOMARKER_ELIGIBLE"].fillna(False)]
    months = R.available_months(eligible_scoped) if not eligible_scoped.empty else []
    for month in S.TARGET_MONTHS:
        if month not in months:
            _, elig = _process_one((biomarker, assay_platform, analysis_type), pd.DataFrame(), month)
            eligibility_rows.append(elig)
            continue
        sample = R.build_biomarker_sample(
            plasma_df, dx_df, value_col, month,
            lot_bias_col=lot_bias_col, exclude_flagged=exclude_flagged, platform=platform,
        )
        rob, elig = _process_one((biomarker, assay_platform, analysis_type), sample, month)
        robustness_rows += rob
        eligibility_rows.append(elig)
    return robustness_rows, eligibility_rows


def build_methods_md(counts, recommendation_text):
    lines = []
    lines.append("# ADNI Final Robustness Pass -- Methods\n")
    lines.append(
        "**Final robustness pass only**, layered on top of the already-approved preprocessing and "
        "primary statistical-analysis stages. Nothing about preprocessing, cohorts, endpoints, "
        "transformations, primary model formulas, eligibility rules, small-cell suppression, or raw "
        "data was changed. `biomarker_dashboard.html` was not built. All outputs are aggregate-only; "
        "no participant identifier appears anywhere.\n"
    )

    lines.append("## 1. HC3 robust inference\n")
    lines.append(
        "Every model the primary stage actually fit (never a suppressed one) was re-covariance'd with "
        "`statsmodels`' `get_robustcov_results(cov_type='HC3')` -- confirmed to preserve the identical "
        "coefficient point estimates and the same underlying model object, changing only the covariance "
        "matrix (and therefore every SE/CI/p-value derived from it). The overall group test's "
        "heteroscedasticity-robust equivalent is a robust Wald joint test (`f_test` under the HC3 "
        "covariance) that all group-dummy coefficients are simultaneously zero -- the standard robust "
        "generalization of the conventional ANOVA F-test used for the primary result, since the "
        "sum-of-squares-based F-test itself has no direct robust-covariance analogue.\n"
    )
    lines.append(
        "A result is flagged `sensitive_flag = True` when the conventional-vs-HC3 comparison crosses "
        "the p<0.05 boundary in either direction, or when CI-exclusion-of-zero changes -- never chosen "
        "to prefer whichever method reaches significance.\n"
    )

    lines.append("## 2. Influential-observation sensitivity\n")
    lines.append(
        "For every fitted model, observations with Cook's D > 4/n (the same threshold already reported "
        "in `adni_model_diagnostics.csv`) were identified and excluded, and the identical model "
        "specification was re-fit on the remaining rows -- sensitivity analysis only; the original "
        "(conventional, full-sample) fit remains primary and is never replaced. If excluding influential "
        "rows would itself collapse a diagnosis group below the unchanged small-cell threshold, the "
        "sensitivity check is reported as not assessable rather than forcing a comparison that would "
        "require weakening that rule.\n"
    )
    lines.append(
        "Flagged (`sensitive_flag = True`) when the sign of an estimate flips (direction change), or "
        "when a pairwise contrast's or the overall test's p<0.05 conclusion changes.\n"
    )

    lines.append("## 3. Dashboard eligibility classification\n")
    lines.append(
        "Every (endpoint/biomarker, assay platform, analysis type, month) combination -- across the "
        "full 7-month grid, not just the months that had a row in the primary summary CSVs -- is "
        "classified into exactly one of four categories, in `adni_dashboard_eligibility.csv`:\n\n"
        f"- **{RB.CLASS_ADJUSTED}** -- ANCOVA fitted in the primary stage and robust to both the HC3 "
        "and influence checks here.\n"
        f"- **{RB.CLASS_DESCRIPTIVE}** -- insufficient subgroup n for ANCOVA (small-cell suppressed), "
        "or the structurally-excluded baseline month.\n"
        f"- **{RB.CLASS_SENSITIVITY_CONCERN}** -- ANCOVA fitted, but HC3 covariance or influential-"
        "observation exclusion materially changes an inference conclusion.\n"
        f"- **{RB.CLASS_NOT_AVAILABLE}** -- no supported visit data exists for that combination at all.\n\n"
        "GFAP and NfL are classified using this identical rule, with no special-casing -- if the "
        "unchanged small-cell rule does not support inference for them (as the primary stage already "
        "found: 0 of their months were fitted on either platform), they land in "
        f"'{RB.CLASS_DESCRIPTIVE}' or '{RB.CLASS_NOT_AVAILABLE}' as a direct, unforced consequence, not "
        "a rule written specifically for them.\n"
    )

    lines.append("## Recommendation: conventional vs. HC3 as the primary displayed inference\n")
    lines.append(recommendation_text + "\n")

    lines.append("## Run counts\n")
    lines.append("| Metric | Count |\n|---|---:|")
    for k, v in counts.items():
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines)


def main():
    if not ADNI_AUDIT_APPROVED:
        raise RuntimeError("ADNI_AUDIT_APPROVED is False -- robustness pass must not run.")

    print("Loading locked processed datasets (read-only, unchanged)...")
    tables = R.load_processed_tables()
    clinical_long = tables["clinical_long"]
    dx_df = clinical_long.drop_duplicates("RID")[["RID", "DX_BASELINE_FIXED"]]

    all_robustness, all_eligibility = [], []

    print("Re-deriving cognitive models for HC3 + influence checks...")
    for endpoint in ["ADAS_COG13", "MMSE"]:
        rob, elig = run_cognitive(clinical_long, endpoint, "primary")
        all_robustness += rob
        all_eligibility += elig
    mmse_excl = R.mmse_sensitivity_exclusion_rids(clinical_long)
    rob, elig = run_cognitive(clinical_long, "MMSE", "sensitivity_interval_excl", exclude_rids=mmse_excl)
    all_robustness += rob
    all_eligibility += elig

    print("Re-deriving biomarker models for HC3 + influence checks...")
    rob, elig = run_biomarker(tables["ptau181_long"], dx_df, "pTau181", "PLASMAPTAU181", "Gothenburg_Simoa", "primary")
    all_robustness += rob
    all_eligibility += elig

    rob, elig = run_biomarker(
        tables["ptau217_long"], dx_df, "pTau217", "PTAU217", "Fujirebio_Lumipulse", "primary",
        lot_bias_col="PTAU217_LOT_BIAS_FLAG", exclude_flagged=True,
    )
    all_robustness += rob
    all_eligibility += elig
    rob, elig = run_biomarker(
        tables["ptau217_long"], dx_df, "pTau217", "PTAU217", "Fujirebio_Lumipulse", "sensitivity_incl_lot_bias",
        lot_bias_col="PTAU217_LOT_BIAS_FLAG", exclude_flagged=False,
    )
    all_robustness += rob
    all_eligibility += elig

    rob, elig = run_biomarker(tables["abeta_ratio_long"], dx_df, "Abeta42_40_ratio", "ABETA_RATIO", "Fujirebio_Lumipulse", "primary")
    all_robustness += rob
    all_eligibility += elig

    for biomarker, value_col, table_key in [("GFAP", "GFAP", "gfap_long"), ("NfL", "NfL", "nfl_long")]:
        rob, elig = run_biomarker(tables[table_key], dx_df, biomarker, value_col, "Quanterix", "primary", platform="Quanterix")
        all_robustness += rob
        all_eligibility += elig
        rob, elig = run_biomarker(tables[table_key], dx_df, biomarker, value_col, "Fujirebio", "sensitivity_fujirebio", platform="Fujirebio")
        all_robustness += rob
        all_eligibility += elig

    robustness_cols = [
        "endpoint_or_biomarker", "assay_platform", "analysis_type", "month", "robustness_check",
        "level", "group_or_comparison",
        "conventional_estimate", "conventional_se", "conventional_ci_lower", "conventional_ci_upper", "conventional_p",
        "alternative_estimate", "alternative_se", "alternative_ci_lower", "alternative_ci_upper", "alternative_p",
        "sensitive_flag",
    ]
    robustness_df = pd.DataFrame(all_robustness)[robustness_cols] if all_robustness else pd.DataFrame(columns=robustness_cols)
    eligibility_df = pd.DataFrame(all_eligibility)

    os.makedirs(ADNI_OUTPUTS_DIR, exist_ok=True)
    robustness_df.to_csv(os.path.join(ADNI_OUTPUTS_DIR, "adni_robustness_summary.csv"), index=False)
    eligibility_df.to_csv(os.path.join(ADNI_OUTPUTS_DIR, "adni_dashboard_eligibility.csv"), index=False)

    n_fitted_models = eligibility_df["classification"].isin([RB.CLASS_ADJUSTED, RB.CLASS_SENSITIVITY_CONCERN]).sum()
    hc3_rows = robustness_df[robustness_df["robustness_check"] == "HC3"]
    influence_rows = robustness_df[robustness_df["robustness_check"].str.startswith("Influence_exclusion", na=False)]
    n_hc3_sensitive_models = (
        hc3_rows.groupby(["endpoint_or_biomarker", "assay_platform", "analysis_type", "month"])["sensitive_flag"].any().sum()
    ) if not hc3_rows.empty else 0
    n_influence_sensitive_models = (
        influence_rows.groupby(["endpoint_or_biomarker", "assay_platform", "analysis_type", "month"])["sensitive_flag"].any().sum()
    ) if not influence_rows.empty else 0
    n_robust_to_hc3 = n_fitted_models - n_hc3_sensitive_models

    class_counts = eligibility_df["classification"].value_counts().to_dict()

    recommended_method = "HC3"
    if n_hc3_sensitive_models == 0:
        recommendation = (
            "No fitted model's inference conclusion changed under HC3 covariance in this run. Given that, "
            "and given the diagnostic-stage finding (adni_model_diagnostics.csv) that Breusch-Pagan "
            "heteroscedasticity was statistically significant for essentially every fitted model, HC3 is "
            "recommended as the primary DISPLAYED inference going forward: it costs nothing here (same "
            "conclusions) and is the more defensible choice given the documented heteroscedasticity, "
            "rather than waiting for a future dataset where the two methods might actually disagree."
        )
    else:
        recommendation = (
            f"{n_hc3_sensitive_models} fitted model(s) had at least one inference conclusion change under "
            "HC3 covariance. Combined with the diagnostic-stage finding of significant heteroscedasticity "
            "in most fitted models, HC3 is recommended as the primary displayed inference -- the "
            "conventional SS-based inference should not be treated as the more trustworthy default when "
            "its own assumption (homoscedasticity) is the one in question. Flagged combinations should be "
            "reviewed individually before display (see adni_robustness_summary.csv, sensitive_flag=True)."
        )

    counts = {
        "Fitted models re-examined": int(n_fitted_models),
        "Fitted models robust to HC3": int(n_robust_to_hc3),
        "Fitted models with an HC3-sensitive inference change": int(n_hc3_sensitive_models),
        "Fitted models with an influence-sensitive inference change": int(n_influence_sensitive_models),
    }
    for cls in [RB.CLASS_ADJUSTED, RB.CLASS_DESCRIPTIVE, RB.CLASS_SENSITIVITY_CONCERN, RB.CLASS_NOT_AVAILABLE]:
        counts[f"Endpoint/timepoints classified {cls}"] = int(class_counts.get(cls, 0))

    methods_md = build_methods_md(counts, recommendation)
    with open(os.path.join(ADNI_OUTPUTS_DIR, "adni_robustness_methods.md"), "w", encoding="utf-8") as f:
        f.write(methods_md)

    print(f"wrote {ADNI_OUTPUTS_DIR}/adni_robustness_summary.csv")
    print(f"wrote {ADNI_OUTPUTS_DIR}/adni_dashboard_eligibility.csv")
    print(f"wrote {ADNI_OUTPUTS_DIR}/adni_robustness_methods.md")

    print()
    print("=== DONE (aggregate summary only) ===")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    print(f"  Recommendation: {recommended_method} -- see adni_robustness_methods.md for full text")


if __name__ == "__main__":
    main()
