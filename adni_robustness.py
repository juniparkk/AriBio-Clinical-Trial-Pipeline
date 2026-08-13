# ============================================================
# ADNI_ROBUSTNESS -- pure functions for the final robustness pass:
# HC3 heteroscedasticity-robust inference and influential-observation
# sensitivity analysis, layered strictly on TOP of the already-
# approved primary models. Nothing here changes preprocessing,
# cohorts, endpoints, transformations, primary model formulas,
# eligibility rules, or small-cell suppression -- it only re-examines
# already-fitted models with an alternative covariance treatment or a
# sensitivity-only exclusion, never both at once, and never refits a
# suppressed model.
#
# Same discipline as adni_stats.py: pure functions, no file I/O, no
# participant identifier ever returned.
# ============================================================

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import OLSInfluence

import adni_stats as S

ALPHA = 0.05


# ------------------------------------------------------------------
# HC3 robust inference
# ------------------------------------------------------------------


def fit_hc3(conventional_model):
    """
    Re-covariance (NOT re-estimate) the identical fitted OLS model with
    HC3 heteroscedasticity-robust standard errors. `get_robustcov_results`
    returns a results object with byte-identical coefficient point
    estimates (verified: same .params) and the same underlying .model
    (so the existing adni_stats.compute_adjusted_means() /
    compute_pairwise_contrasts() machinery -- which only depends on
    .params, .t_test(), and .model.data.design_info -- works unmodified
    against it). Only the covariance matrix, and therefore every SE/CI/
    p-value derived from it, changes.
    """
    return conventional_model.get_robustcov_results(cov_type="HC3")


def compute_group_wald_test_hc3(hc3_model, group_term):
    """
    HC3-robust overall group significance. anova_lm's classical sum-of-
    squares F-test (used for the conventional overall_F/overall_p) has
    no direct heteroscedasticity-robust analogue, so the "methodologically
    available" robust equivalent used here is a robust Wald joint test
    (statsmodels f_test with the HC3 covariance already active on
    hc3_model) that all group-dummy coefficients are simultaneously
    zero -- the standard robust generalization of an ANOVA F-test for
    a categorical term.
    """
    group_params = [p for p in S.param_names_of(hc3_model) if p.startswith(group_term)]
    hypothesis = ", ".join(f"{p} = 0" for p in group_params)
    wald = hc3_model.f_test(hypothesis)
    return {"F": float(np.ravel(wald.fvalue)[0]), "p": float(np.ravel(wald.pvalue)[0])}


def _is_significant(p, ci_lower, ci_upper):
    """p < 0.05 and CI excludes 0 should always agree for a Wald test;
    both are checked so a genuine disagreement (e.g. numerical edge
    case) is visible rather than assumed away."""
    p_sig = pd.notna(p) and p < ALPHA
    ci_excludes_zero = pd.notna(ci_lower) and pd.notna(ci_upper) and not (ci_lower <= 0 <= ci_upper)
    return p_sig, ci_excludes_zero


def compare_conventional_vs_hc3(fit, hc3_model, group_term, group_levels, reference_level):
    """
    Row-level comparison of every conventional-vs-HC3 quantity for one
    fitted model: the overall group Wald/F test, each group's adjusted
    mean, and each pairwise contrast. Returns a list of dicts (one per
    comparison) with both estimates and a `sensitive_flag` set per the
    documented rule: p<0.05 crosses the 0.05 boundary in either
    direction, OR CI-exclusion-of-zero changes. The point estimate
    itself never changes between conventional and HC3 (verified: only
    SE/CI/p do) -- included in the same row for direct comparison,
    not because it was re-estimated.
    """
    rows = []

    conv_overall = fit["anova"]
    hc3_overall = compute_group_wald_test_hc3(hc3_model, group_term)
    conv_sig, _ = _is_significant(conv_overall["p"], None, None)
    hc3_sig, _ = _is_significant(hc3_overall["p"], None, None)
    rows.append(
        {
            "level": "overall_group_test",
            "group_or_comparison": "",
            "conventional_estimate": conv_overall["F"],
            "conventional_se": np.nan,
            "conventional_ci_lower": np.nan,
            "conventional_ci_upper": np.nan,
            "conventional_p": conv_overall["p"],
            "alternative_estimate": hc3_overall["F"],
            "alternative_se": np.nan,
            "alternative_ci_lower": np.nan,
            "alternative_ci_upper": np.nan,
            "alternative_p": hc3_overall["p"],
            "sensitive_flag": bool(conv_sig != hc3_sig),
        }
    )

    hc3_adjusted_means = S.compute_adjusted_means(hc3_model, fit["_df"], fit["_group_col"], group_levels, alpha=ALPHA)
    for level in group_levels:
        conv = fit["adjusted_means"][level]
        hc3v = hc3_adjusted_means[level]
        conv_sig_amt, conv_ci_excl = _is_significant(np.nan, conv["ci_lower"], conv["ci_upper"])
        hc3_sig_amt, hc3_ci_excl = _is_significant(np.nan, hc3v["ci_lower"], hc3v["ci_upper"])
        rows.append(
            {
                "level": "adjusted_mean",
                "group_or_comparison": level,
                "conventional_estimate": conv["mean"],
                "conventional_se": conv["se"],
                "conventional_ci_lower": conv["ci_lower"],
                "conventional_ci_upper": conv["ci_upper"],
                "conventional_p": np.nan,
                "alternative_estimate": hc3v["mean"],
                "alternative_se": hc3v["se"],
                "alternative_ci_lower": hc3v["ci_lower"],
                "alternative_ci_upper": hc3v["ci_upper"],
                "alternative_p": np.nan,
                "sensitive_flag": bool(conv_ci_excl != hc3_ci_excl),
            }
        )

    hc3_pairwise = S.compute_pairwise_contrasts(hc3_model, group_term, group_levels, reference_level, alpha=ALPHA)
    hc3_pairwise_by_comparison = {p["comparison"]: p for p in hc3_pairwise}
    for pw in fit["pairwise"]:
        hc3pw = hc3_pairwise_by_comparison[pw["comparison"]]
        conv_sig_pw, conv_ci_excl_pw = _is_significant(pw["p_value"], pw["ci_lower"], pw["ci_upper"])
        hc3_sig_pw, hc3_ci_excl_pw = _is_significant(hc3pw["p_value"], hc3pw["ci_lower"], hc3pw["ci_upper"])
        rows.append(
            {
                "level": "pairwise_contrast",
                "group_or_comparison": pw["comparison"],
                "conventional_estimate": pw["estimate"],
                "conventional_se": pw["se"],
                "conventional_ci_lower": pw["ci_lower"],
                "conventional_ci_upper": pw["ci_upper"],
                "conventional_p": pw["p_value"],
                "alternative_estimate": hc3pw["estimate"],
                "alternative_se": hc3pw["se"],
                "alternative_ci_lower": hc3pw["ci_lower"],
                "alternative_ci_upper": hc3pw["ci_upper"],
                "alternative_p": hc3pw["p_value"],
                "sensitive_flag": bool((conv_sig_pw != hc3_sig_pw) or (conv_ci_excl_pw != hc3_ci_excl_pw)),
            }
        )
    return rows


# ------------------------------------------------------------------
# Influential-observation sensitivity
# ------------------------------------------------------------------


def cooks_distance(fitted_model):
    influence = OLSInfluence(fitted_model)
    return influence.cooks_distance[0]


def run_influence_sensitivity(fit, group_term, group_levels, reference_level):
    """
    Sensitivity-only re-fit excluding observations with Cook's D > 4/n
    from the SAME model specification used for the primary fit. The
    original (conventional) fit is never replaced -- this function only
    reports a comparison. Returns (rows, n_excluded). rows is a list of
    dicts shaped like compare_conventional_vs_hc3()'s output (same
    columns, "alternative_*" now meaning "influence-excluded", not HC3)
    so both robustness checks can share one output schema.
    """
    df = fit["_df"]
    cooks_d = cooks_distance(fit["model"])
    n = len(df)
    threshold = 4.0 / n
    keep_mask = cooks_d <= threshold
    n_excluded = int((~keep_mask).sum())

    trimmed = df.loc[keep_mask].reset_index(drop=True)
    ok, _, _, _ = S.check_group_sizes(trimmed, group_col=fit["_group_col"], group_levels=group_levels)
    if n_excluded == 0 or not ok:
        # Nothing to exclude, or excluding influential points collapses a
        # group below the (unchanged) small-cell rule -- sensitivity
        # cannot be assessed without silently weakening that rule, so it
        # is reported as not assessable rather than forced.
        return [], n_excluded

    trimmed_fit = S.fit_ancova(
        trimmed,
        outcome_col=fit["_outcome_col"],
        group_col=fit["_group_col"],
        continuous_covariates=fit["_continuous_covariates"],
        categorical_covariates=fit["_categorical_covariates"],
        group_levels=group_levels,
        reference_level=reference_level,
        alpha=ALPHA,
    )

    rows = []
    conv_overall, trim_overall = fit["anova"], trimmed_fit["anova"]
    conv_sig, _ = _is_significant(conv_overall["p"], None, None)
    trim_sig, _ = _is_significant(trim_overall["p"], None, None)
    rows.append(
        {
            "level": "overall_group_test",
            "group_or_comparison": "",
            "conventional_estimate": conv_overall["F"],
            "conventional_se": np.nan,
            "conventional_ci_lower": np.nan,
            "conventional_ci_upper": np.nan,
            "conventional_p": conv_overall["p"],
            "alternative_estimate": trim_overall["F"],
            "alternative_se": np.nan,
            "alternative_ci_lower": np.nan,
            "alternative_ci_upper": np.nan,
            "alternative_p": trim_overall["p"],
            "sensitive_flag": bool(conv_sig != trim_sig),
        }
    )

    for level in group_levels:
        conv = fit["adjusted_means"][level]
        trimv = trimmed_fit["adjusted_means"][level]
        direction_changed = np.sign(conv["mean"]) != np.sign(trimv["mean"]) and conv["mean"] != 0 and trimv["mean"] != 0
        magnitude_pct_change = (
            abs(trimv["mean"] - conv["mean"]) / abs(conv["mean"]) * 100 if conv["mean"] != 0 else np.nan
        )
        _, conv_ci_excl = _is_significant(np.nan, conv["ci_lower"], conv["ci_upper"])
        _, trim_ci_excl = _is_significant(np.nan, trimv["ci_lower"], trimv["ci_upper"])
        rows.append(
            {
                "level": "adjusted_mean",
                "group_or_comparison": level,
                "conventional_estimate": conv["mean"],
                "conventional_se": conv["se"],
                "conventional_ci_lower": conv["ci_lower"],
                "conventional_ci_upper": conv["ci_upper"],
                "conventional_p": np.nan,
                "alternative_estimate": trimv["mean"],
                "alternative_se": trimv["se"],
                "alternative_ci_lower": trimv["ci_lower"],
                "alternative_ci_upper": trimv["ci_upper"],
                "alternative_p": np.nan,
                "sensitive_flag": bool(direction_changed or conv_ci_excl != trim_ci_excl),
            }
        )

    trimmed_pairwise_by_comparison = {p["comparison"]: p for p in trimmed_fit["pairwise"]}
    for pw in fit["pairwise"]:
        trimpw = trimmed_pairwise_by_comparison[pw["comparison"]]
        direction_changed = (
            np.sign(pw["estimate"]) != np.sign(trimpw["estimate"]) and pw["estimate"] != 0 and trimpw["estimate"] != 0
        )
        conv_sig_pw, _ = _is_significant(pw["p_value"], pw["ci_lower"], pw["ci_upper"])
        trim_sig_pw, _ = _is_significant(trimpw["p_value"], trimpw["ci_lower"], trimpw["ci_upper"])
        rows.append(
            {
                "level": "pairwise_contrast",
                "group_or_comparison": pw["comparison"],
                "conventional_estimate": pw["estimate"],
                "conventional_se": pw["se"],
                "conventional_ci_lower": pw["ci_lower"],
                "conventional_ci_upper": pw["ci_upper"],
                "conventional_p": pw["p_value"],
                "alternative_estimate": trimpw["estimate"],
                "alternative_se": trimpw["se"],
                "alternative_ci_lower": trimpw["ci_lower"],
                "alternative_ci_upper": trimpw["ci_upper"],
                "alternative_p": trimpw["p_value"],
                "sensitive_flag": bool(direction_changed or conv_sig_pw != trim_sig_pw),
            }
        )
    return rows, n_excluded


# ------------------------------------------------------------------
# Dashboard eligibility classification
# ------------------------------------------------------------------

CLASS_ADJUSTED = "A. Adjusted analysis"
CLASS_DESCRIPTIVE = "B. Descriptive only"
CLASS_SENSITIVITY_CONCERN = "C. Sensitivity concern"
CLASS_NOT_AVAILABLE = "D. Not available"


def classify_eligibility(inferential_status, hc3_sensitive, influence_sensitive, has_any_data):
    """
    A/B/C/D per the fixed rule -- never a 5th category, never invented.
    `has_any_data` False (no record at all for this endpoint/month
    combination) always wins as "Not available"; STATUS_BASELINE and
    STATUS_SUPPRESSED both read as "Descriptive only" (both are
    genuinely descriptive-only, for different structural reasons,
    documented separately in the methods note); STATUS_FITTED becomes
    "Sensitivity concern" only if the HC3 or influence check actually
    flagged it, never by default.
    """
    if not has_any_data:
        return CLASS_NOT_AVAILABLE, "No supported visit data for this endpoint/month combination."
    if inferential_status == S.STATUS_BASELINE:
        return CLASS_DESCRIPTIVE, "Baseline month -- change is structurally zero, never inferentially tested."
    if inferential_status == S.STATUS_SUPPRESSED or str(inferential_status).startswith("Suppressed"):
        return CLASS_DESCRIPTIVE, str(inferential_status)
    if inferential_status == S.STATUS_FITTED:
        if hc3_sensitive or influence_sensitive:
            reasons = []
            if hc3_sensitive:
                reasons.append("HC3 covariance changes an inference conclusion")
            if influence_sensitive:
                reasons.append("excluding influential observations changes an inference conclusion")
            return CLASS_SENSITIVITY_CONCERN, "; ".join(reasons)
        return CLASS_ADJUSTED, "ANCOVA fitted; robust to HC3 covariance and influential-observation exclusion."
    return CLASS_NOT_AVAILABLE, f"Unrecognized inferential_status: {inferential_status!r}"
