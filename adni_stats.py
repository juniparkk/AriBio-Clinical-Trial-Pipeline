# ============================================================
# ADNI_STATS -- pure statistical engine for the ANCOVA analysis stage.
#
# STATISTICAL ANALYSIS ONLY. This module fits models and reports
# aggregate estimates; it never reads/writes files, never touches
# raw/ or processed/ directly (run_adni_statistics.py does that), and
# never renders a chart or writes a dashboard file. Every function
# here takes a clean, already-assembled analysis-ready DataFrame (one
# row per participant at one month, covariate columns already
# selected/renamed) and returns aggregate statistics only -- no
# function in this module ever returns a participant identifier.
#
# This is an EXPLORATORY analysis: no multiplicity adjustment is
# applied anywhere (labeled explicitly in every pairwise result), and
# nothing here should be read as a treatment-effect estimate --
# baseline diagnosis group is an observational grouping variable, not
# a randomized arm.
# ============================================================

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats as scipy_stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import OLSInfluence

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

TARGET_MONTHS = [0, 6, 12, 18, 24, 36, 48]
INFERENTIAL_MONTHS = [6, 12, 18, 24, 36, 48]  # month 0 is descriptive-only, never inferential
DX_LEVELS = ["CN", "MCI", "Dementia"]
DX_REFERENCE = "CN"
MIN_GROUP_N = 10

STATUS_FITTED = "Fitted"
STATUS_SUPPRESSED = "Suppressed: small cell"
STATUS_BASELINE = "Not tested: baseline (change structurally zero)"


# ------------------------------------------------------------------
# Sample-size gate
# ------------------------------------------------------------------


def check_group_sizes(df, group_col="DX_BASELINE_FIXED", group_levels=DX_LEVELS, min_n=MIN_GROUP_N):
    """
    Returns (ok, group_ns, limiting_group, limiting_n). group_ns is a
    dict of every requested group_level's count in df (0 if a group is
    entirely absent -- absence is itself a small-cell condition, not a
    silently skipped group). ok is False if any group's n < min_n.
    """
    group_ns = {level: int((df[group_col] == level).sum()) for level in group_levels}
    limiting_group = min(group_ns, key=lambda g: group_ns[g])
    limiting_n = group_ns[limiting_group]
    ok = limiting_n >= min_n
    return ok, group_ns, limiting_group, limiting_n


# ------------------------------------------------------------------
# Raw (unadjusted) descriptive statistics
# ------------------------------------------------------------------


def raw_change_stats(df, value_col, group_col="DX_BASELINE_FIXED", group_levels=DX_LEVELS):
    """
    Per-group n / raw mean / raw SD of `value_col` (already the
    change-from-baseline or log-change column). Always computable,
    even when the group is too small for inference.
    """
    out = {}
    for level in group_levels:
        sub = df.loc[df[group_col] == level, value_col].dropna()
        out[level] = {
            "n": int(len(sub)),
            "mean": float(sub.mean()) if len(sub) else np.nan,
            "sd": float(sub.std(ddof=1)) if len(sub) > 1 else np.nan,
        }
    return out


def compute_descriptive_biomarker_stats(sample, raw_stats, month, group_col="DX_BASELINE_FIXED", group_levels=DX_LEVELS):
    """
    Purely DESCRIPTIVE (no model fit, no covariates) per-group biomarker
    statistics, computable for every cell -- including ones the
    small-cell rule suppresses from ANCOVA:
      - raw_geometric_mean: geometric mean of the untransformed
        biomarker level itself (baseline value at month 0; follow-up
        value at month > 0) -- exp(mean(log-level)), a one-sample
        summary, no adjustment.
      - raw_geometric_pct_change: (exp(mean(log_change)) - 1) * 100 --
        the unadjusted analogue of the ANCOVA-adjusted
        geometric_percent_change already reported elsewhere; reuses
        `raw_stats` (this module's own raw_change_stats() output for
        the same sample) rather than recomputing n/mean/sd, so this
        function never duplicates or risks drifting from an existing
        calculation.
      - raw_geometric_pct_change_ci_lower/upper: a plain one-sample
        Student-t interval on log_change (mean ± t_(0.975,n-1) * SE),
        back-transformed the same way -- a genuinely descriptive CI
        (no design matrix, no covariates, no model fit of any kind),
        only computed when n >= 2 (need a sample SD). At month 0,
        log_change is identically 0 for every participant by
        construction, so SD = 0 and the CI collapses to a degenerate
        point at 0% -- handled as a normal, not a special, case.
      - descriptive_status: "Computed" (n >= 1, so a mean exists) or
        "Insufficient data" (n == 0) -- distinct from and independent
        of `inferential_status`, since a cell can have a valid
        descriptive summary whether or not it also supports ANCOVA.
    """
    out = {}
    for level in group_levels:
        r = raw_stats[level]
        n = r["n"]
        if n == 0:
            out[level] = {
                "raw_geometric_mean": np.nan,
                "raw_geometric_pct_change": np.nan,
                "raw_geometric_pct_change_ci_lower": np.nan,
                "raw_geometric_pct_change_ci_upper": np.nan,
                "descriptive_status": "Insufficient data",
            }
            continue

        sub = sample[sample[group_col] == level]
        log_level = sub["log_baseline"] if month == 0 else (sub["log_baseline"] + sub["log_change"])
        log_level = log_level.dropna()
        raw_geo_mean = float(np.exp(log_level.mean())) if len(log_level) else np.nan

        mean_log_change = r["mean"]
        raw_pct = geometric_percent_change(mean_log_change) if pd.notna(mean_log_change) else np.nan

        ci_lo = ci_hi = np.nan
        if n >= 2 and pd.notna(r["sd"]):
            se = r["sd"] / np.sqrt(n)
            if se > 0:
                t_crit = scipy_stats.t.ppf(0.975, df=n - 1)
                ci_lo = geometric_percent_change(mean_log_change - t_crit * se)
                ci_hi = geometric_percent_change(mean_log_change + t_crit * se)
            else:
                ci_lo = ci_hi = raw_pct  # zero variance (e.g. month 0): degenerate point CI

        out[level] = {
            "raw_geometric_mean": raw_geo_mean,
            "raw_geometric_pct_change": raw_pct,
            "raw_geometric_pct_change_ci_lower": ci_lo,
            "raw_geometric_pct_change_ci_upper": ci_hi,
            "descriptive_status": "Computed",
        }
    return out


def descriptive_mean_ci(values):
    """
    Purely descriptive (no model fit, no covariates) one-sample n /
    mean / 95% CI (Student t, mean +/- t_(0.975,n-1) * SE) for a column
    of already-appropriately-scaled values.

    Scale-agnostic by design: callers pass log-transformed values for
    a geometric/multiplicative quantity (biomarker concentrations) and
    back-transform the returned mean/CI via exp() themselves, or pass
    linear-scale values directly (cognitive scores) and use the
    result as-is -- this function only ever computes a plain one-
    sample t-interval, never anything scale-specific.

    n < 2 (no sample SD available) returns a real mean but NaN CI
    bounds -- a degenerate CI is never fabricated. n == 0 returns NaN
    for everything.
    """
    values = pd.Series(values).dropna()
    n = int(len(values))
    if n == 0:
        return {"n": 0, "mean": np.nan, "ci_lower": np.nan, "ci_upper": np.nan}
    mean = float(values.mean())
    ci_lo = ci_hi = np.nan
    if n >= 2:
        sd = float(values.std(ddof=1))
        se = sd / np.sqrt(n)
        if se > 0:
            t_crit = scipy_stats.t.ppf(0.975, df=n - 1)
            ci_lo, ci_hi = mean - t_crit * se, mean + t_crit * se
        else:
            ci_lo = ci_hi = mean  # zero variance: degenerate point CI
    return {"n": n, "mean": mean, "ci_lower": ci_lo, "ci_upper": ci_hi}


def compute_absolute_cognitive_stats(sample, month, group_col="DX_BASELINE_FIXED", group_levels=DX_LEVELS):
    """
    Purely DESCRIPTIVE (no model fit) per-group ABSOLUTE cognitive
    score (not change-from-baseline) -- n, mean, 95% CI -- computable
    for every cell regardless of whether the change-from-baseline
    ANCOVA at that month was fitted or suppressed.

    Absolute score at month 0 is BASELINE_VALUE_FOR_MODEL itself;
    at month > 0 it is BASELINE_VALUE_FOR_MODEL + change_from_baseline
    (both already present on `sample` -- see build_cognitive_sample())
    -- reconstructed here rather than reading a separate raw score
    column, since the change-from-baseline sample is exactly the
    eligible-participant set this must describe, and reconstructing it
    guarantees the same denominator as every other statistic reported
    for this cell.
    """
    out = {}
    for level in group_levels:
        sub = sample[sample[group_col] == level]
        absolute = sub["BASELINE_VALUE_FOR_MODEL"] if month == 0 else (
            sub["BASELINE_VALUE_FOR_MODEL"] + sub["change_from_baseline"]
        )
        stats_ = descriptive_mean_ci(absolute)
        out[level] = {
            "raw_absolute_mean": stats_["mean"],
            "raw_absolute_ci_lower": stats_["ci_lower"],
            "raw_absolute_ci_upper": stats_["ci_upper"],
            "descriptive_status": "Computed" if stats_["n"] > 0 else "Insufficient data",
        }
    return out


def compute_absolute_biomarker_level_ci(sample, month, group_col="DX_BASELINE_FIXED", group_levels=DX_LEVELS):
    """
    A 95% CI on the ABSOLUTE biomarker LEVEL itself (raw_geometric_mean,
    already computed elsewhere by compute_descriptive_biomarker_stats)
    -- computed on the log scale (log_baseline at month 0;
    log_baseline + log_change at month > 0, same construction used
    there) and back-transformed via exp(), so it is the geometric
    (not arithmetic) analogue.

    Deliberately a SEPARATE, purely additive function rather than a
    change to compute_descriptive_biomarker_stats(): that function's
    existing output (raw_geometric_mean and its pct-change CI) is
    validated and must not be touched; this only adds a CI for the
    level itself, which that function never computed.
    """
    out = {}
    for level in group_levels:
        sub = sample[sample[group_col] == level]
        log_level = sub["log_baseline"] if month == 0 else (sub["log_baseline"] + sub["log_change"])
        stats_ = descriptive_mean_ci(log_level.dropna())
        out[level] = {
            "raw_geometric_mean_ci_lower": float(np.exp(stats_["ci_lower"])) if pd.notna(stats_["ci_lower"]) else np.nan,
            "raw_geometric_mean_ci_upper": float(np.exp(stats_["ci_upper"])) if pd.notna(stats_["ci_upper"]) else np.nan,
        }
    return out


def compute_cross_sectional_biomarker_stats(cross_sectional_sample, value_col, group_col="DX_BASELINE_FIXED", group_levels=DX_LEVELS):
    """
    Purely DESCRIPTIVE per-group biomarker LEVEL statistics (n,
    geometric mean, 95% CI) at a single timepoint, computed from every
    participant with a valid (>0) value AT THAT TIMEPOINT -- with NO
    requirement that the same participant also have a paired baseline
    measurement.

    This is the correct denominator for "what is the actual biomarker
    level at month X" (a cross-sectional descriptive question) and is
    deliberately a SEPARATE computation from
    compute_descriptive_biomarker_stats()'s raw_geometric_mean, which
    is restricted to the baseline-and-followup-PAIRED change-from-
    baseline analysis sample -- the correct, necessary denominator for
    THAT question (log_change cannot be computed without both values),
    but not for this one. Using the paired sample for both questions
    was found to understate real cross-sectional support by up to
    ~40-70% at later months across every plasma biomarker, and to drop
    two (biomarker, platform, month) cells to zero support entirely
    even though real cross-sectional data existed -- see
    run_adni_statistics.build_biomarker_cross_sectional_sample()'s
    docstring for the validated comparison.
    """
    out = {}
    for level in group_levels:
        sub = cross_sectional_sample[cross_sectional_sample[group_col] == level]
        log_vals = np.log(sub[value_col].dropna())
        log_vals = log_vals[np.isfinite(log_vals)]
        stats_ = descriptive_mean_ci(log_vals)
        out[level] = {
            "n_cross_sectional": stats_["n"],
            "raw_geometric_mean_cross_sectional": float(np.exp(stats_["mean"])) if pd.notna(stats_["mean"]) else np.nan,
            "raw_geometric_mean_ci_lower_cross_sectional": float(np.exp(stats_["ci_lower"])) if pd.notna(stats_["ci_lower"]) else np.nan,
            "raw_geometric_mean_ci_upper_cross_sectional": float(np.exp(stats_["ci_upper"])) if pd.notna(stats_["ci_upper"]) else np.nan,
        }
    return out


# ------------------------------------------------------------------
# ANCOVA model fitting
# ------------------------------------------------------------------


def param_names_of(fitted_model):
    """
    Ordered coefficient/parameter names for a fitted model, working for
    both a normal formula-fit results object (`.params` is a pandas
    Series with a named `.index`) AND a `get_robustcov_results()`
    (e.g. HC3) results object (`.params` is a bare numpy array with no
    `.index` -- the names instead live on `.model.exog_names`, in the
    same order). Shared by every function here and in adni_robustness.py
    that needs to locate a specific coefficient by name regardless of
    which kind of results object it was handed.
    """
    index = getattr(fitted_model.params, "index", None)
    if index is not None:
        return list(index)
    return list(fitted_model.model.exog_names)


def _build_formula(outcome_col, group_col, continuous_covariates, categorical_covariates, reference_level):
    group_term = f"C({group_col}, Treatment(reference={reference_level!r}))"
    terms = [group_term] + list(continuous_covariates) + [f"C({c})" for c in categorical_covariates]
    return f"{outcome_col} ~ {' + '.join(terms)}", group_term


def _group_exog_mean(fitted_model, df, group_col, group_level):
    """
    G-computation reference row for one group level: sets group_col to
    group_level for every row of df (keeping each participant's own
    covariates), builds the full model design matrix under that
    counterfactual, and averages every column. For a linear model with
    no group-by-covariate interaction terms (this module never fits
    one), this is mathematically identical to plugging the sample mean
    of every continuous covariate and the sample proportion of every
    categorical covariate level into the linear predictor -- i.e. "the
    same covariate distribution" for every group, exactly as
    requested, computed generally rather than hand-coded per covariate
    type.
    """
    design_info = fitted_model.model.data.design_info
    counterfactual = df.copy()
    counterfactual[group_col] = group_level
    exog = patsy.dmatrix(design_info, counterfactual, return_type="dataframe")
    return exog.mean(axis=0).values.reshape(1, -1)


def _unpack_ttest(tt, alpha):
    """
    statsmodels' t_test() result shapes its .effect/.sd/.conf_int()
    inconsistently depending on whether the contrast was a single row
    vector vs. a multi-row matrix (sometimes genuinely scalar,
    sometimes a length-1 array) -- np.atleast_1d/atleast_2d normalizes
    both cases so downstream code can always index position 0.
    """
    # .effect/.sd come back as either a 1-D length-1 array or a (1,1)
    # 2-D array depending on the contrast shape passed to t_test() --
    # np.ravel() flattens either case to a true 1-D array so indexing
    # [0] always yields a genuine scalar.
    ci = np.atleast_2d(tt.conf_int(alpha=alpha))
    return (
        float(np.ravel(tt.effect)[0]),
        float(np.ravel(tt.sd)[0]),
        float(ci[0][0]),
        float(ci[0][1]),
    )


def compute_adjusted_means(fitted_model, df, group_col, group_levels, alpha=0.05):
    """
    Adjusted (covariate-common) group means with SE and (1-alpha) CI.

    Uses `fitted_model.t_test(exog_row)` rather than `get_prediction()`:
    a formula-fit statsmodels model's `get_prediction(exog=...)` always
    re-runs `exog` through patsy against the stored design_info,
    expecting *raw* covariate data (original column names) rather than
    an already-built numeric design matrix -- so a pre-averaged design
    row (see _group_exog_mean()) cannot be passed there directly.
    `t_test()` accepts an arbitrary linear-combination vector directly
    and returns exactly the same quantities (effect = row @ params,
    sd = sqrt(row @ cov_params @ row.T), CI, by design the same
    machinery used for compute_pairwise_contrasts() below), so this is
    both correct and internally consistent.

    Returns {level: {"mean": ..., "se": ..., "ci_lower": ...,
    "ci_upper": ...}}.
    """
    out = {}
    for level in group_levels:
        exog_row = _group_exog_mean(fitted_model, df, group_col, level)
        tt = fitted_model.t_test(exog_row)
        mean, se, ci_lo, ci_hi = _unpack_ttest(tt, alpha)
        out[level] = {"mean": mean, "se": se, "ci_lower": ci_lo, "ci_upper": ci_hi}
    return out


def compute_pairwise_contrasts(fitted_model, group_term, group_levels, reference_level, alpha=0.05):
    """
    Pairwise contrasts among group_levels using t_test() on the fitted
    coefficients directly (exact for a linear model with no group-by-
    covariate interaction -- the difference of two G-computation-
    adjusted means reduces exactly to the difference of the group
    dummy coefficients, since every other term is identical between
    the two counterfactual populations and cancels). Always the
    reference level vs. each non-reference level directly from the
    fitted coefficients, and non-reference-vs-non-reference via an
    explicit coefficient-difference contrast string.

    Returns a list of dicts: comparison ("B - A"), estimate, se,
    ci_lower, ci_upper, p_value. No multiplicity adjustment (labeled
    by the caller, not here).
    """
    non_reference = [lv for lv in group_levels if lv != reference_level]
    param_names = param_names_of(fitted_model)

    def _term_for(level):
        matches = [p for p in param_names if p.startswith(group_term) and f"[T.{level}]" in p]
        if len(matches) != 1:
            raise ValueError(f"Could not uniquely identify coefficient for group level {level!r}: {matches}")
        return matches[0]

    results = []
    for level in non_reference:
        term = _term_for(level)
        tt = fitted_model.t_test(f"{term} = 0")
        est, se, ci_lo, ci_hi = _unpack_ttest(tt, alpha)
        results.append(
            {
                "comparison": f"{level} - {reference_level}",
                "estimate": est,
                "se": se,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
                "p_value": float(np.ravel(tt.pvalue)[0]),
            }
        )
    if len(non_reference) == 2:
        a, b = non_reference
        term_a, term_b = _term_for(a), _term_for(b)
        tt = fitted_model.t_test(f"{term_b} - {term_a} = 0")
        est, se, ci_lo, ci_hi = _unpack_ttest(tt, alpha)
        results.append(
            {
                "comparison": f"{b} - {a}",
                "estimate": est,
                "se": se,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
                "p_value": float(np.ravel(tt.pvalue)[0]),
            }
        )
    return results


def compute_group_anova(fitted_model, group_term):
    """
    Type-II ANOVA F-test / p-value for the group term, plus partial
    eta squared (SS_group / (SS_group + SS_residual)).
    """
    anova_table = sm.stats.anova_lm(fitted_model, typ=2)
    group_rows = [i for i in anova_table.index if i.startswith(group_term)]
    if len(group_rows) != 1:
        raise ValueError(f"Could not uniquely identify the group ANOVA row: {group_rows}")
    row = anova_table.loc[group_rows[0]]
    ss_group = float(row["sum_sq"])
    ss_resid = float(anova_table.loc["Residual", "sum_sq"])
    return {
        "F": float(row["F"]),
        "p": float(row["PR(>F)"]),
        "partial_eta_squared": ss_group / (ss_group + ss_resid) if (ss_group + ss_resid) > 0 else np.nan,
    }


def compute_model_diagnostics(fitted_model):
    """
    Aggregate model diagnostics only -- residual n, Shapiro-Wilk p,
    residual skewness/kurtosis, max Cook's distance, count of
    observations with Cook's D > 4/n, and a Breusch-Pagan
    heteroscedasticity p-value. Never triggers any automatic model
    change -- callers decide whether to flag a model for review.
    """
    resid = fitted_model.resid
    n = len(resid)
    try:
        shapiro_stat, shapiro_p = scipy_stats.shapiro(resid)
    except Exception:
        shapiro_p = np.nan

    influence = OLSInfluence(fitted_model)
    cooks_d = influence.cooks_distance[0]
    max_cooks = float(np.max(cooks_d)) if n else np.nan
    n_high_cooks = int(np.sum(cooks_d > (4.0 / n))) if n else 0

    try:
        bp_stat, bp_p, _, _ = het_breuschpagan(resid, fitted_model.model.exog)
    except Exception:
        bp_p = np.nan

    return {
        "residual_n": int(n),
        "shapiro_wilk_p": float(shapiro_p) if pd.notna(shapiro_p) else np.nan,
        "residual_skewness": float(scipy_stats.skew(resid)) if n else np.nan,
        "residual_kurtosis": float(scipy_stats.kurtosis(resid)) if n else np.nan,
        "max_cooks_distance": max_cooks,
        "n_high_cooks_distance": n_high_cooks,
        "breusch_pagan_p": float(bp_p) if pd.notna(bp_p) else np.nan,
    }


def fit_ancova(
    df,
    outcome_col,
    group_col="DX_BASELINE_FIXED",
    continuous_covariates=("BASELINE_VALUE_FOR_MODEL", "BASELINE_AGE"),
    categorical_covariates=("SEX",),
    group_levels=DX_LEVELS,
    reference_level=DX_REFERENCE,
    alpha=0.05,
):
    """
    Top-level ANCOVA fit: outcome_col ~ C(group_col) + continuous
    covariates + C(categorical covariates). Returns a dict with the
    fitted model, group ANOVA F/p/partial-eta-squared, adjusted means
    per group, pairwise contrasts, and model diagnostics -- or raises
    nothing; callers are responsible for calling check_group_sizes()
    first and skipping this entirely when suppressed.
    """
    formula, group_term = _build_formula(
        outcome_col, group_col, continuous_covariates, categorical_covariates, reference_level
    )
    model = smf.ols(formula, data=df).fit()

    anova = compute_group_anova(model, group_term)
    adjusted_means = compute_adjusted_means(model, df, group_col, group_levels, alpha=alpha)
    pairwise = compute_pairwise_contrasts(model, group_term, group_levels, reference_level, alpha=alpha)
    diagnostics = compute_model_diagnostics(model)

    return {
        "model": model,
        "formula": formula,
        "group_term": group_term,
        "anova": anova,
        "adjusted_means": adjusted_means,
        "pairwise": pairwise,
        "diagnostics": diagnostics,
        # Inputs the fit was built from -- purely additive metadata (not
        # used by fit_ancova() itself) so a downstream robustness pass can
        # reproduce an *identical* re-fit (e.g. HC3 re-covariance, or an
        # influence-sensitivity re-fit on a row subset) without needing to
        # re-derive the model specification from scratch or risk it
        # silently drifting from the primary formula.
        "_df": df,
        "_outcome_col": outcome_col,
        "_group_col": group_col,
        "_continuous_covariates": continuous_covariates,
        "_categorical_covariates": categorical_covariates,
        "_group_levels": group_levels,
        "_reference_level": reference_level,
    }


# ------------------------------------------------------------------
# Biomarker back-transform (natural log -> geometric % change)
# ------------------------------------------------------------------


def geometric_percent_change(log_change_value):
    """(exp(log_change) - 1) * 100 -- natural log throughout, per instructions."""
    return (np.exp(log_change_value) - 1.0) * 100.0
