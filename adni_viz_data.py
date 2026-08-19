# ============================================================
# ADNI_VIZ_DATA -- governed data-loading and assembly layer for the
# biomarker dashboard. VISUALIZATION ONLY:
#
#   - loads ONLY files under ADNI_OUTPUTS_DIR (aggregate CSVs written
#     by the approved preprocessing/statistics/robustness stages)
#   - never opens raw/, interim/, processed/, or any .parquet file
#   - never accepts a CSV containing a participant-identifier column
#   - refits nothing, computes no new inferential statistic; the only
#     arithmetic here is geometric_percent_change(), a pure unit
#     re-expression ((exp(x)-1)*100) of an ALREADY-COMPUTED HC3 log-
#     scale estimate already present in adni_robustness_summary.csv --
#     the identical formula already used and approved in
#     adni_stats.geometric_percent_change(), redefined here (not
#     imported) so this module carries no dependency on the modeling
#     stack (statsmodels/scipy/patsy)
#
# No function in this module ever returns a participant identifier --
# every DataFrame loaded here is already aggregate-only by the time it
# reaches this process (enforced by load_aggregate_csv()'s column
# check, not merely assumed).
# ============================================================

import os

import numpy as np
import pandas as pd

GROUP_COLORS = {"CN": "#2196F3", "MCI": "#FF9800", "Dementia": "#F44336"}
GROUP_ORDER = ["CN", "MCI", "Dementia"]
TARGET_MONTHS = [0, 6, 12, 18, 24, 36, 48]

CLASS_ADJUSTED = "A. Adjusted analysis"
CLASS_DESCRIPTIVE = "B. Descriptive only"
CLASS_SENSITIVITY_CONCERN = "C. Sensitivity concern"
CLASS_NOT_AVAILABLE = "D. Not available"


class DataGovernanceError(Exception):
    """Raised when the visualization layer is asked to load something
    outside the aggregate outputs/ contract. Always fails loudly --
    never caught and silently ignored anywhere in this module."""


_FORBIDDEN_PATH_SEGMENTS = {"raw", "interim", "processed"}
_FORBIDDEN_COLUMNS = {"RID", "PTID", "USUBJID", "SUBJID", "PARTICIPANT_ID"}

REQUIRED_AGGREGATE_FILES = [
    "adni_dashboard_eligibility.csv",
    "adni_cognitive_summary.csv",
    "adni_biomarker_summary.csv",
    "adni_pairwise_results.csv",
    "adni_robustness_summary.csv",
    "adni_sensitivity_summary.csv",
    "adni_target_population_presets.csv",
    "adni_target_population_cohort_attrition.csv",
    "adni_target_population_profile.csv",
    "adni_target_population_cognitive_trajectories.csv",
    "adni_target_population_biomarker_trajectories.csv",
    "adni_target_population_trajectory_status.csv",
    "adni_target_population_pooled_trajectories.csv",
]


def load_aggregate_csv(outputs_dir, filename):
    """
    The ONLY sanctioned way this module (or adni_viz.py) reads a file.
    Refuses anything that resolves outside `outputs_dir`, anything
    under a raw/interim/processed path segment, any .parquet file, and
    any CSV whose columns include a known participant-identifier field
    -- each check raises DataGovernanceError rather than returning
    partial/sanitized data, so a governance failure is always visible,
    never silently patched over.
    """
    path = os.path.join(outputs_dir, filename)
    real_path = os.path.realpath(path)
    real_outputs_dir = os.path.realpath(outputs_dir)

    if os.path.commonpath([real_path, real_outputs_dir]) != real_outputs_dir:
        raise DataGovernanceError(
            f"Refusing to load {filename!r}: resolves outside the aggregate outputs/ "
            f"directory ({real_outputs_dir})."
        )
    path_segments = set(real_path.split(os.sep))
    hit = path_segments & _FORBIDDEN_PATH_SEGMENTS
    if hit:
        raise DataGovernanceError(
            f"Refusing to load {filename!r}: path contains forbidden segment(s) {sorted(hit)}."
        )
    if real_path.lower().endswith(".parquet"):
        raise DataGovernanceError(
            f"Refusing to load {filename!r}: parquet files are participant-level processed data."
        )
    if not os.path.exists(real_path):
        raise DataGovernanceError(f"Aggregate file not found: {filename!r} (expected under {outputs_dir}).")

    df = pd.read_csv(real_path)
    bad_cols = _FORBIDDEN_COLUMNS & {str(c).upper() for c in df.columns}
    if bad_cols:
        raise DataGovernanceError(
            f"Refusing to use {filename!r}: contains forbidden participant-identifier column(s) {sorted(bad_cols)}."
        )
    return df


def geometric_percent_change(log_change_value):
    return (np.exp(log_change_value) - 1.0) * 100.0


def load_all(outputs_dir):
    """Loads every aggregate file the dashboard needs, all through
    load_aggregate_csv() -- the single governed entry point."""
    return {
        "eligibility": load_aggregate_csv(outputs_dir, "adni_dashboard_eligibility.csv"),
        "cognitive": load_aggregate_csv(outputs_dir, "adni_cognitive_summary.csv"),
        "biomarker": load_aggregate_csv(outputs_dir, "adni_biomarker_summary.csv"),
        "pairwise": load_aggregate_csv(outputs_dir, "adni_pairwise_results.csv"),
        "robustness": load_aggregate_csv(outputs_dir, "adni_robustness_summary.csv"),
        "sensitivity": load_aggregate_csv(outputs_dir, "adni_sensitivity_summary.csv"),
    }


# ------------------------------------------------------------------
# POLARIS AD-Aligned trajectory data -- loaded through the identical
# governed load_aggregate_csv() entry point, then reshaped into the
# SAME "data" dict shape load_all() produces so every existing chart-
# data builder below (build_cognitive_chart_data, build_biomarker_
# chart_data, build_results_table_rows, ...) works completely
# unmodified against either population. No new statistic is computed
# anywhere in this reshape -- adni_polaris_trajectory_status.csv
# already carries both the eligibility classification AND the HC3 /
# influence robustness detail (conventional and alternative side by
# side) for every cell, produced entirely by
# run_adni_polaris_trajectories.py (statistical analysis stage, not
# this module); this is a pure column rename/filter, not a new fit.
# ------------------------------------------------------------------

POLARIS_MULTIPLICITY_LABEL = "None (primary exploratory analysis)"


def load_polaris_trajectory_data(outputs_dir):
    """The only sanctioned way the dashboard reads the POLARIS
    trajectory aggregate outputs -- all three go through
    load_aggregate_csv(), inheriting the same governance checks as
    every other file this module loads."""
    return {
        "cognitive": load_aggregate_csv(outputs_dir, "adni_polaris_cognitive_trajectories.csv"),
        "biomarker": load_aggregate_csv(outputs_dir, "adni_polaris_biomarker_trajectories.csv"),
        "status": load_aggregate_csv(outputs_dir, "adni_polaris_trajectory_status.csv"),
    }


def polaris_pairwise_view(status_df):
    """Reshapes adni_polaris_trajectory_status.csv's HC3 pairwise-
    contrast rows (which already carry BOTH the conventional and HC3
    estimate/SE/CI/p for every pairwise comparison, produced by the
    same compare_conventional_vs_hc3() the Overall-ADNI robustness
    stage uses) into the same column shape as
    adni_pairwise_results.csv, so build_results_table_rows() can read
    a POLARIS "pairwise" source without any code change. The
    conventional (non-HC3) columns are used here -- matching
    adni_pairwise_results.csv's own convention of storing the
    conventional fit's pairwise estimate -- not the HC3 columns, which
    remain reachable the normal way via the "robustness" source."""
    sub = status_df[(status_df["level"] == "pairwise_contrast") & (status_df["robustness_check"] == "HC3")].copy()
    out = sub.rename(columns={
        "endpoint_or_biomarker": "endpoint",
        "group_or_comparison": "comparison",
        "conventional_estimate": "adjusted_difference",
        "conventional_se": "se",
        "conventional_ci_lower": "ci_lower",
        "conventional_ci_upper": "ci_upper",
        "conventional_p": "p_value",
    })
    out["multiplicity_adjustment"] = POLARIS_MULTIPLICITY_LABEL
    cols = ["endpoint", "assay_platform", "analysis_type", "month", "comparison", "adjusted_difference", "se", "ci_lower", "ci_upper", "p_value", "multiplicity_adjustment"]
    return out[cols].reset_index(drop=True)


def polaris_data_view(outputs_dir):
    """Assembles the POLARIS-population equivalent of load_all()'s
    dict, sourced entirely from the three POLARIS trajectory aggregate
    files: "eligibility" and "robustness" both point at the same
    already-governed status_df (it carries every column either role
    needs; eligibility_lookup() and build_results_table_rows() each
    only read the columns relevant to them), "pairwise" is the
    reshape above, "cognitive"/"biomarker" are used as-is (identical
    schema to the Overall-ADNI summary files)."""
    raw = load_polaris_trajectory_data(outputs_dir)
    status_df = raw["status"]
    return {
        "eligibility": status_df,
        "cognitive": raw["cognitive"],
        "biomarker": raw["biomarker"],
        "robustness": status_df,
        "pairwise": polaris_pairwise_view(status_df),
    }


# ------------------------------------------------------------------
# Target Population presets (generalized cohort-definition tool) --
# loaded through the identical governed load_aggregate_csv() entry
# point as everything else in this module, then reshaped into the SAME
# dict shapes polaris_data_view()/build_polaris_funnel()/
# build_polaris_profile() already produce, so build_population_payload()
# and every existing chart-data builder above work completely
# unmodified against any preset. No new statistic is computed anywhere
# in this reshape -- adni_target_population_*.csv already carries
# every number (produced entirely by run_adni_target_populations.py,
# the statistical/preprocessing stage, not this module).
# ------------------------------------------------------------------


def load_target_population_data(outputs_dir):
    """The only sanctioned way the dashboard reads the target-population
    aggregate outputs -- all seven go through load_aggregate_csv()."""
    return {
        "presets": load_aggregate_csv(outputs_dir, "adni_target_population_presets.csv"),
        "attrition": load_aggregate_csv(outputs_dir, "adni_target_population_cohort_attrition.csv"),
        "profile": load_aggregate_csv(outputs_dir, "adni_target_population_profile.csv"),
        "cognitive": load_aggregate_csv(outputs_dir, "adni_target_population_cognitive_trajectories.csv"),
        "biomarker": load_aggregate_csv(outputs_dir, "adni_target_population_biomarker_trajectories.csv"),
        "status": load_aggregate_csv(outputs_dir, "adni_target_population_trajectory_status.csv"),
        "pooled": load_aggregate_csv(outputs_dir, "adni_target_population_pooled_trajectories.csv"),
    }


def build_preset_catalog(presets_df):
    """One entry per preset, in adni_eligibility.PRESET_LIBRARY order
    (the CSV is already written in that order) -- for the "Define
    Target Population" picker."""
    return [
        {
            "id": r["id"], "label": r["label"], "description": r["description"],
            "n": int(r["n"]), "isPolarisEquivalent": bool(r["is_polaris_equivalent"]),
        }
        for _, r in presets_df.iterrows()
    ]


def preset_data_view(target_data, preset_id):
    """Assembles one preset's equivalent of load_all()'s dict, filtered
    from the wide adni_target_population_*.csv tables loaded by
    load_target_population_data(). Identical shape to polaris_data_view()
    -- "eligibility" and "robustness" both point at the same status_df,
    "pairwise" is the same reshape polaris_pairwise_view() already
    performs (that function is not actually POLARIS-specific in its
    body; it operates on any status_df sharing the schema, which this
    one -- adni_target_population_trajectory_status.csv, minus its
    preset_id column -- is schema-locked to match, see
    test_adni_target_populations.py)."""
    status_df = target_data["status"][target_data["status"]["preset_id"] == preset_id].drop(columns=["preset_id"]).reset_index(drop=True)
    cognitive_df = target_data["cognitive"][target_data["cognitive"]["preset_id"] == preset_id].drop(columns=["preset_id"]).reset_index(drop=True)
    biomarker_df = target_data["biomarker"][target_data["biomarker"]["preset_id"] == preset_id].drop(columns=["preset_id"]).reset_index(drop=True)
    return {
        "eligibility": status_df,
        "cognitive": cognitive_df,
        "biomarker": biomarker_df,
        "robustness": status_df,
        "pairwise": polaris_pairwise_view(status_df),
    }


TARGET_POPULATION_NUMERIC_VARIABLES = [
    {"variable": "Baseline age (years)"},
    {"variable": "Baseline MMSE"},
    {"variable": "Baseline ADAS-Cog13"},
    {
        "variable": "Baseline Centiloid",
        "note": "Overall ADNI Centiloid reflects only the subset with an available near-baseline PET scan; the two denominators can differ from the Target Population's.",
    },
]

TARGET_POPULATION_CATEGORICAL_VARIABLES = [
    "Baseline diagnosis", "Sex", "APOE4 carrier",
    "pTau181 available", "pTau217 available", "Aβ42/Aβ40 ratio available", "GFAP available", "NfL available",
]


def build_target_population_funnel(attrition_df):
    """Same reshape as build_polaris_funnel() -- that function's body
    is already population-agnostic (a plain iteration over an
    already-governed attrition table's rows, computing only the ratio
    of consecutive remaining_n values), so this is a thin, distinctly-
    named wrapper for API clarity at the new preset-driven call sites,
    not a duplicate implementation."""
    return build_polaris_funnel(attrition_df)


def build_target_population_profile(profile_df, overall_label="Overall ADNI", target_label="Target Population"):
    """Generalized version of build_polaris_profile(): same reshape
    logic, parameterized population labels instead of the hardcoded
    POLARIS_OVERALL_LABEL/POLARIS_ALIGNED_LABEL, and an extended
    variable set that also covers the five biomarker-availability rows
    adni_eligibility.build_preset_profile() adds. Pure lookup/reshape --
    computes no new statistic, makes no inferential or causal claim."""
    rows = []
    for spec in TARGET_POPULATION_NUMERIC_VARIABLES:
        var = spec["variable"]
        sub = profile_df[profile_df["variable"] == var]
        overall = sub[sub["population"] == overall_label]
        target = sub[sub["population"] == target_label]
        if overall.empty and target.empty:
            continue
        rows.append({
            "variable": var, "kind": "numeric",
            "overall": _polaris_numeric_summary(overall),
            "polaris": _polaris_numeric_summary(target),
            "note": spec.get("note"),
        })

    for var in TARGET_POPULATION_CATEGORICAL_VARIABLES:
        sub = profile_df[profile_df["variable"] == var]
        if sub.empty:
            continue
        levels_seen = list(dict.fromkeys(sub["level"].dropna().tolist()))
        levels = []
        for level in levels_seen:
            level_sub = sub[sub["level"] == level]
            levels.append({
                "level": level,
                "overall": _polaris_categorical_summary(level_sub[level_sub["population"] == overall_label]),
                "polaris": _polaris_categorical_summary(level_sub[level_sub["population"] == target_label]),
            })
        rows.append({"variable": var, "kind": "categorical", "levels": levels})
    return rows


def build_pooled_trajectory_chart_data(pooled_df, preset_id, entity, assay_platform="", analysis_type="primary"):
    """Per-(month, population) point for the dual-population default
    chart view -- 2 lines (Overall ADNI vs Target Population), pooled
    across diagnosis group. Same point shape as build_cognitive_chart_
    data()/build_biomarker_chart_data() (month, group, classification,
    reason, n, estimate, ci_lower, ci_upper, is_descriptive_ci), with
    `group` meaning a POPULATION label here, not a diagnosis code --
    downstream JS trace-building treats both the same way. Always
    CLASS_DESCRIPTIVE or CLASS_NOT_AVAILABLE, per
    run_adni_target_populations.py's pooled computation (no ANCOVA
    "group" term applies once there is no diagnosis-group split)."""
    sub = pooled_df[
        (pooled_df["preset_id"] == preset_id) & (pooled_df["entity"] == entity)
        & (pooled_df["assay_platform"].apply(_platform_key) == _platform_key(assay_platform))
        & (pooled_df["analysis_type"] == analysis_type)
    ]
    points = []
    for month in TARGET_MONTHS:
        for population, group_label in (("overall", "Overall ADNI"), ("target", "Target Population")):
            row = sub[(sub["month"] == month) & (sub["population"] == population)]
            n = int(row.iloc[0]["n"]) if not row.empty and pd.notna(row.iloc[0]["n"]) else 0
            if row.empty or n == 0:
                points.append({
                    "month": month, "group": group_label, "n": n, "estimate": None,
                    "ci_lower": None, "ci_upper": None, "is_descriptive_ci": False,
                    "classification": CLASS_NOT_AVAILABLE, "reason": "No data at this timepoint.",
                })
                continue
            r = row.iloc[0]
            points.append({
                "month": month, "group": group_label, "n": n,
                "estimate": _safe_float(r.get("estimate")),
                "ci_lower": _safe_float(r.get("ci_lower")), "ci_upper": _safe_float(r.get("ci_upper")),
                "is_descriptive_ci": n >= 2,
                "classification": CLASS_DESCRIPTIVE,
                "reason": "Pooled (non-diagnosis-stratified) descriptive trend -- not compared statistically to the other population.",
            })
    return points


# ------------------------------------------------------------------
# Eligibility lookup
# ------------------------------------------------------------------


def _platform_key(v):
    return "" if pd.isna(v) else str(v)


def eligibility_lookup(eligibility_df):
    """(endpoint_or_biomarker, assay_platform, analysis_type, month) ->
    (classification, reason)."""
    out = {}
    for _, r in eligibility_df.iterrows():
        key = (r["endpoint_or_biomarker"], _platform_key(r["assay_platform"]), r["analysis_type"], int(r["month"]))
        out[key] = (r["classification"], r["reason"])
    return out


# ------------------------------------------------------------------
# Cognitive chart data
# ------------------------------------------------------------------


def build_cognitive_chart_data(data, endpoint, analysis_type="primary"):
    """
    Per (month, group) display point for one cognitive endpoint, using
    ONLY already-computed aggregate numbers:
      - classification A/B/C/D from adni_dashboard_eligibility.csv
      - descriptive estimate (B) from adni_cognitive_summary.csv's
        raw_mean_change (no CI -- descriptive only)
      - adjusted estimate (A/C) from adni_robustness_summary.csv's HC3
        rows (alternative_estimate/ci) -- HC3 is the only inference
        ever displayed for an adjusted point, per instructions
      - n from adni_cognitive_summary.csv
    """
    elig = eligibility_lookup(data["eligibility"])
    cog = data["cognitive"]
    cog_sub = cog[(cog["endpoint"] == endpoint) & (cog["analysis_type"] == analysis_type)]
    rob = data["robustness"]
    rob_sub = rob[
        (rob["endpoint_or_biomarker"] == endpoint)
        & (rob["analysis_type"] == analysis_type)
        & (rob["robustness_check"] == "HC3")
    ]

    points = []
    for month in TARGET_MONTHS:
        classification, reason = elig.get(
            (endpoint, "", analysis_type, month), (CLASS_NOT_AVAILABLE, "No eligibility record for this month.")
        )
        overall_p_hc3 = None
        if classification in (CLASS_ADJUSTED, CLASS_SENSITIVITY_CONCERN):
            overall_row = rob_sub[(rob_sub["month"] == month) & (rob_sub["level"] == "overall_group_test")]
            if not overall_row.empty:
                overall_p_hc3 = float(overall_row.iloc[0]["alternative_p"])

        for group in GROUP_ORDER:
            row_cog = cog_sub[(cog_sub["month"] == month) & (cog_sub["group"] == group)]
            n = int(row_cog.iloc[0]["n"]) if not row_cog.empty and pd.notna(row_cog.iloc[0]["n"]) else None
            point = {
                "month": month, "group": group, "classification": classification, "reason": reason, "n": n,
                "estimate": None, "ci_lower": None, "ci_upper": None, "is_descriptive_ci": False,
                "overall_p_hc3": overall_p_hc3, "is_hc3": False,
            }
            if classification == CLASS_NOT_AVAILABLE:
                points.append(point)
                continue
            if classification == CLASS_DESCRIPTIVE:
                if not row_cog.empty and pd.notna(row_cog.iloc[0]["raw_mean_change"]):
                    point["estimate"] = float(row_cog.iloc[0]["raw_mean_change"])
                points.append(point)
                continue
            hc3_row = rob_sub[
                (rob_sub["month"] == month) & (rob_sub["level"] == "adjusted_mean")
                & (rob_sub["group_or_comparison"] == group)
            ]
            if not hc3_row.empty:
                r = hc3_row.iloc[0]
                point.update(
                    estimate=float(r["alternative_estimate"]),
                    ci_lower=float(r["alternative_ci_lower"]),
                    ci_upper=float(r["alternative_ci_upper"]),
                    is_hc3=True,
                )
            points.append(point)
    return points


# ------------------------------------------------------------------
# Biomarker chart data
# ------------------------------------------------------------------


def build_biomarker_chart_data(data, biomarker, assay_platform, analysis_type):
    """
    Same shape as build_cognitive_chart_data(), on the geometric-
    percent-change scale for adjusted points.

    Descriptive-only (B) points now plot a real value: the statistical-
    analysis stage (run_adni_statistics.py, which reads participant-
    level processed/ data -- this module never does) persists a purely
    descriptive geometric mean percent change (`raw_geometric_pct_change`
    in adni_biomarker_summary.csv) for every biomarker/group/month cell,
    including suppressed ones, computed WITHOUT fitting any ANCOVA
    model. Its CI (`raw_geometric_pct_change_ci_lower/upper`) is a plain
    one-sample descriptive interval, present only when n >= 2
    (`descriptive_status == "Computed"` and a non-null CI) --
    `is_descriptive_ci` marks this explicitly so the display layer only
    ever draws a CI for a B-status point when it is genuinely
    descriptive, never implying an inferential one.
    """
    elig = eligibility_lookup(data["eligibility"])
    bio = data["biomarker"]
    bio_sub = bio[
        (bio["biomarker"] == biomarker) & (bio["assay_platform"] == assay_platform)
        & (bio["analysis_type"] == analysis_type)
    ]
    rob = data["robustness"]
    rob_sub = rob[
        (rob["endpoint_or_biomarker"] == biomarker) & (rob["assay_platform"] == assay_platform)
        & (rob["analysis_type"] == analysis_type) & (rob["robustness_check"] == "HC3")
    ]

    points = []
    for month in TARGET_MONTHS:
        classification, reason = elig.get(
            (biomarker, assay_platform, analysis_type, month),
            (CLASS_NOT_AVAILABLE, "No eligibility record for this month."),
        )
        overall_p_hc3 = None
        if classification in (CLASS_ADJUSTED, CLASS_SENSITIVITY_CONCERN):
            overall_row = rob_sub[(rob_sub["month"] == month) & (rob_sub["level"] == "overall_group_test")]
            if not overall_row.empty:
                overall_p_hc3 = float(overall_row.iloc[0]["alternative_p"])

        for group in GROUP_ORDER:
            row_bio = bio_sub[(bio_sub["month"] == month) & (bio_sub["group"] == group)]
            n = int(row_bio.iloc[0]["n"]) if not row_bio.empty and pd.notna(row_bio.iloc[0]["n"]) else None
            point = {
                "month": month, "group": group, "classification": classification, "reason": reason, "n": n,
                "estimate": None, "ci_lower": None, "ci_upper": None, "is_descriptive_ci": False,
                "overall_p_hc3": overall_p_hc3, "is_hc3": False,
            }
            if classification == CLASS_NOT_AVAILABLE:
                points.append(point)
                continue
            if classification == CLASS_DESCRIPTIVE:
                if not row_bio.empty:
                    r = row_bio.iloc[0]
                    if pd.notna(r.get("raw_geometric_pct_change")):
                        point["estimate"] = float(r["raw_geometric_pct_change"])
                    if pd.notna(r.get("raw_geometric_pct_change_ci_lower")) and pd.notna(r.get("raw_geometric_pct_change_ci_upper")):
                        point["ci_lower"] = float(r["raw_geometric_pct_change_ci_lower"])
                        point["ci_upper"] = float(r["raw_geometric_pct_change_ci_upper"])
                        point["is_descriptive_ci"] = True
                points.append(point)
                continue
            hc3_row = rob_sub[
                (rob_sub["month"] == month) & (rob_sub["level"] == "adjusted_mean")
                & (rob_sub["group_or_comparison"] == group)
            ]
            if not hc3_row.empty:
                r = hc3_row.iloc[0]
                point.update(
                    estimate=geometric_percent_change(float(r["alternative_estimate"])),
                    ci_lower=geometric_percent_change(float(r["alternative_ci_lower"])),
                    ci_upper=geometric_percent_change(float(r["alternative_ci_upper"])),
                    is_hc3=True,
                )
            points.append(point)
    return points


# ------------------------------------------------------------------
# Absolute-value chart data (Medical Affairs redesign) -- ALWAYS
# descriptive, never ANCOVA-adjusted: the fitted model's outcome was
# change-from-baseline, so there is no "adjusted absolute score" to
# show -- adjusted/HC3 values remain exclusively on the existing
# change-from-baseline charts (build_cognitive_chart_data /
# build_biomarker_chart_data, both unchanged above). These functions
# read the new raw_absolute_*/raw_geometric_mean_ci_* columns added by
# run_adni_statistics.py -- both purely descriptive, added specifically
# to support this view (see adni_stats.compute_absolute_cognitive_stats
# / compute_absolute_biomarker_level_ci) -- never compute anything
# themselves.
# ------------------------------------------------------------------


def build_cognitive_absolute_chart_data(data, endpoint, analysis_type="primary"):
    """Absolute ADAS-Cog13/MMSE score per (month, group) -- descriptive
    n/mean/CI from adni_cognitive_summary.csv's raw_absolute_* columns.
    A cell is CLASS_DESCRIPTIVE whenever n >= 1 (real data exists, just
    never model-adjusted); CLASS_NOT_AVAILABLE only when n == 0."""
    cog = data["cognitive"]
    cog_sub = cog[(cog["endpoint"] == endpoint) & (cog["analysis_type"] == analysis_type)]

    points = []
    for month in TARGET_MONTHS:
        for group in GROUP_ORDER:
            row = cog_sub[(cog_sub["month"] == month) & (cog_sub["group"] == group)]
            n = int(row.iloc[0]["n"]) if not row.empty and pd.notna(row.iloc[0]["n"]) else None
            point = {
                "month": month, "group": group, "n": n,
                "estimate": None, "ci_lower": None, "ci_upper": None,
                "classification": CLASS_NOT_AVAILABLE, "reason": "No data at this timepoint.",
            }
            if not n:
                points.append(point)
                continue
            r = row.iloc[0]
            point["classification"] = CLASS_DESCRIPTIVE
            point["reason"] = "Descriptive absolute score (no ANCOVA adjustment applies to an absolute value)."
            point["estimate"] = _safe_float(r.get("raw_absolute_mean"))
            point["ci_lower"] = _safe_float(r.get("raw_absolute_ci_lower"))
            point["ci_upper"] = _safe_float(r.get("raw_absolute_ci_upper"))
            points.append(point)
    return points


def build_biomarker_absolute_chart_data(data, biomarker, assay_platform, analysis_type):
    """Absolute biomarker concentration per (month, group) -- descriptive
    n/geometric-mean/CI from adni_biomarker_summary.csv's
    *_cross_sectional columns (n_cross_sectional, raw_geometric_mean_
    cross_sectional, raw_geometric_mean_ci_*_cross_sectional).

    Deliberately NOT the plain raw_geometric_mean/raw_geometric_mean_
    ci_* columns (used by the CHANGE-from-baseline descriptive view,
    still correct there): those are computed over the baseline-and-
    followup-PAIRED change-analysis sample, the correct denominator
    for "how much did this change" but not for "what is the level at
    this month" -- a cross-sectional, single-timepoint question that
    should count every participant with a valid value at that month,
    whether or not they also have a paired baseline draw. Using the
    paired columns here was found to understate real support by up to
    ~70% at later months and to hide two (biomarker, platform, month)
    cells with real data entirely -- see run_adni_statistics.
    build_biomarker_cross_sectional_sample()'s docstring for the
    validated comparison. Same always-descriptive treatment as
    build_cognitive_absolute_chart_data() -- never mixes assay/platform
    (scoped to exactly one assay_platform, matching the existing
    change-from-baseline chart's own scoping)."""
    bio = data["biomarker"]
    bio_sub = bio[
        (bio["biomarker"] == biomarker) & (bio["assay_platform"] == assay_platform)
        & (bio["analysis_type"] == analysis_type)
    ]

    points = []
    for month in TARGET_MONTHS:
        for group in GROUP_ORDER:
            row = bio_sub[(bio_sub["month"] == month) & (bio_sub["group"] == group)]
            n = int(row.iloc[0]["n_cross_sectional"]) if not row.empty and pd.notna(row.iloc[0].get("n_cross_sectional")) else None
            point = {
                "month": month, "group": group, "n": n,
                "estimate": None, "ci_lower": None, "ci_upper": None,
                "classification": CLASS_NOT_AVAILABLE, "reason": "No data at this timepoint.",
            }
            if not n:
                points.append(point)
                continue
            r = row.iloc[0]
            point["classification"] = CLASS_DESCRIPTIVE
            point["reason"] = "Descriptive absolute concentration (no ANCOVA adjustment applies to an absolute value)."
            point["estimate"] = _safe_float(r.get("raw_geometric_mean_cross_sectional"))
            point["ci_lower"] = _safe_float(r.get("raw_geometric_mean_ci_lower_cross_sectional"))
            point["ci_upper"] = _safe_float(r.get("raw_geometric_mean_ci_upper_cross_sectional"))
            points.append(point)
    return points


# ------------------------------------------------------------------
# Disease Continuum (Medical Affairs redesign) -- baseline (month=0)
# absolute value per group, across all 7 endpoints in one matrix.
# Cognitive endpoints use their primary analysis_type; each biomarker
# uses its own primary assay/platform (never a sensitivity variant,
# never mixing platforms) -- same primary selection BIOMARKER_SPECS
# already uses in adni_viz.py, duplicated here as plain data (not an
# import of adni_viz.py, to keep this module import-light and free of
# any HTML/CSS/JS concern).
# ------------------------------------------------------------------

DISEASE_CONTINUUM_ENDPOINTS = [
    {"key": "ADAS_COG13", "label": "ADAS-Cog13", "kind": "cognitive", "analysis_type": "primary"},
    {"key": "MMSE", "label": "MMSE", "kind": "cognitive", "analysis_type": "primary"},
    {"key": "pTau181", "label": "pTau181", "kind": "biomarker", "assay_platform": "Gothenburg_Simoa", "analysis_type": "primary"},
    {"key": "pTau217", "label": "pTau217", "kind": "biomarker", "assay_platform": "Fujirebio_Lumipulse", "analysis_type": "primary"},
    {"key": "Abeta42_40_ratio", "label": "Aβ42/Aβ40", "kind": "biomarker", "assay_platform": "Fujirebio_Lumipulse", "analysis_type": "primary"},
    {"key": "GFAP", "label": "GFAP", "kind": "biomarker", "assay_platform": "Quanterix", "analysis_type": "primary"},
    {"key": "NfL", "label": "NfL", "kind": "biomarker", "assay_platform": "Quanterix", "analysis_type": "primary"},
]


def build_disease_continuum_data(data):
    """One row per endpoint (in DISEASE_CONTINUUM_ENDPOINTS order),
    each with a CN/MCI/Dementia cell of {n, value, ci_lower, ci_upper}
    at baseline (month=0) -- the absolute-value data for the Disease
    Continuum heatmap. Purely a reshape of already-computed columns
    (raw_absolute_mean for cognition, raw_geometric_mean for
    biomarkers); computes nothing new."""
    cog = data["cognitive"]
    bio = data["biomarker"]

    rows = []
    for spec in DISEASE_CONTINUUM_ENDPOINTS:
        if spec["kind"] == "cognitive":
            sub = cog[(cog["endpoint"] == spec["key"]) & (cog["analysis_type"] == spec["analysis_type"]) & (cog["month"] == 0)]
            value_col, ci_lo_col, ci_hi_col = "raw_absolute_mean", "raw_absolute_ci_lower", "raw_absolute_ci_upper"
        else:
            sub = bio[
                (bio["biomarker"] == spec["key"]) & (bio["assay_platform"] == spec["assay_platform"])
                & (bio["analysis_type"] == spec["analysis_type"]) & (bio["month"] == 0)
            ]
            value_col, ci_lo_col, ci_hi_col = "raw_geometric_mean", "raw_geometric_mean_ci_lower", "raw_geometric_mean_ci_upper"

        cells = {}
        for group in GROUP_ORDER:
            row = sub[sub["group"] == group]
            n = int(row.iloc[0]["n"]) if not row.empty and pd.notna(row.iloc[0]["n"]) else 0
            if n == 0:
                cells[group] = {"n": 0, "value": None, "ci_lower": None, "ci_upper": None}
                continue
            r = row.iloc[0]
            cells[group] = {
                "n": n,
                "value": _safe_float(r.get(value_col)),
                "ci_lower": _safe_float(r.get(ci_lo_col)),
                "ci_upper": _safe_float(r.get(ci_hi_col)),
            }
        rows.append({"key": spec["key"], "label": spec["label"], "cells": cells})
    return rows


# ------------------------------------------------------------------
# Results-table rows (one row per group + one per pairwise comparison,
# per endpoint/biomarker/platform/analysis_type/month)
# ------------------------------------------------------------------

PAIRWISE_COMPARISONS = ["MCI - CN", "Dementia - CN", "Dementia - MCI"]


def _overall_conventional(summary_df, key_cols, key_vals, month):
    sub = summary_df
    for col, val in zip(key_cols, key_vals):
        sub = sub[sub[col] == val]
    sub = sub[sub["month"] == month]
    if sub.empty or pd.isna(sub.iloc[0].get("overall_F")):
        return None, None, None
    row = sub.iloc[0]
    return (
        float(row["overall_F"]) if pd.notna(row["overall_F"]) else None,
        float(row["overall_p"]) if pd.notna(row["overall_p"]) else None,
        float(row["partial_eta_squared"]) if pd.notna(row["partial_eta_squared"]) else None,
    )


def build_results_table_rows(data, entity, assay_platform, analysis_type, is_cognitive):
    """
    Rows for the collapsible "Statistical Results" table: one row per
    group adjusted-mean (when A/C) plus one row per pairwise comparison
    (when A/C), each carrying its conventional result, HC3 result, and
    -- for A/C cells -- the influence-sensitivity result and the reason
    the cell was (or wasn't) flagged, for the row-expansion detail.
    Descriptive-only (B) and Not-available (D) months are included too,
    with n only, so the table documents every timepoint, not just the
    fitted ones.
    """
    elig = eligibility_lookup(data["eligibility"])
    summary_df = data["cognitive"] if is_cognitive else data["biomarker"]
    entity_col = "endpoint" if is_cognitive else "biomarker"
    summary_sub = summary_df[summary_df[entity_col] == entity]
    if not is_cognitive:
        summary_sub = summary_sub[summary_sub["assay_platform"] == assay_platform]
    summary_sub = summary_sub[summary_sub["analysis_type"] == analysis_type]

    target_platform = assay_platform if not is_cognitive else ""

    pairwise_df = data["pairwise"]
    pairwise_sub = pairwise_df[
        (pairwise_df["endpoint"] == entity)
        & (pairwise_df["analysis_type"] == analysis_type)
        & (pairwise_df["assay_platform"].apply(_platform_key) == target_platform)
    ]

    rob = data["robustness"]
    rob_sub = rob[
        (rob["endpoint_or_biomarker"] == entity)
        & (rob["analysis_type"] == analysis_type)
        & (rob["assay_platform"].apply(_platform_key) == target_platform)
    ]

    rows = []
    for month in TARGET_MONTHS:
        key = (entity, assay_platform if not is_cognitive else "", analysis_type, month)
        classification, reason = elig.get(key, (CLASS_NOT_AVAILABLE, "No eligibility record for this month."))
        conv_F, conv_p, conv_eta = None, None, None
        if classification in (CLASS_ADJUSTED, CLASS_SENSITIVITY_CONCERN):
            key_cols = [entity_col, "analysis_type"] if is_cognitive else [entity_col, "assay_platform", "analysis_type"]
            key_vals = [entity, analysis_type] if is_cognitive else [entity, assay_platform, analysis_type]
            conv_F, conv_p, conv_eta = _overall_conventional(summary_sub, key_cols, key_vals, month)

        month_rob = rob_sub[rob_sub["month"] == month]

        for group in GROUP_ORDER:
            row_summary = summary_sub[(summary_sub["month"] == month) & (summary_sub["group"] == group)]
            n = int(row_summary.iloc[0]["n"]) if not row_summary.empty and pd.notna(row_summary.iloc[0]["n"]) else None
            hc3_row = month_rob[(month_rob["level"] == "adjusted_mean") & (month_rob["group_or_comparison"] == group) & (month_rob["robustness_check"] == "HC3")]
            infl_row = month_rob[(month_rob["level"] == "adjusted_mean") & (month_rob["group_or_comparison"] == group) & (month_rob["robustness_check"].str.startswith("Influence_exclusion", na=False))]
            rows.append(
                {
                    "row_type": "group", "month": month, "group_or_comparison": group, "n": n,
                    "classification": classification, "reason": reason,
                    "estimate": float(hc3_row.iloc[0]["alternative_estimate"]) if not hc3_row.empty else None,
                    "ci_lower": float(hc3_row.iloc[0]["alternative_ci_lower"]) if not hc3_row.empty else None,
                    "ci_upper": float(hc3_row.iloc[0]["alternative_ci_upper"]) if not hc3_row.empty else None,
                    "overall_F": conv_F, "hc3_p": None, "partial_eta_squared": conv_eta,
                    "conventional": _row_to_dict(hc3_row.iloc[0], prefix="conventional_") if not hc3_row.empty else None,
                    "hc3_detail": _row_to_dict(hc3_row.iloc[0], prefix="alternative_") if not hc3_row.empty else None,
                    "influence_detail": _row_to_dict(infl_row.iloc[0], prefix="alternative_") if not infl_row.empty else None,
                }
            )

        if classification in (CLASS_ADJUSTED, CLASS_SENSITIVITY_CONCERN):
            overall_row = month_rob[(month_rob["level"] == "overall_group_test") & (month_rob["robustness_check"] == "HC3")]
            hc3_overall_p = float(overall_row.iloc[0]["alternative_p"]) if not overall_row.empty else None
            for comparison in PAIRWISE_COMPARISONS:
                conv_pw = pairwise_sub[(pairwise_sub["month"] == month) & (pairwise_sub["comparison"] == comparison)]
                hc3_pw = month_rob[(month_rob["level"] == "pairwise_contrast") & (month_rob["group_or_comparison"] == comparison) & (month_rob["robustness_check"] == "HC3")]
                infl_pw = month_rob[(month_rob["level"] == "pairwise_contrast") & (month_rob["group_or_comparison"] == comparison) & (month_rob["robustness_check"].str.startswith("Influence_exclusion", na=False))]
                rows.append(
                    {
                        "row_type": "pairwise", "month": month, "group_or_comparison": comparison, "n": None,
                        "classification": classification, "reason": reason,
                        "estimate": float(hc3_pw.iloc[0]["alternative_estimate"]) if not hc3_pw.empty else None,
                        "ci_lower": float(hc3_pw.iloc[0]["alternative_ci_lower"]) if not hc3_pw.empty else None,
                        "ci_upper": float(hc3_pw.iloc[0]["alternative_ci_upper"]) if not hc3_pw.empty else None,
                        "overall_F": conv_F, "hc3_p": hc3_overall_p, "partial_eta_squared": conv_eta,
                        "multiplicity_adjustment": conv_pw.iloc[0]["multiplicity_adjustment"] if not conv_pw.empty else "None (primary exploratory analysis)",
                        "conventional": _pairwise_to_dict(conv_pw.iloc[0]) if not conv_pw.empty else None,
                        "hc3_detail": _row_to_dict(hc3_pw.iloc[0], prefix="alternative_") if not hc3_pw.empty else None,
                        "influence_detail": _row_to_dict(infl_pw.iloc[0], prefix="alternative_") if not infl_pw.empty else None,
                    }
                )
    return rows


def _row_to_dict(row, prefix):
    return {
        "estimate": _safe_float(row.get(f"{prefix}estimate")),
        "se": _safe_float(row.get(f"{prefix}se")),
        "ci_lower": _safe_float(row.get(f"{prefix}ci_lower")),
        "ci_upper": _safe_float(row.get(f"{prefix}ci_upper")),
        "p": _safe_float(row.get(f"{prefix}p")),
    }


def _pairwise_to_dict(row):
    return {
        "estimate": _safe_float(row.get("adjusted_difference")),
        "se": _safe_float(row.get("se")),
        "ci_lower": _safe_float(row.get("ci_lower")),
        "ci_upper": _safe_float(row.get("ci_upper")),
        "p": _safe_float(row.get("p_value")),
    }


# ------------------------------------------------------------------
# POLARIS AD-aligned cohort (dashboard integration) -- reshapes ONLY
# the two governed aggregate outputs already produced by
# run_adni_pet_eligibility.py (adni_polaris_cohort_attrition.csv,
# adni_polaris_population_profile.csv), loaded exclusively through
# load_aggregate_csv() like everything else in this module. This layer
# never opens processed/adni_pet_eligibility.parquet or any other
# participant-level file -- there is no code path here that could.
# ------------------------------------------------------------------

POLARIS_OVERALL_LABEL = "Overall ADNI"
POLARIS_ALIGNED_LABEL = "POLARIS-aligned ADNI"

POLARIS_NUMERIC_VARIABLES = [
    {"variable": "Baseline age (years)"},
    {"variable": "Baseline MMSE"},
    {"variable": "Baseline ADAS-Cog13"},
    {
        "variable": "Baseline Centiloid",
        "note": (
            "Overall ADNI Centiloid reflects only the subset with an available near-baseline "
            "PET scan; Demo Centiloid reflects the full eligible cohort by "
            "definition -- the two denominators differ."
        ),
    },
]

POLARIS_CATEGORICAL_VARIABLES = ["Baseline diagnosis", "Sex", "APOE4 carrier"]


def load_polaris_data(outputs_dir):
    """The only sanctioned way the dashboard reads the POLARIS-cohort
    aggregate outputs -- both go through load_aggregate_csv(), so the
    same forbidden-path/forbidden-column/parquet checks apply here as
    to every other file this module loads."""
    return {
        "attrition": load_aggregate_csv(outputs_dir, "adni_polaris_cohort_attrition.csv"),
        "profile": load_aggregate_csv(outputs_dir, "adni_polaris_population_profile.csv"),
    }


def build_polaris_funnel(attrition_df):
    """Reshapes the governed cohort-attrition CSV (already one row per
    named step, in step order) into a compact funnel list. The only
    arithmetic performed is percent_retained_of_previous, a plain ratio
    of two already-computed remaining_n values in the same table --
    the CSV's own percent_retained_of_cohort (retained from the
    starting cohort, not the previous step) is passed through
    unchanged alongside it."""
    steps = []
    prev_n = None
    for _, r in attrition_df.iterrows():
        remaining = int(r["remaining_n"])
        pct_of_previous = round(remaining / prev_n * 100, 1) if prev_n else None
        steps.append({
            "step": r["step"],
            "starting_n": int(r["starting_n"]),
            "remaining_n": remaining,
            "excluded_n": int(r["excluded_n"]),
            "percent_retained_of_cohort": _safe_float(r["percent_retained_of_cohort"]),
            "percent_retained_of_previous": pct_of_previous,
        })
        prev_n = remaining
    return steps


def _polaris_numeric_summary(sub):
    if sub.empty:
        return {"n": None, "mean": None, "sd": None, "median": None}
    r = sub.iloc[0]
    return {
        "n": int(r["n"]) if pd.notna(r["n"]) else None,
        "mean": _safe_float(r.get("mean")),
        "sd": _safe_float(r.get("sd")),
        "median": _safe_float(r.get("median")),
    }


def _polaris_categorical_summary(sub):
    if sub.empty:
        return {"n": None, "percent": None}
    r = sub.iloc[0]
    return {
        "n": int(r["n"]) if pd.notna(r["n"]) else None,
        "percent": _safe_float(r.get("percent")),
    }


def build_polaris_profile(profile_df):
    """Reshapes the governed population-profile CSV (already Overall-
    ADNI-vs-POLARIS-aligned aggregate rows -- n/mean/sd/median for
    numeric variables, n/percent per level for categorical ones) into
    per-variable comparison entries for the dashboard's compact profile
    cards. Pure lookup/reshape -- computes no new statistic and makes
    no inferential or causal claim; that judgment is left entirely to
    the display layer's neutral wording."""
    rows = []
    for spec in POLARIS_NUMERIC_VARIABLES:
        var = spec["variable"]
        sub = profile_df[profile_df["variable"] == var]
        overall = sub[sub["population"] == POLARIS_OVERALL_LABEL]
        polaris = sub[sub["population"] == POLARIS_ALIGNED_LABEL]
        if overall.empty and polaris.empty:
            continue
        rows.append({
            "variable": var, "kind": "numeric",
            "overall": _polaris_numeric_summary(overall),
            "polaris": _polaris_numeric_summary(polaris),
            "note": spec.get("note"),
        })

    for var in POLARIS_CATEGORICAL_VARIABLES:
        sub = profile_df[profile_df["variable"] == var]
        if sub.empty:
            continue
        levels_seen = list(dict.fromkeys(sub["level"].dropna().tolist()))
        levels = []
        for level in levels_seen:
            level_sub = sub[sub["level"] == level]
            levels.append({
                "level": level,
                "overall": _polaris_categorical_summary(level_sub[level_sub["population"] == POLARIS_OVERALL_LABEL]),
                "polaris": _polaris_categorical_summary(level_sub[level_sub["population"] == POLARIS_ALIGNED_LABEL]),
            })
        rows.append({"variable": var, "kind": "categorical", "levels": levels})
    return rows


def _safe_float(v):
    try:
        if v is None or pd.isna(v):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None
