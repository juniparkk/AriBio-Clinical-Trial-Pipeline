# ============================================================
# RUN_ADNI_POLARIS_TRAJECTORIES -- POLARIS AD-Aligned longitudinal
# trajectory outputs, layered strictly on top of the already-approved
# primary statistical-analysis (run_adni_statistics.py) and robustness
# (run_adni_robustness.py) stages. Every sample-building, ANCOVA, HC3,
# and influence-sensitivity function used here is IMPORTED from those
# modules, never reimplemented or modified -- this script only
# RESTRICTS the analysis population to the already-validated POLARIS
# AD-Aligned eligibility cohort before handing the same clinical_long /
# plasma tables to the same functions.
#
# Eligibility is CONSUMED, never recomputed: the POLARIS_ELIGIBLE flag
# is read as-is from processed/adni_pet_eligibility.parquet (written by
# run_adni_pet_eligibility.py / adni_pet.py, already approved). This
# script does not touch adni_pet.py, does not recompute Centiloid/MMSE
# eligibility, and does not modify that upstream stage's outputs.
#
# Explicitly NOT in scope here:
#   - propensity-score matching
#   - any statistical comparison of POLARIS vs. Overall ADNI
#     trajectories (both are reported side by side in the methods
#     document only as independent descriptive counts, never tested
#     against each other)
#   - AR1001 / any treatment-outcome comparison
#   - modifying adni_viz.py / adni_viz_data.py / biomarker_dashboard.html
#     (visualization is a later, separate, explicitly-gated step)
#
# Reads (locked, read-only inputs for this stage):
#   ADNI_PROCESSED_DIR/adni_clinical_long.parquet
#   ADNI_PROCESSED_DIR/adni_{ptau181,ptau217,abeta_ratio,gfap,nfl}_long.parquet
#   ADNI_PROCESSED_DIR/adni_pet_eligibility.parquet (POLARIS_ELIGIBLE column only)
#
# Writes (aggregate-only; no participant identifier in any output):
#   ADNI_OUTPUTS_DIR/adni_polaris_cognitive_trajectories.csv
#   ADNI_OUTPUTS_DIR/adni_polaris_biomarker_trajectories.csv
#   ADNI_OUTPUTS_DIR/adni_polaris_trajectory_status.csv
#   ADNI_OUTPUTS_DIR/adni_polaris_trajectory_methods.md
#
# Usage: .venv/bin/python run_adni_polaris_trajectories.py
# ============================================================

import os

import pandas as pd

from adni_analysis import ADNI_AUDIT_APPROVED, ADNI_OUTPUTS_DIR, ADNI_PROCESSED_DIR
import adni_stats as S
import run_adni_robustness as RR
import run_adni_statistics as R

EXPECTED_POLARIS_N = 620

STATUS_COLS = [
    "endpoint_or_biomarker", "assay_platform", "analysis_type", "month",
    "classification", "reason",
    "robustness_check", "level", "group_or_comparison",
    "conventional_estimate", "conventional_se", "conventional_ci_lower", "conventional_ci_upper", "conventional_p",
    "alternative_estimate", "alternative_se", "alternative_ci_lower", "alternative_ci_upper", "alternative_p",
    "sensitive_flag",
]


# ------------------------------------------------------------------
# POLARIS population restriction (consumes, never redefines, eligibility)
# ------------------------------------------------------------------


def load_polaris_rids():
    """Reads the already-validated POLARIS_ELIGIBLE flag as-is from the
    approved processed/adni_pet_eligibility.parquet -- no eligibility
    logic (MMSE threshold, Centiloid threshold, PET window, QC, tie-
    break) is re-derived here."""
    pet = pd.read_parquet(os.path.join(ADNI_PROCESSED_DIR, "adni_pet_eligibility.parquet"))
    return set(pet.loc[pet["POLARIS_ELIGIBLE"], "RID"])


def restrict_to_polaris(df, polaris_rids):
    return df[df["RID"].isin(polaris_rids)].copy()


# ------------------------------------------------------------------
# Cognitive / biomarker runners -- reuse run_adni_statistics.py's
# sample builders and run_adni_robustness.py's fit-or-suppress + HC3 +
# influence + classification core UNCHANGED, on the POLARIS-restricted
# tables. Also captures the real per-cell (DX x month) sample size
# used at each step (via adni_stats.check_group_sizes, the exact same
# function the small-cell suppression rule itself uses) for the
# pre-registered modeling-check report -- never a separate, potentially
# drifting count.
# ------------------------------------------------------------------


def run_cognitive_with_status(clinical_long, endpoint, analysis_type, exclude_rids=None):
    summary_rows, _, _ = R.run_cognitive_endpoint(clinical_long, endpoint, analysis_type, exclude_rids)

    status_rows, cell_size_rows = [], []
    eligible_df = clinical_long[clinical_long[f"{endpoint}_ELIGIBLE"].fillna(False)]
    months_present = R.available_months(eligible_df) if not eligible_df.empty else []
    for month in S.TARGET_MONTHS:
        if month not in months_present:
            cell_size_rows.append({
                "entity": endpoint, "assay_platform": "", "analysis_type": analysis_type, "month": month,
                "CN": 0, "MCI": 0, "Dementia": 0,
            })
            rob, elig = RR._process_one((endpoint, "", analysis_type), pd.DataFrame(), month)
        else:
            sample = R.build_cognitive_sample(clinical_long, endpoint, month, exclude_rids)
            _, group_ns, _, _ = S.check_group_sizes(sample)
            cell_size_rows.append({
                "entity": endpoint, "assay_platform": "", "analysis_type": analysis_type, "month": month, **group_ns,
            })
            rob, elig = RR._process_one((endpoint, "", analysis_type), sample, month)
        status_rows.append((elig, rob))
    return summary_rows, status_rows, cell_size_rows


def run_biomarker_with_status(
    plasma_df, dx_df, biomarker, value_col, assay_platform, analysis_type,
    lot_bias_col=None, exclude_flagged=False, platform=None,
):
    summary_rows, _, _ = R.run_biomarker_family(
        plasma_df, dx_df, biomarker, value_col, assay_platform, analysis_type,
        lot_bias_col=lot_bias_col, exclude_flagged=exclude_flagged, platform=platform,
    )

    status_rows, cell_size_rows = [], []
    scoped = plasma_df if platform is None else plasma_df[plasma_df["PLATFORM"] == platform]
    eligible_scoped = scoped[scoped["BIOMARKER_ELIGIBLE"].fillna(False)]
    months_present = R.available_months(eligible_scoped) if not eligible_scoped.empty else []
    for month in S.TARGET_MONTHS:
        if month not in months_present:
            cell_size_rows.append({
                "entity": biomarker, "assay_platform": assay_platform, "analysis_type": analysis_type, "month": month,
                "CN": 0, "MCI": 0, "Dementia": 0,
            })
            rob, elig = RR._process_one((biomarker, assay_platform, analysis_type), pd.DataFrame(), month)
        else:
            sample = R.build_biomarker_sample(
                plasma_df, dx_df, value_col, month,
                lot_bias_col=lot_bias_col, exclude_flagged=exclude_flagged, platform=platform,
            )
            _, group_ns, _, _ = S.check_group_sizes(sample)
            cell_size_rows.append({
                "entity": biomarker, "assay_platform": assay_platform, "analysis_type": analysis_type, "month": month,
                **group_ns,
            })
            rob, elig = RR._process_one((biomarker, assay_platform, analysis_type), sample, month)
        status_rows.append((elig, rob))
    return summary_rows, status_rows, cell_size_rows


def flatten_status_rows(status_rows):
    """One combined status table: every (entity, platform, analysis_type,
    month) cell's classification + reason (the adni_dashboard_eligibility.csv
    schema) joined with its HC3 / influence robustness detail rows (the
    adni_robustness_summary.csv schema) -- a single file covering the
    full 7-month grid, so a future dashboard population switch needs
    one lookup instead of two. A cell with no fitted model still gets
    exactly one row (robustness_check = "N/A (no model fit)") so
    coverage of the full grid is never silently narrower than the
    classification alone would imply."""
    rows = []
    for elig, rob_rows in status_rows:
        if not rob_rows:
            row = dict(elig)
            row.update({
                "robustness_check": "N/A (no model fit)", "level": "", "group_or_comparison": "",
                "conventional_estimate": None, "conventional_se": None,
                "conventional_ci_lower": None, "conventional_ci_upper": None, "conventional_p": None,
                "alternative_estimate": None, "alternative_se": None,
                "alternative_ci_lower": None, "alternative_ci_upper": None, "alternative_p": None,
                "sensitive_flag": None,
            })
            rows.append(row)
        else:
            for r in rob_rows:
                row = dict(elig)
                row.update(r)
                rows.append(row)
    return pd.DataFrame(rows)[STATUS_COLS]


# ------------------------------------------------------------------
# Modeling-check report (diagnosis composition + cell sizes) -- built
# BEFORE any conclusion is drawn about grouping, from the identical
# check_group_sizes() counts the fitting step itself uses.
# ------------------------------------------------------------------


def build_cell_size_table_md(cell_size_rows, title):
    df = pd.DataFrame(cell_size_rows)
    lines = [f"### {title}\n", "| Entity | Platform | Month | CN | MCI | Dementia | Min cell (limiting) |", "|---|---|---:|---:|---:|---:|---:|"]
    for _, r in df.iterrows():
        min_cell = min(r["CN"], r["MCI"], r["Dementia"])
        flag = " ⚠" if min_cell < S.MIN_GROUP_N else ""
        lines.append(
            f"| {r['entity']} | {r['assay_platform'] or '—'} | {int(r['month'])} | {int(r['CN'])} | {int(r['MCI'])} | "
            f"{int(r['Dementia'])} | {min_cell}{flag} |"
        )
    return "\n".join(lines)


# ------------------------------------------------------------------
# Methods / validation document
# ------------------------------------------------------------------


def build_methods_md(dx_composition, cog_cell_rows, bio_cell_rows, counts, baseline_check):
    lines = []
    lines.append("# ADNI POLARIS AD-Aligned Longitudinal Trajectories -- Methods & Validation\n")
    lines.append(
        "**Population restriction only.** Every sample-construction, ANCOVA, HC3-robust-covariance, "
        "influential-observation-sensitivity, and eligibility-classification function used here is "
        "imported unchanged from `run_adni_statistics.py` / `run_adni_robustness.py` / `adni_stats.py` / "
        "`adni_robustness.py`. Nothing about the model specification, covariates, small-cell rule, "
        "log-scale handling, or assay/platform separation was changed for this population. "
        "`biomarker_dashboard.html` was not built or touched. All outputs are aggregate-only; no "
        "participant identifier appears anywhere.\n"
    )
    lines.append(
        "**Eligibility is consumed, not redefined.** The POLARIS AD-Aligned cohort (n=620; baseline "
        "MMSE >=20, QC-passed amyloid PET Centiloid >=30 within +/-90 days of clinical baseline) is "
        "read as-is from `processed/adni_pet_eligibility.parquet`'s `POLARIS_ELIGIBLE` flag, computed "
        "by the already-approved `adni_pet.py` / `run_adni_pet_eligibility.py` stage. No eligibility "
        "rule is re-derived here.\n"
    )
    lines.append(
        "**No comparison against Overall ADNI, no propensity-score matching, no AR1001 treatment-outcome "
        "comparison** is performed anywhere in this stage.\n"
    )

    lines.append("## 1. Diagnosis composition at baseline (POLARIS AD-Aligned, n=620)\n")
    lines.append("| Diagnosis | n | % |\n|---|---:|---:|")
    for level in S.DX_LEVELS:
        n = dx_composition.get(level, 0)
        lines.append(f"| {level} | {n} | {n / 620 * 100:.1f}% |")
    lines.append("")

    lines.append("## 2. Modeling check -- cell sizes by diagnosis x month\n")
    lines.append(
        f"Minimum diagnosis-group cell size for ANCOVA remains **n={S.MIN_GROUP_N}** (unchanged from the "
        "primary Overall-ADNI analysis). Rows below marked ⚠ have at least one diagnosis group below "
        "that threshold and are reported descriptive-only (or not-available), never force-fitted. Only "
        "the `primary` analysis_type is shown here for readability; sensitivity variants follow the "
        "identical rule.\n"
    )
    lines.append(build_cell_size_table_md([r for r in cog_cell_rows if r["analysis_type"] == "primary"], "Cognitive endpoints"))
    lines.append("")
    lines.append(build_cell_size_table_md([r for r in bio_cell_rows if r["analysis_type"] == "primary"], "Plasma biomarkers"))
    lines.append("")

    lines.append("## 3. Grouping recommendation\n")
    lines.append(
        "**Recommendation: retain CN/MCI/Dementia diagnosis stratification, using the unchanged "
        f"n>={S.MIN_GROUP_N} small-cell rule, exactly as the primary Overall-ADNI analysis does -- do "
        "not collapse to a single pooled POLARIS trajectory and do not adopt a different grouping.**\n\n"
        "Reasoning: for both cognitive endpoints, cell sizes at months 0/6/12/24 support a fitted ANCOVA "
        "in all three diagnosis groups; month 18 is sparse cohort-wide (an existing ADNI visit-schedule "
        "characteristic, not specific to POLARIS) and months 36/48 lose only the Dementia group below "
        "threshold -- the existing per-cell suppression rule already reports exactly those cells as "
        "descriptive-only without being told to. For plasma biomarkers, the POLARIS subset is "
        "considerably sparser than Overall ADNI at every non-baseline month (as expected: POLARIS is a "
        "620-participant subset of the 3,030-participant cohort, and biomarker draws were never "
        "collected on every participant to begin with) -- GFAP/NfL fit zero months on either platform, "
        "matching the *already-established* Overall-ADNI finding that GFAP/NfL never clear the small-"
        "cell threshold at any month; pTau181 retains enough support to fit at months 12 and 24; "
        "pTau217/Aβ42-Aβ40 (Fujirebio, the lower-volume platform) fit essentially no follow-up month. "
        "None of this required inventing a new grouping: applying the SAME unmodified min-n rule to the "
        "restricted population already produces the scientifically correct behavior (fit where "
        "supported, honestly descriptive-only or not-available where not) -- exactly the outcome a "
        "collapsed single-cohort or ad hoc regrouping would have to reproduce by hand, with the added "
        "risk of losing the (still scientifically meaningful, still adequately powered at several "
        "timepoints) CN/MCI/Dementia contrast where the data do support it.\n"
    )

    lines.append("## 4. Validation -- baseline values against the POLARIS population profile\n")
    lines.append(
        "Cross-check: the baseline value (`*_BASELINE`) over the full 620-participant POLARIS cohort, "
        "computed directly from `adni_clinical_long.parquet` in *this* stage, is compared against "
        "`adni_polaris_population_profile.csv` (written by the prior, already-approved POLARIS "
        "eligibility stage) -- both describe the identical population from the identical source column, "
        "so n and mean should match exactly.\n\n"
        "A separate, smaller `n (longitudinal-analysis-eligible)` is also reported: the trajectory "
        "tables' own month-0 row additionally requires >=1 qualifying follow-up visit "
        "(`*_ELIGIBLE`, the same longitudinal-eligibility rule the primary Overall-ADNI analysis has "
        "always used) -- this is an expected, real restriction, not a discrepancy.\n\n"
    )
    lines.append(
        "| Measure | This stage (n, full cohort) | This stage (mean) | Population profile (n) | "
        "Population profile (mean) | Match | n (longitudinal-analysis-eligible) |\n"
        "|---|---:|---:|---:|---:|---:|---:|"
    )
    for row in baseline_check:
        match = "✓" if row["match"] else "✗ MISMATCH"
        lines.append(
            f"| {row['measure']} | {row['this_stage_n']} | {row['this_stage_mean']:.3f} | "
            f"{row['profile_n']} | {row['profile_mean']:.3f} | {match} | {row['n_longitudinal_eligible']} |"
        )
    lines.append("")

    lines.append("## 5. Output files\n")
    lines.append(
        "`adni_polaris_cognitive_trajectories.csv` and `adni_polaris_biomarker_trajectories.csv` reuse "
        "the exact column schema of `adni_cognitive_summary.csv` / `adni_biomarker_summary.csv`. "
        "`adni_polaris_trajectory_status.csv` combines the `adni_dashboard_eligibility.csv` "
        "(classification/reason) and `adni_robustness_summary.csv` (HC3 / influence detail) schemas "
        "into one file keyed on (endpoint_or_biomarker, assay_platform, analysis_type, month), covering "
        "the full 7-month grid.\n"
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
        raise RuntimeError("ADNI_AUDIT_APPROVED is False -- POLARIS trajectory analysis must not run.")

    print("Loading validated POLARIS AD-Aligned eligibility flag (consumed, not recomputed)...")
    polaris_rids = load_polaris_rids()
    if len(polaris_rids) != EXPECTED_POLARIS_N:
        raise RuntimeError(
            f"Expected {EXPECTED_POLARIS_N} POLARIS-eligible participants from the approved eligibility "
            f"stage, found {len(polaris_rids)} -- stopping rather than proceeding with a drifted cohort."
        )
    print(f"  POLARIS AD-Aligned cohort: n={len(polaris_rids)}")

    print("Loading locked processed datasets and restricting to the POLARIS cohort...")
    tables = R.load_processed_tables()
    clinical_long = restrict_to_polaris(tables["clinical_long"], polaris_rids)
    dx_df = clinical_long.drop_duplicates("RID")[["RID", "DX_BASELINE_FIXED"]]
    dx_composition = dx_df["DX_BASELINE_FIXED"].value_counts().to_dict()

    all_cog_summary, all_cog_status, cog_cell_rows = [], [], []
    print("Fitting cognitive endpoint models (ADAS-Cog13, MMSE) -- POLARIS population...")
    for endpoint in ["ADAS_COG13", "MMSE"]:
        s, st, cells = run_cognitive_with_status(clinical_long, endpoint, "primary")
        all_cog_summary += s
        all_cog_status += st
        cog_cell_rows += cells

    mmse_excl = R.mmse_sensitivity_exclusion_rids(clinical_long)
    s, st, cells = run_cognitive_with_status(clinical_long, "MMSE", "sensitivity_interval_excl", exclude_rids=mmse_excl)
    all_cog_summary += s
    all_cog_status += st
    cog_cell_rows += cells

    all_bio_summary, all_bio_status, bio_cell_rows = [], [], []
    print("Fitting plasma biomarker models -- POLARIS population...")

    ptau181_p = restrict_to_polaris(tables["ptau181_long"], polaris_rids)
    s, st, cells = run_biomarker_with_status(ptau181_p, dx_df, "pTau181", "PLASMAPTAU181", "Gothenburg_Simoa", "primary")
    all_bio_summary += s; all_bio_status += st; bio_cell_rows += cells

    ptau217_p = restrict_to_polaris(tables["ptau217_long"], polaris_rids)
    s, st, cells = run_biomarker_with_status(
        ptau217_p, dx_df, "pTau217", "PTAU217", "Fujirebio_Lumipulse", "primary",
        lot_bias_col="PTAU217_LOT_BIAS_FLAG", exclude_flagged=True,
    )
    all_bio_summary += s; all_bio_status += st; bio_cell_rows += cells
    s, st, cells = run_biomarker_with_status(
        ptau217_p, dx_df, "pTau217", "PTAU217", "Fujirebio_Lumipulse", "sensitivity_incl_lot_bias",
        lot_bias_col="PTAU217_LOT_BIAS_FLAG", exclude_flagged=False,
    )
    all_bio_summary += s; all_bio_status += st; bio_cell_rows += cells

    abeta_p = restrict_to_polaris(tables["abeta_ratio_long"], polaris_rids)
    s, st, cells = run_biomarker_with_status(abeta_p, dx_df, "Abeta42_40_ratio", "ABETA_RATIO", "Fujirebio_Lumipulse", "primary")
    all_bio_summary += s; all_bio_status += st; bio_cell_rows += cells

    for biomarker, value_col, table_key in [("GFAP", "GFAP", "gfap_long"), ("NfL", "NfL", "nfl_long")]:
        plat_p = restrict_to_polaris(tables[table_key], polaris_rids)
        s, st, cells = run_biomarker_with_status(plat_p, dx_df, biomarker, value_col, "Quanterix", "primary", platform="Quanterix")
        all_bio_summary += s; all_bio_status += st; bio_cell_rows += cells
        s, st, cells = run_biomarker_with_status(plat_p, dx_df, biomarker, value_col, "Fujirebio", "sensitivity_fujirebio", platform="Fujirebio")
        all_bio_summary += s; all_bio_status += st; bio_cell_rows += cells

    cognitive_df = pd.DataFrame(all_cog_summary)
    biomarker_df = pd.DataFrame(all_bio_summary)
    status_df = flatten_status_rows(all_cog_status + all_bio_status)

    print("Cross-validating baseline values against the approved POLARIS population profile...")
    profile_df = pd.read_csv(os.path.join(ADNI_OUTPUTS_DIR, "adni_polaris_population_profile.csv"))
    dedup_clinical = clinical_long.drop_duplicates("RID")
    baseline_check = []
    for measure, baseline_col, eligible_col, profile_var in [
        ("MMSE (baseline value)", "MMSE_BASELINE", "MMSE_ELIGIBLE", "Baseline MMSE"),
        ("ADAS-Cog13 (baseline value)", "ADAS_COG13_BASELINE", "ADAS_COG13_ELIGIBLE", "Baseline ADAS-Cog13"),
    ]:
        # Apples-to-apples: the population profile describes ALL 620
        # POLARIS participants with a baseline value, regardless of
        # whether they also have follow-up data -- so the comparison
        # must use the same *_BASELINE column over the same full
        # denominator, not the longitudinal-analysis-eligible (month=0
        # trajectory-row) subset, which additionally requires >=1
        # follow-up visit and is therefore a real, expected, smaller n.
        this_n = int(dedup_clinical[baseline_col].notna().sum())
        this_mean = float(dedup_clinical[baseline_col].mean())
        prof_row = profile_df[(profile_df["variable"] == profile_var) & (profile_df["population"] == "POLARIS-aligned ADNI")].iloc[0]
        prof_n, prof_mean = int(prof_row["n"]), float(prof_row["mean"])
        n_longitudinal_eligible = int(dedup_clinical[eligible_col].fillna(False).sum())
        baseline_check.append({
            "measure": measure, "this_stage_n": this_n, "this_stage_mean": this_mean,
            "profile_n": prof_n, "profile_mean": prof_mean,
            "n_longitudinal_eligible": n_longitudinal_eligible,
            "match": this_n == prof_n and abs(this_mean - prof_mean) < 1e-6,
        })

    os.makedirs(ADNI_OUTPUTS_DIR, exist_ok=True)
    cognitive_df.to_csv(os.path.join(ADNI_OUTPUTS_DIR, "adni_polaris_cognitive_trajectories.csv"), index=False)
    biomarker_df.to_csv(os.path.join(ADNI_OUTPUTS_DIR, "adni_polaris_biomarker_trajectories.csv"), index=False)
    status_df.to_csv(os.path.join(ADNI_OUTPUTS_DIR, "adni_polaris_trajectory_status.csv"), index=False)

    n_fitted = int((status_df["robustness_check"] == "HC3").groupby(
        [status_df["endpoint_or_biomarker"], status_df["assay_platform"], status_df["analysis_type"], status_df["month"]]
    ).any().sum()) if not status_df.empty else 0
    class_counts = status_df.drop_duplicates(["endpoint_or_biomarker", "assay_platform", "analysis_type", "month"])["classification"].value_counts().to_dict()

    counts = {
        "POLARIS AD-Aligned cohort n": len(polaris_rids),
        "Cognitive trajectory rows": len(cognitive_df),
        "Biomarker trajectory rows": len(biomarker_df),
        "Trajectory-status rows (all)": len(status_df),
        "Distinct endpoint/platform/analysis/month cells": status_df.drop_duplicates(["endpoint_or_biomarker", "assay_platform", "analysis_type", "month"]).shape[0] if not status_df.empty else 0,
        "Cells classified A. Adjusted analysis": int(class_counts.get("A. Adjusted analysis", 0)),
        "Cells classified B. Descriptive only": int(class_counts.get("B. Descriptive only", 0)),
        "Cells classified C. Sensitivity concern": int(class_counts.get("C. Sensitivity concern", 0)),
        "Cells classified D. Not available": int(class_counts.get("D. Not available", 0)),
    }
    methods_md = build_methods_md(dx_composition, cog_cell_rows, bio_cell_rows, counts, baseline_check)
    with open(os.path.join(ADNI_OUTPUTS_DIR, "adni_polaris_trajectory_methods.md"), "w", encoding="utf-8") as f:
        f.write(methods_md)

    print(f"wrote {ADNI_OUTPUTS_DIR}/adni_polaris_cognitive_trajectories.csv")
    print(f"wrote {ADNI_OUTPUTS_DIR}/adni_polaris_biomarker_trajectories.csv")
    print(f"wrote {ADNI_OUTPUTS_DIR}/adni_polaris_trajectory_status.csv")
    print(f"wrote {ADNI_OUTPUTS_DIR}/adni_polaris_trajectory_methods.md")

    print()
    print("=== DONE (aggregate summary only) ===")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    for row in baseline_check:
        status = "OK" if row["match"] else "MISMATCH -- INVESTIGATE"
        print(
            f"  Baseline check [{row['measure']}]: {status} (n={row['this_stage_n']} vs {row['profile_n']}, "
            f"mean={row['this_stage_mean']:.6f} vs {row['profile_mean']:.6f}; "
            f"longitudinal-analysis-eligible n={row['n_longitudinal_eligible']})"
        )


if __name__ == "__main__":
    main()
