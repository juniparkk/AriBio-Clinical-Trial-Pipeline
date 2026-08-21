# ============================================================
# RUN_ADNI_TARGET_POPULATIONS -- generalized target-population
# eligibility + longitudinal trajectory outputs for the ADNI Natural
# History dashboard's interactive cohort-definition tool.
#
# Generalizes the pattern already validated by run_adni_pet_eligibility.py
# (eligibility funnel + population profile) and run_adni_polaris_
# trajectories.py (restrict-then-reuse trajectory fitting) from ONE
# hardcoded alternate population (POLARIS) to adni_eligibility.
# PRESET_LIBRARY's N named presets -- currently a 3x3 diagnosis
# (CN/MCI/Dementia) x amyloid-status (Overall/Confirmed/Not confirmed)
# grid. Every preset is evaluated identically through
# adni_eligibility.evaluate_preset()/build_preset_attrition()/
# build_preset_profile() -- no preset is special-cased. Every
# sample-building, ANCOVA, HC3, and influence-sensitivity function used
# here is IMPORTED from run_adni_statistics.py / run_adni_robustness.py /
# adni_stats.py / adni_robustness.py, never reimplemented or modified.
#
# Pooled (non-diagnosis-stratified) trajectories are NEW: no existing
# stage computes a whole-population (not split by CN/MCI/Dementia)
# change-from-baseline trend. This is composition only, not new
# statistics -- adni_stats.descriptive_mean_ci() already takes a plain
# ungrouped Series; this script just calls it on the whole sample
# instead of grouping by DX_BASELINE_FIXED first. Always descriptive-
# only (no ANCOVA "group" term applies once there is no group split),
# consistent with the same subset/superset-independence reasoning
# run_adni_polaris_trajectories.py's own header already documents for
# why POLARIS is never inferentially tested against Overall ADNI.
#
# Explicitly NOT in scope here (same non-goals as run_adni_polaris_
# trajectories.py, generalized to every preset):
#   - propensity-score matching
#   - any statistical (p-value/test-statistic) comparison of a Target
#     Population against Overall ADNI -- Target is always a SUBSET of
#     Overall (nested, non-independent samples), so no independent-
#     samples test in this codebase is valid here; every comparison
#     produced by this script is purely descriptive, side by side
#   - AR1001 / any treatment-outcome comparison
#   - modifying adni_viz.py / adni_viz_data.py / biomarker_dashboard.html
#
# Reads (locked, read-only inputs for this stage):
#   ADNI_PROCESSED_DIR/adni_pet_eligibility.parquet
#   ADNI_PROCESSED_DIR/adni_clinical_long.parquet
#   ADNI_PROCESSED_DIR/adni_{ptau181,ptau217,abeta_ratio,gfap,nfl}_long.parquet
#
# Writes (aggregate-only; no participant identifier in any output):
#   ADNI_OUTPUTS_DIR/adni_target_population_presets.csv
#   ADNI_OUTPUTS_DIR/adni_target_population_cohort_attrition.csv
#   ADNI_OUTPUTS_DIR/adni_target_population_profile.csv
#   ADNI_OUTPUTS_DIR/adni_target_population_cognitive_trajectories.csv
#   ADNI_OUTPUTS_DIR/adni_target_population_biomarker_trajectories.csv
#   ADNI_OUTPUTS_DIR/adni_target_population_trajectory_status.csv
#   ADNI_OUTPUTS_DIR/adni_target_population_pooled_trajectories.csv
#   ADNI_OUTPUTS_DIR/adni_target_population_methods.md
#
# Usage: .venv/bin/python run_adni_target_populations.py
# ============================================================

import os

import pandas as pd

from adni_analysis import ADNI_AUDIT_APPROVED, ADNI_OUTPUTS_DIR, ADNI_PROCESSED_DIR
import adni_eligibility as E
import adni_robustness as RB
import adni_stats as S
import run_adni_robustness as RR
import run_adni_statistics as R
from run_adni_polaris_trajectories import (
    STATUS_COLS,
    flatten_status_rows,
    run_biomarker_with_status,
    run_cognitive_with_status,
)

OVERALL_LABEL = "Overall ADNI"

# (biomarker, value_col, assay_platform, table_key, platform, lot_bias_col)
# -- primary analysis_type only, mirroring run_adni_polaris_trajectories.py's
# primary-only selections. Pooled trajectories intentionally skip the
# sensitivity variants (screening-interval, lot-bias-inclusion, alternate
# platform) that the per-diagnosis-group trajectory files still carry in
# full via run_cognitive_with_status/run_biomarker_with_status below --
# keeping the pooled default-view line simple, not exhaustive.
POOLED_BIOMARKER_SPECS = [
    ("pTau181", "PLASMAPTAU181", "Gothenburg_Simoa", "ptau181_long", None, None),
    ("pTau217", "PTAU217", "Fujirebio_Lumipulse", "ptau217_long", None, "PTAU217_LOT_BIAS_FLAG"),
    ("Abeta42_40_ratio", "ABETA_RATIO", "Fujirebio_Lumipulse", "abeta_ratio_long", None, None),
    ("GFAP", "GFAP", "Quanterix", "gfap_long", "Quanterix", None),
    ("NfL", "NfL", "Quanterix", "nfl_long", "Quanterix", None),
]


# ------------------------------------------------------------------
# I/O
# ------------------------------------------------------------------


def load_master_eligibility_table():
    pet = pd.read_parquet(os.path.join(ADNI_PROCESSED_DIR, "adni_pet_eligibility.parquet"))
    biomarker_eligible_rids = {}
    for col, table_key in [
        ("HAS_PTAU181", "ptau181_long"), ("HAS_PTAU217", "ptau217_long"),
        ("HAS_ABETA_RATIO", "abeta_ratio_long"), ("HAS_GFAP", "gfap_long"), ("HAS_NFL", "nfl_long"),
    ]:
        df = pd.read_parquet(os.path.join(ADNI_PROCESSED_DIR, f"adni_{table_key}.parquet"))
        elig = df[df["BIOMARKER_ELIGIBLE"].fillna(False)]
        biomarker_eligible_rids[col] = set(elig["RID"])
    return E.build_master_eligibility_table(pet, biomarker_eligible_rids)


def restrict_to_rids(df, rids):
    return df[df["RID"].isin(rids)].copy()


# ------------------------------------------------------------------
# Pooled (non-diagnosis-stratified) descriptive trajectory
# ------------------------------------------------------------------


def compute_pooled_trajectory_rows(preset_id, population_label, clinical_long, plasma_tables):
    rows = []
    for endpoint in ["ADAS_COG13", "MMSE"]:
        for month in S.TARGET_MONTHS:
            sample = R.build_cognitive_sample(clinical_long, endpoint, month)
            stats = S.descriptive_mean_ci(sample["change_from_baseline"]) if not sample.empty else {"n": 0, "mean": None, "ci_lower": None, "ci_upper": None}
            status = RB.CLASS_DESCRIPTIVE if stats["n"] >= 1 else RB.CLASS_NOT_AVAILABLE
            rows.append({
                "preset_id": preset_id, "population": population_label, "entity": endpoint,
                "assay_platform": "", "analysis_type": "primary", "month": month,
                "n": stats["n"], "estimate": stats["mean"], "ci_lower": stats["ci_lower"], "ci_upper": stats["ci_upper"],
                "descriptive_status": status,
            })

    dx_df = clinical_long.drop_duplicates("RID")[["RID", "DX_BASELINE_FIXED"]]
    for biomarker, value_col, assay_platform, table_key, platform, lot_bias_col in POOLED_BIOMARKER_SPECS:
        plasma_df = plasma_tables[table_key]
        for month in S.TARGET_MONTHS:
            sample = R.build_biomarker_sample(
                plasma_df, dx_df, value_col, month,
                lot_bias_col=lot_bias_col, exclude_flagged=lot_bias_col is not None, platform=platform,
            )
            if sample.empty:
                rows.append({
                    "preset_id": preset_id, "population": population_label, "entity": biomarker,
                    "assay_platform": assay_platform, "analysis_type": "primary", "month": month,
                    "n": 0, "estimate": None, "ci_lower": None, "ci_upper": None,
                    "descriptive_status": RB.CLASS_NOT_AVAILABLE,
                })
                continue
            stats = S.descriptive_mean_ci(sample["log_change"])
            est = S.geometric_percent_change(stats["mean"]) if stats["n"] >= 1 else None
            ci_lo = S.geometric_percent_change(stats["ci_lower"]) if pd.notna(stats["ci_lower"]) else None
            ci_hi = S.geometric_percent_change(stats["ci_upper"]) if pd.notna(stats["ci_upper"]) else None
            rows.append({
                "preset_id": preset_id, "population": population_label, "entity": biomarker,
                "assay_platform": assay_platform, "analysis_type": "primary", "month": month,
                "n": stats["n"], "estimate": est, "ci_lower": ci_lo, "ci_upper": ci_hi,
                "descriptive_status": RB.CLASS_DESCRIPTIVE if stats["n"] >= 1 else RB.CLASS_NOT_AVAILABLE,
            })
    return rows


# ------------------------------------------------------------------
# Per-diagnosis-group trajectory for one preset's target population
# (reuses run_adni_polaris_trajectories.py's helpers unmodified)
# ------------------------------------------------------------------


def run_target_trajectories(preset_id, rids, tables):
    clinical_long = restrict_to_rids(tables["clinical_long"], rids)
    dx_df = clinical_long.drop_duplicates("RID")[["RID", "DX_BASELINE_FIXED"]]

    all_cog_summary, all_cog_status = [], []
    for endpoint in ["ADAS_COG13", "MMSE"]:
        s, st, _ = run_cognitive_with_status(clinical_long, endpoint, "primary")
        all_cog_summary += s
        all_cog_status += st

    mmse_excl = R.mmse_sensitivity_exclusion_rids(clinical_long)
    s, st, _ = run_cognitive_with_status(clinical_long, "MMSE", "sensitivity_interval_excl", exclude_rids=mmse_excl)
    all_cog_summary += s
    all_cog_status += st

    all_bio_summary, all_bio_status = [], []

    ptau181_p = restrict_to_rids(tables["ptau181_long"], rids)
    s, st, _ = run_biomarker_with_status(ptau181_p, dx_df, "pTau181", "PLASMAPTAU181", "Gothenburg_Simoa", "primary")
    all_bio_summary += s; all_bio_status += st

    ptau217_p = restrict_to_rids(tables["ptau217_long"], rids)
    s, st, _ = run_biomarker_with_status(
        ptau217_p, dx_df, "pTau217", "PTAU217", "Fujirebio_Lumipulse", "primary",
        lot_bias_col="PTAU217_LOT_BIAS_FLAG", exclude_flagged=True,
    )
    all_bio_summary += s; all_bio_status += st
    s, st, _ = run_biomarker_with_status(
        ptau217_p, dx_df, "pTau217", "PTAU217", "Fujirebio_Lumipulse", "sensitivity_incl_lot_bias",
        lot_bias_col="PTAU217_LOT_BIAS_FLAG", exclude_flagged=False,
    )
    all_bio_summary += s; all_bio_status += st

    abeta_p = restrict_to_rids(tables["abeta_ratio_long"], rids)
    s, st, _ = run_biomarker_with_status(abeta_p, dx_df, "Abeta42_40_ratio", "ABETA_RATIO", "Fujirebio_Lumipulse", "primary")
    all_bio_summary += s; all_bio_status += st

    for biomarker, value_col, table_key in [("GFAP", "GFAP", "gfap_long"), ("NfL", "NfL", "nfl_long")]:
        plat_p = restrict_to_rids(tables[table_key], rids)
        s, st, _ = run_biomarker_with_status(plat_p, dx_df, biomarker, value_col, "Quanterix", "primary", platform="Quanterix")
        all_bio_summary += s; all_bio_status += st
        s, st, _ = run_biomarker_with_status(plat_p, dx_df, biomarker, value_col, "Fujirebio", "sensitivity_fujirebio", platform="Fujirebio")
        all_bio_summary += s; all_bio_status += st

    cognitive_df = pd.DataFrame(all_cog_summary)
    biomarker_df = pd.DataFrame(all_bio_summary)
    status_df = flatten_status_rows(all_cog_status + all_bio_status)
    for df in (cognitive_df, biomarker_df, status_df):
        df.insert(0, "preset_id", preset_id)

    plasma_tables_restricted = {
        "ptau181_long": ptau181_p, "ptau217_long": ptau217_p, "abeta_ratio_long": abeta_p,
        "gfap_long": restrict_to_rids(tables["gfap_long"], rids), "nfl_long": restrict_to_rids(tables["nfl_long"], rids),
    }
    pooled_rows = compute_pooled_trajectory_rows(preset_id, "target", clinical_long, plasma_tables_restricted)

    return cognitive_df, biomarker_df, status_df, pooled_rows


# ------------------------------------------------------------------
# Methods document
# ------------------------------------------------------------------


def build_methods_md(preset_results):
    lines = ["# ADNI Target Population Presets -- Methods & Validation\n"]
    lines.append(
        "**Population restriction only.** Every sample-construction, ANCOVA, HC3-robust-covariance, "
        "influential-observation-sensitivity, and eligibility-classification function used here is "
        "imported unchanged from `run_adni_statistics.py` / `run_adni_robustness.py` / `adni_stats.py` / "
        "`adni_robustness.py`. Eligibility criteria are declarative (`adni_eligibility.PRESET_LIBRARY`), "
        "evaluated against fields already present in the approved `adni_pet_eligibility.parquet` / "
        "`adni_clinical_long.parquet` / biomarker long tables -- no new participant-level data, no new "
        "disease-stage granularity, no new statistical test. `biomarker_dashboard.html` was not built or "
        "touched by this stage.\n"
    )
    lines.append(
        "**No statistical comparison between any Target Population and Overall ADNI is performed "
        "anywhere in this stage.** A Target Population is always a SUBSET of Overall ADNI -- nested, "
        "non-independent samples -- so no independent-samples test in this codebase is methodologically "
        "valid here, matching the identical reasoning already documented in "
        "`run_adni_polaris_trajectories.py` for POLARIS vs. Overall ADNI. Every comparison this stage "
        "produces (population profile, pooled trajectory) is purely descriptive: n / mean / SD / CI "
        "reported side by side, never tested against each other.\n"
    )
    lines.append(
        "**Pooled trajectories are new composition, not new statistics.** "
        "`adni_stats.descriptive_mean_ci()` (already approved, used elsewhere for other descriptive "
        "summaries) is called on the whole population's change-from-baseline / log-change column "
        "without a diagnosis-group split -- always classified `B. Descriptive only` or `D. Not "
        "available`, never `A`/`C`, since there is no ANCOVA group term once there is no group split.\n"
    )
    lines.append("## Presets\n")
    lines.append("| Preset | n (target) | Diagnosis composition |\n|---|---:|---|")
    for r in preset_results:
        dx = ", ".join(f"{k}={v}" for k, v in r["dx_composition"].items())
        lines.append(f"| {r['label']} | {r['n']} | {dx} |")
    lines.append("")
    lines.append(
        "## Grouping recommendation (all presets)\n\n"
        "Per-diagnosis-group (CN/MCI/Dementia) trajectory files retain the unchanged "
        f"n≥{S.MIN_GROUP_N} small-cell rule, exactly as the primary Overall-ADNI analysis and the "
        "POLARIS trajectory stage do -- no preset collapses or regroups diagnosis. Presets with a "
        "smaller resulting n (typically the amyloid-confirmed and amyloid-not-confirmed columns, "
        "since both require a valid amyloid-PET scan on top of the diagnosis restriction) are "
        "expected to show more `B. Descriptive only`/`D. Not available` cells at later follow-up "
        "months; this is the existing suppression rule working correctly on a smaller n, not a new rule.\n"
    )
    return "\n".join(lines)


# ------------------------------------------------------------------
# Orchestration
# ------------------------------------------------------------------


def main():
    if not ADNI_AUDIT_APPROVED:
        raise RuntimeError("ADNI_AUDIT_APPROVED is False -- target-population analysis must not run.")

    print("Building master eligibility table (validated cohort + biomarker availability)...")
    master = load_master_eligibility_table()
    print(f"  Master eligibility table: n={len(master)}")

    print("Loading locked processed datasets...")
    tables = R.load_processed_tables()

    preset_catalog_rows = []
    attrition_frames, profile_frames = [], []
    cognitive_frames, biomarker_frames, status_frames, pooled_frames = [], [], [], []
    preset_methods_results = []

    # Overall ADNI's own pooled trajectory does not depend on the
    # preset, but is written once per preset row (denormalized) so the
    # dashboard loader never needs preset-independent special-casing --
    # computed once here, reused for every preset.
    print("Computing Overall ADNI pooled (non-diagnosis-stratified) trajectory (once, reused per preset)...")
    overall_pooled_rows = compute_pooled_trajectory_rows("__overall__", "overall", tables["clinical_long"], tables)

    for preset in E.PRESET_LIBRARY:
        print(f"--- {preset.label} ---")
        target_mask = E.evaluate_preset(master, preset)
        rids = set(master.loc[target_mask, "RID"])
        attrition_df = E.build_preset_attrition(master, preset)
        profile_df = E.build_preset_profile(master, target_mask, overall_label=OVERALL_LABEL, target_label="Target Population")

        n = len(rids)
        dx_composition = master.loc[master["RID"].isin(rids), "DX_BASELINE_FIXED"].value_counts().to_dict()
        print(f"  n={n}  ({', '.join(f'{k}={v}' for k, v in dx_composition.items())})")

        attrition_df = attrition_df.copy()
        attrition_df.insert(0, "preset_id", preset.id)
        profile_df = profile_df.copy()
        profile_df.insert(0, "preset_id", preset.id)
        attrition_frames.append(attrition_df)
        profile_frames.append(profile_df)

        preset_catalog_rows.append({
            "id": preset.id, "label": preset.label, "description": preset.description, "n": n,
        })

        cog_df, bio_df, status_df, target_pooled_rows = run_target_trajectories(preset.id, rids, tables)
        cognitive_frames.append(cog_df)
        biomarker_frames.append(bio_df)
        status_frames.append(status_df)
        pooled_frames.append(pd.DataFrame(target_pooled_rows))
        pooled_frames.append(pd.DataFrame([dict(r, preset_id=preset.id) for r in overall_pooled_rows]))

        preset_methods_results.append({"label": preset.label, "n": n, "dx_composition": dx_composition})

    presets_df = pd.DataFrame(preset_catalog_rows)
    attrition_df = pd.concat(attrition_frames, ignore_index=True)
    profile_df = pd.concat(profile_frames, ignore_index=True)
    cognitive_df = pd.concat(cognitive_frames, ignore_index=True)
    biomarker_df = pd.concat(biomarker_frames, ignore_index=True)
    status_df = pd.concat(status_frames, ignore_index=True)[["preset_id"] + STATUS_COLS]
    pooled_df = pd.concat(pooled_frames, ignore_index=True)

    os.makedirs(ADNI_OUTPUTS_DIR, exist_ok=True)
    outputs = {
        "adni_target_population_presets.csv": presets_df,
        "adni_target_population_cohort_attrition.csv": attrition_df,
        "adni_target_population_profile.csv": profile_df,
        "adni_target_population_cognitive_trajectories.csv": cognitive_df,
        "adni_target_population_biomarker_trajectories.csv": biomarker_df,
        "adni_target_population_trajectory_status.csv": status_df,
        "adni_target_population_pooled_trajectories.csv": pooled_df,
    }
    for filename, df in outputs.items():
        path = os.path.join(ADNI_OUTPUTS_DIR, filename)
        df.to_csv(path, index=False)
        print(f"wrote {path}  ({len(df)} rows)")

    methods_md = build_methods_md(preset_methods_results)
    methods_path = os.path.join(ADNI_OUTPUTS_DIR, "adni_target_population_methods.md")
    with open(methods_path, "w", encoding="utf-8") as f:
        f.write(methods_md)
    print(f"wrote {methods_path}")

    print()
    print("=== DONE (aggregate summary only) ===")
    for r in preset_methods_results:
        print(f"  {r['label']}: n={r['n']}")


if __name__ == "__main__":
    main()
