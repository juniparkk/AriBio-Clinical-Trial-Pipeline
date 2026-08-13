# ============================================================
# RUN_ADNI_PET_ELIGIBILITY -- orchestration entry point for the
# POLARIS AD-aligned ADNI eligibility cohort. PET ELIGIBILITY
# PREPROCESSING ONLY:
#   - does not modify raw/, interim/, or the LOCKED clinical baseline
#     definition / MMSE baseline logic in adni_cohort.py (read-only
#     input here)
#   - does not touch biomarker_dashboard.html / adni_viz.py / the
#     ANCOVA statistics or robustness stages
#   - does not perform propensity-score matching -- this is
#     ELIGIBILITY FILTERING only, never call the output "matched ADNI"
#   - writes ONE new participant-level file (local-only, gitignored,
#     never read by anything under adni_viz*.py) and three new
#     aggregate-only outputs; no participant identifier is ever
#     written to ADNI_OUTPUTS_DIR
#
# Reads (locked, read-only inputs for this stage):
#   ADNI_PROCESSED_DIR/adni_clinical_long.parquet
#   ADNI_INTERIM_DIR/UCBERKELEY_AMY_6MM.csv
#
# Writes:
#   ADNI_PROCESSED_DIR/adni_pet_eligibility.parquet   (participant-level, local-only)
#   ADNI_OUTPUTS_DIR/adni_polaris_cohort_attrition.csv
#   ADNI_OUTPUTS_DIR/adni_polaris_population_profile.csv
#   ADNI_OUTPUTS_DIR/adni_polaris_eligibility_metadata.md
#
# Usage: .venv/bin/python run_adni_pet_eligibility.py
# ============================================================

import datetime
import os

import numpy as np
import pandas as pd

from adni_analysis import ADNI_AUDIT_APPROVED, ADNI_INTERIM_DIR, ADNI_OUTPUTS_DIR, ADNI_PROCESSED_DIR
import adni_pet

assert ADNI_AUDIT_APPROVED, "ADNI audit must be approved before running any ADNI preprocessing stage."


# ------------------------------------------------------------------
# I/O
# ------------------------------------------------------------------


def load_clinical_baseline_table():
    """One row per RID from the locked adni_clinical_long.parquet:
    baseline diagnosis/demographics/cognitive scores, plus
    CLINICAL_BASELINE_DATE (the VISIT_MONTH == 0 row's EXAMDATE --
    NaT, not silently dropped, for the 39 participants who have no
    VISIT_MONTH == 0 row at all)."""
    clinical = pd.read_parquet(os.path.join(ADNI_PROCESSED_DIR, "adni_clinical_long.parquet"))
    static_cols = ["RID", "DX_BASELINE_FIXED", "BASELINE_AGE", "SEX", "APOE4_CARRIER",
                   "MMSE_BASELINE", "ADAS_COG13_BASELINE"]
    static = clinical.drop_duplicates("RID")[static_cols]
    baseline_dates = (
        clinical.loc[clinical["VISIT_MONTH"] == 0, ["RID", "EXAMDATE"]]
        .drop_duplicates("RID")
        .rename(columns={"EXAMDATE": "CLINICAL_BASELINE_DATE"})
    )
    return static.merge(baseline_dates, on="RID", how="left")


def load_pet_table():
    path = os.path.join(ADNI_INTERIM_DIR, "UCBERKELEY_AMY_6MM.csv")
    return pd.read_csv(path, low_memory=False)


def validated_cohort(clinical_baseline_df):
    """The 3,030-participant validated cohort: a fixed baseline
    diagnosis of CN/MCI/Dementia (same definition used everywhere else
    in this pipeline -- never re-derived here)."""
    return clinical_baseline_df[clinical_baseline_df["DX_BASELINE_FIXED"].isin(["CN", "MCI", "Dementia"])].copy()


# ------------------------------------------------------------------
# Aggregate output A: cohort attrition
# ------------------------------------------------------------------


def build_cohort_attrition(eligibility_df):
    """Sequential attrition through the named steps -- each step's
    "remaining n" is the previous step's population narrowed by one
    more condition, so the table is self-consistent and every
    exclusion is individually traceable back to a real condition, not
    a single opaque final filter."""
    steps = []
    pop = eligibility_df.copy()
    start_n = len(pop)
    steps.append(("Validated ADNI cohort", start_n, start_n, 0))

    pop2 = pop[pop["CLINICAL_BASELINE_DATE"].notna()]
    steps.append(("Valid clinical baseline date", len(pop), len(pop2), len(pop) - len(pop2)))

    pop3 = pop2[pop2["MMSE_BASELINE"].notna()]
    steps.append(("Baseline MMSE available", len(pop2), len(pop3), len(pop2) - len(pop3)))

    pop4 = pop3[pop3["MMSE_BASELINE"] >= adni_pet.MMSE_THRESHOLD]
    steps.append((f"MMSE >= {adni_pet.MMSE_THRESHOLD}", len(pop3), len(pop4), len(pop3) - len(pop4)))

    pop5 = pop4[pop4["CENTILOID_ELIGIBLE"]]
    steps.append((f"QC-passed PET within +/-{adni_pet.PET_WINDOW_DAYS} days", len(pop4), len(pop5), len(pop4) - len(pop5)))

    pop6 = pop5[pop5["CENTILOID_BASELINE"] >= adni_pet.CENTILOID_THRESHOLD]
    steps.append((f"Centiloid >= {adni_pet.CENTILOID_THRESHOLD}", len(pop5), len(pop6), len(pop5) - len(pop6)))

    steps.append(("Final POLARIS-aligned cohort", len(pop6), len(pop6), 0))

    rows = []
    for label, starting_n, remaining_n, excluded_n in steps:
        pct = round(100.0 * remaining_n / start_n, 1) if start_n else 0.0
        rows.append({
            "step": label, "starting_n": starting_n, "remaining_n": remaining_n,
            "excluded_n": excluded_n, "percent_retained_of_cohort": pct,
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Aggregate output B: population profile (Overall ADNI vs
# POLARIS-aligned ADNI)
# ------------------------------------------------------------------


def _numeric_summary(series):
    s = series.dropna()
    if len(s) == 0:
        return {"n": 0, "mean": np.nan, "sd": np.nan, "median": np.nan}
    return {"n": int(len(s)), "mean": float(s.mean()), "sd": float(s.std(ddof=1)) if len(s) > 1 else np.nan,
            "median": float(s.median())}


def build_population_profile(eligibility_df):
    overall = eligibility_df
    polaris = eligibility_df[eligibility_df["POLARIS_ELIGIBLE"]]

    rows = []

    def add_numeric_rows(field, label):
        for pop_name, pop_df in (("Overall ADNI", overall), ("POLARIS-aligned ADNI", polaris)):
            s = _numeric_summary(pop_df[field])
            rows.append({"variable": label, "population": pop_name, **s})

    add_numeric_rows("BASELINE_AGE", "Baseline age (years)")
    add_numeric_rows("MMSE_BASELINE", "Baseline MMSE")
    add_numeric_rows("ADAS_COG13_BASELINE", "Baseline ADAS-Cog13")
    add_numeric_rows("CENTILOID_BASELINE", "Baseline Centiloid")

    def add_categorical_rows(field, label):
        for pop_name, pop_df in (("Overall ADNI", overall), ("POLARIS-aligned ADNI", polaris)):
            total = len(pop_df)
            counts = pop_df[field].value_counts(dropna=False)
            for level, n in counts.items():
                level_label = "Missing" if pd.isna(level) else str(level)
                pct = round(100.0 * n / total, 1) if total else 0.0
                rows.append({"variable": f"{label}: {level_label}", "population": pop_name,
                             "n": int(n), "mean": np.nan, "sd": np.nan, "median": np.nan, "percent": pct})

    profile_numeric = pd.DataFrame(rows)

    cat_rows = []
    for pop_name, pop_df in (("Overall ADNI", overall), ("POLARIS-aligned ADNI", polaris)):
        total = len(pop_df)
        for field, label in (("DX_BASELINE_FIXED", "Baseline diagnosis"), ("SEX", "Sex"), ("APOE4_CARRIER", "APOE4 carrier")):
            counts = pop_df[field].value_counts(dropna=False)
            for level, n in counts.items():
                level_label = "Missing" if pd.isna(level) else str(level)
                pct = round(100.0 * n / total, 1) if total else 0.0
                cat_rows.append({"variable": label, "level": level_label, "population": pop_name,
                                 "n": int(n), "percent_of_population": pct})
    profile_categorical = pd.DataFrame(cat_rows)

    profile_numeric["level"] = ""
    profile_numeric = profile_numeric.rename(columns={"variable": "variable"})
    combined_numeric = profile_numeric[["variable", "level", "population", "n", "mean", "sd", "median"]]
    combined_categorical = profile_categorical.rename(columns={"percent_of_population": "percent"})
    combined_categorical = combined_categorical.assign(mean=np.nan, sd=np.nan, median=np.nan)[
        ["variable", "level", "population", "n", "mean", "sd", "median", "percent"]
    ]
    if "percent" not in combined_numeric.columns:
        combined_numeric = combined_numeric.assign(percent=np.nan)
    return pd.concat([combined_numeric, combined_categorical], ignore_index=True)


# ------------------------------------------------------------------
# Aggregate output C: eligibility metadata
# ------------------------------------------------------------------


def build_eligibility_metadata_md():
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""# POLARIS AD-Aligned ADNI Eligibility -- Rule Definition

Generated: {now}

This describes ELIGIBILITY FILTERING only. This population is **not**
a propensity-score-matched cohort and must never be referred to as
"matched ADNI."

## Eligibility rule

A participant is POLARIS-eligible iff ALL of the following hold:

- Baseline MMSE &ge; {adni_pet.MMSE_THRESHOLD}
- A QC-passed (`qc_flag == {adni_pet.PET_QC_PASS}`, "Pass") amyloid-PET
  Centiloid scan exists within &plusmn;{adni_pet.PET_WINDOW_DAYS} days of
  the participant's own already-validated clinical baseline date
  (`VISIT_MONTH == 0`'s exam date in `adni_clinical_long.parquet`)
- That scan's Centiloid value &ge; {adni_pet.CENTILOID_THRESHOLD}

## PET scan selection (when more than one QC-passed scan falls in the window)

Deterministic tie-break, applied in order:

1. Smallest absolute day difference from the clinical baseline date.
2. If tied, the earlier scan (more negative day difference).
3. If still tied, the lower `LONIUID` (PET image ID).

## Data source

- Table: `UCBERKELEY_AMY_6MM` (UC Berkeley amyloid-PET quantification
  pipeline, raw ADNIMERGE2 package).
- Field used: `CENTILOIDS` (summary cortical SUVR normalized to whole
  cerebellum, transformed to Centiloids -- tracer-harmonized across
  FBB/FBP/NAV by UC Berkeley's own published methodology; this
  pipeline does not independently re-derive that harmonization).
- QC gate: `qc_flag == {adni_pet.PET_QC_PASS}` ("Pass") only.
  `qc_flag` values of 1 ("Partial pass"), 0 ("Fail"), -1 ("Not
  assessed"), and -2 ("Cannot be processed") are excluded.

## Explicit handling of missing clinical baseline dates

A participant with no `VISIT_MONTH == 0` row (no valid clinical
baseline date) is recorded with `CENTILOID_BASELINE_STATUS ==
"{adni_pet.CENTILOID_STATUS_NO_BASELINE_DATE}"` and is excluded from
POLARIS eligibility -- never silently classified as eligible or
ineligible without this explicit, auditable reason.

## What this is not

- Not propensity-score matching.
- Not a treatment-effect analysis.
- Not a claim of comparability to any specific external trial arm
  beyond sharing this eligibility definition.
"""


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main():
    print("Loading locked clinical baseline table (read-only)...")
    clinical_baseline = load_clinical_baseline_table()
    cohort = validated_cohort(clinical_baseline)
    print(f"Validated ADNI cohort: {len(cohort)} participants")

    print("Loading UCBERKELEY_AMY_6MM (amyloid-PET Centiloid) from interim/...")
    pet_df = load_pet_table()

    print("Selecting near-baseline QC-passed Centiloid scans...")
    pet_baseline = adni_pet.build_pet_baseline(pet_df, cohort[["RID", "CLINICAL_BASELINE_DATE"]])

    eligibility = cohort.merge(pet_baseline, on="RID", how="left")
    eligibility = adni_pet.add_polaris_eligibility(eligibility)

    os.makedirs(ADNI_PROCESSED_DIR, exist_ok=True)
    processed_path = os.path.join(ADNI_PROCESSED_DIR, "adni_pet_eligibility.parquet")
    eligibility.to_parquet(processed_path, index=False)
    print(f"wrote {processed_path}  ({len(eligibility)} rows, participant-level, local-only)")

    attrition_df = build_cohort_attrition(eligibility)
    profile_df = build_population_profile(eligibility)
    metadata_md = build_eligibility_metadata_md()

    os.makedirs(ADNI_OUTPUTS_DIR, exist_ok=True)
    attrition_path = os.path.join(ADNI_OUTPUTS_DIR, "adni_polaris_cohort_attrition.csv")
    profile_path = os.path.join(ADNI_OUTPUTS_DIR, "adni_polaris_population_profile.csv")
    metadata_path = os.path.join(ADNI_OUTPUTS_DIR, "adni_polaris_eligibility_metadata.md")

    attrition_df.to_csv(attrition_path, index=False)
    profile_df.to_csv(profile_path, index=False)
    with open(metadata_path, "w", encoding="utf-8") as f:
        f.write(metadata_md)

    print(f"wrote {attrition_path}")
    print(f"wrote {profile_path}")
    print(f"wrote {metadata_path}")

    final_n = int(eligibility["POLARIS_ELIGIBLE"].sum())
    dx_counts = eligibility.loc[eligibility["POLARIS_ELIGIBLE"], "DX_BASELINE_FIXED"].value_counts()

    print()
    print("=== DONE (aggregate summary only) ===")
    print(f"Final POLARIS-aligned eligible cohort: {final_n}")
    print(f"  CN: {int(dx_counts.get('CN', 0))}")
    print(f"  MCI: {int(dx_counts.get('MCI', 0))}")
    print(f"  Dementia: {int(dx_counts.get('Dementia', 0))}")

    return eligibility, attrition_df, profile_df


if __name__ == "__main__":
    main()
