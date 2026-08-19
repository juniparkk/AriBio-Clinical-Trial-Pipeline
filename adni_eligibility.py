# ============================================================
# ADNI_ELIGIBILITY -- generalized target-population eligibility engine.
# Same participant-level-data/pure-function discipline as adni_pet.py /
# adni_cohort.py / adni_plasma.py: every function here is pure (no file
# I/O, no printing at import time), and no function returns anything
# intended for outputs/ that still carries a participant identifier --
# aggregation for outputs/ happens exclusively in
# run_adni_target_populations.py.
#
# This module generalizes the POLARIS-specific pattern in adni_pet.py /
# run_adni_pet_eligibility.py (one hardcoded eligibility rule, one
# hardcoded 5-step attrition funnel, one hardcoded Overall-vs-POLARIS
# profile) into a data-driven PRESET_LIBRARY: any number of named
# eligibility presets, each defined by a small declarative PresetSpec,
# evaluated against a single unified per-participant master eligibility
# table using only dimensions already computed elsewhere in this
# pipeline (diagnosis, MMSE, Centiloid, age, sex, biomarker
# availability) -- no new participant-level data, no new statistic, no
# new threshold logic beyond simple comparisons against fields that
# already exist.
#
# The `polaris_like` preset is a DESCRIPTIVE placeholder in
# PRESET_LIBRARY only -- run_adni_target_populations.py deliberately
# does NOT evaluate it through evaluate_preset()/build_preset_attrition()
# below. It reuses the already-approved POLARIS_ELIGIBLE flag and the
# already-approved adni_polaris_cohort_attrition.csv /
# adni_polaris_population_profile.csv outputs directly, so the
# already-validated POLARIS cohort (n=620) can never numerically drift
# from re-deriving the same rule through a different code path.
# ============================================================

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Preset definitions
# ------------------------------------------------------------------


@dataclass(frozen=True)
class PresetSpec:
    id: str
    label: str
    description: str
    # Empty tuple = no diagnosis restriction (all of CN/MCI/Dementia).
    diagnosis: tuple = ()
    mmse_min: float = None
    mmse_max: float = None
    centiloid_min: float = None
    age_min: float = None
    age_max: float = None
    # Column names on the master eligibility table, e.g. "HAS_PTAU217".
    require_biomarkers: tuple = ()
    is_polaris_equivalent: bool = False


# Column -> human-readable label, used in both attrition-step names and
# the population-profile's biomarker-availability rows.
BIOMARKER_COLUMNS = {
    "HAS_PTAU181": "pTau181",
    "HAS_PTAU217": "pTau217",
    "HAS_ABETA_RATIO": "Aβ42/Aβ40 ratio",
    "HAS_GFAP": "GFAP",
    "HAS_NFL": "NfL",
}

# All criteria are built strictly from fields already computed upstream
# in this pipeline (adni_pet_eligibility.parquet's DX_BASELINE_FIXED /
# MMSE_BASELINE / CENTILOID_BASELINE / CENTILOID_ELIGIBLE / BASELINE_AGE,
# plus the 5 biomarker families' BIOMARKER_ELIGIBLE flags joined in by
# build_master_eligibility_table()) -- no criterion here requires new
# data collection or a disease-stage granularity finer than
# CN/MCI/Dementia, which does not exist upstream.
PRESET_LIBRARY = [
    PresetSpec(
        id="polaris_like",
        label="Preset: Broad amyloid-confirmed cohort",
        description="Baseline MMSE ≥ 20 and QC-passed amyloid-PET Centiloid ≥ 30 within ±90 days of clinical baseline. No diagnosis restriction.",
        mmse_min=20,
        centiloid_min=30,
        is_polaris_equivalent=True,
    ),
    PresetSpec(
        id="mild_moderate_amyloid",
        label="Preset: Mild-to-moderate AD, amyloid-confirmed",
        description=(
            "MCI or Dementia at baseline, MMSE 14–26, amyloid-confirmed (Centiloid ≥ 30). "
            "The MMSE band is a general placeholder for a commonly-used mild-to-moderate AD trial "
            "range -- NOT a specific trial's actual protocol -- pending clinical review."
        ),
        diagnosis=("MCI", "Dementia"),
        mmse_min=14,
        mmse_max=26,
        centiloid_min=30,
    ),
    PresetSpec(
        id="mild_dementia_amyloid",
        label="Preset: Mild dementia, amyloid-confirmed",
        description=(
            "Dementia at baseline, MMSE 20–26, amyloid-confirmed (Centiloid ≥ 30). "
            "The MMSE band is a general mild-dementia placeholder, pending clinical review."
        ),
        diagnosis=("Dementia",),
        mmse_min=20,
        mmse_max=26,
        centiloid_min=30,
    ),
    PresetSpec(
        id="prodromal_mci_amyloid",
        label="Preset: Prodromal/MCI, amyloid-confirmed",
        description="MCI at baseline, MMSE ≥ 24, amyloid-confirmed (Centiloid ≥ 30).",
        diagnosis=("MCI",),
        mmse_min=24,
        centiloid_min=30,
    ),
    PresetSpec(
        id="biomarker_complete",
        label="Preset: Amyloid + full plasma panel available",
        description=(
            "MCI or Dementia at baseline, amyloid-confirmed (Centiloid ≥ 30), with both pTau217 "
            "and Aβ42/Aβ40 plasma measurements available. Small cohort -- likely to hit small-cell "
            "suppression at later follow-up months."
        ),
        diagnosis=("MCI", "Dementia"),
        centiloid_min=30,
        require_biomarkers=("HAS_PTAU217", "HAS_ABETA_RATIO"),
    ),
    PresetSpec(
        id="age_restricted_sensitivity",
        label="Preset: Amyloid-confirmed, age 65–85",
        description="MCI or Dementia at baseline, amyloid-confirmed (Centiloid ≥ 30), age 65–85 at baseline.",
        diagnosis=("MCI", "Dementia"),
        centiloid_min=30,
        age_min=65,
        age_max=85,
    ),
]

PRESET_BY_ID = {p.id: p for p in PRESET_LIBRARY}


# ------------------------------------------------------------------
# Master eligibility table
# ------------------------------------------------------------------


def build_master_eligibility_table(pet_eligibility_df, biomarker_eligible_rids):
    """
    Joins the already-approved adni_pet_eligibility.parquet (one row per
    validated-cohort RID: DX_BASELINE_FIXED, BASELINE_AGE, SEX,
    MMSE_BASELINE, CENTILOID_BASELINE, CENTILOID_ELIGIBLE,
    POLARIS_ELIGIBLE, ...) with new per-biomarker availability booleans.

    `biomarker_eligible_rids`: dict of {column_name: set_of_RIDs}, e.g.
    {"HAS_PTAU217": {12, 45, ...}, ...} -- one entry per
    BIOMARKER_COLUMNS key. A RID not in the set gets False, never NaN,
    since "no biomarker draw for this participant" is a real, known
    fact (absence of data), not missing information about a value that
    should otherwise exist.

    Recomputes nothing from adni_pet_eligibility.parquet itself -- every
    existing column (including POLARIS_ELIGIBLE) passes through
    unchanged.
    """
    out = pet_eligibility_df.copy()
    for col in BIOMARKER_COLUMNS:
        rids = biomarker_eligible_rids.get(col, set())
        out[col] = out["RID"].isin(rids)
    return out


# ------------------------------------------------------------------
# Preset evaluation
# ------------------------------------------------------------------


def evaluate_preset(master_df, preset):
    """Boolean membership Series over master_df's index. NaN in any
    compared field evaluates the criterion to False (pandas comparison
    semantics), never silently True -- a participant with unknown MMSE
    is never assumed to meet an MMSE criterion."""
    mask = pd.Series(True, index=master_df.index)
    if preset.diagnosis:
        mask &= master_df["DX_BASELINE_FIXED"].isin(preset.diagnosis)
    if preset.mmse_min is not None:
        mask &= master_df["MMSE_BASELINE"].notna() & (master_df["MMSE_BASELINE"] >= preset.mmse_min)
    if preset.mmse_max is not None:
        mask &= master_df["MMSE_BASELINE"].notna() & (master_df["MMSE_BASELINE"] <= preset.mmse_max)
    if preset.centiloid_min is not None:
        mask &= master_df["CENTILOID_ELIGIBLE"] & (master_df["CENTILOID_BASELINE"] >= preset.centiloid_min)
    if preset.age_min is not None:
        mask &= master_df["BASELINE_AGE"].notna() & (master_df["BASELINE_AGE"] >= preset.age_min)
    if preset.age_max is not None:
        mask &= master_df["BASELINE_AGE"].notna() & (master_df["BASELINE_AGE"] <= preset.age_max)
    for col in preset.require_biomarkers:
        mask &= master_df[col].fillna(False)
    return mask


# ------------------------------------------------------------------
# Attrition funnel -- generalized, data-driven version of
# run_adni_pet_eligibility.build_cohort_attrition(): an ordered list of
# named criteria, each narrowing the population by exactly one
# condition, so every exclusion is individually traceable. A distinct
# "<field> available" step precedes each threshold step for a field
# with real missingness (MMSE, Centiloid, biomarkers) -- so a
# participant excluded for lacking data is never counted the same way
# as one excluded for not meeting a threshold.
# ------------------------------------------------------------------


def final_step_label(preset):
    """"Final <preset name> cohort" -- appends "cohort" only if the
    preset's own label doesn't already end with that word (e.g. "Broad
    amyloid-confirmed cohort" would otherwise read "...cohort cohort").
    Shared by build_preset_attrition() below and
    run_adni_target_populations.py's polaris_like special case, so both
    paths use the identical rule."""
    name = preset.label.replace("Preset: ", "")
    return name if name.lower().endswith("cohort") else f"{name} cohort"


def build_preset_attrition(master_df, preset):
    steps = []
    pop = master_df
    start_n = len(pop)
    steps.append(("Validated ADNI cohort", start_n, start_n, 0))

    def _add_step(label, next_pop):
        nonlocal pop
        steps.append((label, len(pop), len(next_pop), len(pop) - len(next_pop)))
        pop = next_pop

    if preset.diagnosis:
        _add_step(f"Diagnosis in {{{', '.join(preset.diagnosis)}}}", pop[pop["DX_BASELINE_FIXED"].isin(preset.diagnosis)])

    if preset.mmse_min is not None or preset.mmse_max is not None:
        _add_step("Baseline MMSE available", pop[pop["MMSE_BASELINE"].notna()])
        if preset.mmse_min is not None:
            _add_step(f"MMSE ≥ {preset.mmse_min}", pop[pop["MMSE_BASELINE"] >= preset.mmse_min])
        if preset.mmse_max is not None:
            _add_step(f"MMSE ≤ {preset.mmse_max}", pop[pop["MMSE_BASELINE"] <= preset.mmse_max])

    if preset.centiloid_min is not None:
        _add_step("Amyloid PET (Centiloid) data available (QC-passed, within ±90 days of baseline)", pop[pop["CENTILOID_ELIGIBLE"]])
        _add_step(f"Centiloid ≥ {preset.centiloid_min}", pop[pop["CENTILOID_BASELINE"] >= preset.centiloid_min])

    if preset.age_min is not None or preset.age_max is not None:
        _add_step("Baseline age available", pop[pop["BASELINE_AGE"].notna()])
        if preset.age_min is not None:
            _add_step(f"Age ≥ {preset.age_min}", pop[pop["BASELINE_AGE"] >= preset.age_min])
        if preset.age_max is not None:
            _add_step(f"Age ≤ {preset.age_max}", pop[pop["BASELINE_AGE"] <= preset.age_max])

    for col in preset.require_biomarkers:
        _add_step(f"{BIOMARKER_COLUMNS.get(col, col)} available", pop[pop[col].fillna(False)])

    steps.append((f"Final {final_step_label(preset)}", len(pop), len(pop), 0))

    rows = []
    for label, starting_n, remaining_n, excluded_n in steps:
        pct = round(100.0 * remaining_n / start_n, 1) if start_n else 0.0
        rows.append({
            "step": label, "starting_n": starting_n, "remaining_n": remaining_n,
            "excluded_n": excluded_n, "percent_retained_of_cohort": pct,
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Population profile -- generalized version of
# run_adni_pet_eligibility.build_population_profile(): same
# (variable, level, population, n, mean, sd, median, percent) schema,
# purely descriptive (no p-value/test-statistic column), parameterized
# by an arbitrary target-population label and boolean mask instead of
# the hardcoded POLARIS_ELIGIBLE column.
# ------------------------------------------------------------------


def _numeric_summary(series):
    s = series.dropna()
    if len(s) == 0:
        return {"n": 0, "mean": np.nan, "sd": np.nan, "median": np.nan}
    return {"n": int(len(s)), "mean": float(s.mean()), "sd": float(s.std(ddof=1)) if len(s) > 1 else np.nan,
            "median": float(s.median())}


def build_preset_profile(master_df, target_mask, overall_label="Overall ADNI", target_label="Target Population"):
    overall = master_df
    target = master_df[target_mask]

    rows = []

    def add_numeric_rows(field, label):
        for pop_name, pop_df in ((overall_label, overall), (target_label, target)):
            s = _numeric_summary(pop_df[field])
            rows.append({"variable": label, "level": "", "population": pop_name, **s})

    add_numeric_rows("BASELINE_AGE", "Baseline age (years)")
    add_numeric_rows("MMSE_BASELINE", "Baseline MMSE")
    add_numeric_rows("ADAS_COG13_BASELINE", "Baseline ADAS-Cog13")
    add_numeric_rows("CENTILOID_BASELINE", "Baseline Centiloid")

    cat_rows = []

    def add_categorical_rows(field, label, pop_name, pop_df):
        total = len(pop_df)
        counts = pop_df[field].value_counts(dropna=False)
        for level, n in counts.items():
            level_label = "Missing" if pd.isna(level) else str(level)
            pct = round(100.0 * n / total, 1) if total else 0.0
            cat_rows.append({"variable": label, "level": level_label, "population": pop_name,
                              "n": int(n), "mean": np.nan, "sd": np.nan, "median": np.nan, "percent": pct})

    for pop_name, pop_df in ((overall_label, overall), (target_label, target)):
        add_categorical_rows("DX_BASELINE_FIXED", "Baseline diagnosis", pop_name, pop_df)
        add_categorical_rows("SEX", "Sex", pop_name, pop_df)
        add_categorical_rows("APOE4_CARRIER", "APOE4 carrier", pop_name, pop_df)
        for col, label in BIOMARKER_COLUMNS.items():
            if col not in pop_df.columns:
                continue
            add_categorical_rows(col, f"{label} available", pop_name, pop_df)

    numeric_df = pd.DataFrame(rows)
    if "percent" not in numeric_df.columns:
        numeric_df = numeric_df.assign(percent=np.nan)
    categorical_df = pd.DataFrame(cat_rows)
    cols = ["variable", "level", "population", "n", "mean", "sd", "median", "percent"]
    return pd.concat([numeric_df[cols], categorical_df[cols]], ignore_index=True)
