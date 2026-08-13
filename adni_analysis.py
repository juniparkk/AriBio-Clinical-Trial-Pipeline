# ============================================================
# ADNI (Alzheimer's Disease Neuroimaging Initiative) biomarker data
# analysis — kept entirely separate from pipeline_viz.py's
# ClinicalTrials.gov pipeline. These are two unrelated data sources
# that happen to feed the same dashboard: ADNI requires an approved
# LONI/ADNI data-use agreement and is never fetched automatically the
# way ClinicalTrials.gov data is (see ctgov_client.py) — there is
# nothing here that downloads, guesses, or fabricates ADNI data.
#
# The ADNI data audit (outputs/adni_data_audit.md under ADNI_ROOT) has
# been reviewed and approved -- ADNI_AUDIT_APPROVED is now True. Real
# ADNI files now live under ADNI_ROOT below, a directory *outside* this
# git repository on purpose: participant-level ADNI data must never be
# written to git, so raw/, interim/, and processed/ under ADNI_ROOT all
# stay local-only. See adni_cohort.py / adni_plasma.py for the
# preprocessing pipeline that reads from there and
# run_adni_preprocessing.py for the orchestration entry point. Only
# aggregate-only QC outputs (ADNI_ROOT/outputs/*.md, *.csv) and this
# project's own outputs/ dashboard code may ever reference ADNI
# findings in git-tracked files.
#
# Every function below still degrades honestly to "no data" rather
# than inventing a plausible-looking biomarker summary when a file is
# absent — same rule competitive_attention_viz.py's PLACEHOLDER
# follows: an honest empty state, not stale/wrong data.
#
# Like drug_classification.py and competitive_attention.py, this
# module only DEFINES functions — no file I/O or printing at import
# time.
# ============================================================

import os

import pandas as pd

# ADNI lives entirely outside this git repository (not a git repo
# itself) so that participant-level data is never at risk of being
# committed. All ADNI preprocessing modules import these paths rather
# than hardcoding them.
ADNI_ROOT = os.path.expanduser("~/Desktop/ADNI")
ADNI_RAW_CLINICAL_PKG = os.path.join(ADNI_ROOT, "raw", "clinical", "ADNIMERGE2")
ADNI_RAW_PLASMA_DIR = os.path.join(ADNI_ROOT, "raw", "plasma")
# R-exported flat CSVs of the raw .rda eCRF tables Python needs (see
# adni_export_raw.R) -- a format bridge only, never edited by hand,
# never committed, and never derived from anything but raw/ itself.
ADNI_INTERIM_DIR = os.path.join(ADNI_ROOT, "interim")
# Final analysis-ready longitudinal tables (participant-level) -- local
# only, gitignored, read by nothing outside this project's own scripts.
ADNI_PROCESSED_DIR = os.path.join(ADNI_ROOT, "processed")
# Aggregate-only QC/audit reports -- safe to read, safe to reference,
# never contains a participant ID or a participant-level row.
ADNI_OUTPUTS_DIR = os.path.join(ADNI_ROOT, "outputs")

# Where a flat single-file ADNI export (e.g. a copy of ADNIMERGE.csv)
# would be placed for the summarize_adni_data() path below, kept for
# backward compatibility with the dashboard's existing empty-state
# handling. The real preprocessing pipeline (adni_cohort.py /
# adni_plasma.py) reads the full raw/ package directly instead and
# does not use this path.
ADNI_DATA_PATH = "data/adni/adni_merge.csv"

# The ADNI data-use audit (verifying LONI/ADNI DUA compliance,
# column-schema review, and the aggregate-only / no-participant-level-
# data rule for this dashboard) has been reviewed and approved -- see
# ADNI_ROOT/outputs/adni_data_audit.md. Every analysis function below
# still checks this first before computing anything, so a future
# revocation only requires flipping this one switch back to False.
ADNI_AUDIT_APPROVED = True


def load_adni_data(path=ADNI_DATA_PATH):
    """
    Read an ADNI export from disk. Missing-file-tolerant, same pattern
    as load_official_pipeline()/load_scope_overrides() elsewhere in
    this project: a missing file means "no ADNI data configured yet",
    not a crash.

    Returns None (never an empty DataFrame) when unavailable, so
    callers can tell "no data source configured" apart from "data
    source configured but genuinely produced zero rows" without
    inspecting length.
    """
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, low_memory=False)


def summarize_adni_data(adni_df):
    """
    Top-line summary for the Biomarker Dashboard. Returns None when
    adni_df is None (see load_adni_data), so adni_viz.py can render
    its honest empty state instead of a broken/empty-looking dashboard.

    NOT YET IMPLEMENTED beyond a raw row/column count: this project has
    never seen a real ADNI file, so there is no verified column schema
    to summarize against. ADNIMERGE alone carries 100+ columns spanning
    demographics, MRI volumetrics, PET SUVRs, CSF Aβ42/tau/p-tau,
    cognitive batteries, and APOE genotype — which of those actually
    belong on this dashboard is a scope decision for whoever adds the
    first real file, not something to guess at here. Once real data
    lands, extend this the same way build_resolved_drugs_dataframe()
    etc. were built: pure functions, unit-tested against known,
    verified column names — never inferred from column names alone.
    """
    if adni_df is None:
        return None
    return {
        "row_count": len(adni_df),
        "column_count": len(adni_df.columns),
        "columns": list(adni_df.columns),
    }


# ------------------------------------------------------------------
# Longitudinal analysis / aggregate statistics — NOT YET IMPLEMENTED.
#
# These are the eventual homes for the real work (cognitive-composite
# and plasma-biomarker trajectories by diagnosis group, ANCOVA group
# comparisons, attrition by visit), but every one of them is gated on
# ADNI_AUDIT_APPROVED first, ahead of even checking whether adni_df is
# None. Two independent reasons, both must clear before either of
# these functions does real work:
#   1. No verified ADNIMERGE-equivalent column schema exists in this
#      project yet (see summarize_adni_data's docstring) — writing
#      group-by logic against guessed column names risks silently
#      computing something wrong and confidently wrong is worse than
#      visibly absent.
#   2. The ADNI data-use audit itself hasn't been signed off, so even
#      a correct computation over a real file isn't cleared for use.
#
# Callers (adni_viz.py) always get a clear "pending" signal back
# rather than a crash, so the dashboard can render an honest
# placeholder section instead of raising.
# ------------------------------------------------------------------


def compute_longitudinal_trajectories(adni_df):
    """
    Group-level (never participant-level) trajectories of cognitive
    composites and plasma biomarkers over visit/time, aggregated by
    diagnosis group. Powers the "Cognitive progression" and "Plasma
    biomarker progression" dashboard sections once implemented.

    Returns None until ADNI_AUDIT_APPROVED is True — see module
    docstring above.
    """
    if not ADNI_AUDIT_APPROVED:
        return None
    raise NotImplementedError(
        "compute_longitudinal_trajectories: audit approved, but no verified "
        "ADNI column schema exists yet to build this against."
    )


def compute_ancova_results(adni_df):
    """
    Group-comparison ANCOVA (adjusted mean differences, CIs, p-values)
    across diagnosis/treatment groups, covariates TBD once a verified
    column schema exists (likely baseline age, education, APOE
    status). Powers the "ANCOVA results" dashboard section.

    Returns None until ADNI_AUDIT_APPROVED is True — see module
    docstring above.
    """
    if not ADNI_AUDIT_APPROVED:
        return None
    raise NotImplementedError(
        "compute_ancova_results: audit approved, but no verified ADNI "
        "column schema exists yet to build this against."
    )


def compute_attrition_summary(adni_df):
    """
    Sample size by visit and cumulative attrition rate, aggregated
    only (counts, never participant identities). Powers the
    "Sample-size / attrition information" dashboard section.

    Returns None until ADNI_AUDIT_APPROVED is True — see module
    docstring above.
    """
    if not ADNI_AUDIT_APPROVED:
        return None
    raise NotImplementedError(
        "compute_attrition_summary: audit approved, but no verified ADNI "
        "column schema exists yet to build this against."
    )
