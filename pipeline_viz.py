# ============================================================
# AD PIPELINE VISUALIZATION
# Data source: clinicaltrials.gov (public, no restrictions)
#
# BEFORE RUNNING:
#   1. Go to clinicaltrials.gov
#   2. Search "Alzheimer's Disease"
#   3. Download results as CSV → save as trials.csv in this folder
#   4. Run: python pipeline_viz.py
# ============================================================

import json
import os
import re
from datetime import date

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import plotly.offline as pyo
from plotly.subplots import make_subplots

from drug_classification import (
    load_official_pipeline,
    build_interventions_dataframe,
    resolve_developed_drug,
    build_resolved_drugs_dataframe,
    build_unresolved_trials_dataframe,
    build_target_phase_counts,
    build_resolved_drug_trial_links_df,
    load_scope_overrides,
    build_scope_audit_dataframe,
    THERAPEUTIC_SCOPE,
)

# ============================================================
# AriBio brand colors + shade helpers — defined up front (rather than
# down in "STEP 4: COLORS" where they used to live) because STEP 3.7
# now also needs them to build the verification/confidence/review
# palettes. Every categorical color in this dashboard is now a shade of
# ARIBIO_BLUE, with ARIBIO_ACCENT used sparingly for a single "needs
# attention" value per group — EXCEPT TARGET_COLORS (pathway), which
# stays its own fully distinct rainbow so pathways stay tellable apart
# at a glance. This keeps the page reading as one coherent brand color
# instead of a different independent rainbow per chart/column.
# ============================================================
ARIBIO_ACCENT = "#c2255c"  # magenta/crimson from the "Ari" wordmark
ARIBIO_BLUE = "#2e5fa3"    # blue from the "Bio" wordmark — the base for every blue-ramp shade in this file


def darken(hex_color, amount=0.15):
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    r, g, b = (max(0, int(c * (1 - amount))) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def lighten(hex_color, amount):
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    r, g, b = (int(c + (255 - c) * amount) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


ARIBIO_BLUE_HOVER = darken(ARIBIO_BLUE, 0.18)
ARIBIO_ACCENT_HOVER = darken(ARIBIO_ACCENT, 0.18)
ARIBIO_BLUE_SUBTLE = lighten(ARIBIO_BLUE, 0.75)      # light blue text-on-dark tint (topbar subtitle)
ARIBIO_ACCENT_BG = lighten(ARIBIO_ACCENT, 0.93)       # pale pink spotlight card background
ARIBIO_ACCENT_BORDER = lighten(ARIBIO_ACCENT, 0.7)    # spotlight card border

# Neutral-but-branded UI surface tokens — every "near-white, faintly
# blue" background/border in the page (row hover, detail-panel fill,
# section dividers) is derived from ARIBIO_BLUE at a high lighten()
# value, rather than separately hand-picked grays that only
# coincidentally look blue-ish. One brand hue, one set of derived
# tints — a real design-system token approach instead of ad hoc hex.
SURFACE_TINT = lighten(ARIBIO_BLUE, 0.96)     # subtle panel/hover backgrounds
SURFACE_BORDER = lighten(ARIBIO_BLUE, 0.85)   # subtle dividers/borders on white
CARD_RADIUS = "12px"                          # one border-radius for every card in the system
CARD_SHADOW = "0 1px 3px rgba(20, 40, 70, 0.09)"           # resting, in-flow cards
ELEVATED_SHADOW = "0 6px 20px rgba(20, 40, 70, 0.14)"      # floating/fixed elements (sidebar)

# ============================================================
# STEP 1: LOAD THE DATA
# ============================================================

df = pd.read_csv("trials.csv", low_memory=False)

print("=== RAW DATA LOADED ===")
print(f"Total rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print()

# ============================================================
# STEP 2: CLEAN AND FILTER
# clinicaltrials.gov CSV columns vary slightly — we'll handle both
# old and new column naming conventions
# ============================================================

column_map = {
    "NCT Number": "nct_id",
    "nctId": "nct_id",
    "Title": "title",
    "briefTitle": "title",
    "Study Title": "title",
    "Status": "status",
    "overallStatus": "status",
    "Overall Status": "status",
    "Study Status": "status",
    "Phases": "phase",
    "phase": "phase",
    "Interventions": "interventions",
    "interventions": "interventions",
    "Enrollment": "enrollment",
    "enrollmentCount": "enrollment",
    "Sponsor": "sponsor",
    "leadSponsor": "sponsor",
    "Sponsor/Collaborators": "sponsor",
    "Conditions": "conditions",
    "conditions": "conditions",
    "Start Date": "start_date",
    "primaryCompletionDate": "start_date",
}

df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

# Keep only Phase 1, 2, and 3 trials
# CT.gov writes phases like "PHASE1", "PHASE2", "PHASE3", or "PHASE1|PHASE2"
def extract_highest_phase(phase_str):
    if pd.isna(phase_str):
        return None
    phase_str = str(phase_str).upper()
    if "3" in phase_str:
        return "Phase 3"
    elif "2" in phase_str:
        return "Phase 2"
    elif "1" in phase_str:
        return "Phase 1"
    return None

df["phase_clean"] = df["phase"].apply(extract_highest_phase)
df = df[df["phase_clean"].notna()].copy()

print(f"=== AFTER FILTERING TO PHASE 1/2/3: {len(df)} trials ===")
print(df["phase_clean"].value_counts())
print()

# Clean up status — match against clinicaltrials.gov's actual status enum first
# (substring matching alone is a trap: "ACTIVE_NOT_RECRUITING" contains the
# substring "RECRUIT", so a naive "RECRUIT in s" check before "ACTIVE" mis-files
# every active-but-closed trial — including AR1001's own Phase 3 — as Recruiting)
STATUS_MAP = {
    "RECRUITING": "Recruiting",
    "NOT_YET_RECRUITING": "Recruiting",
    "ENROLLING_BY_INVITATION": "Recruiting",
    "ACTIVE_NOT_RECRUITING": "Active",
    "COMPLETED": "Completed",
    "TERMINATED": "Discontinued",
    "WITHDRAWN": "Discontinued",
    "SUSPENDED": "Discontinued",
    "APPROVED_FOR_MARKETING": "FDA Approved",
    "UNKNOWN": "Unknown",
}

def clean_status(s):
    if pd.isna(s):
        return "Unknown"
    key = str(s).strip().upper().replace(" ", "_")
    return STATUS_MAP.get(key, "Other")

df["status_clean"] = df["status"].apply(clean_status)

# ============================================================
# STEP 3: CLASSIFY DRUG TYPE AND TARGET PATHWAY
#
# Two-tier approach:
#   1. KNOWN_COMPOUNDS — a curated lookup of ~150 named/code-named
#      compounds from the public AD pipeline literature (secretase
#      inhibitors, anti-amyloid/anti-tau antibodies, PET tracers,
#      symptomatic/psychiatric drugs, etc). This is what actually
#      moves the needle — most trials list a research code
#      ("BMS-708163", "KarXT") that no keyword search will catch.
#   2. TARGET_KEYWORDS — broad mechanism-word fallback on the
#      combined title + intervention text, for anything not in
#      the lookup.
# Anything matching neither stays "Other" — review/correct it in
# pipeline_annotated.csv.
# ============================================================

def guess_drug_type(intervention):
    if pd.isna(intervention):
        return "Unknown"
    parts = [p.strip() for p in str(intervention).split("|")]
    # drop placebo/comparator arms — classify by the actual study drug
    real_parts = [p for p in parts if "placebo" not in p.lower() and p.lower() != "other: no intervention"]
    if not real_parts:
        real_parts = parts

    type_prefixes = [p.split(":", 1)[0].strip().upper() for p in real_parts if ":" in p]
    text = " ".join(real_parts).lower()

    if any(kw in text for kw in ["stem cell", "mesenchymal", "cell therapy", "cord blood"]):
        return "Cell/Gene Therapy"
    if "GENETIC" in type_prefixes or "gene therapy" in text or "viral vector" in text:
        return "Cell/Gene Therapy"
    if any(kw in text for kw in ["mab", "umab", "zumab", "nemab", "antibody", "immunoglobulin",
                                   "vaccine", "immunotherapy"]):
        return "Biologic"
    if "BIOLOGICAL" in type_prefixes:
        return "Biologic"
    if "DEVICE" in type_prefixes:
        return "Device"
    if "DIETARY_SUPPLEMENT" in type_prefixes:
        return "Dietary Supplement"
    if type_prefixes and all(p in ("BEHAVIORAL", "PROCEDURE", "DIAGNOSTIC_TEST", "RADIATION", "OTHER") for p in type_prefixes):
        return "Non-Drug/Behavioral"
    if "DRUG" in type_prefixes or not type_prefixes:
        return "Small Molecule"
    return "Other"


# --- Tier 1: curated named/code-named compound lookup ---
KNOWN_COMPOUNDS = {
    # --- Amyloid: BACE / gamma-secretase inhibitors (USAN stem "-cestat" covers both) ---
    "bms-708163": "Amyloid", "avagacestat": "Amyloid", "verubecestat": "Amyloid",
    "mk-8931": "Amyloid", "mk8931": "Amyloid", "lanabecestat": "Amyloid", "azd3293": "Amyloid",
    "elenbecestat": "Amyloid", "e2609": "Amyloid", "atabecestat": "Amyloid",
    "semagacestat": "Amyloid", "ly450139": "Amyloid", "begacestat": "Amyloid",
    "gsi-953": "Amyloid", "umibecestat": "Amyloid", "cnp520": "Amyloid",
    "ly2886721": "Amyloid", "jnj-54861911": "Amyloid",
    # --- Amyloid: anti-amyloid antibodies / vaccines ---
    "bapineuzumab": "Amyloid", "aab-003": "Amyloid", "pf-05236812": "Amyloid",
    "solanezumab": "Amyloid", "gantenerumab": "Amyloid", "crenezumab": "Amyloid",
    "aducanumab": "Amyloid", "lecanemab": "Amyloid", "donanemab": "Amyloid",
    "remternetug": "Amyloid", "trontinemab": "Amyloid", "ponezumab": "Amyloid",
    "pf-04360365": "Amyloid", "acc-001": "Amyloid", "cad106": "Amyloid",
    "abvac40": "Amyloid", "tb006": "Amyloid", "shr-1707": "Amyloid", "khk6640": "Amyloid",
    "qs-21": "Amyloid", "ub-311": "Amyloid", "affitope": "Amyloid", "lu af20513": "Amyloid",
    "pq912": "Amyloid", "varoglutamstat": "Amyloid", "ngp 555": "Amyloid", "ngp555": "Amyloid",
    "sar228810": "Amyloid", "mabt5102a": "Amyloid", "pf-04494700": "Amyloid", "azeliragon": "Amyloid",
    "bms-984923": "Amyloid", "lx1001": "Amyloid",
    # --- Amyloid: aggregation inhibitors / modulators ---
    "elnd005": "Amyloid", "scyllo-inositol": "Amyloid", "ct1812": "Amyloid",
    "tramiprosate": "Amyloid", "alz-801": "Amyloid", "3aps": "Amyloid",
    "chf 5074": "Amyloid", "chf5074": "Amyloid", "csp-1103": "Amyloid",
    "mpc-7869": "Amyloid", "tarenflurbil": "Amyloid", "pbt2": "Amyloid",
    "gv-971": "Amyloid", "sodium oligomannate": "Amyloid", "simufilam": "Amyloid",
    # --- Amyloid: amyloid PET tracers ---
    "florbetapir": "Amyloid", "florbetaben": "Amyloid", "flutemetamol": "Amyloid",
    "av-45": "Amyloid", "av45": "Amyloid", "azd4694": "Amyloid", "nav4694": "Amyloid",

    # --- Tau: antibodies / aggregation inhibitors / ASO ---
    "gosuranemab": "Tau", "biib092": "Tau", "tilavonemab": "Tau", "abbv-8e12": "Tau",
    "semorinemab": "Tau", "ro7105705": "Tau", "zagotenemab": "Tau", "ly3303560": "Tau",
    "bepranemab": "Tau", "ucb0107": "Tau", "e2814": "Tau", "jnj-63733657": "Tau",
    "biib080": "Tau", "nio752": "Tau", "pnt001": "Tau", "trx0014": "Tau",
    "lmtm": "Tau", "hydromethylthionine": "Tau", "tideglusib": "Tau", "spg302": "Tau",
    "apn-1607": "Tau", "thk-5351": "Tau", "thk5351": "Tau", "trx0037": "Tau",
    "abbv-1758": "Tau", "asn51": "Tau",
    # --- Tau PET tracers ---
    "flortaucipir": "Tau", "av-1451": "Tau", "av1451": "Tau", "gtp1": "Tau",
    "mk-6240": "Tau", "mk6240": "Tau", "pi-2620": "Tau", "pi2620": "Tau", "mni-187": "Tau",

    # --- Inflammation ---
    "etanercept": "Inflammation", "sargramostim": "Inflammation", "minocycline": "Inflammation",
    "al002": "Inflammation", "al003": "Inflammation",
    "pbr28": "Inflammation", "fedaa1106": "Inflammation", "dpa713": "Inflammation",
    "xpro1595": "Inflammation", "gsk2647544": "Inflammation", "ntrx-07": "Inflammation",
    "naproxen": "Inflammation",

    # --- Neuroprotection ---
    "dimebon": "Neuroprotection", "latrepirdine": "Neuroprotection",
    "cerebrolysin": "Neuroprotection", "t-817ma": "Neuroprotection",
    "bryostatin": "Neuroprotection", "posiphen": "Neuroprotection",
    "buntanetap": "Neuroprotection", "anavex2-73": "Neuroprotection",
    "blarcamesine": "Neuroprotection", "nilotinib": "Neuroprotection",
    "bexarotene": "Neuroprotection", "cilostazol": "Neuroprotection",
    "tadalafil": "Neuroprotection", "sildenafil": "Neuroprotection",
    "mirodenafil": "Neuroprotection", "ar1001": "Neuroprotection",
    "bpn14770": "Neuroprotection", "ath-1017": "Neuroprotection",
    "fosgonimeton": "Neuroprotection", "st101": "Neuroprotection",
    "allopregnanolone": "Neuroprotection", "dasatinib": "Neuroprotection",
    "quercetin": "Neuroprotection", "estrogen": "Neuroprotection", "gv1001": "Neuroprotection",
    "mem 1003": "Neuroprotection", "xaliproden": "Neuroprotection", "pf-04447943": "Neuroprotection",

    # --- Metabolism ---
    "rosiglitazone": "Metabolism", "pioglitazone": "Metabolism", "metformin": "Metabolism",
    "liraglutide": "Metabolism", "semaglutide": "Metabolism", "exenatide": "Metabolism",
    "t3d-959": "Metabolism", "simvastatin": "Metabolism", "atorvastatin": "Metabolism",
    "nicotinamide": "Metabolism", "mib-626": "Metabolism", "tricaprilin": "Metabolism",
    "ac-1202": "Metabolism",

    # --- Symptomatic: cholinesterase inhibitors / NMDA / nicotinic cognitive enhancers ---
    "donepezil": "Symptomatic", "aricept": "Symptomatic", "e2020": "Symptomatic", "galantamine": "Symptomatic",
    "rivastigmine": "Symptomatic", "memantine": "Symptomatic", "tacrine": "Symptomatic",
    "huperzine": "Symptomatic", "octohydroaminoacridine": "Symptomatic",
    "abt-089": "Symptomatic", "sam-531": "Symptomatic", "gsk239512": "Symptomatic",
    "abt-126": "Symptomatic", "azd3480": "Symptomatic", "bi 409306": "Symptomatic",
    "bi409306": "Symptomatic", "rasagiline": "Symptomatic", "jnj-39393406": "Symptomatic",
    "evp-6124": "Symptomatic", "sb-742457": "Symptomatic", "lecozotan": "Symptomatic",
    "talsaclidine": "Symptomatic", "htl0009936": "Symptomatic", "ac-3933": "Symptomatic",
    "pf-05212377": "Symptomatic",

    # --- Neuropsychiatric ---
    "karxt": "Neuropsychiatric", "xanomeline": "Neuropsychiatric", "trospium": "Neuropsychiatric",
    "brexpiprazole": "Neuropsychiatric", "pimavanserin": "Neuropsychiatric",
    "risperidone": "Neuropsychiatric", "daridorexant": "Neuropsychiatric",
    "lemborexant": "Neuropsychiatric", "suvorexant": "Neuropsychiatric", "iti-007": "Neuropsychiatric",
}

# --- Tier 2: broad mechanism-keyword fallback ---
TARGET_KEYWORDS = {
    "Amyloid": ["amyloid", "abeta", "a-beta", "aβ", "plaque", "secretase", "cestat", "fibril", "oligomer"],
    "Tau": ["tau", "tangle", "ptau", "mapt", "neurofibrillary"],
    "Inflammation": ["inflam", "neuroinflamm", "microglia", "tnf", "cytokine", "complement",
                      "nlrp3", "csf1r", "trem2", "immune"],
    "Neuroprotection": ["neuroprotect", "neurotroph", "mitochondri", "oxidative stress",
                         "pde5", "pde4", "pde9", "bdnf"],
    "Metabolism": ["insulin", "glp-1", "glp1", "metabol", "glucose", "diabet", "ppar"],
    "Symptomatic": ["acetylcholin", "cholinester", "nmda", "glutamate", "nicotinic"],
    "Neuropsychiatric": ["sleep", "agitat", "depress", "anxiet", "psychi", "psychosis", "apathy", "sundowning"],
}
TARGET_ORDER = ["Amyloid", "Tau", "Inflammation", "Neuroprotection", "Metabolism", "Symptomatic", "Neuropsychiatric"]


def guess_target(title, intervention):
    intervention_text = str(intervention).lower()
    combined = str(title).lower() + " " + intervention_text

    # Tier 1 — check each named compound in the KNOWN_COMPOUNDS dict against
    # the intervention field (more precise than scanning the free-text title)
    for compound, pathway in KNOWN_COMPOUNDS.items():
        if compound in intervention_text or compound in combined:
            return pathway

    # Tier 2 — mechanism keywords over title + intervention text
    for pathway in TARGET_ORDER:
        if any(kw in combined for kw in TARGET_KEYWORDS[pathway]):
            return pathway

    return "Other"


intervention_col = "interventions" if "interventions" in df.columns else None
title_col = "title" if "title" in df.columns else df.columns[1]

if intervention_col:
    df["drug_type"] = df[intervention_col].apply(guess_drug_type)
    df["target"] = df.apply(lambda r: guess_target(r[title_col], r[intervention_col]), axis=1)
else:
    df["drug_type"] = "Unknown"
    df["target"] = df[title_col].apply(lambda t: guess_target(t, ""))

# Flag AR1001 (AriBio's drug) — match on title, sponsor, or intervention name
df["is_aribio"] = (
    df[title_col].str.contains("AR1001|mirodenafil|aribio", case=False, na=False)
    | df["sponsor"].str.contains("aribio", case=False, na=False)
    | df[intervention_col].str.contains("AR1001|mirodenafil", case=False, na=False)
)

print("=== DRUG TYPE BREAKDOWN ===")
print(df["drug_type"].value_counts())
print()
print("=== TARGET PATHWAY BREAKDOWN ===")
target_counts_preview = df["target"].value_counts()
print(target_counts_preview)
other_pct = 100 * target_counts_preview.get("Other", 0) / len(df)
print(f"'Other' = {other_pct:.1f}% of trials (was 71.6% before reclassification)")
print()
print(f"=== AR1001 ROWS FOUND: {df['is_aribio'].sum()} ===")
print()

# ============================================================
# STEP 3.5 REMOVED (Phase 0 data-source consolidation).
# This used to build `legacy_drugs_df` via primary_intervention_name()
# (picks the FIRST "DRUG:"/"BIOLOGICAL:" entry in a trial and discards
# every sibling intervention) → clean_drug_name() → canonical_drug_key()
# (substring match against KNOWN_COMPOUNDS). It was the sole remaining
# source for the heatmap and the Phase 3 leaderboard; every other
# dashboard component (KPI tiles, the visible table, pipeline_drugs.csv)
# had already migrated to `resolved_drugs_df` in an earlier checkpoint.
# Per MIGRATION_PLAN.md Phase 0, the heatmap and leaderboard are now
# migrated too (see STEP 5.5 below), which leaves this entire legacy
# chain — primary_intervention_name(), canonical_drug_key(),
# summarize_drug(), mode_or_first(), df["primary_drug_raw"]/
# ["primary_drug_clean"]/["drug_key"], legacy_drugs_df itself — with NO
# remaining consumers anywhere in this file (confirmed by grep before
# removal). Deleted rather than left as dead code.
# ============================================================

# ============================================================
# STEP 3.6: PER-INTERVENTION CLASSIFICATION (the classify_intervention()/
# resolve_developed_drug()-based pipeline). Produces developed_drug,
# drug_classification, and related columns on `df`, used below by
# STEP 3.7 to build resolved_drugs_df — the one drug-level source of
# truth for every dashboard component (see STEP 3.75/3.8).
# ============================================================

pipeline_records = load_official_pipeline("data/official_pipeline.csv")
scope_overrides = load_scope_overrides("data/reference/intervention_scope_overrides.csv")

interventions_df = build_interventions_dataframe(df, pipeline_records, scope_overrides)

print("=== PER-INTERVENTION CLASSIFICATION ===")
print(f"{len(interventions_df)} individual interventions parsed from {len(df)} trials")
print(interventions_df["classification"].value_counts())
print()
print("=== PER-INTERVENTION PIPELINE SCOPE (Phase 1A) ===")
print(interventions_df["pipeline_scope"].value_counts())
print(f"{len(scope_overrides)} curated override(s) loaded from data/reference/intervention_scope_overrides.csv")
print()

_resolved_records = []
for _nct_id, _group in interventions_df.groupby("nct_id", sort=False):
    _resolved = resolve_developed_drug(_group.to_dict("records"))
    _resolved["nct_id"] = _nct_id
    _resolved_records.append(_resolved)
resolved_drug_df = pd.DataFrame(_resolved_records)

df = df.merge(resolved_drug_df, on="nct_id", how="left")

# trials with no parsed interventions at all (blank/NaN Interventions
# cell) won't have a matching row in resolved_drug_df — fill those in
# as "no candidate" rather than leaving NaN
_NEW_COLUMN_DEFAULTS = {
    "developed_drug": "",
    "developed_drug_normalized": "",
    "drug_classification": "no_therapeutic_candidate",
    "classification_reason": "trial has no parsed interventions",
    "official_pipeline_match": False,
    "official_source_url": "",
    "verification_status": "not_applicable",
    "classification_confidence": "high",
    "needs_manual_review": False,
    "pipeline_scope": "Exclude",
    "scope_reason": "trial has no parsed interventions",
    "scope_method": "rule_type",
    "scope_confidence": "high",
    "manual_review_required": False,
}
for _col, _default in _NEW_COLUMN_DEFAULTS.items():
    df[_col] = df[_col].fillna(_default)

print("=== DEVELOPED-DRUG RESOLUTION (per trial) ===")
print(df["drug_classification"].value_counts())
print(f"{df['needs_manual_review'].sum()} trials flagged needs_manual_review")
print()

# ============================================================
# STEP 3.7: RESOLVED DRUG-LEVEL ROLLUP — the ONE drug-level source of
# truth for the whole dashboard as of Phase 0 (data-source
# consolidation): the visible HTML/JS drug table, pipeline_drugs.csv,
# the KPI tiles, the heatmap, the Phase 3 leaderboard, and the
# drug-type/target pies all derive from this dataframe now — no
# component computes its own separate drug-level rollup anymore.
# ============================================================

resolved_drugs_df = build_resolved_drugs_dataframe(df)

# AR1001 mechanistically touches multiple pathways — special case for
# this dashboard's Target/Pathway column display.
resolved_drugs_df["target_display"] = resolved_drugs_df.apply(
    lambda r: "Multi (Amyloid/Tau/Neuroprotection)" if r["is_aribio"] else r["target"], axis=1
)

resolved_drugs_df["study_url"] = "https://clinicaltrials.gov/study/" + resolved_drugs_df["nct_id"]

# Human-readable labels for the raw internal enum values — the UI must
# never show a raw value like "sponsor_developed_therapeutic" or
# "pipeline_record_match_without_source". Every value
# classify_intervention()/resolve_developed_drug()/build_resolved_drugs_dataframe()
# can actually produce is mapped explicitly (verified against the real
# regenerated pipeline_drugs.csv); anything unrecognized falls back to
# "Manual review required" — NEVER to "Confirmed", since silently
# treating an unknown value as confirmed would be actively misleading.
VERIFICATION_LABELS = {
    "confirmed_official_match": "Confirmed official match",
    "pipeline_record_match_without_source": "Pipeline match; source needed",
    "mixed": "Mixed evidence",
    "no_match": "Unverified investigational",
    "ambiguous_multiple_matches": "Manual review required",
}
DEFAULT_VERIFICATION_LABEL = "Manual review required"

CONFIDENCE_LABELS = {"high": "High", "medium": "Medium", "low": "Low"}
DEFAULT_CONFIDENCE_LABEL = "Low"

resolved_drugs_df["verification_label"] = (
    resolved_drugs_df["verification_status"].map(VERIFICATION_LABELS).fillna(DEFAULT_VERIFICATION_LABEL)
)
resolved_drugs_df["confidence_label"] = (
    resolved_drugs_df["classification_confidence"].map(CONFIDENCE_LABELS).fillna(DEFAULT_CONFIDENCE_LABEL)
)
resolved_drugs_df["review_label"] = resolved_drugs_df["needs_manual_review"].map(
    {True: "Needs manual review", False: "Reviewed / no review needed"}
)

# synonyms from data/official_pipeline.csv, so searching "BAN2401" still
# finds the "Lecanemab" row — not invented, just surfaced from data
# already loaded above
_synonyms_by_drug_name = {r["drug_name"]: r["synonyms"] for r in pipeline_records}
resolved_drugs_df["synonyms"] = resolved_drugs_df["display_name"].map(
    lambda name: "; ".join(_synonyms_by_drug_name.get(name, []))
)


def _sponsor_display(sponsor_field):
    # compact visible cell text: first sponsor + "+N more" when a drug
    # has multiple distinct sponsors. The FULL list is always preserved
    # separately (title tooltip + row detail panel) — never silently
    # truncated, per the requirement that multi-sponsor rows stay
    # visibly flagged rather than implying single ownership.
    parts = [p for p in str(sponsor_field).split("; ") if p]
    if len(parts) <= 1:
        return sponsor_field
    return f"{parts[0]} +{len(parts) - 1} more"


resolved_drugs_df["sponsor_display"] = resolved_drugs_df["sponsor"].apply(_sponsor_display)

# ============================================================
# PHASE 1A: therapeutic_drugs_df — the DEFAULT dashboard population.
# resolved_drugs_df (above) stays the one drug-level source of truth and
# keeps every scope Phase 1A didn't outright exclude (Therapeutic Drug,
# Diagnostic Agent, Non-Drug Intervention, Supportive Treatment, Needs
# Review), so the dashboard's optional "reveal non-therapeutic records"
# filter has real data to show. Components that must show ONLY actual
# therapeutic drugs (KPI tiles, heatmap, Phase 3 leaderboard, drug-type/
# target pies, and the table's default filter state) read THIS narrower
# view instead.
# ============================================================
therapeutic_drugs_df = resolved_drugs_df[resolved_drugs_df["pipeline_scope"] == THERAPEUTIC_SCOPE].copy()

print("=== RESOLVED DRUG ROLLUP (new table source) ===")
print(f"{len(resolved_drugs_df)} total resolved drug rows ({len(therapeutic_drugs_df)} Therapeutic Drug / "
      f"{len(resolved_drugs_df) - len(therapeutic_drugs_df)} other scope)")
print(resolved_drugs_df["verification_label"].value_counts())
print("pipeline_scope breakdown:")
print(resolved_drugs_df["pipeline_scope"].value_counts())
print()

# ============================================================
# STEP 3.75: NAMED, GRANULARITY-EXPLICIT DATASETS
# trials_df: one row per unique NCT ID (trial-level) — an explicit,
# clearly-named alias for `df` at this point in the pipeline, where it
# has stabilized (every remaining assignment below adds columns to
# individual dashboard components, not new/reassigned rows).
# resolved_drug_trial_links_df: the explicit drug<->trial join table —
# one row per (canonical drug, contributing trial) pair, rather than an
# implicit semicolon-joined string every caller has to re-split.
# ============================================================

trials_df = df

resolved_drug_trial_links_df = build_resolved_drug_trial_links_df(resolved_drugs_df)

# ============================================================
# STEP 3.8: DATA RECONCILIATION REPORT (temporary, Phase 0 only)
# One place to see that every number downstream traces back
# consistently: raw trials -> parsed interventions -> classified
# (therapeutic / non-therapeutic / unresolved) -> resolved canonical
# drugs. This exists specifically because Phase 0 just changed which
# dashboard components read from which dataset — it's the sanity check
# that consolidating onto resolved_drugs_df didn't silently drop or
# duplicate anything relative to the old dual-source setup.
# ============================================================

_therapeutic_labels = ["sponsor_developed_therapeutic", "investigational_therapeutic_unverified"]
_unresolved_labels = ["uncertain"]
_intervention_classification_counts = interventions_df["classification"].value_counts()
_therapeutic_record_count = int(_intervention_classification_counts.reindex(_therapeutic_labels).fillna(0).sum())
_unresolved_record_count = int(_intervention_classification_counts.reindex(_unresolved_labels).fillna(0).sum())
_non_therapeutic_record_count = len(interventions_df) - _therapeutic_record_count - _unresolved_record_count

print("=== DATA RECONCILIATION ===")
print(f"unique raw trials: {trials_df['nct_id'].nunique()}")
print(f"raw intervention records: {len(interventions_df)}")
print(f"resolved canonical drugs: {len(resolved_drugs_df)}")
print(f"therapeutic drugs (intervention-level: sponsor_developed_therapeutic + investigational_therapeutic_unverified): {_therapeutic_record_count}")
print(f"non-therapeutic records (placebo/diagnostic/procedure/device/behavioral/comparator/other): {_non_therapeutic_record_count}")
print(f"unresolved records (uncertain): {_unresolved_record_count}")
print(f"  [check: therapeutic + non-therapeutic + unresolved == raw intervention records? "
      f"{_therapeutic_record_count + _non_therapeutic_record_count + _unresolved_record_count == len(interventions_df)}]")
print(f"Phase 1 drugs: {int((resolved_drugs_df['phase_reached'] == 'Phase 1').sum())}")
print(f"Phase 2 drugs: {int((resolved_drugs_df['phase_reached'] == 'Phase 2').sum())}")
print(f"Phase 3 drugs: {int((resolved_drugs_df['phase_reached'] == 'Phase 3').sum())}")
print("sum by target/pathway (resolved drugs):")
print(resolved_drugs_df["target"].value_counts().to_string())
print("sum by drug type (resolved drugs):")
print(resolved_drugs_df["drug_type"].value_counts().to_string())
print(f"[supplementary] resolved_drug_trial_links_df: {len(resolved_drug_trial_links_df)} drug<->trial links "
      f"across {resolved_drug_trial_links_df['nct_id'].nunique()} distinct contributing trials")
print()

# --- Phase 1A extension: pipeline_scope breakdown, both at the raw
# intervention level (every parsed intervention, regardless of whether it
# ever became a developed_drug candidate) and at the resolved-drug level
# (resolved_drugs_df, which now excludes "Exclude"/"Placebo or Comparator"
# scope trials entirely — see build_resolved_drugs_dataframe). ---
_scope_counts = interventions_df["pipeline_scope"].value_counts()

def _scope_count(label):
    return int(_scope_counts.get(label, 0))

print("=== PHASE 1A — PIPELINE SCOPE RECONCILIATION ===")
print(f"all resolved drug records (resolved_drugs_df): {len(resolved_drugs_df)}")
print(f"  therapeutic drugs (pipeline_scope == 'Therapeutic Drug'): {len(therapeutic_drugs_df)}")
print(f"  diagnostic agents: {int((resolved_drugs_df['pipeline_scope'] == 'Diagnostic Agent').sum())}")
print(f"  non-drug interventions: {int((resolved_drugs_df['pipeline_scope'] == 'Non-Drug Intervention').sum())}")
print(f"  supportive treatments: {int((resolved_drugs_df['pipeline_scope'] == 'Supportive Treatment').sum())}")
print(f"  needs-review records: {int((resolved_drugs_df['pipeline_scope'] == 'Needs Review').sum())}")
print(f"raw intervention records by pipeline_scope (interventions_df, {len(interventions_df)} total):")
print(f"  Therapeutic Drug: {_scope_count('Therapeutic Drug')}")
print(f"  Diagnostic Agent: {_scope_count('Diagnostic Agent')}")
print(f"  Non-Drug Intervention: {_scope_count('Non-Drug Intervention')}")
print(f"  Supportive Treatment: {_scope_count('Supportive Treatment')}")
print(f"  Placebo or Comparator: {_scope_count('Placebo or Comparator')}")
print(f"  Exclude: {_scope_count('Exclude')}")
print(f"  Needs Review: {_scope_count('Needs Review')}")
print("raw intervention records by ClinicalTrials.gov intervention type:")
print(interventions_df["original_type"].value_counts(dropna=False).to_string())
print(f"[check: scope counts sum to raw intervention records? "
      f"{sum(_scope_count(l) for l in ['Therapeutic Drug','Diagnostic Agent','Non-Drug Intervention','Supportive Treatment','Placebo or Comparator','Exclude','Needs Review']) == len(interventions_df)}]")
print(f"records removed from the default Therapeutic Drug view (resolved_drugs_df rows with pipeline_scope != 'Therapeutic Drug'): "
      f"{len(resolved_drugs_df) - len(therapeutic_drugs_df)}")
print()

scope_audit_df = build_scope_audit_dataframe(interventions_df)
os.makedirs("outputs", exist_ok=True)
scope_audit_df.to_csv("outputs/classification_gap_audit.csv", index=False)
print(f"=== SAVED: outputs/classification_gap_audit.csv ({len(scope_audit_df)} distinct intervention name/type records) ===")
print()

# ============================================================
# STEP 4: COLORS
# (brand colors + darken()/lighten() helpers now live near the top of
# the file, right after the imports — STEP 3.7 needs them too)
# ============================================================

PHASE_COLORS = {
    "Phase 3": darken(ARIBIO_BLUE, 0.35),
    "Phase 2": ARIBIO_BLUE,
    "Phase 1": lighten(ARIBIO_BLUE, 0.45),
}

# The one deliberately-distinct categorical palette left — pathway is
# worth telling apart at a glance in the pie/heatmap, so it keeps its
# own colors rather than folding into the blue ramp.
TARGET_COLORS = {
    "Amyloid":          "#e53935",
    "Tau":              "#8e24aa",
    "Inflammation":     "#f4511e",
    "Neuroprotection":  "#00897b",
    "Metabolism":       "#f9a825",
    "Symptomatic":      "#1e88e5",
    "Neuropsychiatric": "#6d4c41",
    "Other":            "#9e9e9e",
    "Unknown":          "#bdbdbd",
}

DRUG_TYPE_COLORS = {
    "Biologic":            darken(ARIBIO_BLUE, 0.30),
    "Small Molecule":      ARIBIO_BLUE,
    "Cell/Gene Therapy":   lighten(ARIBIO_BLUE, 0.25),
    "Dietary Supplement":  lighten(ARIBIO_BLUE, 0.50),
    "Device":              lighten(ARIBIO_BLUE, 0.65),
    "Non-Drug/Behavioral": "#9e9e9e",
    "Unknown":             "#bdbdbd",
    "Other":               ARIBIO_ACCENT,
}

STATUS_COLORS = {
    "FDA Approved":  darken(ARIBIO_BLUE, 0.35),
    "Active":        darken(ARIBIO_BLUE, 0.12),
    "Recruiting":    ARIBIO_BLUE,
    "Completed":     lighten(ARIBIO_BLUE, 0.35),
    "Unknown":       "#bdbdbd",
    "Other":         "#9e9e9e",
    "Discontinued":  ARIBIO_ACCENT,
}

# --- Compute counts for pie charts ---
# Phase 0 data-source consolidation: "By Drug Type" and "By Target
# Pathway" become genuinely drug-level — their titles never said "trial"
# in the first place, so this actually makes them match what they always
# claimed to show. "By Phase" and "By Trial Status" stay trial-level
# (trials_df) deliberately: "how many TRIALS are at each phase/status" is
# a real, different metric from the KPI tiles' "how many DRUGS have
# reached each phase" — not an inconsistency to fix, a distinct metric
# worth keeping. Subplot titles below say which basis each pie uses.
#
# Phase 1A: "By Drug Type"/"By Target Pathway" further narrow from
# resolved_drugs_df to therapeutic_drugs_df (pipeline_scope == "Therapeutic
# Drug" only) — per the requirement that these two pies "must use only
# Therapeutic Drug records". They are NOT the same population as the
# table's full JSON payload anymore (that stays the broader
# resolved_drugs_df so the table's optional filter can reveal the rest).
phase_counts = trials_df["phase_clean"].value_counts()
type_counts = therapeutic_drugs_df["drug_type"].value_counts()
target_counts = therapeutic_drugs_df["target"].value_counts()
status_counts = trials_df["status_clean"].value_counts()

# ============================================================
# STEP 5: BUILD THE 4-PIE FIGURE
# ============================================================

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=["By Phase (trials)", "By Drug Type (therapeutic drugs)", "By Target Pathway (therapeutic drugs)", "By Trial Status (trials)"],
    specs=[[{"type": "pie"}, {"type": "pie"}],
           [{"type": "pie"}, {"type": "pie"}]]
)

# Trace order matters: it's used client-side (in the injected JS below) to map
# a clicked pie slice back to the dataframe column it filters on.
fig.add_trace(go.Pie(
    labels=phase_counts.index.tolist(),
    values=phase_counts.values.tolist(),
    marker_colors=[PHASE_COLORS.get(p, "#999") for p in phase_counts.index],
    name="Phase", hole=0.35, textinfo="label+percent",
    hovertemplate="<b>%{label}</b><br>%{value} trials<br>%{percent}<extra></extra>"
), row=1, col=1)

fig.add_trace(go.Pie(
    labels=type_counts.index.tolist(),
    values=type_counts.values.tolist(),
    marker_colors=[DRUG_TYPE_COLORS.get(t, "#999") for t in type_counts.index],
    name="Drug Type", hole=0.35, textinfo="label+percent",
    hovertemplate="<b>%{label}</b><br>%{value} therapeutic drugs<br>%{percent}<extra></extra>"
), row=1, col=2)

fig.add_trace(go.Pie(
    labels=target_counts.index.tolist(),
    values=target_counts.values.tolist(),
    marker_colors=[TARGET_COLORS.get(t, "#999") for t in target_counts.index],
    name="Target", hole=0.35, textinfo="label+percent",
    hovertemplate="<b>%{label}</b><br>%{value} therapeutic drugs<br>%{percent}<extra></extra>"
), row=2, col=1)

fig.add_trace(go.Pie(
    labels=status_counts.index.tolist(),
    values=status_counts.values.tolist(),
    marker_colors=[STATUS_COLORS.get(s, "#999") for s in status_counts.index],
    name="Status", hole=0.35, textinfo="label+percent",
    hovertemplate="<b>%{label}</b><br>%{value} trials<br>%{percent}<extra></extra>"
), row=2, col=2)

fig.update_layout(
    paper_bgcolor="white",
    font=dict(family="Arial", size=13),
    height=640,
    showlegend=False,
    margin=dict(t=50, b=20),
)

# ============================================================
# STEP 5.5: "AT A GLANCE" PANEL — target×phase heatmap + Phase 3 leaderboard.
#
# Uses go.Heatmap (native trace, fires real plotly_click events) rather than
# hand-rolled geometry.
#
# Scope note: "faceted heatmap panels" (one per drug category) assumed N
# similarly-sized categories. The real data has only 2 categories with
# enough volume to facet meaningfully — Small Molecule (703 drugs) and
# Biologic (86) — Cell/Gene Therapy (8) and Device (7) are too sparse to
# earn their own panel. The heatmap facets via a 3-way tab (All / Small
# Molecule / Biologic) instead of simultaneous panels. Mini population
# icons inside cells are dropped in favor of exact counts — more precise
# than eyeballing an icon grid once counts exceed ~15.
# ============================================================

def readable_text_color(hex_color):
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (0, 2, 4))
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#1a1a1a" if luminance > 0.55 else "#ffffff"


PHASES_ASC = ["Phase 1", "Phase 2", "Phase 3"]

# --- Target × Phase heatmap (magnitude → one hue light-to-dark, not the
# categorical target colors — a sequential ramp is the correct encoding for
# "how many", built from the AriBio blue rather than Plotly's default) ---
# Phase 0: built from resolved_drugs_df (was legacy_drugs_df). Phase 1A:
# narrowed to therapeutic_drugs_df (pipeline_scope == "Therapeutic Drug"
# only), per the requirement that the heatmap show only real drugs.
HEATMAP_TABS = [("All", therapeutic_drugs_df), ("Small Molecule", therapeutic_drugs_df[therapeutic_drugs_df["drug_type"] == "Small Molecule"]),
                 ("Biologic", therapeutic_drugs_df[therapeutic_drugs_df["drug_type"] == "Biologic"])]
HEATMAP_COLORSCALE = [[0, "#eef2f8"], [1, ARIBIO_BLUE]]


def build_heatmap(sub_df):
    # count grid extracted into drug_classification.build_target_phase_counts()
    # so it's unit-testable independent of this Plotly figure
    z = build_target_phase_counts(sub_df, TARGET_ORDER, PHASES_ASC)
    fig = go.Figure(go.Heatmap(
        z=z, x=PHASES_ASC, y=TARGET_ORDER, colorscale=HEATMAP_COLORSCALE,
        text=z, texttemplate="%{text}", textfont=dict(size=13, color="#1a237e"),
        hovertemplate="<b>%{y}</b> &middot; %{x}: %{z} drugs<extra></extra>",
        showscale=False, xgap=4, ygap=4,
    ))
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=12))
    fig.update_xaxes(tickfont=dict(size=12), side="top")
    fig.update_layout(height=290, margin=dict(t=36, b=6, l=6, r=6), paper_bgcolor="white", plot_bgcolor="white")
    return fig


heatmap_figs = {label: build_heatmap(sub) for label, sub in HEATMAP_TABS}

# --- Phase 3 leaderboard: same drug-level data, sorted by target ---
# Phase 0: built from resolved_drugs_df (was legacy_drugs_df). Phase 1A:
# narrowed to therapeutic_drugs_df, per the requirement that the
# leaderboard show only real drugs.
phase3_df = therapeutic_drugs_df[therapeutic_drugs_df["phase_reached"] == "Phase 3"].sort_values(
    ["target", "is_aribio", "display_name"], ascending=[True, False, True]
)
PHASE3_PREVIEW_N = 8

print("=== PER-DRUG TABLE PREVIEW (therapeutic_drugs_df, first 30 rows, sorted by phase) ===")
preview = therapeutic_drugs_df.sort_values("phase_reached", ascending=False)
preview_cols = ["display_name", "phase_reached", "drug_type", "target", "status_summary", "is_aribio"]
print(preview[preview_cols].head(30).to_string(index=False))
print()

# ============================================================
# STEP 6: BUILD THE INTERACTIVE HTML PAGE
# Header bar + KPI tiles + AR1001 spotlight + pill filters + the
# trial-composition pies (plotly) + a hand-rolled sortable/filterable
# per-drug table. Plotly's Table trace doesn't support header-click
# sorting or being filtered by another chart's click events, so the
# table is plain HTML/JS driven off a JSON blob — still a single,
# fully standalone file, no server required.
# ============================================================

# Visible table now sourced from resolved_drugs_df (STEP 3.7) — the
# classify_intervention()/resolve_developed_drug()-based rollup — NOT
# legacy_drugs_df. "Review Status" sorts by the underlying boolean
# needs_manual_review (see the JS sort function below), not by the
# rendered label text.
TABLE_COLUMNS = [
    # (data key, header label, column width %) — the width is what lets
    # the table use table-layout:fixed (see CSS below): with the default
    # table-layout:auto, the browser recomputes every column's width from
    # whichever ROWS ARE CURRENTLY VISIBLE, so filtering in the sidebar
    # (which changes the visible row set) reshuffles column widths on
    # every click. Fixed layout + explicit widths means columns are set
    # once and never move again regardless of what's filtered.
    ("display_name", "Drug", 21),
    ("sponsor", "Sponsor", 19),
    ("phase_reached", "Highest Phase", 10),
    ("status_summary", "Status", 10),
    ("target_display", "Target / Pathway", 13),
    ("drug_type", "Drug Type", 12),
    ("trial_count", "Trial Count", 8),
    ("max_enrollment", "Enrollment", 7),
    # Verification/Confidence/Review Status columns removed from the main
    # table per request — still available per-row via the details toggle
    # (the underlying data columns are kept in table_df below for that).
]

# Phase 1A: table_df stays sourced from the BROADER resolved_drugs_df
# (every scope except Exclude/Placebo or Comparator, which never get a
# row at all) — not therapeutic_drugs_df — so the browser has the data
# to power the "reveal non-therapeutic records" toggle below. The JS
# applies a client-side pipeline_scope === "Therapeutic Drug" filter by
# default, matching the requirement that the DEFAULT table view show
# only therapeutic drugs while the data itself preserves the rest.
table_df = resolved_drugs_df[[
    "display_name", "phase_reached", "drug_type", "target", "target_display",
    "status_summary", "trial_count", "max_enrollment", "sponsor", "sponsor_display",
    "synonyms", "is_aribio", "study_url",
    "verification_status", "verification_label",
    "classification_confidence", "confidence_label",
    "needs_manual_review", "review_label",
    "confirmed_trial_count", "unverified_trial_count",
    "official_source_url", "classification_reason", "nct_ids",
    "pipeline_scope", "scope_reason", "manual_review_required",
]].copy()
table_records = json.loads(table_df.to_json(orient="records"))

# KPI tiles: Phase 1A narrows these to therapeutic_drugs_df — "Total
# drugs" must mean actual therapeutic drugs, not every resolved record.
total_drugs = len(therapeutic_drugs_df)
total_resolved_records = len(resolved_drugs_df)
phase3_agents = int((therapeutic_drugs_df["phase_reached"] == "Phase 3").sum())
phase2_agents = int((therapeutic_drugs_df["phase_reached"] == "Phase 2").sum())
phase1_agents = int((therapeutic_drugs_df["phase_reached"] == "Phase 1").sum())

plotlyjs_lib = pyo.get_plotlyjs()  # loaded once at the top of the page; every figure below skips its own copy
pies_html = pio.to_html(fig, include_plotlyjs=False, full_html=False, div_id="pieDiv")

HEATMAP_DIV_IDS = {"All": "heatmapAll", "Small Molecule": "heatmapSmallMolecule", "Biologic": "heatmapBiologic"}
heatmap_html = {
    label: pio.to_html(f, include_plotlyjs=False, full_html=False, div_id=HEATMAP_DIV_IDS[label])
    for label, f in heatmap_figs.items()
}

def status_text(status):
    if status == "FDA Approved":
        return f'<span style="color:{STATUS_COLORS["FDA Approved"]};font-weight:700;">FDA &#10003;</span>'
    if status == "Discontinued":
        return '<span style="color:#999;font-style:italic;">Discontinued</span>'
    return f'<span style="color:{STATUS_COLORS.get(status, "#666")};font-weight:600;">{status}</span>'

def phase3_row_html(row):
    name = f'<a href="{row["study_url"]}" target="_blank" rel="noopener">{row["display_name"]}</a>'
    if row["is_aribio"]:
        name_html = f'<b style="color:{ARIBIO_ACCENT}">{name} &#9733;</b>'
    else:
        name_html = name
    dot_color = STATUS_COLORS.get(row["status_summary"], "#999")
    return (
        f'<tr><td><span style="display:inline-block;width:8px;height:8px;border-radius:50%;'
        f'background:{dot_color};margin-right:8px;"></span>{name_html}</td>'
        f'<td>{row["drug_type"]}</td><td>{row["target"]}</td><td>{status_text(row["status_summary"])}</td></tr>'
    )

phase3_rows_html = "".join(phase3_row_html(r) for _, r in phase3_df.head(PHASE3_PREVIEW_N).iterrows())
phase3_shown = min(PHASE3_PREVIEW_N, len(phase3_df))

header_cells = "".join(
    f'<th data-key="{key}" style="width:{width}%" onclick="sortTable(\'{key}\')">{label} <span class="sort-arrow" id="arrow-{key}"></span></th>'
    for key, label, width in TABLE_COLUMNS
)

# AriBio-blue ramp (darkest = most confirmed) for verification/
# confidence/review pills and row accents, matching every other
# categorical color in the dashboard now. ARIBIO_ACCENT is reserved for
# "Manual review required" only — the one value across this whole group
# genuinely worth a second glance — everything else is a shade of blue.
# An "unverified investigational" drug is still a real, possible
# candidate, just not yet confirmed, so it gets a lighter blue, not a
# warning color.
VERIFICATION_COLORS = {
    "Confirmed official match": darken(ARIBIO_BLUE, 0.30),
    "Pipeline match; source needed": ARIBIO_BLUE,
    "Mixed evidence": lighten(ARIBIO_BLUE, 0.20),
    "Unverified investigational": lighten(ARIBIO_BLUE, 0.40),
    "Manual review required": ARIBIO_ACCENT,
}
CONFIDENCE_COLORS = {
    "High": darken(ARIBIO_BLUE, 0.30),
    "Medium": ARIBIO_BLUE,
    "Low": lighten(ARIBIO_BLUE, 0.40),
}
REVIEW_COLORS = {
    "Needs manual review": ARIBIO_ACCENT,
    "Reviewed / no review needed": lighten(ARIBIO_BLUE, 0.45),
}

# Verification/Confidence/Review Status filter groups removed per
# request — VERIFICATION_COLORS/CONFIDENCE_COLORS/REVIEW_COLORS are
# still defined above and still used by the row-detail panel's pills.
PILL_GROUPS = [
    ("phase", "Phase", ["Phase 1", "Phase 2", "Phase 3"], PHASE_COLORS),
    ("drugType", "Drug Type", list(DRUG_TYPE_COLORS.keys())[:3], DRUG_TYPE_COLORS),
    ("target", "Target", [t for t in TARGET_COLORS if t not in ("Other", "Unknown")], TARGET_COLORS),
    ("status", "Status", [s for s in STATUS_COLORS if s != "Other"], STATUS_COLORS),
]

def render_pill_group(field, title, values, colors):
    # Target is the one group that keeps its full color coding (a
    # colored dot per value); every other group is plain monochrome
    # text — a single shared --pill-color (brand blue) used only for
    # the active/selected state, no per-value hue and no dot at all.
    show_dot = field == "target"
    dot_class = " filter-pill--dot" if show_dot else ""
    pills = "".join(
        f'<button class="filter-pill{dot_class}" data-field="{field}" data-value="{v}" '
        + (f'style="--pill-color:{colors.get(v, "#999")}" ' if show_dot else "")
        + f'onclick="togglePill(this)">{v}</button>'
        for v in values
    )
    return f'<div class="filter-group"><div class="filter-group-title">{title.upper()}</div><div class="filter-pills">{pills}</div></div>'

pill_groups_html = "".join(render_pill_group(f, t, v, c) for f, t, v, c in PILL_GROUPS)

target_colors_js = json.dumps(TARGET_COLORS)
phase_colors_js = json.dumps(PHASE_COLORS)
status_colors_js = json.dumps(STATUS_COLORS)
type_colors_js = json.dumps(DRUG_TYPE_COLORS)
verification_colors_js = json.dumps(VERIFICATION_COLORS)
confidence_colors_js = json.dumps(CONFIDENCE_COLORS)
records_js = json.dumps(table_records)
table_column_count = len(TABLE_COLUMNS)

# pie trace order added above: 0=phase, 1=drug_type, 2=target, 3=status
# (phase/status pies read trials_df; drug_type/target pies read
# resolved_drugs_df as of Phase 0 — see the pie-counts comment above.
# The click-to-filter mapping below is unaffected either way: it maps a
# clicked slice's LABEL text to a table filter, and the table's filter
# values — Phase 1/2/3, Small Molecule/Biologic/etc, Amyloid/Tau/etc,
# Recruiting/Completed/etc — are the same vocabulary regardless of
# which dataset produced the slice.)
pie_field_map_js = json.dumps(["phase", "drugType", "target", "status"])
today_str = date.today().isoformat()

html_template = f"""
<style>
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: #f4f5f8; margin: 0; padding: 0 0 40px; color: #1a1a1a;
  }}
  /* sidebar floats as a detached card near the left edge (margin + radius
     + all-around shadow, not flush/docked); page-content is shifted right
     to clear it. No max-width on main/topbar-inner anymore — the whole
     point of pulling the sidebar out here is to give the table (and
     everything else) the full remaining viewport width to span, instead
     of being capped at 1280px and centered. */
  .page-content {{ margin-left: 276px; }}
  /* asymmetric padding: sidebar already provides its own 16px gap on the
     left, so the extra breathing room belongs on the right, otherwise
     wide tables/cards run flush to the browser edge and feel crowded */
  main {{ margin: 0; padding: 0 96px 0 24px; }}

  /* topbar is a full-bleed top-level element now (a sibling of the
     sidebar/page-content, not nested inside page-content's margin), so
     it spans the entire page width; the sidebar sits below it (see
     positionSidebar() in the script, which measures its real height) */
  .topbar {{
    background: {ARIBIO_BLUE}; color: white; padding: 20px 96px 20px 24px; margin-bottom: 24px;
    box-shadow: 0 2px 10px rgba(20, 40, 70, 0.16);
  }}
  .topbar-inner {{ margin: 0; }}
  .topbar-title {{ font-size: 21px; font-weight: 700; letter-spacing: -0.01em; }}
  .topbar-sub {{ font-size: 12.5px; color: {ARIBIO_BLUE_SUBTLE}; margin-top: 4px; }}

  .spotlight {{
    background: {ARIBIO_ACCENT_BG}; border: 1px solid {ARIBIO_ACCENT_BORDER}; border-left: 5px solid {ARIBIO_ACCENT};
    border-radius: {CARD_RADIUS}; padding: 14px 20px; margin-bottom: 20px; font-size: 14px; color: #4a1230;
  }}
  .spotlight b {{ color: {ARIBIO_ACCENT}; }}

  .kpi-row {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin-bottom: 20px; }}
  .kpi-tile {{
    background: white; border-radius: {CARD_RADIUS}; border-top: 3px solid {ARIBIO_BLUE};
    padding: 16px 18px; box-shadow: {CARD_SHADOW};
  }}
  .kpi-value {{ font-size: 27px; font-weight: 700; color: {ARIBIO_BLUE}; letter-spacing: -0.01em; }}
  .kpi-label {{ font-size: 12.5px; color: #666; margin-top: 3px; }}

  .sidebar {{
    /* top: 96px is a fallback (topbar height + gap) for the instant
       before JS measures the real header height and overrides it via
       positionSidebar() below — keeps things sane with JS disabled too.
       transition: top so any later re-measurement (window resize, text
       reflow) eases into place instead of snapping. */
    position: fixed; left: 16px; top: 96px; bottom: 16px; width: 240px;
    background: white; border-radius: {CARD_RADIUS}; box-shadow: {ELEVATED_SHADOW};
    z-index: 20; display: flex; flex-direction: column; overflow: hidden;
    transition: top 0.2s ease;
  }}
  .sidebar-header {{
    display: flex; align-items: center; justify-content: space-between; flex-shrink: 0;
    padding: 14px 16px; background: {ARIBIO_BLUE_SUBTLE}; border-bottom: 1px solid {SURFACE_BORDER};
  }}
  .sidebar-title {{
    display: flex; align-items: center; gap: 7px; font-size: 12.5px; font-weight: 700;
    color: {ARIBIO_BLUE}; text-transform: uppercase; letter-spacing: 0.05em;
  }}
  .sidebar-title svg {{ color: {ARIBIO_BLUE}; flex-shrink: 0; }}
  .filter-groups {{ flex: 1 1 auto; overflow-y: auto; padding: 6px 16px 16px; display: flex; flex-direction: column; }}
  .filter-group {{ padding: 14px 0; }}
  .filter-group:first-child {{ padding-top: 12px; }}
  .filter-group + .filter-group {{ border-top: 1px solid {SURFACE_BORDER}; }}
  .filter-group-title {{ font-size: 10.5px; letter-spacing: 0.06em; color: #9aa0ab; margin-bottom: 8px; font-weight: 600; }}
  .filter-pills {{ display: flex; flex-direction: column; gap: 1px; }}
  @media (max-width: 960px) {{
    .sidebar {{
      position: static; width: auto; height: auto; box-shadow: {CARD_SHADOW};
      border-radius: {CARD_RADIUS}; margin: 0 24px 20px;
    }}
    .page-content {{ margin-left: 0; }}
    .filter-pills {{ flex-direction: row; flex-wrap: wrap; }}
  }}
  /* Clean list style, not a bubble/badge: plain text, bigger for
     readability. No color-coding except the Target group (the one
     dimension worth telling apart at a glance) — those pills get the
     .filter-pill--dot modifier and a small colored dot; every other
     group is plain monochrome text. Active state reads like a selected
     nav item: a colored left accent bar + soft tinted background,
     rather than a filled pill. */
  .filter-pill {{
    --pill-color: {ARIBIO_BLUE}; display: flex; align-items: center; gap: 8px;
    background: none; border: none; border-left: 3px solid transparent; border-radius: 0 6px 6px 0;
    font-size: 14.5px; color: #444; padding: 6px 8px 6px 9px; width: 100%; text-align: left;
    cursor: pointer; font-family: inherit;
    transition: background-color 0.15s ease, border-left-color 0.15s ease, color 0.15s ease;
  }}
  .filter-pill--dot::before {{
    content: ""; width: 8px; height: 8px; min-width: 8px; border-radius: 50%; background: var(--pill-color);
  }}
  .filter-pill:hover {{ background: rgba(0,0,0,0.045); }}
  .filter-pill.active {{
    color: var(--pill-color); font-weight: 700; background: rgba(0,0,0,0.045); border-left-color: var(--pill-color);
  }}
  #clear-filter {{
    font-size: 12px; font-weight: 600; color: {ARIBIO_ACCENT}; background: none; border: none;
    cursor: pointer; padding: 2px 0; visibility: hidden; font-family: inherit;
  }}
  #clear-filter:hover {{ text-decoration: underline; }}
  #clear-filter.visible {{ visibility: visible; }}

  h2.section-title {{ color: #1a1a1a; font-size: 18px; font-weight: 700; letter-spacing: -0.01em; margin: 30px 0 6px; }}
  .section-hint {{ font-size: 13px; color: #666; margin-bottom: 4px; }}

  #controls {{ display: flex; align-items: center; gap: 12px; margin: 10px 0 12px; flex-wrap: wrap; }}
  #controls input[type="text"] {{
    padding: 8px 12px; border: 1px solid #ccc; border-radius: 8px; font-size: 13px; width: 280px;
    font-family: inherit; transition: border-color 0.12s ease, box-shadow 0.12s ease;
  }}
  #controls input[type="text"]:focus {{
    outline: none; border-color: {ARIBIO_BLUE}; box-shadow: 0 0 0 3px {lighten(ARIBIO_BLUE, 0.82)};
  }}
  #scope-toggle-label {{
    font-size: 12.5px; color: #444; display: flex; align-items: center; gap: 6px; cursor: pointer; user-select: none;
  }}
  #scope-toggle-label input {{ cursor: pointer; }}
  #row-count {{ font-size: 12px; color: #666; margin: 0 0 6px; }}

  #table-wrap {{
    max-height: 640px; border-radius: {CARD_RADIUS}; background: white; box-shadow: {CARD_SHADOW};
    /* overflow-y:scroll (not auto) + scrollbar-gutter:stable — the space
       for the scrollbar is reserved whether or not it's currently
       needed, so toggling a sidebar filter (which changes the row count,
       which flips the scrollbar on/off) never shifts the table's column
       widths left/right. scroll+gutter is the belt-and-suspenders pair:
       gutter is the modern/clean way, scroll is the fallback for
       browsers that don't support it yet — both reserve the same space. */
    overflow-y: scroll; scrollbar-gutter: stable;
  }}
  /* table-layout:fixed + explicit per-column widths (set inline on each
     <th>, from TABLE_COLUMNS) — with the default table-layout:auto, the
     browser recomputes every column's width from whatever ROWS ARE
     CURRENTLY VISIBLE, so filtering in the sidebar (which changes the
     visible row set) reshuffled column widths on every click. Fixed
     layout locks widths in from the header row alone, once, so the
     table never shifts again regardless of what's filtered. */
  table#drug-table {{ border-collapse: collapse; table-layout: fixed; width: 100%; font-size: 13px; }}
  table#drug-table thead th {{
    position: sticky; top: 0; background: {ARIBIO_BLUE}; color: white; text-align: left;
    padding: 11px 14px; cursor: pointer; user-select: none; white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis; z-index: 1;
  }}
  table#drug-table thead th:hover {{ background: {ARIBIO_BLUE_HOVER}; }}
  table#drug-table td {{
    padding: 9px 14px; border-bottom: 1px solid #eee; vertical-align: top; overflow-wrap: break-word;
  }}
  table#drug-table tbody tr:hover td {{ background: {SURFACE_TINT}; }}
  .sort-arrow {{ font-size: 10px; }}
  #drug-table td a, table.phase3-table td a {{ color: inherit; text-decoration: none; }}
  #drug-table td a:hover, table.phase3-table td a:hover {{ color: {ARIBIO_BLUE}; text-decoration: underline; }}
  tr.aribio-row td a:hover {{ color: white; text-decoration: underline; }}
  .phase-3-row td {{ font-weight: 700; font-size: 14px; }}
  .phase-1-row td {{ font-weight: 400; font-size: 12px; color: #555; }}
  .discontinued td {{ color: #999 !important; text-decoration: line-through; }}
  tr.aribio-row td {{
    background: {ARIBIO_ACCENT} !important; color: white !important;
    font-weight: 700; text-decoration: none !important;
  }}
  tr.aribio-row:hover td {{ background: {ARIBIO_ACCENT_HOVER} !important; }}
  .pill {{ font-weight: 600; white-space: nowrap; }}  /* plain colored text, not a badge — color set inline per-value in JS */

  .details-toggle {{
    background: none; border: none; cursor: pointer; font-size: 11px; color: #999;
    padding: 0 4px 0 0; font-family: inherit; vertical-align: middle;
  }}
  .details-toggle:hover {{ color: {ARIBIO_BLUE}; }}
  tr.detail-row td {{ background: {SURFACE_TINT}; padding: 14px 20px; border-bottom: 1px solid {SURFACE_BORDER}; }}
  .detail-panel {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 10px 22px; font-size: 12.5px; color: #444; }}
  .detail-panel strong {{ display: block; color: #999; font-size: 10.5px; letter-spacing: 0.04em; text-transform: uppercase; margin-bottom: 3px; }}
  .detail-panel ul {{ margin: 0; padding-left: 16px; }}
  .sponsor-cell {{ cursor: help; }}

  .glance-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 8px; }}
  .glance-panel {{ background: white; border-radius: {CARD_RADIUS}; padding: 18px; box-shadow: {CARD_SHADOW}; }}
  .glance-panel-title {{ font-size: 15px; font-weight: 700; letter-spacing: -0.01em; margin-bottom: 4px; }}
  /* wraps the Plotly pies so "Trial Composition" lives in a white card
     like every other section, instead of sitting directly on the page
     background — the pies themselves are unchanged (still built in
     STEP 5), this is presentation-only */
  .pies-card {{ background: white; border-radius: {CARD_RADIUS}; padding: 12px 18px 4px; box-shadow: {CARD_SHADOW}; }}

  .heatmap-tabs {{ display: flex; gap: 6px; margin: 10px 0 6px; }}
  .heatmap-tab {{
    font-size: 12.5px; padding: 5px 12px; border-radius: 14px; border: 1px solid #ddd;
    background: white; color: #666; cursor: pointer; font-family: inherit;
  }}
  .heatmap-tab.active {{ background: {ARIBIO_BLUE}; border-color: {ARIBIO_BLUE}; color: white; font-weight: 600; }}
  table.phase3-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  table.phase3-table th {{
    text-align: left; font-size: 10.5px; letter-spacing: 0.04em; color: #999;
    padding: 6px 8px; border-bottom: 1px solid #eee;
  }}
  table.phase3-table td {{ padding: 8px; border-bottom: 1px solid #f2f2f2; }}
  #show-all-phase3 {{
    display: inline-block; margin-top: 10px; font-size: 12.5px; color: {ARIBIO_BLUE};
    cursor: pointer; text-decoration: underline;
  }}
  @media (max-width: 860px) {{ .glance-grid {{ grid-template-columns: 1fr; }} }}
</style>

<script>{plotlyjs_lib}</script>

<header class="topbar">
  <div class="topbar-inner">
    <div class="topbar-title">Alzheimer's Disease Clinical Trial Pipeline</div>
    <div class="topbar-sub">Source: clinicaltrials.gov &middot; Updated {today_str} &middot; {len(df)} trials analyzed &middot; {total_drugs} therapeutic drugs ({total_resolved_records} total resolved records)</div>
  </div>
</header>

<aside class="sidebar">
  <div class="sidebar-header">
    <div class="sidebar-title">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round">
        <line x1="4" y1="6" x2="20" y2="6"></line>
        <line x1="4" y1="12" x2="20" y2="12"></line>
        <line x1="4" y1="18" x2="20" y2="18"></line>
        <circle cx="9" cy="6" r="2" fill="white" stroke-width="1.6"></circle>
        <circle cx="15" cy="12" r="2" fill="white" stroke-width="1.6"></circle>
        <circle cx="9" cy="18" r="2" fill="white" stroke-width="1.6"></circle>
      </svg>
      Filters
    </div>
    <button id="clear-filter" onclick="clearFilters()">Clear</button>
  </div>
  <div class="filter-groups">
    {pill_groups_html}
  </div>
</aside>

<div class="page-content">
  <main>
    <div class="spotlight">
      &#9733; <b>AR1001 (AriBio)</b> &mdash; Phase 3 &middot; Oral PDE5 inhibitor &middot; Amyloid + Tau + Neuroprotection &middot;
      POLARIS AD trial &middot; 1,535 patients enrolled. Phase 2 showed improvements in pTau181, A&beta;42/40 ratio, and
      ADAS-Cog13 vs. ADNI external controls (AAIC 2026).
    </div>

    <div class="kpi-row">
      <div class="kpi-tile"><div class="kpi-value">{total_drugs}</div><div class="kpi-label">Total therapeutic drugs</div></div>
      <div class="kpi-tile"><div class="kpi-value" style="color:{PHASE_COLORS['Phase 3']}">{phase3_agents}</div><div class="kpi-label">Phase 3 agents</div></div>
      <div class="kpi-tile"><div class="kpi-value" style="color:{PHASE_COLORS['Phase 2']}">{phase2_agents}</div><div class="kpi-label">Phase 2 agents</div></div>
      <div class="kpi-tile"><div class="kpi-value" style="color:{PHASE_COLORS['Phase 1']}">{phase1_agents}</div><div class="kpi-label">Phase 1 agents</div></div>
    </div>

    <div class="glance-grid">
      <div class="glance-panel">
        <div class="glance-panel-title">Target &times; phase heatmap</div>
        <div class="heatmap-tabs">
          {"".join(f'<button class="heatmap-tab{" active" if label == "All" else ""}" data-tab="{HEATMAP_DIV_IDS[label]}" onclick="switchHeatmapTab(this)">{label}</button>' for label, _ in HEATMAP_TABS)}
        </div>
        {"".join(f'<div class="heatmap-pane" id="pane-{HEATMAP_DIV_IDS[label]}" style="display:{"block" if label == "All" else "none"}">{heatmap_html[label]}</div>' for label, _ in HEATMAP_TABS)}
        <div class="section-hint">Darker = more drugs. Click a cell to filter the table below.</div>
      </div>
      <div class="glance-panel">
        <div class="glance-panel-title">Phase 3 agents &mdash; sorted by target
          <span style="font-weight:400;font-size:12px;color:#999;">(showing {phase3_shown} of {len(phase3_df)})</span>
        </div>
        <table class="phase3-table">
          <thead><tr><th>Drug</th><th>Type</th><th>Target</th><th>Status</th></tr></thead>
          <tbody>{phase3_rows_html}</tbody>
        </table>
        <span id="show-all-phase3" onclick="showAllPhase3()">Show all {len(phase3_df)} Phase 3 agents in the table below &darr;</span>
      </div>
    </div>

    <h2 class="section-title">Trial Composition</h2>
    <div class="section-hint">Click any pie slice to filter the table below (in addition to the filters in the sidebar).</div>
    <div class="pies-card">
      {pies_html}
    </div>

    <h2 class="section-title">Drug Pipeline Table</h2>
    <div id="controls">
      <input type="text" id="search-box" placeholder="Search by drug, sponsor, target, or verification...">
      <label id="scope-toggle-label">
        <input type="checkbox" id="scope-toggle"> Show non-therapeutic / needs-review records
      </label>
    </div>
    <div id="row-count"></div>
    <div id="table-wrap">
      <table id="drug-table">
        <thead><tr>{header_cells}</tr></thead>
        <tbody id="drug-table-body"></tbody>
      </table>
    </div>
  </main>
</div>

<script id="drug-data" type="application/json">{records_js}</script>
<script>
  const ALL_ROWS = JSON.parse(document.getElementById('drug-data').textContent);
  const TARGET_COLORS = {target_colors_js};
  const PHASE_COLORS = {phase_colors_js};
  const STATUS_COLORS = {status_colors_js};
  const TYPE_COLORS = {type_colors_js};
  const VERIFICATION_COLORS = {verification_colors_js};
  const CONFIDENCE_COLORS = {confidence_colors_js};
  const PIE_FIELD_MAP = {pie_field_map_js};
  const TABLE_COLUMN_COUNT = {table_column_count};
  // maps a pill/pie "field" key to the actual column name on each row
  const FIELD_TO_COLUMN = {{
    phase: 'phase_reached', drugType: 'drug_type', target: 'target', status: 'status_summary',
  }};

  let sortKey = 'phase_reached';
  let sortAsc = false;
  let filters = {{
    phase: new Set(), drugType: new Set(), target: new Set(), status: new Set(),
  }};
  let searchTerm = '';
  // Phase 1A: ALL_ROWS carries every resolved_drugs_df record (every
  // pipeline_scope except "Exclude"/"Placebo or Comparator", which never
  // get a row at all — see table_df in pipeline_viz.py). The DEFAULT
  // table view still shows only "Therapeutic Drug" records; checking
  // the "Show non-therapeutic / needs-review records" box reveals the
  // rest (Diagnostic Agent / Non-Drug Intervention / Supportive
  // Treatment / Needs Review).
  let showNonTherapeutic = false;
  const THERAPEUTIC_SCOPE = 'Therapeutic Drug';
  // drug display_name -> expanded/collapsed state for the row-detail
  // panel, tracked across re-renders (filter/sort/search all rebuild
  // the table body from scratch, so this can't just live in the DOM)
  let expandedRows = new Set();

  function escapeHtml(value) {{
    return String(value == null ? '' : value).replace(/[&<>"']/g, ch => ({{
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    }})[ch]);
  }}

  function textSafeColor(hexColor) {{
    // Pills are now plain colored TEXT on a white background, not a
    // filled badge — several of the blue-ramp shades are light enough
    // (by design, for use as backgrounds/dots elsewhere) that using them
    // directly as text color would be hard to read on white. This
    // darkens only the colors that need it, so the value still reads as
    // "that color family" without disappearing into the page.
    const h = hexColor.replace('#', '');
    const r = parseInt(h.substring(0, 2), 16), g = parseInt(h.substring(2, 4), 16), b = parseInt(h.substring(4, 6), 16);
    const luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
    if (luminance <= 0.55) return hexColor;
    const factor = 0.55;
    const clamp = (v) => Math.round(v * factor).toString(16).padStart(2, '0');
    return `#${{clamp(r)}}${{clamp(g)}}${{clamp(b)}}`;
  }}

  function pill(text, colorMap) {{
    const c = textSafeColor(colorMap[text] || '#666666');
    return `<span class="pill" style="color:${{c}}">${{text || 'Unknown'}}</span>`;
  }}

  function anyFiltersActive() {{
    return Object.values(filters).some(s => s.size > 0);
  }}

  function updateClearButton() {{
    document.getElementById('clear-filter').classList.toggle('visible', anyFiltersActive());
  }}

  function matchesSearch(r, term) {{
    const haystack = [
      r.display_name, r.sponsor, r.synonyms, r.verification_label, r.confidence_label, r.target_display,
    ].filter(Boolean).join(' ').toLowerCase();
    return haystack.includes(term);
  }}

  function renderDetailRow(r) {{
    const sponsors = String(r.sponsor || '').split('; ').filter(Boolean);
    const sponsorList = sponsors.length
      ? `<ul>${{sponsors.map(s => `<li>${{escapeHtml(s)}}</li>`).join('')}}</ul>`
      : '—';
    return `<tr class="detail-row"><td colspan="${{TABLE_COLUMN_COUNT}}"><div class="detail-panel">
      <div><strong>Sponsors</strong>${{sponsorList}}</div>
      <div><strong>Verification</strong>${{pill(r.verification_label, VERIFICATION_COLORS)}}</div>
      <div><strong>Confidence</strong>${{pill(r.confidence_label, CONFIDENCE_COLORS)}}</div>
      <div><strong>Pipeline scope</strong>${{escapeHtml(r.pipeline_scope || '—')}}</div>
      <div><strong>Confirmed trials</strong>${{r.confirmed_trial_count}}</div>
      <div><strong>Unverified trials</strong>${{r.unverified_trial_count}}</div>
      <div><strong>Notes</strong>${{escapeHtml(r.classification_reason || '—')}}</div>
      <div><strong>Scope reason</strong>${{escapeHtml(r.scope_reason || '—')}}</div>
      <div><strong>Trial IDs</strong>${{escapeHtml(r.nct_ids || '—')}}</div>
    </div></td></tr>`;
  }}

  function renderTable() {{
    const body = document.getElementById('drug-table-body');

    if (ALL_ROWS.length === 0) {{
      body.innerHTML = `<tr><td colspan="${{TABLE_COLUMN_COUNT}}" style="text-align:center;color:#999;padding:28px;">No therapeutic drug candidates were resolved.</td></tr>`;
      document.getElementById('row-count').textContent = '0 of 0 drugs shown';
      return;
    }}

    const scopeBase = showNonTherapeutic ? ALL_ROWS : ALL_ROWS.filter(r => r.pipeline_scope === THERAPEUTIC_SCOPE);

    let rows = scopeBase.filter(r => {{
      for (const field in filters) {{
        const set = filters[field];
        if (set.size > 0 && !set.has(r[FIELD_TO_COLUMN[field]])) return false;
      }}
      if (searchTerm && !matchesSearch(r, searchTerm)) return false;
      return true;
    }});

    rows.sort((a, b) => {{
      let av, bv;
      if (sortKey === 'trial_count' || sortKey === 'max_enrollment') {{
        av = a[sortKey] || 0; bv = b[sortKey] || 0;
      }} else {{
        av = String(a[sortKey] || '').toLowerCase(); bv = String(b[sortKey] || '').toLowerCase();
      }}
      if (av < bv) return sortAsc ? -1 : 1;
      if (av > bv) return sortAsc ? 1 : -1;
      return 0;
    }});

    const hiddenCount = ALL_ROWS.length - scopeBase.length;
    const hiddenNote = (!showNonTherapeutic && hiddenCount > 0) ? ` (${{hiddenCount}} non-therapeutic/needs-review record${{hiddenCount === 1 ? '' : 's'}} hidden)` : '';
    const scopeLabel = showNonTherapeutic ? 'records' : 'therapeutic drugs';
    document.getElementById('row-count').textContent = `${{rows.length}} of ${{scopeBase.length}} ${{scopeLabel}} shown${{hiddenNote}}`;

    if (rows.length === 0) {{
      body.innerHTML = `<tr><td colspan="${{TABLE_COLUMN_COUNT}}" style="text-align:center;color:#999;padding:28px;">No drugs match the current filters.</td></tr>`;
      return;
    }}

    body.innerHTML = rows.map(r => {{
      const classes = [];
      if (r.phase_reached === 'Phase 3') classes.push('phase-3-row');
      if (r.phase_reached === 'Phase 1') classes.push('phase-1-row');
      if (r.status_summary === 'Discontinued') classes.push('discontinued');
      if (r.is_aribio) classes.push('aribio-row');
      const star = r.is_aribio ? '\\u2605 ' : '';
      const enrollment = r.max_enrollment ? Math.round(r.max_enrollment).toLocaleString() : '—';
      const isExpanded = expandedRows.has(r.display_name);
      const toggle = `<button class="details-toggle" data-drug-key="${{escapeHtml(r.display_name)}}" title="Show details">${{isExpanded ? '\\u25be' : '\\u25b8'}}</button>`;
      const mainRow = `<tr class="${{classes.join(' ')}}">
        <td>${{toggle}} ${{star}}<a href="${{r.study_url}}" target="_blank" rel="noopener">${{r.display_name}}</a></td>
        <td class="sponsor-cell" title="${{escapeHtml(r.sponsor || '')}}">${{r.sponsor_display || ''}}</td>
        <td>${{pill(r.phase_reached, PHASE_COLORS)}}</td>
        <td>${{pill(r.status_summary, STATUS_COLORS)}}</td>
        <td>${{pill(r.target_display, TARGET_COLORS)}}</td>
        <td>${{pill(r.drug_type, TYPE_COLORS)}}</td>
        <td>${{r.trial_count}}</td>
        <td>${{enrollment}}</td>
      </tr>`;
      return isExpanded ? mainRow + renderDetailRow(r) : mainRow;
    }}).join('');
  }}

  function sortTable(key) {{
    if (sortKey === key) {{ sortAsc = !sortAsc; }}
    else {{ sortKey = key; sortAsc = true; }}
    document.querySelectorAll('.sort-arrow').forEach(el => el.textContent = '');
    const arrow = document.getElementById('arrow-' + key);
    if (arrow) arrow.textContent = sortAsc ? '\\u25B2' : '\\u25BC';
    renderTable();
  }}

  function togglePill(btn) {{
    const field = btn.dataset.field, value = btn.dataset.value;
    if (filters[field].has(value)) {{ filters[field].delete(value); btn.classList.remove('active'); }}
    else {{ filters[field].add(value); btn.classList.add('active'); }}
    updateClearButton();
    renderTable();
  }}

  function clearFilters() {{
    for (const field in filters) filters[field].clear();
    document.querySelectorAll('.filter-pill.active').forEach(el => el.classList.remove('active'));
    updateClearButton();
    renderTable();
  }}

  document.getElementById('search-box').addEventListener('input', (e) => {{
    searchTerm = e.target.value.toLowerCase();
    renderTable();
  }});

  document.getElementById('scope-toggle').addEventListener('change', (e) => {{
    showNonTherapeutic = e.target.checked;
    renderTable();
  }});

  document.getElementById('drug-table-body').addEventListener('click', (e) => {{
    const btn = e.target.closest('.details-toggle');
    if (!btn) return;
    const key = btn.dataset.drugKey;
    if (expandedRows.has(key)) expandedRows.delete(key); else expandedRows.add(key);
    renderTable();
  }});

  // clicking a pie slice toggles the same filter set as its matching pill
  // (falls back silently for slices with no matching pill, e.g. "Other")
  const pieDiv = document.getElementById('pieDiv');
  pieDiv.on('plotly_click', function(data) {{
    const pt = data.points[0];
    const field = PIE_FIELD_MAP[pt.curveNumber];
    if (!field || !filters[field]) return;
    const matchingPill = document.querySelector(`.filter-pill[data-field="${{field}}"][data-value="${{pt.label}}"]`);
    if (matchingPill) {{ togglePill(matchingPill); return; }}
    if (filters[field].has(pt.label)) filters[field].delete(pt.label);
    else filters[field].add(pt.label);
    updateClearButton();
    renderTable();
  }});

  function togglePillByValue(field, value) {{
    const pillEl = document.querySelector(`.filter-pill[data-field="${{field}}"][data-value="${{value}}"]`);
    if (pillEl) togglePill(pillEl);
  }}

  // clicking a heatmap cell toggles Phase + Target (drug-type stays whatever tab is active, not auto-filtered)
  ['heatmapAll', 'heatmapSmallMolecule', 'heatmapBiologic'].forEach((divId) => {{
    document.getElementById(divId).on('plotly_click', function(data) {{
      const pt = data.points[0];
      togglePillByValue('phase', pt.x);
      togglePillByValue('target', pt.y);
    }});
  }});

  function switchHeatmapTab(btn) {{
    document.querySelectorAll('.heatmap-tab').forEach(t => t.classList.remove('active'));
    btn.classList.add('active');
    document.querySelectorAll('.heatmap-pane').forEach(p => p.style.display = 'none');
    document.getElementById('pane-' + btn.dataset.tab).style.display = 'block';
    // panes start hidden (display:none), so Plotly measured 0 width on first
    // render — force a resize now that the container has real dimensions
    Plotly.Plots.resize(document.getElementById(btn.dataset.tab));
  }}

  function showAllPhase3() {{
    const phase3Pill = document.querySelector('.filter-pill[data-field="phase"][data-value="Phase 3"]');
    if (phase3Pill && !phase3Pill.classList.contains('active')) togglePill(phase3Pill);
    document.getElementById('table-wrap').scrollIntoView({{ behavior: 'smooth', block: 'start' }});
  }}

  // The topbar now spans the full page width (it's a sibling of the
  // sidebar, not nested inside it), so the floating sidebar needs to
  // start below it rather than at the very top of the viewport. This
  // measures the topbar's REAL rendered height (rather than hardcoding
  // an estimate) so it stays correct if text wraps differently across
  // browsers/zoom levels.
  function positionSidebar() {{
    const header = document.querySelector('.topbar');
    const sidebar = document.querySelector('.sidebar');
    if (!header || !sidebar) return;
    sidebar.style.top = (header.getBoundingClientRect().height + 16) + 'px';
  }}
  window.addEventListener('load', positionSidebar);
  window.addEventListener('resize', positionSidebar);
  positionSidebar();

  renderTable();
</script>
"""

with open("pipeline_overview.html", "w") as f:
    f.write("<!DOCTYPE html><html><head><meta charset='utf-8'><title>AD Pipeline Dashboard</title></head><body>")
    f.write(html_template)
    f.write("</body></html>")

print("=== SAVED: pipeline_overview.html ===")
print("Open this file in any browser — no Python needed")
print()

# ============================================================
# STEP 7: SAVE ANNOTATED DATA FOR MANUAL REVIEW
# Open these CSVs in Excel to check/fix the auto-annotations —
# the KNOWN_COMPOUNDS dictionary is a best-effort pass from public
# literature, not a verified source. Spot-check "Other" rows and
# anything pathway-critical before using this for external comms.
# ============================================================

review_cols = [
    # nct_id/title/phase_clean/status_clean/is_aribio: trial-level fields.
    # drug_type/target: STEP 3's guess_drug_type()/guess_target() — still
    # the only drug-type/target classification logic that exists (Phase 0
    # only consolidated WHICH trials produce a drug row, not how drug_type/
    # target get computed for them — see MIGRATION_PLAN.md Phase 2).
    "nct_id", "title", "phase_clean", "drug_type", "target", "status_clean", "is_aribio",
    # classify_intervention()/resolve_developed_drug()-based resolution,
    # produced alongside the columns above so both can be compared
    "developed_drug", "developed_drug_normalized", "drug_classification",
    "classification_reason", "official_pipeline_match", "official_source_url",
    "verification_status", "classification_confidence", "needs_manual_review",
    # Phase 1A intervention-scope gap closure — the trial-level scope
    # forwarded from whichever intervention resolve_developed_drug()
    # picked as this trial's developed_drug (see resolve_developed_drug()'s
    # scope_fields() helper in drug_classification.py).
    "pipeline_scope", "scope_reason", "scope_method", "scope_confidence", "manual_review_required",
]
review_cols = [c for c in review_cols if c in df.columns]
df[review_cols].to_csv("pipeline_annotated.csv", index=False)
print("=== SAVED: pipeline_annotated.csv (per-trial) ===")

# pipeline_drugs.csv comes from resolved_drugs_df (STEP 3.7) — the same
# one drug-level source of truth every other dashboard component now
# uses (Phase 0 data-source consolidation). This is what removes
# placebo/diagnostic-tracer/procedure/device/behavioral/non-treatment-
# control rows from pipeline_drugs.csv. Phase 1A: still the FULL
# resolved_drugs_df (every pipeline_scope except Exclude/Placebo or
# Comparator, which never get a row at all) — not narrowed to
# therapeutic_drugs_df — so this CSV stays the same "one drug-level
# source of truth" the dashboard's table/optional-filter reads, with the
# new pipeline_scope column making it trivial to filter to
# "Therapeutic Drug" only in Excel.
drug_review_cols = [
    "display_name", "phase_reached", "drug_type", "target", "status_summary",
    "trial_count", "max_enrollment", "sponsor", "is_aribio",
    "verification_status", "classification_confidence", "needs_manual_review",
    "confirmed_trial_count", "unverified_trial_count",
    "pipeline_scope", "scope_reason", "scope_method", "scope_confidence", "manual_review_required",
]
resolved_drugs_df[drug_review_cols].sort_values(["phase_reached", "display_name"], ascending=[False, True]).to_csv(
    "pipeline_drugs.csv", index=False
)
print("=== SAVED: pipeline_drugs.csv (per-drug rollup, now from developed_drug resolution) ===")
print(f"{len(resolved_drugs_df)} drug rows ({len(therapeutic_drugs_df)} Therapeutic Drug scope) "
      f"({int((resolved_drugs_df['confirmed_trial_count'] > 0).sum())} with a confirmed/pipeline-matched trial, "
      f"{int(((resolved_drugs_df['unverified_trial_count'] > 0) & (resolved_drugs_df['confirmed_trial_count'] == 0)).sum())} unverified-only)")
print("Open in Excel to review and fix any wrong drug_type or target labels")
print()

# ============================================================
# STEP 7.6: SAVE UNRESOLVED-TRIAL AUDIT (NEW)
# Trials that build_resolved_drugs_dataframe() left OUT of the drug
# rollup because their developed_drug couldn't be confidently resolved
# (multiple candidates, or unresolved "uncertain" interventions) — kept
# here so nothing is silently dropped, only cleanly non-therapeutic
# trials (placebo/diagnostic/procedure/device/behavioral-only) are
# absent from BOTH pipeline_drugs.csv and this file.
# ============================================================

unresolved_trials_df = build_unresolved_trials_dataframe(df)
unresolved_trials_df.to_csv("pipeline_unresolved_trials.csv", index=False)
print(f"=== SAVED: pipeline_unresolved_trials.csv ({len(unresolved_trials_df)} trials needing manual review) ===")
print()

# ============================================================
# STEP 7.5: SAVE PER-INTERVENTION CLASSIFICATION (NEW)
# One row per individual trial-intervention pair — the full detail
# behind each trial's developed_drug resolution above. Use this to
# audit/improve the classification rules or data/official_pipeline.csv.
# ============================================================

interventions_output = interventions_df.rename(columns={
    "nct_id": "NCT Number",
    "sponsor": "Sponsor",
    "title": "Study Title",
    "original_type": "intervention_type",
    "original_name": "intervention_name",
    "reason": "classification_reason",
    "confidence": "confidence",
})[[
    "NCT Number", "Sponsor", "Study Title", "intervention_type", "intervention_name",
    "normalized_name", "classification", "classification_reason", "official_pipeline_match",
    "matched_pipeline_drug", "official_source_url", "verification_status", "confidence",
    "needs_manual_review",
    # Phase 1A: every intervention's scope verdict, preserved here
    # regardless of whether it ever became a developed_drug candidate —
    # this is the traceability record for records that never reach
    # resolved_drugs_df at all (e.g. Exclude/Placebo or Comparator scope).
    "pipeline_scope", "scope_reason", "scope_method", "scope_confidence", "manual_review_required",
]]
interventions_output.to_csv("pipeline_interventions.csv", index=False)
print("=== SAVED: pipeline_interventions.csv (per-intervention detail) ===")
print()

# ============================================================
# COMMON ERRORS:
#   "FileNotFoundError: trials.csv" → put the downloaded CSV in
#       the same folder as this script, named exactly trials.csv
#   "KeyError" on a column → print(df.columns.tolist()) and
#       tell Claude which columns you actually have
# ============================================================
