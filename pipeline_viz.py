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

from drug_classification import (
    load_official_pipeline,
    load_generic_drug_aliases,
    canonicalize_developed_drug_names,
    build_interventions_dataframe,
    resolve_developed_drug,
    build_resolved_drugs_dataframe,
    build_unresolved_trials_dataframe,
    build_target_phase_counts,
    build_relevance_matrix,
    summarize_relevance_scores,
    MATURITY_PHASE_ORDER,
    build_resolved_drug_trial_links_df,
    build_drug_date_rollup,
    load_scope_overrides,
    build_scope_audit_dataframe,
    build_diagnostic_agent_audit_dataframe,
    build_resolved_drugs_exclusion_audit_dataframe,
    THERAPEUTIC_SCOPE,
    normalize_text,
)
from nih_reference import parse_nih_dataset
from scientific_classification import (
    load_drug_classification_overrides,
    build_nih_name_lookup,
    build_official_pipeline_classification_lookup,
    gather_structured_evidence_for_drug,
    resolve_drug_classification,
    build_classification_conflicts_dataframe,
    classify_pipeline_quadrant,
)
from competitive_intelligence import compute_relevance_score
from dashboard_nav import NAV_BG, NAV_CSS, render_nav_bar
from fda_status import load_fda_status_reference, match_drug_to_fda_status
from competitive_attention_viz import (
    COMPETITIVE_ATTENTION_CSS,
    PLACEHOLDER as COMPETITIVE_ATTENTION_PLACEHOLDER,
    MILESTONES_PLACEHOLDER as COMPETITIVE_MILESTONES_PLACEHOLDER,
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
    "startDate": "start_date",
    "Primary Completion Date": "primary_completion_date",
    "primaryCompletionDate": "primary_completion_date",
    "Completion Date": "completion_date",
    "completionDate": "completion_date",
}

df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

# ct.gov's own export mixes date granularity — "2016-11" (month only)
# alongside "2025-11-20" (full date) — pd.to_datetime handles both,
# defaulting a month-only value to that month's 1st. errors="coerce"
# turns anything unparseable (or blank) into NaT rather than raising,
# since ~1-3% of trials are missing one of these fields.
def parse_ct_date(value):
    if pd.isna(value) or not str(value).strip():
        return pd.NaT
    return pd.to_datetime(str(value).strip(), errors="coerce")


df["start_date_parsed"] = df["start_date"].apply(parse_ct_date) if "start_date" in df.columns else pd.NaT
df["primary_completion_date_parsed"] = (
    df["primary_completion_date"].apply(parse_ct_date) if "primary_completion_date" in df.columns else pd.NaT
)

# Multicenter status per trial, derived from the "Locations" column
# (pipe-separated "facility, city, state, zip, country" per site — see
# _format_locations() in ctgov_normalize.py). ct.gov's API has no
# dedicated multicenter field, so this is the only available signal.
# De-dupe on (facility, city) before counting: a handful of trials list
# the exact same site twice (e.g. NCT06730438 lists the same hospital
# twice), which would otherwise inflate a 1-site trial into a false
# "multicenter" at the raw-count>=2 boundary. Missing Locations data
# (246 trials) is reported as "unknown", never silently folded into
# "single" — the CSV genuinely doesn't say.
def _site_facility_city(entry):
    parts = entry.split(",")
    facility = parts[0].strip().lower() if parts else ""
    city = parts[1].strip().lower() if len(parts) > 1 else ""
    return (facility, city)


def classify_site_status(locations_str):
    if not isinstance(locations_str, str) or not locations_str.strip():
        return "unknown"
    entries = [e.strip() for e in locations_str.split("|") if e.strip()]
    if not entries:
        return "unknown"
    unique_sites = {_site_facility_city(e) for e in entries}
    return "multicenter" if len(unique_sites) >= 2 else "single"


df["site_status"] = df["Locations"].apply(classify_site_status) if "Locations" in df.columns else "unknown"
TRIAL_SITE_STATUS = dict(zip(df["nct_id"], df["site_status"])) if "nct_id" in df.columns else {}

# Every trial is kept, regardless of phase — including NA (no phase
# assigned, e.g. observational/expanded-access records), Phase 4
# (post-marketing), Early Phase 1 (aka Phase 0), and the combined
# dual-phase designations ct.gov itself uses (PHASE1|PHASE2, PHASE2|PHASE3).
# This is an EXACT map against ct.gov's own phase enum, not a substring
# search — a substring check on "1" would previously have wrongly
# folded EARLY_PHASE1 into plain "Phase 1" (EARLY_PHASE1 contains the
# character "1"), which is a real, different trial-design designation.
PHASE_LABELS = {
    "NA": "NA",
    "EARLY_PHASE1": "Early Phase 1",
    "PHASE1": "Phase 1",
    "PHASE1|PHASE2": "Phase 1/Phase 2",
    "PHASE2": "Phase 2",
    "PHASE2|PHASE3": "Phase 2/Phase 3",
    "PHASE3": "Phase 3",
    "PHASE4": "Phase 4",
}
# clinical-progression order, ascending — every chart/pill list that
# enumerates phases explicitly (heatmap x-axis, sidebar filter) uses
# this, so a new/unrecognized ct.gov phase value can't silently vanish
# from those (it still gets a "NA" fallback below, which IS in this list)
PHASE_ORDER = ["NA", "Early Phase 1", "Phase 1", "Phase 1/Phase 2", "Phase 2", "Phase 2/Phase 3", "Phase 3", "Phase 4"]


def clean_phase(phase_str):
    if pd.isna(phase_str):
        return "NA"
    key = str(phase_str).strip().upper()
    return PHASE_LABELS.get(key, "NA")  # any future/unrecognized ct.gov phase value degrades to NA, never dropped


df["phase_clean"] = df["phase"].apply(clean_phase)

print(f"=== ALL PHASES INCLUDED: {len(df)} trials ===")
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
    "diagnostic_subtype": "",
}
for _col, _default in _NEW_COLUMN_DEFAULTS.items():
    df[_col] = df[_col].fillna(_default)

print("=== DEVELOPED-DRUG RESOLUTION (per trial) ===")
print(df["drug_classification"].value_counts())
print(f"{df['needs_manual_review'].sum()} trials flagged needs_manual_review")
print()

# ============================================================
# STEP 3.65: GENERIC-DRUG NAME CANONICALIZATION
#
# Collapses salt/brand/formulation/route/typo variants of the SAME old,
# multi-sponsor generic active agent (e.g. "Donepezil Hydrochloride",
# "Aricept", "Donepezil TDS" -> "Donepezil") into one canonical
# developed_drug value BEFORE STEP 3.7 groups by it — see
# drug_classification.load_generic_drug_aliases()'s docstring for why
# this can't just reuse official_pipeline.csv (that mechanism requires
# a sponsor match, which fails for a generic studied by dozens of
# unrelated sponsors). Never touches combination products, prodrugs, or
# genuinely distinct active moieties — only verified variants of one
# agent, curated in data/reference/generic_drug_aliases.csv.
# ============================================================

_generic_drug_aliases = load_generic_drug_aliases("data/reference/generic_drug_aliases.csv")
df = canonicalize_developed_drug_names(df, _generic_drug_aliases)
print(f"{len(_generic_drug_aliases)} generic-drug alias(es) loaded from "
      f"data/reference/generic_drug_aliases.csv")
print()

# ============================================================
# STEP 3.7: RESOLVED DRUG-LEVEL ROLLUP — the ONE drug-level source of
# truth for the whole dashboard as of Phase 0 (data-source
# consolidation): the visible HTML/JS drug table, pipeline_drugs.csv,
# the KPI tiles, the heatmap, and the Phase 3 leaderboard all derive
# from this dataframe now — no component computes its own separate
# drug-level rollup anymore.
# ============================================================

resolved_drugs_df = build_resolved_drugs_dataframe(df)

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

# ============================================================
# STEP 3.71: SCIENTIFIC CLASSIFICATION RESOLUTION (Phase 2)
#
# Replaces guess_drug_type()/guess_target() (STEP 3 above, which still
# runs and still populates the raw per-TRIAL df["drug_type"]/df["target"]
# columns used only by pipeline_annotated.csv's legacy audit column —
# left untouched per this phase's "do not change... change tracking"
# instruction) as the SOURCE for resolved_drugs_df's drug_type/target:
# every canonical drug's modality/target_pathways/mechanism_of_action
# now come from scientific_classification.resolve_drug_classification(),
# using verified, drug-specific evidence only (curated override -> NIH
# reference -> official_pipeline.csv -> exact known-compound match ->
# this drug's OWN structured ct.gov intervention evidence — never the
# raw, possibly-multi-intervention trial text guess_drug_type() reads).
#
# pipeline_scope, drug identity/grouping (build_resolved_drugs_dataframe
# above), FDA status, company/sponsor fields, and change-tracking are
# NOT touched by this step — only drug_type/target and the new
# target_pathways/mechanism_of_action/molecular_targets columns.
# ============================================================

_drug_classification_overrides = load_drug_classification_overrides(
    "data/reference/drug_classification_overrides.csv"
)
_nih_reference_df = parse_nih_dataset("nih_data.csv")
_nih_name_lookup = build_nih_name_lookup(_nih_reference_df)
_official_pipeline_classification_lookup = build_official_pipeline_classification_lookup(pipeline_records)

_previous_drug_type = resolved_drugs_df["drug_type"].copy()
_previous_target = resolved_drugs_df["target"].copy()

_sci_results = []
for _, _row in resolved_drugs_df.iterrows():
    _drug_synonyms = _synonyms_by_drug_name.get(_row["display_name"], [])
    _evidence = gather_structured_evidence_for_drug(interventions_df, normalize_text(_row["display_name"]))
    _sci_results.append(resolve_drug_classification(
        _row["display_name"], _drug_synonyms, _evidence,
        overrides=_drug_classification_overrides, nih_name_lookup=_nih_name_lookup,
        official_pipeline_lookup=_official_pipeline_classification_lookup,
    ))

resolved_drugs_df["modality"] = [r["modality"] for r in _sci_results]
resolved_drugs_df["target_pathways_list"] = [r["target_pathways"] for r in _sci_results]
resolved_drugs_df["target"] = [
    (r["target_pathways"][0] if r["target_pathways"] else "Other") for r in _sci_results
]
resolved_drugs_df["target_pathways"] = ["; ".join(r["target_pathways"]) for r in _sci_results]
resolved_drugs_df["mechanism_of_action"] = [r["mechanism_of_action"] for r in _sci_results]
resolved_drugs_df["molecular_targets"] = ["; ".join(r["molecular_targets"]) for r in _sci_results]
resolved_drugs_df["classification_source"] = [r["classification_source"] for r in _sci_results]
resolved_drugs_df["classification_method"] = [r["classification_method"] for r in _sci_results]
resolved_drugs_df["scientific_classification_confidence"] = [r["classification_confidence"] for r in _sci_results]
resolved_drugs_df["scientific_classification_reason"] = [r["classification_reason"] for r in _sci_results]
resolved_drugs_df["evidence_used"] = [r["evidence_used"] for r in _sci_results]
resolved_drugs_df["scientific_manual_review_required"] = [r["manual_review_required"] for r in _sci_results]
# NIH-sourced, display-only supplementary fields — blank whenever no NIH
# match exists for this drug (never fabricated for the drugs NIH doesn't cover).
resolved_drugs_df["therapeutic_purpose_class"] = [r["therapeutic_purpose_class"] for r in _sci_results]
resolved_drugs_df["therapeutic_purpose_category"] = [r["therapeutic_purpose_category"] for r in _sci_results]
resolved_drugs_df["cadro"] = [r["cadro"] for r in _sci_results]

# ============================================================
# STEP 3.72: FDA STATUS RESOLUTION
#
# Genuinely separate FDA regulatory status, sourced ONLY from the
# small, hand-curated data/reference/fda_status_reference.csv — see
# fda_status.py's module docstring for why this is a curated file
# rather than a live openFDA integration, and why "no match" resolves
# to Unknown, never "Not FDA Approved" (absence from a small reviewed
# file is not evidence of non-approval). Deliberately never derived
# from trial-level status_summary/phase_reached — see STATUS_MAP above,
# whose "APPROVED_FOR_MARKETING" -> "FDA Approved" TRIAL-status
# relabeling is exactly the trial-status/FDA-status conflation this
# step exists to stop repeating for anything new.
# ============================================================

_fda_status_reference = load_fda_status_reference("data/reference/fda_status_reference.csv")
_fda_results = [
    match_drug_to_fda_status(
        _row["display_name"], _synonyms_by_drug_name.get(_row["display_name"], []), _fda_status_reference
    )
    for _, _row in resolved_drugs_df.iterrows()
]
resolved_drugs_df["fda_status"] = [r["fda_status"] for r in _fda_results]
resolved_drugs_df["fda_indication"] = [r["indication"] for r in _fda_results]
resolved_drugs_df["fda_approval_type"] = [r["approval_type"] for r in _fda_results]
resolved_drugs_df["fda_approval_date"] = [r["approval_date"] for r in _fda_results]
resolved_drugs_df["fda_withdrawal_date"] = [r["withdrawal_date"] for r in _fda_results]
resolved_drugs_df["fda_application_status"] = [r["application_status"] for r in _fda_results]
resolved_drugs_df["fda_source_title"] = [r["source_title"] for r in _fda_results]
resolved_drugs_df["fda_source_url"] = [r["source_url"] for r in _fda_results]
resolved_drugs_df["fda_notes"] = [r["notes"] for r in _fda_results]
print(f"FDA status resolved for {sum(1 for s in resolved_drugs_df['fda_status'] if s != 'Unknown')} of "
      f"{len(resolved_drugs_df)} drugs ({sum(1 for s in resolved_drugs_df['fda_status'] if s == 'FDA Approved')} FDA Approved)")

# drug_type is now the 4-category "pipeline quadrant" scheme (Disease-
# Targeted Biologic / Disease-Targeted Small Molecule / Cognition
# Enhancer / Neuropsychiatric Symptom Tx), matching the published AD
# drug-development pipeline chart's own categorization — NIH-sourced
# where available (~103 drugs), INFERRED from this drug's already-
# resolved modality/target for the rest, and flagged as such via
# drug_type_source/drug_type_inferred so an inferred bucket never reads
# as equally certain as a real NIH citation. The finer-grained modality
# (Small Molecule/Biologic/Cell-Gene Therapy/etc) is preserved separately
# above as resolved_drugs_df["modality"], not discarded.
_quadrant_results = [
    classify_pipeline_quadrant(
        r["modality"], (r["target_pathways"][0] if r["target_pathways"] else "Other"),
        r["therapeutic_purpose_class"], r["therapeutic_purpose_category"],
    )
    for r in _sci_results
]
resolved_drugs_df["drug_type"] = [q[0] for q in _quadrant_results]
resolved_drugs_df["drug_type_source"] = [q[1] for q in _quadrant_results]
resolved_drugs_df["drug_type_inferred"] = [q[2] for q in _quadrant_results]

# target_display: generalizes the old AR1001-only hardcode to EVERY drug
# with more than one target_pathway — "do not force one drug into only
# one pathway" (Phase 2 requirement), now driven by real evidence
# (including AR1001's own curated override) rather than an is_aribio check.
resolved_drugs_df["target_display"] = resolved_drugs_df.apply(
    lambda r: f"Multi ({'/'.join(r['target_pathways_list'])})" if len(r["target_pathways_list"]) > 1 else r["target"],
    axis=1,
)

_classification_conflict_records = [
    {
        "canonical_drug_name": name,
        "previous_modality": prev_type,
        "previous_target": prev_target,
        "new_modality": r["modality"],
        "new_target_pathways": r["target_pathways"],
        "classification_source": r["classification_source"],
        "classification_confidence": r["classification_confidence"],
        "classification_reason": r["classification_reason"],
        "manual_review_required": r["manual_review_required"],
    }
    for name, prev_type, prev_target, r in zip(
        resolved_drugs_df["display_name"], _previous_drug_type, _previous_target, _sci_results
    )
]
classification_conflicts_df = build_classification_conflicts_dataframe(_classification_conflict_records)
os.makedirs("outputs", exist_ok=True)
classification_conflicts_df.to_csv("outputs/classification_conflicts.csv", index=False)

print("=== SCIENTIFIC CLASSIFICATION RESOLUTION (Phase 2) ===")
print(f"{len(_nih_reference_df)} NIH reference rows loaded; {len(_drug_classification_overrides)} curated AriBio override(s)")
print("classification_source breakdown:")
print(resolved_drugs_df["classification_source"].value_counts().to_string())
_drug_type_changed = int((_previous_drug_type != resolved_drugs_df["drug_type"]).sum())
_target_changed = int((_previous_target != resolved_drugs_df["target"]).sum())
_multi_target = int((resolved_drugs_df["target_pathways_list"].apply(len) > 1).sum())
print(f"drug_type corrected: {_drug_type_changed} / {len(resolved_drugs_df)}")
print(f"target corrected: {_target_changed} / {len(resolved_drugs_df)}")
print(f"drugs with multiple target_pathways: {_multi_target}")
print(f"remaining 'Other' target: {int((resolved_drugs_df['target'] == 'Other').sum())}")
print(f"remaining 'Unknown' modality: {int((resolved_drugs_df['drug_type'] == 'Unknown').sum())}")
print(f"=== SAVED: outputs/classification_conflicts.csv ({len(classification_conflicts_df)} rows) ===")
print()


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
# STEP 3.72: DRUG-LEVEL DATE ROLLUP
# earliest_start_date / latest_primary_completion_date per canonical
# drug, across ALL of its contributing trials (not just whichever trial
# happens to be its highest-phase one) — computed here, before
# therapeutic_drugs_df is sliced off below, so both resolved_drugs_df
# and therapeutic_drugs_df carry it.
# ============================================================
_links_for_dates = build_resolved_drug_trial_links_df(resolved_drugs_df)
_date_rollup_df = build_drug_date_rollup(_links_for_dates, df)
resolved_drugs_df = resolved_drugs_df.merge(_date_rollup_df, on="display_name", how="left")

# "%b %Y" (e.g. "Nov 2026"), never day-level — ct.gov's own data mixes
# month-only and full-date granularity (see parse_ct_date above), and
# showing a specific day would imply false precision for the month-only
# rows. "TBD" (not "—") for a drug with no parseable date at all, since
# that's genuinely "not yet determined" from this data, not just blank.
resolved_drugs_df["start_date_display"] = resolved_drugs_df["earliest_start_date"].dt.strftime("%b %Y").fillna("TBD")
resolved_drugs_df["primary_completion_date_display"] = (
    resolved_drugs_df["latest_primary_completion_date"].dt.strftime("%b %Y").fillna("TBD")
)

# ============================================================
# STEP 3.73: ARIBIO RELEVANCE SCORE (competitive intelligence)
# A deterministic, rule-based similarity score (0-100) between every
# resolved drug's already-computed profile and AR1001's — NOT an AI/LLM
# output (see competitive_intelligence.py's module docstring for why
# that distinction matters). Every point is tied to a plain-language
# reason, so it's fully auditable in the drug detail panel/comparator.
# ============================================================
_ar1001_rows = resolved_drugs_df[resolved_drugs_df["display_name"] == "AR1001"]
if _ar1001_rows.empty:
    _ar1001_rows = resolved_drugs_df[resolved_drugs_df["is_aribio"]]

if not _ar1001_rows.empty:
    _ar1001_row = _ar1001_rows.iloc[0]
    _ar1001_target_pathways = _ar1001_row["target_pathways_list"]
    _ar1001_modality = _ar1001_row["modality"]
    _ar1001_purpose_class = _ar1001_row["therapeutic_purpose_class"]
    _ar1001_phase = _ar1001_row["phase_reached"]

    _relevance_results = [
        compute_relevance_score(
            r["target_pathways_list"], r["modality"], r["therapeutic_purpose_class"], r["phase_reached"],
            _ar1001_target_pathways, _ar1001_modality, _ar1001_purpose_class, _ar1001_phase,
        )
        for _, r in resolved_drugs_df.iterrows()
    ]
    resolved_drugs_df["aribio_relevance_score"] = [s for s, _ in _relevance_results]
    resolved_drugs_df["aribio_relevance_reasons"] = ["; ".join(rs) if rs else "No shared profile with AR1001" for _, rs in _relevance_results]
else:
    # AR1001 itself absent from this trials.csv pull (shouldn't happen
    # for the AD pipeline dataset this dashboard is built for, but stay
    # honest rather than crash if it ever is) — score everything 0/absent
    # rather than silently comparing against a made-up reference.
    resolved_drugs_df["aribio_relevance_score"] = 0
    resolved_drugs_df["aribio_relevance_reasons"] = "AR1001 not found in this dataset"

# ============================================================
# PHASE 1A: therapeutic_drugs_df — the DEFAULT dashboard population.
# resolved_drugs_df (above) stays the one drug-level source of truth and
# keeps every scope Phase 1A didn't outright exclude (Therapeutic Drug,
# Diagnostic Agent, Non-Drug Intervention, Supportive Treatment, Needs
# Review), so the dashboard's optional "reveal non-therapeutic records"
# filter has real data to show. Components that must show ONLY actual
# therapeutic drugs (KPI tiles, heatmap, Phase 3 leaderboard, and the
# table's default filter state) read THIS narrower view instead.
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
for _phase_label in PHASE_ORDER:
    print(f"{_phase_label} drugs: {int((resolved_drugs_df['phase_reached'] == _phase_label).sum())}")
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

diagnostic_agent_audit_df = build_diagnostic_agent_audit_dataframe(interventions_df)
diagnostic_agent_audit_df.to_csv("outputs/diagnostic_agent_audit.csv", index=False)
_leaked_count = int(diagnostic_agent_audit_df["previously_leaked_into_therapeutic_dashboard"].sum()) if len(diagnostic_agent_audit_df) else 0
print(f"=== SAVED: outputs/diagnostic_agent_audit.csv ({len(diagnostic_agent_audit_df)} suspected imaging/radiotracer records, "
      f"{_leaked_count} currently/previously leaking into the therapeutic view) ===")
print()

non_drug_exclusion_audit_df = build_resolved_drugs_exclusion_audit_dataframe(interventions_df)
non_drug_exclusion_audit_df.to_csv("outputs/non_drug_exclusion_audit.csv", index=False)
print(f"=== SAVED: outputs/non_drug_exclusion_audit.csv ({len(non_drug_exclusion_audit_df)} non-drug/non-biologic "
      f"interventions excluded from resolved_drugs_df) ===")
print()

# ============================================================
# STEP 4: COLORS
# (brand colors + darken()/lighten() helpers now live near the top of
# the file, right after the imports — STEP 3.7 needs them too)
# ============================================================

# Blue ramp ordered by clinical progression (darkest = furthest along),
# matching PHASE_ORDER — NA is the one true non-phase value and gets a
# neutral gray instead of implying it's "before Phase 1" on the ramp.
PHASE_COLORS = {
    "Phase 4":         darken(ARIBIO_BLUE, 0.45),
    "Phase 3":         darken(ARIBIO_BLUE, 0.35),
    "Phase 2/Phase 3": darken(ARIBIO_BLUE, 0.15),
    "Phase 2":         ARIBIO_BLUE,
    "Phase 1/Phase 2": lighten(ARIBIO_BLUE, 0.25),
    "Phase 1":         lighten(ARIBIO_BLUE, 0.45),
    "Early Phase 1":   lighten(ARIBIO_BLUE, 0.65),
    "NA":              "#9e9e9e",
}

# The one deliberately-distinct categorical palette left — pathway is
# worth telling apart at a glance in the heatmap, so it keeps its
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

# "Drug Type" is the 4-category pipeline-quadrant scheme (see
# scientific_classification.classify_pipeline_quadrant) — matches the
# published AD drug-development pipeline chart's own four categories.
DRUG_TYPE_COLORS = {
    "Disease-Targeted Biologic":       darken(ARIBIO_BLUE, 0.30),
    "Disease-Targeted Small Molecule": ARIBIO_BLUE,
    "Cognition Enhancer":              lighten(ARIBIO_BLUE, 0.25),
    "Neuropsychiatric Symptom Tx":     lighten(ARIBIO_BLUE, 0.50),
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


PHASES_ASC = PHASE_ORDER  # NA, Early Phase 1, Phase 1, Phase 1/Phase 2, Phase 2, Phase 2/Phase 3, Phase 3, Phase 4

# --- Target × Phase heatmap (magnitude → one hue light-to-dark, not the
# categorical target colors — a sequential ramp is the correct encoding for
# "how many", built from the AriBio blue rather than Plotly's default) ---
# Phase 0: built from resolved_drugs_df (was legacy_drugs_df). Phase 1A:
# narrowed to therapeutic_drugs_df (pipeline_scope == "Therapeutic Drug"
# only), per the requirement that the heatmap show only real drugs.
# Filters on `modality` (clean vocabulary: Small Molecule/Biologic/
# Other/Cell-Gene Therapy/...), NOT `drug_type` (a different column
# with values like "Disease-Targeted Small Molecule" that never
# equals the bare "Small Molecule"/"Biologic" tab labels -- comparing
# against drug_type here always evaluated to zero rows, which is why
# the Small Molecule/Biologic tabs previously showed an all-zero grid.
HEATMAP_TABS = [("All", therapeutic_drugs_df), ("Small Molecule", therapeutic_drugs_df[therapeutic_drugs_df["modality"] == "Small Molecule"]),
                 ("Biologic", therapeutic_drugs_df[therapeutic_drugs_df["modality"] == "Biologic"])]
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

# --- "AR1001 Competitive Landscape" — a 2x2 executive-summary view of
# per-drug AR1001 relevance data: one bubble per top-N competitor (by
# aribio_relevance_score),
# x=development maturity (Early Phase 1 -> Phase 4), y=AR1001 relevance
# score, color=target pathway (reuses TARGET_COLORS). Divider lines
# split the plot into four labeled quadrants at fixed, deterministic
# thresholds — x between Phase 2 and Phase 2/Phase 3 (the conventional
# late-stage/pivotal-trial boundary), y at relevance=70 (the "high
# relevance" cutoff) — not data-driven medians, so the quadrant a drug
# lands in doesn't shift meaning from one refresh to the next. AR1001
# itself is plotted separately as a star reference marker at its own
# phase — see build_relevance_matrix()'s docstring for why it's
# excluded from the ranked competitor bubbles.
#
# Same-phase/same-score competitors get a small deterministic visual
# jitter (build_relevance_matrix()'s jitter_x/jitter_y columns) so they
# don't render as a single overlapping dot — the plotted position is
# nudged, but hover always reads the true phase_reached/
# aribio_relevance_score, never the jittered one. ---
RELEVANCE_MATRIX_TOP_N = 40
MATURITY_X_DIVIDER = 3.5  # late-stage = "Phase 2/Phase 3" (index 4) or later
RELEVANCE_Y_DIVIDER = 70  # "high relevance" cutoff
RELEVANCE_MATRIX_LABEL_COUNT = 4  # top 4 highest-priority competitors get a visible name label


def build_relevance_matrix_figure(resolved_drugs_df, top_n=RELEVANCE_MATRIX_TOP_N):
    matrix_df = build_relevance_matrix(resolved_drugs_df, top_n)

    # score report (item 10): printed, not silently used to redesign the
    # scoring algorithm — see the accompanying MIGRATION_PLAN/chat report
    # for what the ties mean and how scoring could become more discriminating
    score_report = summarize_relevance_scores(matrix_df["aribio_relevance_score"])
    print("=== AR1001 RELEVANCE SCORE DISTRIBUTION (Top {} competitors) ===".format(top_n))
    print(f"min={score_report['min']}  max={score_report['max']}  median={score_report['median']}  "
          f"unique scores={score_report['n_unique']} of {score_report['n_total']} competitors")
    for score, count in score_report["score_counts"].items():
        print(f"  score {score}: {count} competitor(s)")
    print()

    fig = go.Figure()
    for target in TARGET_ORDER + ["Other", "Unknown"]:
        rows = matrix_df[matrix_df["target"] == target]
        if rows.empty:
            continue
        plot_x = (rows["maturity_x"] + rows["jitter_x"]).tolist()
        plot_y = (rows["aribio_relevance_score"] + rows["jitter_y"]).clip(1, 99).tolist()
        fig.add_trace(go.Scatter(
            x=plot_x, y=plot_y, mode="markers",
            marker=dict(size=16, color=TARGET_COLORS.get(target, "#999"), opacity=0.85,
                        line=dict(width=1, color="white")),
            name=target, text=rows["display_name"].tolist(),
            customdata=list(zip(rows["sponsor"], rows["phase_reached"], rows["target"],
                                 rows["modality"], rows["aribio_relevance_score"], rows["aribio_relevance_reasons"])),
            hovertemplate=(
                "<b>%{text}</b><br>%{customdata[0]}<br>"
                "Phase: %{customdata[1]}<br>"
                "Target: %{customdata[2]} &middot; Modality: %{customdata[3]}<br>"
                "AR1001 relevance: %{customdata[4]}/100<br>"
                "%{customdata[5]}<extra></extra>"
            ),
        ))

    # Top 4 highest-priority competitors (highest relevance, later stage
    # preferred among ties) get a visible name label; every other point's
    # name is hover-only, per the "label only important competitors"
    # requirement — this is a presentation choice, doesn't touch matrix_df.
    # These competitors tend to cluster tightly (they're tied on the same
    # 1-2 top scores — see summarize_relevance_scores()): in real data
    # the 4 labeled dots have sat within ~0.55 maturity-x units and ~5.5
    # relevance-y units of each other, which cycling through Plotly
    # textposition anchors ("top center"/"bottom center"/etc, each only
    # a few px offset) was nowhere near enough to pull apart — hence the
    # explicit 2x2 grid below instead. Because the y-axis is heavily
    # compressed relative to x (the full y-range 0-118 spans the same
    # pixel height as roughly a 6-7-unit slice of the x-range), spacing
    # is deliberately x-led: each grid COLUMN carries almost all the
    # separation, each ROW only a small vertical stagger. A thin leader
    # line ties each relocated label back to its true (undisturbed) dot
    # position so the label never reads as "a 5th, unplotted competitor."
    labeled = matrix_df.sort_values(
        ["aribio_relevance_score", "maturity_x", "display_name"], ascending=[False, False, True]
    ).head(RELEVANCE_MATRIX_LABEL_COUNT)
    if not labeled.empty:
        LABEL_COL_SPACING = 1.15   # x units between the grid's two columns
        LABEL_ROW_OFFSET = 8.0     # y units the two rows sit above/below the cluster
        dot_x = (labeled["maturity_x"] + labeled["jitter_x"]).tolist()
        dot_y = (labeled["aribio_relevance_score"] + labeled["jitter_y"]).clip(1, 99).tolist()
        cluster_center_x = sum(dot_x) / len(dot_x)
        label_x, label_y = [], []
        for i in range(len(labeled)):
            col, row = i // 2, i % 2
            lx = cluster_center_x + (col - 0.5) * LABEL_COL_SPACING
            ly = dot_y[i] + (LABEL_ROW_OFFSET if row == 0 else -LABEL_ROW_OFFSET)
            # never climb into AR1001's own reference marker/label at
            # y=100, and never drop past the "high relevance" y=70
            # divider into a different quadrant's territory
            ly = max(RELEVANCE_Y_DIVIDER + 2, min(ly, 96))
            label_x.append(lx)
            label_y.append(ly)

        leader_x, leader_y = [], []
        for dx, dy, lx, ly in zip(dot_x, dot_y, label_x, label_y):
            leader_x += [dx, lx, None]
            leader_y += [dy, ly, None]
        fig.add_trace(go.Scatter(
            x=leader_x, y=leader_y, mode="lines",
            line=dict(color="#c7ccd6", width=1),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=label_x, y=label_y,
            mode="text", text=labeled["display_name"].tolist(), textposition="middle center",
            textfont=dict(size=10.5, color="#2a2a2a", family="Arial Black, Arial"),
            showlegend=False, hoverinfo="skip",
        ))

    ar1001_rows = resolved_drugs_df[resolved_drugs_df["is_aribio"]]
    if not ar1001_rows.empty:
        ar1001 = ar1001_rows.iloc[0]
        ar1001_x = MATURITY_PHASE_ORDER.index(ar1001["phase_reached"]) if ar1001["phase_reached"] in MATURITY_PHASE_ORDER else None
        if ar1001_x is not None:
            fig.add_trace(go.Scatter(
                x=[ar1001_x], y=[100], mode="markers+text",
                marker=dict(size=24, symbol="star", color=ARIBIO_ACCENT, line=dict(width=1.5, color="white")),
                # "middle right" (not "top center") keeps AR1001's own
                # label from climbing toward the quadrant caption band
                # above it — the real competitor cluster already crowds
                # the top of the chart (see the jitter comment above),
                # so nothing here should compete with that space vertically.
                text=["AR1001"], textposition="middle right",
                textfont=dict(size=12, color=ARIBIO_ACCENT, family="Arial Black, Arial"),
                name="AR1001 (AriBio)",
                hovertemplate="<b>AR1001</b> &middot; AriBio's own program &middot; %{customdata}<extra></extra>",
                customdata=[ar1001["phase_reached"]],
            ))

    x_max = len(MATURITY_PHASE_ORDER) - 1
    fig.add_shape(type="line", x0=MATURITY_X_DIVIDER, x1=MATURITY_X_DIVIDER, y0=-5, y1=108,
                  line=dict(color="#d8d8d8", width=1, dash="dash"))
    fig.add_shape(type="line", x0=-0.5, x1=x_max + 0.5, y0=RELEVANCE_Y_DIVIDER, y1=RELEVANCE_Y_DIVIDER,
                  line=dict(color="#d8d8d8", width=1, dash="dash"))

    # Quadrant captions are pinned near the top/bottom edge of the AXES
    # rectangle itself (yref="y domain", a 0-1 fraction of the plot area,
    # not the data scale) rather than at a fixed data y-value — the real
    # Top-40 dataset clusters almost entirely between relevance 75-100,
    # so a data-coordinate caption near the top risked colliding with
    # actual points/labels (as seen before this fix). Domain placement
    # at 0.965/0.025, combined with AR1001's label now sitting beside
    # its star instead of above it, keeps this row genuinely empty.
    quadrant_x_fracs = [
        (MATURITY_X_DIVIDER / 2 + 0.5) / (x_max + 1),
        ((MATURITY_X_DIVIDER + x_max) / 2 + 0.5) / (x_max + 1),
    ]
    quadrant_labels = [
        ("WATCH", quadrant_x_fracs[0], 0.965),
        ("PRIORITY COMPETITORS", quadrant_x_fracs[1], 0.965),
        ("LOWER PRIORITY", quadrant_x_fracs[0], 0.025),
        ("LATE-STAGE / DIFFERENT MECHANISM", quadrant_x_fracs[1], 0.025),
    ]
    for text, x, y in quadrant_labels:
        fig.add_annotation(x=x, y=y, xref="x domain", yref="y domain", text=text,
                            showarrow=False, xanchor="center",
                            font=dict(size=10.5, color="#b0b0b0"))

    fig.update_xaxes(
        tickvals=list(range(len(MATURITY_PHASE_ORDER))), ticktext=MATURITY_PHASE_ORDER,
        range=[-0.5, x_max + 0.5], tickfont=dict(size=11), showgrid=True, gridcolor="#f2f2f2",
        title=dict(text="Development Maturity — Early → Late", font=dict(size=12.5, color="#555")),
    )
    fig.update_yaxes(
        range=[-10, 118], tickfont=dict(size=11), showgrid=True, gridcolor="#f2f2f2",
        title=dict(text="Competitive Relevance to AR1001", font=dict(size=12.5, color="#555")),
    )
    fig.update_layout(
        height=480, margin=dict(t=40, b=44, l=6, r=50), paper_bgcolor="white", plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=11)),
        hoverlabel=dict(bgcolor="white", font_size=12, bordercolor="#ddd"),
    )
    return fig


relevance_matrix_fig = build_relevance_matrix_figure(therapeutic_drugs_df)

# --- Phase 3 leaderboard: same drug-level data. phase3_df itself stays
# sorted by target (used for the "Show all N Phase 3 agents" count/link
# below, which reflects the FULL Phase 3 population including AR1001).
# Phase 0: built from resolved_drugs_df (was legacy_drugs_df). Phase 1A:
# narrowed to therapeutic_drugs_df, per the requirement that the
# leaderboard show only real drugs.
phase3_df = therapeutic_drugs_df[therapeutic_drugs_df["phase_reached"] == "Phase 3"].sort_values(
    ["target", "is_aribio", "display_name"], ascending=[True, False, True]
)
PHASE3_PREVIEW_N = 10
# The PREVIEW table (unlike phase3_df/the "show all" link above) ranks
# by AR1001 relevance score instead -- top N Phase 3 competitors most
# relevant to AR1001, not just alphabetized by target. AR1001 itself is
# excluded (its own self-score isn't a meaningful "how relevant" rank),
# same convention as build_relevance_matrix() elsewhere on this dashboard.
phase3_top_relevant_df = phase3_df[~phase3_df["is_aribio"]].sort_values(
    ["aribio_relevance_score", "display_name"], ascending=[False, True]
)

print("=== PER-DRUG TABLE PREVIEW (therapeutic_drugs_df, first 30 rows, sorted by phase) ===")
preview = therapeutic_drugs_df.sort_values("phase_reached", ascending=False)
preview_cols = ["display_name", "phase_reached", "drug_type", "target", "status_summary", "is_aribio"]
print(preview[preview_cols].head(30).to_string(index=False))
print()

# ============================================================
# STEP 6: BUILD THE INTERACTIVE HTML PAGE
# Header bar + KPI tiles + AR1001 spotlight + pill filters + a
# hand-rolled sortable/filterable per-drug table. Plotly's Table trace
# doesn't support header-click sorting or being filtered by another
# chart's click events, so the table is plain HTML/JS driven off a
# JSON blob — still a single, fully standalone file, no server required.
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
    # AR1001 Relevance removed from this table — full score + reasons
    # still available via each row's detail-panel "Compare to AR1001"
    # button, the Phase 3 leaderboard, and the AR1001 Competitive
    # Landscape chart. Drug/Sponsor absorb the freed-up width.
    # Rebalanced from the original 24/21/10/9/12/11/7/6 split: Drug and
    # Sponsor had far more width than their content ever used, while
    # the short-header/short-value columns (Highest Phase, Status,
    # Trial Count, Enrollment) were narrow enough to truncate their own
    # header text into an ellipsis and hard-wrap single words like
    # "Completed" mid-word. Drug/Sponsor still get the two largest
    # shares (they hold the longest real content) but give up some
    # surplus room to the columns that actually needed it.
    ("display_name", "Drug", 19),
    ("sponsor", "Sponsor", 16),
    ("phase_reached", "Highest Phase", 12),
    ("status_summary", "Status", 11),
    ("target_display", "Target / Pathway", 13),
    ("drug_type", "Drug Type", 12),
    ("trial_count", "Trial Count", 9),
    ("max_enrollment", "Enrollment", 8),
    # Verification/Confidence/Review Status columns removed from the main
    # table per request — still available per-row via the details toggle
    # (the underlying data columns are kept in table_df below for that).
]

# Per-drug aggregate of TRIAL_SITE_STATUS (site_status, defined near the
# top of the file from trials.csv's Locations column): "Multicenter" if
# ANY of the drug's trials are multicenter, else "Single-site" if any
# trial has known single-site data, else "Unknown". A competitor running
# even one multicenter trial is worth flagging as a larger, later-stage
# program — unlike the "which drugs matter" framing earlier, this filter
# is specifically about spotting well-resourced competitor programs, so
# one multicenter trial dominating the label (rather than needing ALL
# trials to be multicenter) is the right rule here.
def _drug_site_design_status(nct_ids_str):
    ids = [i.strip() for i in str(nct_ids_str or "").split(";") if i.strip()]
    statuses = {TRIAL_SITE_STATUS.get(i, "unknown") for i in ids}
    if "multicenter" in statuses:
        return "Multicenter"
    if "single" in statuses:
        return "Single-site"
    return "Unknown"


resolved_drugs_df["site_design_status"] = resolved_drugs_df["nct_ids"].apply(_drug_site_design_status)

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
    "official_source_url", "classification_reason", "nct_ids", "brief_summary", "name_variants",
    "pipeline_scope", "scope_reason", "manual_review_required", "site_design_status",
    # Phase 2 — scientific classification (modality/target_pathways),
    # for the drug detail panel
    "target_pathways", "mechanism_of_action", "molecular_targets",
    "classification_source", "scientific_classification_confidence",
    "scientific_classification_reason", "scientific_manual_review_required",
    "therapeutic_purpose_class", "therapeutic_purpose_category", "cadro",
    "modality", "drug_type_source", "drug_type_inferred",
    "start_date_display", "primary_completion_date_display",
    "aribio_relevance_score", "aribio_relevance_reasons",
    # FDA status (STEP 3.72) — a genuinely separate regulatory signal,
    # never derived from trial-level status_summary; see fda_status.py.
    # Only the status itself is shown in the dashboard (as an
    # Intelligence pill) — the richer fields (indication/approval date/
    # source/notes/etc.) still exist on resolved_drugs_df and the
    # reference CSV for later use, just not sent to the browser while
    # nothing displays them.
    "fda_status",
]].copy()
table_records = json.loads(table_df.to_json(orient="records"))

# KPI tiles: Phase 1A narrows these to therapeutic_drugs_df — "Total
# drugs" must mean actual therapeutic drugs, not every resolved record.
total_drugs = len(therapeutic_drugs_df)
total_resolved_records = len(resolved_drugs_df)
phase3_agents = int((therapeutic_drugs_df["phase_reached"] == "Phase 3").sum())
phase2_agents = int((therapeutic_drugs_df["phase_reached"] == "Phase 2").sum())
phase1_agents = int((therapeutic_drugs_df["phase_reached"] == "Phase 1").sum())
# Reuses the same RELEVANCE_Y_DIVIDER (70) the AR1001 Competitive
# Landscape chart's quadrant split is built on -- one consistent "high
# relevance" threshold across the dashboard. AR1001 itself is excluded
# (its own self-score is never a meaningful "how relevant is this
# competitor" count).
high_relevance_agents = int((
    (therapeutic_drugs_df["aribio_relevance_score"] >= RELEVANCE_Y_DIVIDER) & (~therapeutic_drugs_df["is_aribio"])
).sum())

plotlyjs_lib = pyo.get_plotlyjs()  # loaded once at the top of the page; every figure below skips its own copy
HEATMAP_DIV_IDS = {"All": "heatmapAll", "Small Molecule": "heatmapSmallMolecule", "Biologic": "heatmapBiologic"}
heatmap_html = {
    label: pio.to_html(f, include_plotlyjs=False, full_html=False, div_id=HEATMAP_DIV_IDS[label])
    for label, f in heatmap_figs.items()
}
relevance_matrix_html = pio.to_html(relevance_matrix_fig, include_plotlyjs=False, full_html=False, div_id="relevanceMatrix")
RELEVANCE_MATRIX_SUBTITLE = "Which clinical-stage programs are most relevant to AR1001?"
RELEVANCE_MATRIX_EXPLANATION = (
    f"Top {RELEVANCE_MATRIX_TOP_N} competitors by AR1001 relevance score (AR1001 itself shown separately as "
    "the reference star). Further right = later development stage &middot; Higher = greater rule-based "
    "competitive relevance to AR1001 &middot; Upper-right = programs that may warrant closer competitive "
    "monitoring. Relevance is a deterministic, rule-based score &mdash; not a measure of clinical efficacy "
    "or probability of success."
)

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
    score = row.get("aribio_relevance_score")
    if pd.notna(score):
        score = int(score)
        if score >= 65:
            score_color = RELEVANCE_HIGH_COLOR
        elif score >= 35:
            score_color = RELEVANCE_MID_COLOR
        else:
            score_color = RELEVANCE_LOW_COLOR
        score_html = f'<span style="color:{score_color};font-weight:700;">{score}</span>'
    else:
        score_html = "&mdash;"
    return (
        f'<tr><td><span style="display:inline-block;width:8px;height:8px;border-radius:50%;'
        f'background:{dot_color};margin-right:8px;"></span>{name_html}</td>'
        f'<td>{row["target"]}</td><td>{status_text(row["status_summary"])}</td><td>{score_html}</td></tr>'
    )

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
# Same rule as everywhere else in this palette: ARIBIO_ACCENT reserved
# for the one value genuinely worth a second glance (an approval that
# was withdrawn); "Unknown"/"Not Applicable" get the same neutral gray
# STATUS_COLORS already uses for those concepts, not a shade of blue
# that would imply more certainty than a blank reference-file lookup has.
FDA_STATUS_COLORS = {
    "FDA Approved": darken(ARIBIO_BLUE, 0.30),
    "Under Review": ARIBIO_BLUE,
    "Not FDA Approved": "#9e9e9e",
    "Approval Withdrawn": ARIBIO_ACCENT,
    "Not Applicable": "#bdbdbd",
    "Unknown": "#bdbdbd",
}

# Same rule as FDA_STATUS_COLORS: ARIBIO_BLUE for the state worth a
# second look (Multicenter — the signal for a larger, later-stage
# competitor program), neutral grays for Single-site/Unknown so neither
# reads as a warning.
SITE_DESIGN_COLORS = {
    "Multicenter": ARIBIO_BLUE,
    "Single-site": "#9e9e9e",
    "Unknown": "#bdbdbd",
}

# Verification/Confidence/Review Status filter groups removed per
# request — VERIFICATION_COLORS/CONFIDENCE_COLORS/REVIEW_COLORS are
# still defined above and still used by the row-detail panel's pills.
PILL_GROUPS = [
    ("phase", "Phase", PHASE_ORDER, PHASE_COLORS),
    ("drugType", "Drug Type", list(DRUG_TYPE_COLORS.keys()), DRUG_TYPE_COLORS),
    ("target", "Target", [t for t in TARGET_COLORS if t not in ("Other", "Unknown")], TARGET_COLORS),
    ("status", "Status", [s for s in STATUS_COLORS if s != "Other"], STATUS_COLORS),
    ("siteDesign", "Trial Sites", ["Multicenter", "Single-site", "Unknown"], SITE_DESIGN_COLORS),
]

# Shortened DISPLAY text only — data-value (what filtering/sorting
# actually keys off) always stays the full canonical string, and the
# full name is still shown via the button's title="" tooltip. This
# exists purely so Phase (8 values) and Drug Type (4 long values) fit a
# compact 2-column grid without wrapping to 2-3 lines per pill; the
# group's own title (e.g. "DRUG TYPE") already supplies the context a
# shortened label like "Biologic" needs to stay unambiguous.
_PILL_SHORT_LABELS = {
    "Early Phase 1": "Early Ph 1",
    "Phase 1/Phase 2": "Ph 1/2",
    "Phase 2/Phase 3": "Ph 2/3",
    "Disease-Targeted Biologic": "Biologic",
    "Disease-Targeted Small Molecule": "Small Molecule",
    "Cognition Enhancer": "Cognition Enh.",
    "Neuropsychiatric Symptom Tx": "Neuropsychiatric",
}


def render_pill_group(field, title, values, colors):
    # Target is the one group that keeps its full color coding (a
    # colored dot per value) AND a single, full-label column — it's the
    # one dimension worth telling apart at a glance, per the existing
    # design. Phase/Drug Type/Status are plain monochrome text in a
    # tight 2-column grid — a single shared --pill-color (brand blue)
    # used only for the active/selected state, no per-value hue.
    show_dot = field == "target"
    dot_class = " filter-pill--dot" if show_dot else ""
    layout_class = " filter-pills--single" if field == "target" else ""
    pills = "".join(
        f'<button class="filter-pill{dot_class}" data-field="{field}" data-value="{v}" title="{v}" '
        + (f'style="--pill-color:{colors.get(v, "#999")}" ' if show_dot else "")
        + f'onclick="togglePill(this)">{_PILL_SHORT_LABELS.get(v, v)}</button>'
        for v in values
    )
    return (
        f'<div class="filter-group"><div class="filter-group-title">{title.upper()}</div>'
        f'<div class="filter-pills{layout_class}">{pills}</div></div>'
    )

pill_groups_html = "".join(render_pill_group(f, t, v, c) for f, t, v, c in PILL_GROUPS)

target_colors_js = json.dumps(TARGET_COLORS)
trial_site_status_js = json.dumps(TRIAL_SITE_STATUS)
phase_colors_js = json.dumps(PHASE_COLORS)
status_colors_js = json.dumps(STATUS_COLORS)
type_colors_js = json.dumps(DRUG_TYPE_COLORS)
verification_colors_js = json.dumps(VERIFICATION_COLORS)
confidence_colors_js = json.dumps(CONFIDENCE_COLORS)
fda_status_colors_js = json.dumps(FDA_STATUS_COLORS)

# AriBio-relevance-score color thresholds — same blue ramp as everything
# else on this dashboard, not a separate red/yellow/green "traffic
# light" palette.
RELEVANCE_HIGH_COLOR = darken(ARIBIO_BLUE, 0.30)
RELEVANCE_MID_COLOR = ARIBIO_BLUE
RELEVANCE_LOW_COLOR = "#9e9e9e"

phase3_rows_html = "".join(phase3_row_html(r) for _, r in phase3_top_relevant_df.head(PHASE3_PREVIEW_N).iterrows())
phase3_shown = min(PHASE3_PREVIEW_N, len(phase3_top_relevant_df))

records_js = json.dumps(table_records)
table_column_count = len(TABLE_COLUMNS)

today_str = date.today().isoformat()
dashboard_nav_bar = render_nav_bar("pipeline")

html_template = f"""
<style>
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: #f4f5f8; margin: 0; padding: 0 0 40px; color: #1a1a1a;
  }}
  /* sidebar floats as a detached card near the left edge (margin + radius
     + all-around shadow, not flush/docked); page-content is shifted right
     to clear it. No max-width on main anymore — the whole point of
     pulling the sidebar out here is to give the table (and everything
     else) the full remaining viewport width to span, instead of being
     capped at 1280px and centered. */
  .page-content {{ margin-left: 276px; }}
  /* asymmetric padding: sidebar already provides its own 16px gap on the
     left, so the extra breathing room belongs on the right, otherwise
     wide tables/cards run flush to the browser edge and feel crowded */
  main {{ margin: 0; padding: 24px 96px 0 24px; }}

  /* page title now lives in-flow at the top of main (above the AR1001
     spotlight) instead of a separate sticky header bar — dashboard_nav.py's
     nav bar is the only persistent top-of-page chrome now. */
  .page-title-block {{ margin-bottom: 20px; }}
  .page-title {{ font-size: 22px; font-weight: 700; letter-spacing: -0.01em; color: {NAV_BG}; }}
  .page-title-sub {{ font-size: 12.5px; color: #666; margin-top: 4px; }}

  .spotlight {{
    background: {ARIBIO_ACCENT_BG}; border: 1px solid {ARIBIO_ACCENT_BORDER}; border-left: 5px solid {ARIBIO_ACCENT};
    border-radius: {CARD_RADIUS}; padding: 14px 20px; margin-bottom: 20px; font-size: 14px; color: #4a1230;
  }}
  .spotlight b {{ color: {ARIBIO_ACCENT}; }}

  .kpi-row {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 14px; margin-bottom: 20px; }}
  @media (max-width: 1000px) {{ .kpi-row {{ grid-template-columns: repeat(3, 1fr); }} }}
  @media (max-width: 640px) {{ .kpi-row {{ grid-template-columns: repeat(2, 1fr); }} }}
  .kpi-tile {{
    background: white; border-radius: {CARD_RADIUS}; border-top: 3px solid {ARIBIO_BLUE};
    padding: 16px 18px; box-shadow: {CARD_SHADOW};
  }}
  .kpi-value {{ font-size: 27px; font-weight: 700; color: {ARIBIO_BLUE}; letter-spacing: -0.01em; }}
  .kpi-label {{ font-size: 12.5px; color: #666; margin-top: 3px; }}

  .sidebar {{
    /* top: 60px is a fallback (dash-nav's 44px height + 16px gap) for the
       instant before JS measures the real nav bar height and overrides it
       via positionSidebar() below — keeps things sane with JS disabled
       too. transition: top so any later re-measurement (window resize,
       text reflow) eases into place instead of snapping. */
    position: fixed; left: 16px; top: 60px; bottom: 16px; width: 240px;
    background: white; border-radius: {CARD_RADIUS}; box-shadow: {ELEVATED_SHADOW};
    z-index: 20; display: flex; flex-direction: column; overflow: hidden;
    transition: top 0.2s ease;
  }}
  .sidebar-header {{
    display: flex; align-items: center; justify-content: space-between; flex-shrink: 0;
    padding: 11px 14px; background: {NAV_BG};
  }}
  .sidebar-title {{
    display: flex; align-items: center; gap: 7px; font-size: 11.5px; font-weight: 700;
    color: white; text-transform: uppercase; letter-spacing: 0.05em;
  }}
  .sidebar-title svg {{ color: white; flex-shrink: 0; }}
  /* overflow-y stays auto as a fallback for genuinely short viewports —
     the sizing below is tuned to fit all 4 groups without scrolling on
     ordinary laptop/desktop viewport heights, not to physically prevent
     scrolling from ever being possible. */
  .filter-groups {{ flex: 1 1 auto; overflow-y: auto; padding: 4px 14px 12px; display: flex; flex-direction: column; }}
  .filter-group {{ padding: 9px 0; }}
  .filter-group:first-child {{ padding-top: 8px; }}
  .filter-group + .filter-group {{ border-top: 1px solid {SURFACE_BORDER}; }}
  .filter-group-title {{ font-size: 10px; letter-spacing: 0.06em; color: #9aa0ab; margin-bottom: 5px; font-weight: 600; }}
  /* 2-column grid is what makes Phase (8 values) and Status/Drug Type
     fit without scrolling — Target keeps its own single-column legend
     (see .filter-pills--single) since it's the one group meant to be
     scanned as a color-coded list, not a dense grid. */
  .filter-pills {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0 4px; }}
  .filter-pills--single {{ display: flex; flex-direction: column; gap: 0; }}
  @media (max-width: 960px) {{
    .sidebar {{
      position: static; width: auto; height: auto; box-shadow: {CARD_SHADOW};
      border-radius: {CARD_RADIUS}; margin: 0 24px 20px;
    }}
    .page-content {{ margin-left: 0; }}
    .filter-pills {{ grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); }}
  }}
  /* Clean list style, not a bubble/badge: plain text. No color-coding
     except the Target group (the one dimension worth telling apart at
     a glance) — those pills get the .filter-pill--dot modifier and a
     small colored dot; every other group is plain monochrome text.
     Active state reads like a selected nav item: a colored left accent
     bar + soft tinted background, rather than a filled pill. */
  .filter-pill {{
    --pill-color: {ARIBIO_BLUE}; display: flex; align-items: center; gap: 7px;
    background: none; border: none; border-radius: 5px;
    font-size: 12.5px; color: #444; padding: 4px 8px; width: 100%; text-align: left;
    cursor: pointer; font-family: inherit; line-height: 1.3;
    overflow-wrap: break-word; white-space: normal;
    transition: background-color 0.15s ease, color 0.15s ease,
                box-shadow 0.15s ease, transform 0.1s ease;
  }}
  .filter-pill--dot::before {{
    content: ""; width: 7px; height: 7px; min-width: 7px; border-radius: 50%; background: var(--pill-color);
    transition: box-shadow 0.15s ease;
  }}
  .filter-pill:hover {{ background: color-mix(in srgb, var(--pill-color) 7%, white); }}
  .filter-pill:active {{ transform: scale(0.97); }}
  /* Active/selected state: a soft tint of the pill's OWN accent color
     (color-mix, not a flat gray) plus a matching inset ring — reads as
     a real "selected chip," and for Target specifically the highlight
     is literally that pathway's color, not one generic active-blue for
     everything. No left accent bar — the tint + ring alone carry the
     "selected" signal. Falls back gracefully to the old flat-gray look
     on any browser without color-mix() support (Safari <16.4 etc.). */
  .filter-pill.active {{
    /* No font-weight bump here on purpose: the tint + ring already
       carry the "selected" signal (see comment above), and bold text
       is measurably WIDER than regular text at the same font-size --
       in this pill's fixed-width grid cell, that was enough extra
       width to wrap a longer label (e.g. "FDA Approved") onto a
       second line the moment it became active. */
    color: var(--pill-color);
    background: color-mix(in srgb, var(--pill-color) 13%, white);
    box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--pill-color) 30%, transparent);
  }}
  .filter-pill.active:hover {{ background: color-mix(in srgb, var(--pill-color) 19%, white); }}
  .filter-pill.active.filter-pill--dot::before {{
    box-shadow: 0 0 0 3px color-mix(in srgb, var(--pill-color) 22%, transparent);
  }}
  #clear-filter {{
    font-size: 12px; font-weight: 600; color: {ARIBIO_ACCENT}; background: none; border: none;
    cursor: pointer; padding: 2px 0; visibility: hidden; font-family: inherit;
  }}
  #clear-filter:hover {{ text-decoration: underline; }}
  #clear-filter.visible {{ visibility: visible; }}

  h2.section-title {{ color: {NAV_BG}; font-size: 18px; font-weight: 700; letter-spacing: -0.01em; margin: 30px 0 6px; }}
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
    position: sticky; top: 0; background: {NAV_BG}; color: white; text-align: left;
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
  /* plain colored text, not a badge — color set inline per-value in JS.
     NOT white-space:nowrap: with table-layout:fixed, a nowrap pill
     longer than its column (e.g. a multi-target "Multi (Amyloid/Tau/
     Neuroprotection)" label) doesn't get clipped by the cell — it
     visually overflows into the NEXT column's text instead, since
     table cells don't clip overflowing children by default. Letting it
     wrap (the default) keeps long pills inside their own cell/row. */
  .pill {{ font-weight: 600; overflow-wrap: break-word; }}

  .details-toggle {{
    background: none; border: none; cursor: pointer; font-size: 11px; color: #999;
    padding: 0 4px 0 0; font-family: inherit; vertical-align: middle;
  }}
  .details-toggle:hover {{ color: {ARIBIO_BLUE}; }}
  /* caret rotates smoothly instead of swapping glyphs, so opening/
     closing a row's detail panel reads as one continuous motion */
  .details-toggle .caret {{ display: inline-block; transition: transform 0.15s ease; }}
  .details-toggle .caret.expanded {{ transform: rotate(90deg); }}
  /* Drug name in the main table acts as a second trigger for the same
     row-expand detail panel as the caret toggle -- deliberately styled
     as plain text, not a hyperlink, since it no longer navigates
     anywhere itself; individual trial links live inside the detail
     panel (see the Trial IDs row in renderDetailRow). */
  .drug-name-toggle {{
    background: none; border: none; cursor: pointer; font-family: inherit; font-size: inherit;
    color: inherit; padding: 0; text-decoration: none;
  }}
  .drug-name-toggle:hover {{ color: {ARIBIO_BLUE}; }}
  .detail-trial-links {{ display: flex; flex-wrap: wrap; gap: 4px 10px; }}
  .detail-trial-link {{ display: inline-flex; align-items: center; gap: 5px; }}
  .detail-trial-links a {{ color: {ARIBIO_BLUE} !important; text-decoration: none !important; }}
  .detail-trial-links a:hover {{ text-decoration: underline !important; }}
  .site-badge {{ display: inline-block; padding: 1px 6px; border-radius: 8px; font-size: 10px; font-weight: 600; line-height: 1.5; white-space: nowrap; }}
  .site-badge--multi {{ background: color-mix(in srgb, {ARIBIO_BLUE} 15%, white); color: {darken(ARIBIO_BLUE, 0.15)}; }}
  .site-badge--unknown {{ background: #ececec; color: #888; }}
  tr.detail-row td {{ background: {SURFACE_TINT}; padding: 14px 20px; border-bottom: 1px solid {SURFACE_BORDER}; }}
  /* CSS transitions don't run on elements created via innerHTML (they
     already exist in their end state) — a keyframe animation, in
     contrast, plays automatically whenever the element is inserted
     into the DOM, which is what happens every time renderTable()
     rebuilds the tbody after a toggle. That's what makes the detail
     panel fade/slide in smoothly instead of just popping into place. */
  @keyframes detailPanelIn {{
    from {{ opacity: 0; transform: translateY(-4px); }}
    to {{ opacity: 1; transform: translateY(0); }}
  }}
  /* played on the CLOSING panel before it's removed from the DOM (see
     the click handler below, which waits for 'animationend' before
     actually re-rendering the table without this row) — the reverse of
     detailPanelIn, so opening and closing read as one continuous motion */
  @keyframes detailPanelOut {{
    from {{ opacity: 1; transform: translateY(0); }}
    to {{ opacity: 0; transform: translateY(-4px); }}
  }}
  .detail-panel {{
    font-size: 12.5px; color: #444; animation: detailPanelIn 0.18s ease;
  }}
  .detail-panel.closing {{ animation: detailPanelOut 0.15s ease forwards; }}
  .detail-panel strong {{ display: block; color: #999; font-size: 10.5px; letter-spacing: 0.04em; text-transform: uppercase; margin-bottom: 3px; }}
  .detail-panel ul {{ margin: 0; padding-left: 16px; }}
  /* Detail panel is grouped into Clinical / Scientific / Mechanism /
     Intelligence side-by-side, then Provenance as a full-width band
     underneath — Provenance answers "how much to trust the fields
     above", a different kind of information than the facts themselves,
     so it doesn't compete with them as a 5th equal-weight column. FDA
     status lives in Intelligence as a single pill (see FDA_STATUS_COLORS/
     fda_status.py) rather than its own column — deliberately never
     folded into the existing trial-derived Status pill, but not
     substantial enough on its own to earn a whole group either. */
  .detail-groups {{
    display: grid; grid-template-columns: 0.85fr 1.25fr 1.5fr 0.95fr; gap: 0;
  }}
  .detail-group {{ padding-right: 18px; }}
  .detail-group + .detail-group {{ border-left: 1px solid {SURFACE_BORDER}; padding-left: 18px; }}
  .detail-group h4 {{
    font-size: 10px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase;
    color: #9aa0ab; margin: 0 0 11px;
  }}
  .detail-group > div {{ margin-bottom: 11px; }}
  .detail-group > div:last-child {{ margin-bottom: 0; }}
  .detail-mechanism-text {{ font-size: 12.5px; line-height: 1.55; color: #444; max-width: 48ch; }}
  .detail-provenance {{
    grid-column: 1 / -1; border-left: none !important; padding-left: 0 !important;
    border-top: 1px solid {SURFACE_BORDER}; margin-top: 16px; padding-top: 14px;
  }}
  .detail-provenance-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 10px 22px;
  }}
  @media (max-width: 1050px) {{
    .detail-groups {{ grid-template-columns: 1fr; }}
    .detail-group + .detail-group {{
      border-left: none; padding-left: 0;
      border-top: 1px solid {SURFACE_BORDER}; padding-top: 14px; margin-top: 3px;
    }}
  }}
  .sponsor-cell {{ cursor: help; }}

  .compare-btn {{
    background: none; border: none; cursor: pointer; font-size: 12px; color: #999;
    padding: 1px 3px; border-radius: 4px; font-family: inherit; vertical-align: middle;
    transition: background-color 0.15s ease, color 0.15s ease;
  }}
  .compare-btn:hover {{ color: {ARIBIO_BLUE}; background: color-mix(in srgb, {ARIBIO_BLUE} 8%, white); }}

  /* Drug Comparator modal — AR1001 vs. a selected drug, side by side,
     using only fields this pipeline actually resolves with real
     evidence (Mechanism of Action, Modality, Target Pathway(s), Phase,
     Sponsor, Status) — no invented Route/Biomarker/Population rows. */
  #comparator-overlay {{
    display: none; position: fixed; inset: 0; background: rgba(20, 30, 45, 0.45);
    z-index: 100; align-items: center; justify-content: center; padding: 24px;
    animation: overlayIn 0.15s ease;
  }}
  #comparator-overlay.visible {{ display: flex; }}
  @keyframes overlayIn {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
  #comparator-card {{
    background: white; border-radius: {CARD_RADIUS}; box-shadow: {ELEVATED_SHADOW};
    max-width: 640px; width: 100%; max-height: 84vh; overflow-y: auto;
    animation: comparatorIn 0.18s ease;
  }}
  @keyframes comparatorIn {{
    from {{ opacity: 0; transform: translateY(8px) scale(0.98); }}
    to {{ opacity: 1; transform: translateY(0) scale(1); }}
  }}
  #comparator-header {{
    display: flex; align-items: center; justify-content: space-between;
    padding: 16px 20px; border-bottom: 1px solid {SURFACE_BORDER}; position: sticky; top: 0;
    background: white; border-radius: {CARD_RADIUS} {CARD_RADIUS} 0 0;
  }}
  #comparator-header h3 {{ margin: 0; font-size: 16px; font-weight: 700; letter-spacing: -0.01em; }}
  #comparator-close {{
    background: none; border: none; cursor: pointer; font-size: 20px; line-height: 1; color: #999;
    padding: 4px 6px; border-radius: 6px; font-family: inherit;
  }}
  #comparator-close:hover {{ color: #1a1a1a; background: {SURFACE_TINT}; }}
  #comparator-relevance {{
    margin: 16px 20px 0; padding: 12px 14px; border-radius: {CARD_RADIUS};
    background: {SURFACE_TINT}; font-size: 13px;
  }}
  #comparator-relevance .score {{ font-size: 20px; font-weight: 700; }}
  #comparator-relevance ul {{ margin: 6px 0 0; padding-left: 18px; color: #555; }}
  table#comparator-table {{ width: 100%; border-collapse: collapse; margin: 16px 20px 20px; width: calc(100% - 40px); }}
  table#comparator-table th {{
    text-align: left; font-size: 10.5px; letter-spacing: 0.04em; text-transform: uppercase; color: #999;
    padding: 8px; border-bottom: 1px solid {SURFACE_BORDER};
  }}
  table#comparator-table td {{ padding: 10px 8px; border-bottom: 1px solid #f2f2f2; font-size: 13.5px; vertical-align: top; }}
  table#comparator-table td:first-child {{ color: #999; font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.03em; white-space: nowrap; }}

  .glance-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 8px; align-items: start; }}
  .glance-panel {{ background: white; border-radius: {CARD_RADIUS}; padding: 18px; box-shadow: {CARD_SHADOW}; }}
  .glance-panel-title {{ font-size: 15px; font-weight: 700; letter-spacing: -0.01em; margin-bottom: 4px; color: {NAV_BG}; }}
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
  {COMPETITIVE_ATTENTION_CSS}
  {NAV_CSS}
</style>

<script>{plotlyjs_lib}</script>

{dashboard_nav_bar}

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
    <div class="page-title-block">
      <div class="page-title">Alzheimer's Disease Clinical Trial Pipeline</div>
      <div class="page-title-sub">Source: clinicaltrials.gov &middot; Updated {today_str} &middot; {len(df)} trials analyzed &middot; {total_drugs} therapeutic drugs ({total_resolved_records} total resolved records)</div>
    </div>

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
      <div class="kpi-tile"><div class="kpi-value" style="color:{ARIBIO_ACCENT}">{high_relevance_agents}</div><div class="kpi-label">High relevance to AR1001 (&ge;{RELEVANCE_Y_DIVIDER})</div></div>
    </div>

    {COMPETITIVE_ATTENTION_PLACEHOLDER}

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
        <div class="glance-panel-title">Phase 3 agents &mdash; top by AR1001 relevance
          <span style="font-weight:400;font-size:12px;color:#999;">(showing top {phase3_shown} of {len(phase3_top_relevant_df)} competitors)</span>
        </div>
        <table class="phase3-table">
          <thead><tr><th>Drug</th><th>Target</th><th>Status</th><th>Relevance</th></tr></thead>
          <tbody>{phase3_rows_html}</tbody>
        </table>
        <span id="show-all-phase3" onclick="showAllPhase3()">Show all {len(phase3_df)} Phase 3 agents in the table below &darr;</span>
      </div>
    </div>

    <div class="glance-panel" style="margin-bottom: 20px;">
      <div class="glance-panel-title">AR1001 Competitive Landscape</div>
      <div class="section-hint" style="margin-top:-2px;">{RELEVANCE_MATRIX_SUBTITLE}</div>
      <div style="margin-top:10px;">{relevance_matrix_html}</div>
      <div class="section-hint">{RELEVANCE_MATRIX_EXPLANATION}</div>
    </div>

    {COMPETITIVE_MILESTONES_PLACEHOLDER}

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

<div id="comparator-overlay" onclick="if (event.target === this) closeComparator();">
  <div id="comparator-card">
    <div id="comparator-header">
      <h3 id="comparator-title">Drug Comparator</h3>
      <button id="comparator-close" onclick="closeComparator()" title="Close">&times;</button>
    </div>
    <div id="comparator-relevance"></div>
    <table id="comparator-table">
      <thead><tr><th>Feature</th><th>AR1001</th><th id="comparator-other-header">Competitor</th></tr></thead>
      <tbody id="comparator-body"></tbody>
    </table>
  </div>
</div>

<script id="drug-data" type="application/json">{records_js}</script>
<script>
  const ALL_ROWS = JSON.parse(document.getElementById('drug-data').textContent);
  const TARGET_COLORS = {target_colors_js};
  const TRIAL_SITE_STATUS = {trial_site_status_js};
  const PHASE_COLORS = {phase_colors_js};
  const STATUS_COLORS = {status_colors_js};
  const TYPE_COLORS = {type_colors_js};
  const VERIFICATION_COLORS = {verification_colors_js};
  const CONFIDENCE_COLORS = {confidence_colors_js};
  const FDA_STATUS_COLORS = {fda_status_colors_js};
  const TABLE_COLUMN_COUNT = {table_column_count};
  // maps a pill "field" key to the actual column name on each row
  const FIELD_TO_COLUMN = {{
    phase: 'phase_reached', drugType: 'drug_type', target: 'target', status: 'status_summary',
    siteDesign: 'site_design_status',
  }};

  let sortKey = 'phase_reached';
  let sortAsc = false;
  let filters = {{
    phase: new Set(), drugType: new Set(), target: new Set(), status: new Set(),
    siteDesign: new Set(),
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

  // rule-based score thresholds (see competitive_intelligence.py) — NOT
  // a red/yellow/green traffic light, the same brand blue ramp as
  // everything else on this dashboard.
  function relevanceColor(score) {{
    if (score >= 65) return '{RELEVANCE_HIGH_COLOR}';
    if (score >= 35) return '{RELEVANCE_MID_COLOR}';
    return '{RELEVANCE_LOW_COLOR}';
  }}

  function anyFiltersActive() {{
    return Object.values(filters).some(s => s.size > 0);
  }}

  function updateClearButton() {{
    document.getElementById('clear-filter').classList.toggle('visible', anyFiltersActive());
  }}

  // mirrors drug_classification.py's normalize_text() — lowercase, and
  // collapse any run of non-alphanumeric characters to a single space —
  // so a search for "ACP204" still finds "ACP-204" (same drug, just
  // hyphen vs. no hyphen), matching how the rest of this project already
  // treats punctuation/spacing as insignificant everywhere else.
  function normalizeSearchText(value) {{
    return String(value == null ? '' : value).toLowerCase().replace(/[^a-z0-9]+/g, ' ').trim();
  }}

  function matchesSearch(r, term) {{
    const haystack = normalizeSearchText([
      r.display_name, r.sponsor, r.synonyms, r.verification_label, r.confidence_label, r.target_display,
    ].filter(Boolean).join(' '));
    return haystack.includes(term);
  }}

  // Badge is deliberately silent for "single" — a single confirmed
  // site is the unmarked default, so only the two cases worth a second
  // look (multicenter, or no location data on file) get a visible tag.
  function siteStatusBadge(nctId) {{
    const status = TRIAL_SITE_STATUS[nctId];
    if (status === 'multicenter') return '<span class="site-badge site-badge--multi" title="2+ distinct sites in this trial\\'s location data">Multicenter</span>';
    if (status === 'unknown') return '<span class="site-badge site-badge--unknown" title="No location data on file for this trial">No site data</span>';
    return '';
  }}

  function renderTrialLinks(nctIdsField) {{
    const trialIds = String(nctIdsField || '').split(';').map(s => s.trim()).filter(Boolean);
    if (!trialIds.length) return '—';
    return `<div class="detail-trial-links">${{trialIds.map(id =>
      `<span class="detail-trial-link"><a href="https://clinicaltrials.gov/study/${{encodeURIComponent(id)}}" target="_blank" rel="noopener">${{escapeHtml(id)}}</a>${{siteStatusBadge(id)}}</span>`
    ).join('')}}</div>`;
  }}

  function renderDetailRow(r) {{
    const sponsors = String(r.sponsor || '').split('; ').filter(Boolean);
    const sponsorList = sponsors.length
      ? `<ul>${{sponsors.map(s => `<li>${{escapeHtml(s)}}</li>`).join('')}}</ul>`
      : '—';
    // target_pathways is a "; "-joined string of canonical TARGET_COLORS
    // vocabulary (Amyloid/Tau/.../Metabolism/...) — reuse the same pill()
    // helper the main table/filters use, one chip per pathway, rather
    // than a fixed color for the whole field.
    const targetChips = String(r.target_pathways || '').split('; ').filter(Boolean)
      .map(t => pill(t, TARGET_COLORS)).join(' ') || '—';
    const relevanceBlock = r.display_name === 'AR1001'
      ? '<span style="color:#999;">Reference</span>'
      : `<span style="color:${{relevanceColor(r.aribio_relevance_score)}};font-weight:700;">${{r.aribio_relevance_score}}/100</span>
         <div style="height:5px;border-radius:3px;background:{SURFACE_BORDER};overflow:hidden;margin:5px 0 8px;max-width:140px;">
           <div style="height:100%;border-radius:3px;width:${{r.aribio_relevance_score}}%;background:${{relevanceColor(r.aribio_relevance_score)}};"></div>
         </div>
         <button class="compare-btn" onclick="event.stopPropagation(); openComparator('${{escapeHtml(r.display_name)}}')" title="Compare to AR1001">&#8646; Compare to AR1001</button>`;
    // No verified mechanism on file for most rows (see brief_summary's
    // definition in drug_classification.py) — fall back to the trial's
    // own Brief Summary, labeled as exactly that, never as "Mechanism",
    // so a trial-design blurb never reads as pharmacology.
    const hasMechanism = !!(r.mechanism_of_action && String(r.mechanism_of_action).trim());
    const mechanismTitle = hasMechanism ? 'Mechanism' : 'Brief Summary';
    const mechanismText = hasMechanism ? r.mechanism_of_action : (r.brief_summary || '—');
    return `<tr class="detail-row"><td colspan="${{TABLE_COLUMN_COUNT}}"><div class="detail-panel"><div class="detail-groups">
      <div class="detail-group">
        <h4>Clinical</h4>
        <div><strong>Start date</strong>${{escapeHtml(r.start_date_display || 'TBD')}}</div>
        <div><strong>Primary completion</strong>${{escapeHtml(r.primary_completion_date_display || 'TBD')}}</div>
        <div><strong>Trial IDs</strong>${{renderTrialLinks(r.nct_ids)}}</div>
      </div>
      <div class="detail-group">
        <h4>Scientific</h4>
        <div><strong>Modality</strong>${{escapeHtml(r.modality || '—')}}</div>
        <div><strong>Drug type category${{r.drug_type_inferred ? ' (inferred)' : ' (NIH-sourced)'}}</strong>${{escapeHtml(r.drug_type || '—')}}</div>
        <div><strong>Target pathway(s)</strong>${{targetChips}}</div>
        <div><strong>Molecular target(s)</strong>${{escapeHtml(r.molecular_targets || '—')}}</div>
        <div><strong>CADRO category</strong>${{escapeHtml(r.cadro || '—')}}</div>
        <div><strong>Therapeutic purpose</strong>${{escapeHtml([r.therapeutic_purpose_class, r.therapeutic_purpose_category].filter(Boolean).join(' · ') || '—')}}</div>
      </div>
      <div class="detail-group">
        <h4>${{mechanismTitle}}</h4>
        <div class="detail-mechanism-text">${{escapeHtml(mechanismText)}}</div>
      </div>
      <div class="detail-group">
        <h4>Intelligence</h4>
        <div><strong>Verification</strong>${{pill(r.verification_label, VERIFICATION_COLORS)}}</div>
        <div><strong>Confidence</strong>${{pill(r.confidence_label, CONFIDENCE_COLORS)}}</div>
        <div><strong>FDA status</strong>${{pill(r.fda_status, FDA_STATUS_COLORS)}}</div>
        <div><strong>AR1001 Relevance</strong>${{relevanceBlock}}</div>
      </div>
      <div class="detail-group detail-provenance">
        <h4>Provenance &mdash; how much to trust the fields above</h4>
        <div class="detail-provenance-grid">
          <div><strong>Sponsors</strong>${{sponsorList}}</div>
          <div><strong>Name variants merged</strong>${{escapeHtml(r.name_variants || '—')}}</div>
          <div><strong>Pipeline scope</strong>${{escapeHtml(r.pipeline_scope || '—')}}</div>
          <div><strong>Scope reason</strong>${{escapeHtml(r.scope_reason || '—')}}</div>
          <div><strong>Notes</strong>${{escapeHtml(r.classification_reason || '—')}}</div>
          <div><strong>Confirmed trials</strong>${{r.confirmed_trial_count}}</div>
          <div><strong>Unverified trials</strong>${{r.unverified_trial_count}}</div>
          <div><strong>Classification source</strong>${{escapeHtml(r.classification_source || '—')}} (${{escapeHtml(r.scientific_classification_confidence || '—')}} confidence)</div>
        </div>
      </div>
    </div></div></td></tr>`;
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
        if (set.size === 0) continue;
        if (field === 'status') {{
          // The "FDA Approved" pill checks the curated fda_status field
          // (see fda_status.py) — never status_summary, even though
          // STATUS_MAP can also produce a trial-level "FDA Approved"
          // label (from ct.gov's APPROVED_FOR_MARKETING). Those two are
          // deliberately different claims; this pill means the real one.
          const matchesStatus = [...set].some(v =>
            v === 'FDA Approved' ? r.fda_status === 'FDA Approved' : r.status_summary === v
          );
          if (!matchesStatus) return false;
          continue;
        }}
        if (!set.has(r[FIELD_TO_COLUMN[field]])) return false;
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
      const toggle = `<button class="details-toggle" data-drug-key="${{escapeHtml(r.display_name)}}" title="Show details"><span class="caret${{isExpanded ? ' expanded' : ''}}">\\u25b8</span></button>`;
      // Drug name is a second trigger for the SAME row-expand detail
      // panel as the caret (same class + data-drug-key, so the
      // existing #drug-table-body click handler picks it up as-is) --
      // the detail panel is where individual trial links live now
      // (see renderDetailRow's Trial links row), not a direct
      // navigation link to a single, arbitrarily-chosen trial.
      const nameBtn = `<button type="button" class="details-toggle drug-name-toggle" data-drug-key="${{escapeHtml(r.display_name)}}">${{escapeHtml(r.display_name)}}</button>`;
      const mainRow = `<tr class="${{classes.join(' ')}}">
        <td>${{toggle}} ${{star}}${{nameBtn}}</td>
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
    searchTerm = normalizeSearchText(e.target.value);
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

    if (expandedRows.has(key)) {{
      // Closing: renderTable() rebuilds the tbody via innerHTML, which
      // would otherwise delete the detail row INSTANTLY with no chance
      // to animate — so play the closing keyframe on the row that's
      // already in the DOM first, and only drop it from expandedRows
      // (then re-render) once that animation actually finishes.
      const mainRow = btn.closest('tr');
      const panel = mainRow && mainRow.nextElementSibling
        ? mainRow.nextElementSibling.querySelector('.detail-panel') : null;
      if (panel) {{
        panel.classList.add('closing');
        panel.addEventListener('animationend', () => {{
          expandedRows.delete(key);
          renderTable();
        }}, {{ once: true }});
        return;
      }}
      expandedRows.delete(key);
    }} else {{
      expandedRows.add(key);
    }}
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

  // --- Drug Comparator: AR1001 vs. a selected drug, side by side.
  // Only fields this pipeline actually resolves with real evidence are
  // shown (Mechanism of Action, Modality, Target Pathway(s), Phase,
  // Sponsor, Status) — no Route/Biomarker/Population rows, since this
  // dataset doesn't track those and inventing values for them would be
  // worse than not showing them at all.
  const COMPARATOR_FIELDS = [
    ['Mechanism of action', r => r.mechanism_of_action || 'Not reported'],
    ['Modality', r => r.modality || 'Unknown'],
    ['Target pathway(s)', r => r.target_pathways || 'Other'],
    ['Phase', r => r.phase_reached],
    ['Sponsor', r => r.sponsor_display || r.sponsor || 'Unknown'],
    ['Status', r => r.status_summary],
  ];

  function openComparator(displayName) {{
    const other = ALL_ROWS.find(r => r.display_name === displayName);
    const reference = ALL_ROWS.find(r => r.display_name === 'AR1001');
    if (!other || !reference) return;

    document.getElementById('comparator-other-header').textContent = other.display_name;

    document.getElementById('comparator-body').innerHTML = COMPARATOR_FIELDS.map(([label, get]) =>
      `<tr><td>${{escapeHtml(label)}}</td><td>${{escapeHtml(get(reference))}}</td><td>${{escapeHtml(get(other))}}</td></tr>`
    ).join('');

    const score = other.aribio_relevance_score;
    const reasonsList = (other.aribio_relevance_reasons || '').split('; ').filter(Boolean);
    document.getElementById('comparator-relevance').innerHTML = `
      <div>Relevance to AR1001: <span class="score" style="color:${{relevanceColor(score)}}">${{score}}/100</span></div>
      ${{reasonsList.length
        ? `<ul>${{reasonsList.map(x => `<li>${{escapeHtml(x)}}</li>`).join('')}}</ul>`
        : '<div style="color:#999;margin-top:4px;">No shared profile with AR1001 on the dimensions this pipeline tracks.</div>'}}
      <div style="color:#999;font-size:11px;margin-top:6px;">Rule-based score, not AI-generated — see competitive_intelligence.py.</div>
    `;

    document.getElementById('comparator-overlay').classList.add('visible');
  }}

  function closeComparator() {{
    document.getElementById('comparator-overlay').classList.remove('visible');
  }}

  document.addEventListener('keydown', (e) => {{
    if (e.key === 'Escape') closeComparator();
  }});

  // dashboard_nav.py's nav bar spans the full page width (it's a sibling
  // of the sidebar, not nested inside it), so the floating sidebar needs
  // to start below it rather than at the very top of the viewport. This
  // measures the nav bar's REAL rendered height (rather than hardcoding
  // an estimate) so it stays correct if text wraps differently across
  // browsers/zoom levels — and correctly collapses to ~0 when the nav
  // bar hides itself (see dashboard_nav.py) because this page is loaded
  // inside dashboard.html's own tab shell.
  function positionSidebar() {{
    const header = document.querySelector('.dash-nav');
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
    "diagnostic_subtype",
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
    "diagnostic_subtype",
    # Phase 2 — scientific classification (drug_type/target above are
    # now SOURCED from this resolution; these columns carry the full
    # provenance for review in Excel)
    "target_pathways", "mechanism_of_action", "molecular_targets",
    "classification_source", "classification_method",
    "scientific_classification_confidence", "scientific_classification_reason",
    "evidence_used", "scientific_manual_review_required",
    "therapeutic_purpose_class", "therapeutic_purpose_category", "cadro",
    "modality", "drug_type_source", "drug_type_inferred",
    "start_date_display", "primary_completion_date_display",
    "aribio_relevance_score", "aribio_relevance_reasons",
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
    "diagnostic_subtype",
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
