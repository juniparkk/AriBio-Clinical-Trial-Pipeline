# ============================================================
# SCIENTIFIC CLASSIFICATION RESOLVER — Phase 2
#
# Re-derives each canonical THERAPEUTIC DRUG's modality/target-pathway/
# mechanism-of-action from verified, drug-centric evidence, replacing
# pipeline_viz.py's legacy guess_drug_type()/guess_target() as the
# SOURCE for resolved_drugs_df's drug_type/target columns. Those two
# legacy functions still exist in pipeline_viz.py and still populate
# the raw TRIAL-level df["drug_type"]/df["target"] columns used in
# pipeline_annotated.csv (kept intentionally, as a legacy point of
# comparison, per this phase's "do not change... change tracking"
# instruction) — they are simply no longer the source of truth for the
# per-DRUG dashboard fields once this resolver runs.
#
# Classification priority (highest wins; see resolve_drug_classification()):
#   1. Curated AriBio overrides   (data/reference/drug_classification_overrides.csv)
#   2. NIH reference              (CADRO + DTT modality, from nih_reference.py)
#   3. Official pipeline reference (data/official_pipeline.csv, extended
#      with optional modality/target_pathways columns)
#   4. Exact alias/known-compound match (KNOWN_COMPOUND_TARGETS below —
#      the same curated ~150-compound table pipeline_viz.py's
#      guess_target() already uses, exact-match only, not fuzzy)
#   5. Structured ClinicalTrials.gov evidence — the VERIFIED intervention
#      type/name text tied to this specific drug (via candidate_name/
#      developed_drug_normalized), never the whole trial's raw,
#      possibly-multi-intervention Interventions cell
#   6. Claude inference — NOT implemented in this offline, deterministic
#      pipeline (no LLM API call is wired into pipeline_viz.py's batch
#      run). This tier is a documented placeholder: see
#      claude_infer_classification()'s docstring. It always defers to
#      tier 7 today rather than fabricate an unverified guess.
#   7. Needs Review (Unknown modality / Other target, low confidence)
#
# Like drug_classification.py and nih_reference.py, this module only
# DEFINES functions — no file I/O or printing at import time.
# ============================================================

import re

import pandas as pd

from drug_classification import normalize_text
from nih_reference import extract_canonical_and_aliases, infer_nih_target, _extract_nih_modality

# ============================================================
# TIER 1: CURATED ARIBIO OVERRIDES
# ============================================================

REQUIRED_DRUG_OVERRIDE_COLUMNS = [
    "normalized_drug_name", "modality", "target_pathways", "molecular_targets",
    "mechanism_of_action", "reason", "source", "reviewer", "verified_date",
]


def load_drug_classification_overrides(path):
    """
    Read data/reference/drug_classification_overrides.csv into a dict
    keyed by normalized canonical drug name — same read-only,
    missing-file-tolerant pattern as load_official_pipeline() and
    load_scope_overrides(): a missing file degrades to "no curated
    overrides" ({}) rather than crashing the pipeline.

    target_pathways/molecular_targets are semicolon-separated in the CSV
    (e.g. "Amyloid; Tau; Neuroprotection") and parsed into lists here.
    """
    try:
        raw_df = pd.read_csv(path, dtype=str)
    except FileNotFoundError:
        return {}

    missing = [c for c in REQUIRED_DRUG_OVERRIDE_COLUMNS if c not in raw_df.columns]
    if missing:
        raise ValueError(f"drug_classification_overrides.csv at {path!r} is missing required column(s): {missing}")

    raw_df = raw_df.fillna("")

    overrides = {}
    for _, row in raw_df.iterrows():
        key = normalize_text(row["normalized_drug_name"])
        if not key:
            continue
        overrides[key] = {
            "modality": str(row["modality"]).strip(),
            "target_pathways": [t.strip() for t in str(row["target_pathways"]).split(";") if t.strip()],
            "molecular_targets": [t.strip() for t in str(row["molecular_targets"]).split(";") if t.strip()],
            "mechanism_of_action": str(row["mechanism_of_action"]).strip(),
            "reason": str(row["reason"]).strip(),
            "source": str(row["source"]).strip(),
            "reviewer": str(row["reviewer"]).strip(),
            "verified_date": str(row["verified_date"]).strip(),
        }
    return overrides


# ============================================================
# TIER 4: EXACT KNOWN-COMPOUND / ALIAS MATCH
#
# TEMPORARY DUPLICATION (same caveat as drug_classification.py's
# KNOWN_COMPOUND_NAMES): copied from pipeline_viz.py's KNOWN_COMPOUNDS
# rather than imported, because importing pipeline_viz.py directly would
# execute its entire read-trials.csv/write-output pipeline as an import
# side effect. If either copy is edited, edit the other to match.
# Unlike drug_classification.py's copy (names only, no pathway), this
# one keeps the pathway VALUE — that's the whole point of tier 4.
# ============================================================

KNOWN_COMPOUND_TARGETS = {
    # --- Amyloid: BACE / gamma-secretase inhibitors ---
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
    # --- Amyloid: amyloid PET tracers (retained for tier-4 name lookups;
    # these never reach resolve_drug_classification() as a THERAPEUTIC
    # drug in the first place — classify_pipeline_scope() already routes
    # them to Diagnostic Agent — kept here only for completeness/parity
    # with the source table) ---
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
    # --- Tau PET tracers (see amyloid-tracer note above) ---
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
    "mirodenafil": "Neuroprotection",
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

    # --- Symptomatic: cholinesterase inhibitors / NMDA / nicotinic ---
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
# "ar1001" deliberately EXCLUDED from this dict (unlike pipeline_viz.py's
# copy, which maps it to a single "Neuroprotection" bucket): AR1001 is a
# genuinely multi-pathway compound (amyloid + tau + neuroprotection
# evidence per AAIC 2026 Phase 2 data) and belongs in the curated
# AriBio-override tier (1), which outranks this tier and correctly
# supports multiple target_pathways — a single-value dict entry here
# would silently re-introduce the single-pathway limitation this phase
# is meant to fix. See data/reference/drug_classification_overrides.csv.

# modality (small molecule vs. biologic vs. ...) exact-name reference —
# the same compounds tier 4 already covers, tagged by known modality
# where publicly established. Deliberately small and conservative: only
# includes names where the modality is unambiguous public knowledge
# (monoclonal antibodies ending -mab, well-known biologics/vaccines).
KNOWN_COMPOUND_MODALITY = {
    name: "Biologic" for name in [
        "bapineuzumab", "solanezumab", "gantenerumab", "crenezumab", "aducanumab",
        "lecanemab", "donanemab", "remternetug", "trontinemab", "ponezumab",
        "gosuranemab", "biib092", "tilavonemab", "abbv-8e12", "semorinemab",
        "ro7105705", "zagotenemab", "ly3303560", "bepranemab", "ucb0107",
        "jnj-63733657", "biib080", "etanercept", "sargramostim", "al002", "al003",
        "xpro1595", "shr-1707",
    ]
}

# Both dicts above are keyed by the RAW compound string (e.g. "abt-089",
# with a literal hyphen — matching how they were written in
# pipeline_viz.py's KNOWN_COMPOUNDS). Matching must compare against
# normalize_text()'d text (hyphens collapsed to spaces, lowercased) —
# same as drug_classification.py's KNOWN_COMPOUND_NAMES/classify_intervention()
# pattern (`compound in candidate_normalized`, a SUBSTRING check on the
# normalized string, not a token-set membership check, which would
# never match a multi-token normalized key like "abt-089" -> "abt 089"
# against a token SET). Pre-normalizing the keys once here, rather than
# renormalizing on every lookup, keeps resolve_drug_classification() fast.
KNOWN_COMPOUND_TARGETS_NORMALIZED = {normalize_text(k): v for k, v in KNOWN_COMPOUND_TARGETS.items()}
KNOWN_COMPOUND_MODALITY_NORMALIZED = {normalize_text(k): v for k, v in KNOWN_COMPOUND_MODALITY.items()}


# ============================================================
# TIER 5: STRUCTURED CLINICALTRIALS.GOV EVIDENCE (verified, not raw)
# ============================================================

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

_MODALITY_NAME_KEYWORDS = ["mab", "umab", "zumab", "nemab", "antibody", "immunoglobulin", "vaccine", "immunotherapy"]
_MODALITY_CELL_GENE_KEYWORDS = ["stem cell", "mesenchymal", "cell therapy", "cord blood", "gene therapy", "viral vector"]


def gather_structured_evidence_for_drug(interventions_df, developed_drug_normalized):
    """
    The tier-5 evidence set: every intervention ROW (type + name) that
    ACTUALLY won as the developed_drug for some trial contributing to
    this canonical drug — i.e. rows where interventions_df["classification"]
    is a therapeutic label AND normalize_text(candidate_name or
    original_name) equals this drug's developed_drug_normalized.

    This is the key correctness improvement over the legacy
    guess_drug_type()/guess_target(): those scan the trial's WHOLE raw
    Interventions cell (every sibling — placebo, a co-administered
    device, an unrelated comparator), which is how a real drug ends up
    mislabeled "Device" just because a DEVICE-typed sibling appeared
    first in the same trial's cell (see MIGRATION_PLAN.md / Phase 0
    completion report). Restricting to the VERIFIED winning intervention
    only means a drug's OWN evidence, never a sibling's.

    Returns a list of {"type": ..., "name": ..., "title": ...} dicts
    (may be empty — e.g. a confirmed official-pipeline match whose
    candidate_name text doesn't literally appear this way in any single
    intervention row, which is fine, tier 5 just contributes nothing and
    an earlier tier is expected to have already supplied the
    classification). "title" is the STUDY title of the trial this
    specific verified intervention belongs to — legitimate evidence
    about this drug (it's the title of a trial studying THIS drug, not
    a sibling's), used only for target-pathway keyword inference (see
    infer_target_pathways_from_structured_evidence), never for modality.
    """
    if interventions_df.empty:
        return []

    therapeutic_mask = interventions_df["classification"].isin(
        ["sponsor_developed_therapeutic", "investigational_therapeutic_unverified"]
    )
    candidate_norm = interventions_df.get("candidate_name", interventions_df["original_name"]).apply(normalize_text)
    original_norm = interventions_df["original_name"].apply(normalize_text)
    name_mask = (candidate_norm == developed_drug_normalized) | (original_norm == developed_drug_normalized)

    matched = interventions_df[therapeutic_mask & name_mask]
    return [
        {"type": r["original_type"], "name": r["original_name"], "title": r.get("title", "")}
        for _, r in matched.iterrows()
    ]


def infer_modality_from_structured_evidence(evidence):
    """
    tier 5 modality guess — same decision logic as pipeline_viz.py's
    guess_drug_type(), but applied ONLY to this drug's own verified
    evidence rows (never a trial's other, unrelated interventions).
    Returns "" (not "Unknown") when there's no evidence at all, so the
    resolver can tell "no tier-5 evidence" apart from "tier 5 evidence
    says Unknown" — a real, if rare, outcome for e.g. an OTHER-typed row.
    """
    if not evidence:
        return ""
    types = {(e["type"] or "").strip().upper() for e in evidence}
    text = " ".join((e["name"] or "") for e in evidence).lower()

    if any(kw in text for kw in _MODALITY_CELL_GENE_KEYWORDS) or "GENETIC" in types:
        return "Cell/Gene Therapy"
    if any(kw in text for kw in _MODALITY_NAME_KEYWORDS) or "BIOLOGICAL" in types:
        return "Biologic"
    if "DEVICE" in types:
        return "Device"
    if "DIETARY_SUPPLEMENT" in types:
        return "Dietary Supplement"
    if "DRUG" in types:
        return "Small Molecule"
    if types:
        return "Other"
    return ""


def infer_target_pathways_from_structured_evidence(evidence):
    """
    tier 5 target-pathway guess — collects EVERY TARGET_KEYWORDS pathway
    whose keyword appears in this drug's own verified evidence text,
    preserving TARGET_ORDER. Unlike the legacy guess_target(), which
    returns the FIRST keyword match only, this returns ALL matches — the
    mechanism this phase uses to support multi-target drugs from keyword
    evidence.

    Uses BOTH the intervention name AND the STUDY TITLE of each
    contributing trial: a bare development code like "AVP-786" carries
    no pathway keyword by itself, but the trial studying it is often
    titled something like "...for Agitation in Alzheimer's Disease" —
    real evidence ABOUT this drug (it's titling a trial where this drug
    is the verified winning intervention), not noise from an unrelated
    sibling intervention the way scanning a trial's raw Interventions
    cell would be. Title text is intentionally excluded from modality
    inference (infer_modality_from_structured_evidence) — a title
    doesn't reliably indicate small-molecule vs. biologic the way an
    intervention's own type/name does.
    """
    if not evidence:
        return []
    name_text = " ".join((e["name"] or "") for e in evidence).lower()
    title_text = " ".join((e.get("title") or "") for e in evidence).lower()
    text = f"{name_text} {title_text}"
    return [pathway for pathway in TARGET_ORDER if any(kw in text for kw in TARGET_KEYWORDS[pathway])]


# ============================================================
# TIER 2: NIH REFERENCE MATCHING (dashboard drug -> NIH row)
# ============================================================

def build_nih_name_lookup(nih_df):
    """
    Every NIH row's canonical name AND every alias/component name/alias,
    normalized, pointing back at the row — the reverse-direction index
    of nih_reference.build_dashboard_name_lookup() (that one goes
    dashboard-name -> ..., this one goes NIH-name -> nih_row), needed
    because tier 2 here starts from the DASHBOARD drug and looks
    outward at NIH, not the other way around (nih_match_audit.csv, in
    contrast, is built NIH-row-outward).

    TWO PASSES, deliberately: a combination row like "Etalanetug (E2814)
    + Lecanemab (BAN2401)" has "Lecanemab" as one of its COMPONENT names,
    but that combo row's own CADRO/mechanism/modality describe the
    COMBINATION, not solo Lecanemab — attributing them to a lookup of
    plain "Lecanemab" would be wrong (e.g. its combo mechanism text says
    "anti-tau ... plus anti-amyloid", which is not true of Lecanemab
    alone). Pass 1 indexes only single-agent (non-combination) rows'
    own canonical name + aliases; pass 2 fills in combination-component
    names ONLY for keys pass 1 didn't already claim, so a real
    standalone NIH row for a component always wins over a combination
    row that merely happens to mention it.
    """
    lookup = {}

    def add(key_text, row):
        key = normalize_text(key_text)
        if key and key not in lookup:
            lookup[key] = row

    # pass 1: single-agent rows only
    for _, row in nih_df.iterrows():
        if len(row["components"]) > 1:
            continue
        add(row["canonical_name"], row)
        for alias in row["aliases"]:
            add(alias, row)

    # pass 2: combination-agent components, lowest priority
    for _, row in nih_df.iterrows():
        if len(row["components"]) <= 1:
            continue
        for comp in row["components"]:
            if comp["name"]:
                add(comp["name"], row)
            if comp["alias"]:
                add(comp["alias"], row)

    return lookup


def match_drug_to_nih(display_name, synonyms, nih_name_lookup):
    """
    Try the dashboard drug's own display_name, then each known synonym,
    against the NIH name lookup — normalized-exact only (no fuzzy match
    is EVER used to feed an automatic classification decision; fuzzy
    NIH matches are for nih_match_audit.csv's human-reviewed queue only,
    per Phase 1B, not for silently driving Phase 2's numbers).

    Returns the matched NIH row (a pandas Series) or None.
    """
    for name in [display_name] + list(synonyms or []):
        key = normalize_text(name)
        if key in nih_name_lookup:
            return nih_name_lookup[key]
    return None


# ============================================================
# TIER 6: CLAUDE INFERENCE (documented no-op — see module docstring)
# ============================================================

def claude_infer_classification(display_name, evidence_summary):
    """
    Placeholder for tier 6. pipeline_viz.py's run is a deterministic,
    offline batch script with no LLM API call wired into it — this
    function does NOT invent a plausible-sounding classification from
    the drug name alone, because an unverified guess presented with the
    same confidence as sourced evidence would be actively misleading on
    a dashboard used for real competitive-intelligence decisions.

    Always returns None (i.e. "no tier-6 result"), so the resolver falls
    through to tier 7 ("Needs Review"). Left as a real, wired stage in
    the priority chain — not deleted — so a future revision can replace
    this function body with an actual API call without touching the
    resolver's control flow.
    """
    return None


# ============================================================
# THE RESOLVER
# ============================================================

CLASSIFICATION_SOURCES = [
    "curated_override", "nih_reference", "official_pipeline_reference",
    "known_compound_exact_match", "structured_ct_gov_evidence",
    "claude_inference", "needs_review",
]


def _result(modality, target_pathways, molecular_targets, mechanism_of_action,
            source, method, confidence, reason, evidence_used, manual_review_required):
    return {
        "modality": modality or "Unknown",
        "target_pathways": target_pathways or [],
        "molecular_targets": molecular_targets or [],
        "mechanism_of_action": mechanism_of_action or "",
        "classification_source": source,
        "classification_method": method,
        "classification_confidence": confidence,
        "classification_reason": reason,
        "evidence_used": "; ".join(evidence_used) if evidence_used else "",
        "manual_review_required": manual_review_required,
    }


def resolve_drug_classification(display_name, synonyms, structured_evidence,
                                 overrides=None, nih_name_lookup=None, official_pipeline_lookup=None):
    """
    The Phase 2 resolver — produces modality/target_pathways/
    molecular_targets/mechanism_of_action + full provenance for ONE
    canonical drug, using ONLY verified/structured evidence about that
    specific drug (never a trial's raw, mixed Interventions text).

    overrides: load_drug_classification_overrides() dict (tier 1)
    nih_name_lookup: build_nih_name_lookup() dict (tier 2)
    official_pipeline_lookup: dict normalized drug_name -> {"modality":..,
        "target_pathways": [...]} (tier 3 — see
        scientific_classification.build_official_pipeline_classification_lookup())
    structured_evidence: gather_structured_evidence_for_drug() output (tier 5)
    """
    overrides = overrides or {}
    nih_name_lookup = nih_name_lookup or {}
    official_pipeline_lookup = official_pipeline_lookup or {}
    normalized_name = normalize_text(display_name)
    candidate_normalized = normalized_name  # normalize_text(display_name), full string — substring-matched below

    # --- Tier 1: curated AriBio override — wins outright ---
    override = overrides.get(normalized_name)
    if override:
        return _result(
            override["modality"], override["target_pathways"], override["molecular_targets"],
            override["mechanism_of_action"], "curated_override", "tier1_curated_override", "high",
            override.get("reason") or "curated AriBio override", [f"curated override ({override.get('source', '')})"],
            manual_review_required=False,
        )

    # --- Gather tier 2 (NIH) and tiers 3-5 ("existing evidence") independently ---
    nih_row = match_drug_to_nih(display_name, synonyms, nih_name_lookup)
    nih_modality = ""
    nih_targets = []
    nih_mechanism = ""
    nih_evidence_note = ""
    if nih_row is not None:
        nih_modality_raw = _extract_nih_modality(nih_row["purpose_class"], nih_row["purpose_detail"])
        nih_modality = {"small molecule": "Small Molecule", "biologic": "Biologic"}.get(
            normalize_text(nih_modality_raw), ""
        )
        inferred = infer_nih_target(nih_row["cadro"], nih_row["purpose_class"], nih_row["purpose_detail"])
        if inferred:
            nih_targets = [inferred]
        nih_mechanism = nih_row.get("mechanism_of_action", "") or ""
        nih_evidence_note = f"NIH agent={nih_row['canonical_name']!r} CADRO={nih_row['cadro']!r}"

    official = official_pipeline_lookup.get(normalized_name)
    known_compound_target = None
    known_compound_modality = None
    for key, pathway in KNOWN_COMPOUND_TARGETS_NORMALIZED.items():
        if key and key in candidate_normalized:
            known_compound_target = pathway
            break
    for key, modality in KNOWN_COMPOUND_MODALITY_NORMALIZED.items():
        if key and key in candidate_normalized:
            known_compound_modality = modality
            break

    structured_modality = infer_modality_from_structured_evidence(structured_evidence)
    structured_targets = infer_target_pathways_from_structured_evidence(structured_evidence)

    # "existing" evidence = tiers 3, 4, 5 in priority order (first non-empty wins per field)
    existing_modality, existing_modality_source = "", ""
    existing_targets, existing_target_source = [], ""
    evidence_used = []

    if official and official.get("modality"):
        existing_modality, existing_modality_source = official["modality"], "official_pipeline_reference"
        evidence_used.append(f"official_pipeline modality={official['modality']!r}")
    elif known_compound_modality:
        existing_modality, existing_modality_source = known_compound_modality, "known_compound_exact_match"
        evidence_used.append(f"known-compound modality={known_compound_modality!r}")
    elif structured_modality:
        existing_modality, existing_modality_source = structured_modality, "structured_ct_gov_evidence"
        evidence_used.append(f"ct.gov structured modality={structured_modality!r}")

    if official and official.get("target_pathways"):
        existing_targets, existing_target_source = official["target_pathways"], "official_pipeline_reference"
        evidence_used.append(f"official_pipeline target={official['target_pathways']}")
    elif known_compound_target:
        existing_targets, existing_target_source = [known_compound_target], "known_compound_exact_match"
        evidence_used.append(f"known-compound target={known_compound_target!r}")
    elif structured_targets:
        existing_targets, existing_target_source = structured_targets, "structured_ct_gov_evidence"
        evidence_used.append(f"ct.gov structured target(s)={structured_targets}")

    if nih_evidence_note:
        evidence_used.append(nih_evidence_note)

    # --- Reconcile NIH (tier 2) vs "existing" (tiers 3-5) ---
    conflict_notes = []
    manual_review_required = False

    def reconcile(nih_value, existing_value, existing_source, field_name):
        nonlocal manual_review_required
        if nih_value and existing_value:
            if (isinstance(nih_value, list) and isinstance(existing_value, list)
                    and set(nih_value) == set(existing_value)) or nih_value == existing_value:
                return nih_value, "high"
            conflict_notes.append(f"{field_name}: NIH={nih_value!r} vs existing={existing_value!r}")
            manual_review_required = True
            return nih_value, "medium"  # NIH (tier 2) outranks tiers 3-5 on disagreement, but confidence drops
        if nih_value:
            return nih_value, "medium"
        if existing_value:
            # per requirement: do NOT downgrade confidence solely because NIH is missing
            return existing_value, "high" if existing_source in ("official_pipeline_reference", "known_compound_exact_match") else "medium"
        return None, None

    resolved_modality, modality_confidence = reconcile(nih_modality, existing_modality, existing_modality_source, "modality")
    resolved_targets, target_confidence = reconcile(nih_targets, existing_targets, existing_target_source, "target_pathways")

    if resolved_modality is None and resolved_targets is None:
        claude_result = claude_infer_classification(display_name, "; ".join(evidence_used))
        if claude_result is not None:
            return _result(
                claude_result.get("modality"), claude_result.get("target_pathways"), [],
                claude_result.get("mechanism_of_action", ""), "claude_inference", "tier6_claude_inference",
                "low", "Claude-inferred (unverified) — needs human confirmation", evidence_used,
                manual_review_required=True,
            )
        return _result(
            "Unknown", [], [], "", "needs_review", "tier7_no_evidence_found", "low",
            "no curated override, NIH match, official-pipeline record, known-compound match, or structured "
            "ct.gov evidence found for this drug", evidence_used, manual_review_required=True,
        )

    confidences = [c for c in (modality_confidence, target_confidence) if c]
    overall_confidence = "low" if "medium" in confidences and len(conflict_notes) else (
        "medium" if "medium" in confidences else "high"
    )
    if manual_review_required:
        overall_confidence = "medium" if overall_confidence == "high" else overall_confidence

    if conflict_notes:
        reason = "NIH and existing (official-pipeline/known-compound/ct.gov) evidence disagree: " + "; ".join(conflict_notes)
        source = "nih_reference" if nih_modality or nih_targets else (existing_modality_source or existing_target_source)
    elif nih_modality or nih_targets:
        reason = "NIH reference evidence" + (" (agrees with existing evidence)" if (existing_modality or existing_targets) else "")
        source = "nih_reference"
    else:
        reason = f"existing evidence only ({existing_modality_source or existing_target_source}); no NIH match found"
        source = existing_modality_source or existing_target_source

    mechanism = nih_mechanism  # only NIH currently supplies free-text mechanism narrative (see module docstring)
    molecular_targets = []  # no structured molecular-target (protein/receptor-level) data source available yet

    return _result(
        resolved_modality or existing_modality or nih_modality or "Unknown",
        resolved_targets or existing_targets or nih_targets or [],
        molecular_targets, mechanism, source, "tier2_6_resolver", overall_confidence, reason,
        evidence_used, manual_review_required,
    )


def build_official_pipeline_classification_lookup(pipeline_records):
    """
    Tier 3 — data/official_pipeline.csv, extended (optionally) with
    `modality`/`target_pathways` columns. Rows without those columns
    (or with them blank) contribute nothing at this tier and fall
    through to tier 4/5, exactly like any other missing-evidence case.
    """
    lookup = {}
    for record in pipeline_records:
        modality = (record.get("modality") or "").strip()
        target_pathways_raw = (record.get("target_pathways") or "").strip()
        if not modality and not target_pathways_raw:
            continue
        key = normalize_text(record["drug_name"])
        lookup[key] = {
            "modality": modality,
            "target_pathways": [t.strip() for t in target_pathways_raw.split(";") if t.strip()],
        }
    return lookup


# ============================================================
# outputs/classification_conflicts.csv
# ============================================================

def build_classification_conflicts_dataframe(records):
    """
    records: list of dicts, one per canonical drug, each already
    carrying: canonical_drug_name, previous_modality, previous_target,
    new_modality, new_target_pathways (list), classification_source,
    classification_confidence, classification_reason,
    manual_review_required (the resolver's own internal flag) — i.e.
    what pipeline_viz.py assembles per resolved_drugs_df row, comparing
    the resolver's new values against the LEGACY drug_type/target this
    drug had before Phase 2.

    conflict_reason combines BOTH kinds of disagreement this phase
    tracks: (a) the resolver's own internal NIH-vs-existing-evidence
    conflict (classification_reason, when it mentions "disagree"), and
    (b) a plain before/after difference versus the legacy heuristic
    value, even when the resolver itself had no internal conflict (i.e.
    the new evidence-based value simply corrects a wrong legacy guess).
    """
    rows = []
    for r in records:
        new_target_str = "; ".join(r["new_target_pathways"])
        modality_changed = r["previous_modality"] != r["new_modality"]
        # an EMPTY new_target_pathways list displays as "Other" on the
        # dashboard (see pipeline_viz.py's target column assignment) —
        # compare against that same effective value, not against an
        # empty set, or every drug with no keyword evidence would be
        # wrongly flagged "changed" even when it was already "Other"
        effective_new_targets = r["new_target_pathways"] or ["Other"]
        target_changed = normalize_text(r["previous_target"]) not in {normalize_text(t) for t in effective_new_targets}
        internal_conflict = "disagree" in r["classification_reason"]

        reasons = []
        if internal_conflict:
            reasons.append(r["classification_reason"])
        if modality_changed:
            reasons.append(f"modality corrected: {r['previous_modality']!r} -> {r['new_modality']!r}")
        if target_changed:
            reasons.append(f"target corrected: {r['previous_target']!r} -> {new_target_str!r}")
        if not reasons:
            reasons.append("no change from legacy classification")

        rows.append({
            "canonical_drug_name": r["canonical_drug_name"],
            "previous_modality": r["previous_modality"],
            "new_modality": r["new_modality"],
            "previous_target": r["previous_target"],
            "new_target_pathways": new_target_str,
            "classification_source": r["classification_source"],
            "confidence": r["classification_confidence"],
            "conflict_reason": " | ".join(reasons),
            "manual_review_required": bool(r["manual_review_required"]) or modality_changed or target_changed,
        })

    columns = [
        "canonical_drug_name", "previous_modality", "new_modality", "previous_target",
        "new_target_pathways", "classification_source", "confidence", "conflict_reason",
        "manual_review_required",
    ]
    return pd.DataFrame(rows, columns=columns)
