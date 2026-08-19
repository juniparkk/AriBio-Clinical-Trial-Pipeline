# ============================================================
# NIH REFERENCE DATASET — Phase 1B (audit-only)
#
# Pure helper functions for profiling nih_data.csv and reconciling it
# against resolved_drugs_df (here read from the already-generated
# pipeline_drugs.csv, NOT by re-running pipeline_viz.py — Phase 1B is
# explicitly audit-only and must not regenerate/alter the dashboard or
# its output files).
#
# nih_data.csv is a small, curated "Alzheimer's Disease Drug Development
# Pipeline"-style report export (CADRO = Common Alzheimer's Disease
# Research Ontology, the NIA's mechanism/target classification system).
# It is NOT a raw ClinicalTrials.gov export and does NOT claim to cover
# every trial — see NIH_INTEGRATION_PLAN.md for exactly what it is and
# is not authoritative for.
#
# Like drug_classification.py, this module only DEFINES functions — it
# does not read/write any file at import time and prints nothing on
# import, so it stays safely importable from a test file.
# ============================================================

import csv
import difflib
import io
import re

import pandas as pd

from drug_classification import normalize_text, _company_matches, STALE_PHASE3_DISCONTINUED_LABEL

# ============================================================
# STEP 1: PARSE THE MULTI-SECTION NIH CSV
# ============================================================

REQUIRED_NIH_COLUMNS = [
    "Agent", "Therapeutic purpose", "CADRO", "Mechanism of action",
    "Clinical trial", "Lead sponsor", "Start date", "Primary completion date",
]

_PHASE_SECTION_RE = re.compile(r"^Phase\s*([123])\s*$")

# "Name (Alias)" or "Name (Alias1) + Name2 (Alias2)" — a parenthetical
# alias is only extracted when it looks like a short code/acronym (no
# spaces, or a short hyphen/digit-heavy token) so a genuinely descriptive
# parenthetical (rare in the Agent column, but this guards against it)
# isn't mistaken for an alias.
_PAREN_ALIAS_RE = re.compile(r"^(.*?)\s*\(([^)]+)\)\s*$")


def _looks_like_alias_code(text):
    text = text.strip()
    if not text or len(text) > 20:
        return False
    # short, code-shaped: letters/digits/hyphens/spaces only, no long
    # descriptive words (a real alias like "BAN2401", "E2814", "CT1812"
    # has no more than 2 space-separated tokens)
    if not re.match(r"^[A-Za-z0-9\-‐‑ ]+$", text):
        return False
    return len(text.split()) <= 2


def extract_canonical_and_aliases(agent_raw):
    """
    Split one NIH "Agent" cell into:
      canonical_name: the full agent text as written (combination agents
          like "KarXT + KarX-EC" keep their full combined name here —
          Phase 1B does not invent a single canonical name for a
          combination product, matching the same caution
          classify_pipeline_scope() applies to ct.gov COMBINATION_PRODUCT
          entries)
      aliases: parenthetical short codes found anywhere in the cell,
          e.g. "Zervimesine (CT1812)" -> aliases=["CT1812"];
          "Etalanetug (E2814) + Lecanemab (BAN2401)" -> ["E2814", "BAN2401"]
      components: for "+"-joined combination agents, the individual
          component names (each itself run through the same
          name/alias split) — e.g.
          "Etalanetug (E2814) + Lecanemab (BAN2401)" ->
          [{"name": "Etalanetug", "alias": "E2814"},
           {"name": "Lecanemab", "alias": "BAN2401"}]
          A non-combination agent has exactly one component (itself).
    """
    agent_raw = (agent_raw or "").strip()
    parts = [p.strip() for p in agent_raw.split("+") if p.strip()]

    components = []
    aliases = []
    for part in parts:
        m = _PAREN_ALIAS_RE.match(part)
        if m and _looks_like_alias_code(m.group(2)):
            name, alias = m.group(1).strip(), m.group(2).strip()
            components.append({"name": name, "alias": alias})
            aliases.append(alias)
        else:
            components.append({"name": part, "alias": ""})

    canonical_name = agent_raw
    return canonical_name, aliases, components


def parse_nih_dataset(path):
    """
    Parse nih_data.csv's multi-section report format (a "Phase N" marker
    row, a repeated header row, then that phase's data rows — repeated
    for Phase 3/2/1) into one row per Agent entry.

    Uses Python's csv module (not pandas.read_csv) because several
    fields (Clinical trial / Lead sponsor / Start date / Primary
    completion date) are quoted multi-line cells — one line per trial
    for agents studied in more than one trial — and csv's reader handles
    embedded newlines inside quoted fields correctly, which a naive
    line-by-line read would not.

    Returns a DataFrame with columns:
      row_number, phase, agent_raw, canonical_name, aliases (list),
      components (list of {"name","alias"}), therapeutic_purpose,
      purpose_class ("DTT"/"STT"/""), purpose_detail, cadro,
      mechanism_of_action, nct_ids (list), lead_sponsors (list),
      start_dates (list), primary_completion_dates (list)
    """
    with open(path, encoding="utf-8") as f:
        text = f.read()
    raw_rows = list(csv.reader(io.StringIO(text)))

    rows = []
    current_phase = None
    for row_number, row in enumerate(raw_rows):
        row = row + [""] * (len(REQUIRED_NIH_COLUMNS) - len(row))  # pad short rows defensively
        first_cell = row[0].strip()

        phase_match = _PHASE_SECTION_RE.match(first_cell)
        if phase_match:
            current_phase = f"Phase {phase_match.group(1)}"
            continue
        if first_cell == "Agent":
            continue  # repeated header row
        if not any(c.strip() for c in row):
            continue  # blank section-spacer row

        (agent_raw, therapeutic_purpose, cadro, mechanism_of_action,
         clinical_trial, lead_sponsor, start_date, primary_completion_date) = row[:8]

        canonical_name, aliases, components = extract_canonical_and_aliases(agent_raw)

        purpose_parts = [p.strip() for p in therapeutic_purpose.split(";", 1)]
        purpose_class = purpose_parts[0] if purpose_parts and purpose_parts[0] in ("DTT", "STT") else ""
        purpose_detail = purpose_parts[1].strip() if len(purpose_parts) > 1 else ""

        rows.append({
            "row_number": row_number,
            "phase": current_phase,
            "agent_raw": agent_raw,
            "canonical_name": canonical_name,
            "aliases": aliases,
            "components": components,
            "therapeutic_purpose": therapeutic_purpose,
            "purpose_class": purpose_class,
            "purpose_detail": purpose_detail,
            "cadro": cadro.strip(),
            "mechanism_of_action": mechanism_of_action.strip(),
            "nct_ids": [x.strip() for x in clinical_trial.split("\n") if x.strip()],
            "lead_sponsors": [x.strip() for x in lead_sponsor.split("\n") if x.strip()],
            "start_dates": [x.strip() for x in start_date.split("\n") if x.strip()],
            "primary_completion_dates": [x.strip() for x in primary_completion_date.split("\n") if x.strip()],
        })

    return pd.DataFrame(rows)


# ============================================================
# STEP 2: PROFILE THE PARSED DATASET
# ============================================================

def profile_nih_dataset(nih_df):
    """
    outputs/nih_dataset_profile.csv source — one row per FIELD (not per
    record), describing what's actually usable for matching/import:
    non-null coverage, distinct-value coverage, and a plain-language
    note on what the field can support (canonical name, alias, modality,
    mechanism, target/pathway, company, phase — per the Phase 1B
    requirement to identify which fields support which purpose).
    """
    total = len(nih_df)

    def coverage(series_of_lists_or_str, is_list=False):
        if is_list:
            non_empty = series_of_lists_or_str.apply(lambda v: bool(v))
        else:
            non_empty = series_of_lists_or_str.astype(str).str.strip().ne("")
        return int(non_empty.sum())

    fields = [
        ("agent_raw", "canonical name (raw, as published)", False),
        ("canonical_name", "canonical name candidate (Agent cell verbatim; combination agents keep the full '+'-joined text)", False),
        ("aliases", "alias/development-code candidate (parenthetical short codes only)", True),
        ("therapeutic_purpose", "DTT/STT classification + (DTT-only) modality / (STT-only) symptomatic category", False),
        ("purpose_class", "DTT (disease-targeted) vs STT (symptomatic) — reliable for every row", False),
        ("purpose_detail", "modality (small molecule/biologic) for DTT rows ONLY; a symptomatic-category label for STT rows, NOT a modality", False),
        ("cadro", "target/pathway/mechanism-category candidate (NIA's CADRO ontology — finer-grained than the dashboard's 7-bucket target field)", False),
        ("mechanism_of_action", "free-text mechanism narrative — not structured, supporting/reference only", False),
        ("nct_ids", "trial linkage (one or more NCT IDs per agent)", True),
        ("lead_sponsors", "company/sponsor candidate (one per listed trial, parallel to nct_ids)", True),
        ("phase", "phase candidate (the report SECTION the agent is listed under — a single snapshot value, not a full trial-status history)", False),
    ]

    profile_rows = []
    for field, note, is_list in fields:
        non_null = coverage(nih_df[field], is_list=is_list)
        if is_list:
            distinct = nih_df[field].apply(lambda v: tuple(v)).nunique()
            example = next((v for v in nih_df[field] if v), [])
        else:
            distinct = nih_df[field].astype(str).nunique()
            example = next((v for v in nih_df[field].astype(str) if v.strip()), "")
        profile_rows.append({
            "field": field,
            "non_null_count": non_null,
            "total_rows": total,
            "pct_populated": round(100 * non_null / total, 1) if total else 0.0,
            "distinct_values": distinct,
            "example_value": str(example)[:120],
            "supports": note,
        })

    return pd.DataFrame(profile_rows)


def summarize_nih_dataset_shape(nih_df):
    """
    Row-granularity / duplicate summary for NIH_INTEGRATION_PLAN.md —
    NOT written to a CSV (the profile CSV is field-level; this is the
    narrative-level summary the plan doc reports verbatim).
    """
    canonical_counts = nih_df["canonical_name"].apply(normalize_text).value_counts()
    duplicate_canonical_names = canonical_counts[canonical_counts > 1]

    multi_trial_agents = int((nih_df["nct_ids"].apply(len) > 1).sum())
    combination_agents = int((nih_df["components"].apply(len) > 1).sum())

    return {
        "total_agent_rows": len(nih_df),
        "rows_by_phase": nih_df["phase"].value_counts().to_dict(),
        "duplicate_canonical_name_count": int(len(duplicate_canonical_names)),
        "duplicate_canonical_names": duplicate_canonical_names.index.tolist(),
        "multi_trial_agent_rows": multi_trial_agents,
        "combination_agent_rows": combination_agents,
        "has_date_or_version_field_for_the_dataset_itself": False,
    }


# ============================================================
# STEP 3: MATCH NIH RECORDS <-> resolved_drugs_df
# ============================================================

MATCH_TIERS = ["exact_canonical", "exact_alias", "normalized_exact", "fuzzy_suggestion", "unmatched"]
_MATCH_TIER_RANK = {t: i for i, t in enumerate(MATCH_TIERS)}  # lower rank = stronger match

FUZZY_MATCH_CUTOFF = 0.85


def build_dashboard_name_lookup(resolved_drugs_df):
    """normalize_text(display_name) -> display_name, for every resolved_drugs_df row."""
    lookup = {}
    for name in resolved_drugs_df["display_name"]:
        lookup[normalize_text(name)] = name
    return lookup


def _generate_match_candidates(nih_row):
    """
    Ordered (text, kind) candidates to try against the dashboard name
    lookup for one NIH row — canonical name first (strongest signal),
    then its own aliases, then each combination component's name/alias.
    """
    candidates = [(nih_row["canonical_name"], "canonical")]
    for alias in nih_row["aliases"]:
        candidates.append((alias, "alias"))
    for component in nih_row["components"]:
        if component["name"] and component["name"] != nih_row["canonical_name"]:
            candidates.append((component["name"], "component_canonical"))
        if component["alias"]:
            candidates.append((component["alias"], "component_alias"))
    return candidates


def match_nih_row_to_dashboard(nih_row, dashboard_names, name_lookup):
    """
    Try every candidate name for one NIH row against resolved_drugs_df,
    in tier order (strongest first): raw exact string match against a
    real display_name (canonical-candidate -> "exact_canonical",
    alias/component-candidate -> "exact_alias"), then normalize_text()
    equality ("normalized_exact"), then a fuzzy suggestion
    (difflib ratio >= FUZZY_MATCH_CUTOFF — NEVER auto-accepted, always
    needs_manual_review) — falling back to "unmatched" only if nothing
    at all is close.

    dashboard_names: the raw resolved_drugs_df display_name Series/list
        (for exact, case-sensitive string comparison).
    name_lookup: build_dashboard_name_lookup(resolved_drugs_df) — for
        normalized comparison.

    Returns dict: match_tier, matched_dashboard_name, matched_candidate_text,
    matched_candidate_kind.
    """
    dashboard_name_set = set(dashboard_names)
    candidates = _generate_match_candidates(nih_row)

    best = {"match_tier": "unmatched", "matched_dashboard_name": "",
            "matched_candidate_text": "", "matched_candidate_kind": ""}
    best_rank = _MATCH_TIER_RANK["unmatched"]

    for text, kind in candidates:
        if not text:
            continue

        if text in dashboard_name_set:
            tier = "exact_canonical" if kind == "canonical" else "exact_alias"
            rank = _MATCH_TIER_RANK[tier]
            if rank < best_rank:
                best = {"match_tier": tier, "matched_dashboard_name": text,
                        "matched_candidate_text": text, "matched_candidate_kind": kind}
                best_rank = rank
            continue

        normalized = normalize_text(text)
        if normalized in name_lookup:
            rank = _MATCH_TIER_RANK["normalized_exact"]
            if rank < best_rank:
                best = {"match_tier": "normalized_exact", "matched_dashboard_name": name_lookup[normalized],
                        "matched_candidate_text": text, "matched_candidate_kind": kind}
                best_rank = rank
            continue

        if best_rank > _MATCH_TIER_RANK["fuzzy_suggestion"]:
            close = difflib.get_close_matches(normalized, name_lookup.keys(), n=1, cutoff=FUZZY_MATCH_CUTOFF)
            if close:
                best = {"match_tier": "fuzzy_suggestion", "matched_dashboard_name": name_lookup[close[0]],
                        "matched_candidate_text": text, "matched_candidate_kind": kind}
                best_rank = _MATCH_TIER_RANK["fuzzy_suggestion"]

    return best


_DASHBOARD_UNMATCHED_BUCKETS = [
    "non_therapeutic_or_ambiguous", "unresolved_naming_alias_issue",
    "historical_or_discontinued", "current_missing_from_nih",
]


def bucket_unmatched_dashboard_drug(dashboard_row, has_fuzzy_suggestion):
    """
    Per requirement 3: split dashboard drugs with NO confirmed NIH match
    (exact_canonical/exact_alias/normalized_exact) into 4 categories.
    Checked in this priority order:
      1. non_therapeutic_or_ambiguous — pipeline_scope isn't "Therapeutic
         Drug"; NIH's list only covers therapeutic programs, so these
         were never expected to match in the first place.
      2. unresolved_naming_alias_issue — a fuzzy candidate exists (some
         NIH agent's name is CLOSE but not confidently equal) — a human
         should confirm whether it's the same drug under a different
         spelling/alias before deciding anything else.
      3. historical_or_discontinued — status_summary == "Discontinued"
         (ct.gov-reported) or STALE_PHASE3_DISCONTINUED_LABEL (a Phase 3
         trial years past completion with no FDA approval, never
         formally closed by ct.gov -- see drug_classification.py).
         Per the explicit instruction, this is NOT treated as an error —
         NIH's report is a current-pipeline snapshot and is not expected
         to carry historical/discontinued programs.
      4. current_missing_from_nih — everything else: a live therapeutic
         drug NIH's curated list simply doesn't happen to cover (NIH is
         not assumed complete).
    """
    if dashboard_row.get("pipeline_scope") != "Therapeutic Drug":
        return "non_therapeutic_or_ambiguous"
    if has_fuzzy_suggestion:
        return "unresolved_naming_alias_issue"
    if dashboard_row.get("status_summary") in ("Discontinued", STALE_PHASE3_DISCONTINUED_LABEL):
        return "historical_or_discontinued"
    return "current_missing_from_nih"


def build_nih_match_audit(nih_df, resolved_drugs_df):
    """
    outputs/nih_match_audit.csv source — a two-sided reconciliation
    table in ONE dataframe (record_type distinguishes the two halves):

      record_type == "nih_record": one row per NIH agent entry (165 in
          the current dataset), with its best match tier/candidate
          against resolved_drugs_df.
      record_type == "dashboard_only": one row per dashboard drug that
          NO NIH row matched at exact_canonical/exact_alias/
          normalized_exact confidence, bucketed per
          bucket_unmatched_dashboard_drug() above. A dashboard drug that
          only received a fuzzy suggestion is EXCLUDED from
          "auto-matched" but still appears here as
          "unresolved_naming_alias_issue", not silently dropped.
    """
    dashboard_names = resolved_drugs_df["display_name"].tolist()
    name_lookup = build_dashboard_name_lookup(resolved_drugs_df)

    nih_rows_out = []
    matched_dashboard_names = set()          # confidently matched (exact_canonical/exact_alias/normalized_exact)
    fuzzy_suggested_dashboard_names = set()  # only ever reached fuzzy_suggestion tier

    for _, nih_row in nih_df.iterrows():
        result = match_nih_row_to_dashboard(nih_row, dashboard_names, name_lookup)
        nih_rows_out.append({
            "record_type": "nih_record",
            "name": nih_row["canonical_name"],
            "phase": nih_row["phase"],
            "sponsor": "; ".join(nih_row["lead_sponsors"]),
            "match_tier": result["match_tier"],
            "matched_dashboard_name": result["matched_dashboard_name"],
            "matched_candidate_text": result["matched_candidate_text"],
            "matched_candidate_kind": result["matched_candidate_kind"],
            "dashboard_bucket": "",
            "pipeline_scope": "",
            "nct_ids": "; ".join(nih_row["nct_ids"]),
        })
        if result["match_tier"] in ("exact_canonical", "exact_alias", "normalized_exact"):
            matched_dashboard_names.add(result["matched_dashboard_name"])
        elif result["match_tier"] == "fuzzy_suggestion":
            fuzzy_suggested_dashboard_names.add(result["matched_dashboard_name"])

    dashboard_rows_out = []
    for _, drow in resolved_drugs_df.iterrows():
        if drow["display_name"] in matched_dashboard_names:
            continue
        has_fuzzy = drow["display_name"] in fuzzy_suggested_dashboard_names
        bucket = bucket_unmatched_dashboard_drug(drow, has_fuzzy)
        dashboard_rows_out.append({
            "record_type": "dashboard_only",
            "name": drow["display_name"],
            "phase": drow.get("phase_reached", ""),
            "sponsor": drow.get("sponsor", ""),
            "match_tier": "fuzzy_suggestion" if has_fuzzy else "unmatched",
            "matched_dashboard_name": "",
            "matched_candidate_text": "",
            "matched_candidate_kind": "",
            "dashboard_bucket": bucket,
            "pipeline_scope": drow.get("pipeline_scope", ""),
            "nct_ids": "",
        })

    columns = [
        "record_type", "name", "phase", "sponsor", "match_tier",
        "matched_dashboard_name", "matched_candidate_text", "matched_candidate_kind",
        "dashboard_bucket", "pipeline_scope", "nct_ids",
    ]
    return pd.DataFrame(nih_rows_out + dashboard_rows_out, columns=columns)


# ============================================================
# STEP 4: CONFLICT COMPARISON (matched pairs only)
# ============================================================

# CADRO (NIA ontology) -> dashboard's 7-bucket `target` field. Deliberately
# NOT a total mapping — CADRO is finer-grained than the dashboard's target
# taxonomy, and several categories (Proteostasis/proteinopathies, Growth
# factors and hormones, Oxidative stress, Circadian rhythm, Vasculature,
# Gut-brain axis, Epigenetic regulators, Cell death, APOE/lipids) have NO
# real dashboard equivalent — those map to None ("no_dashboard_equivalent"
# in the conflict report) rather than being force-fit into "Other", which
# would silently hide the taxonomy gap instead of surfacing it.
# keys are normalize_text()-shaped (lowercase, hyphens/punctuation
# already collapsed to single spaces) so map_cadro_to_target() can look
# up normalize_text(cadro) directly with no further munging
CADRO_TO_TARGET_MAP = {
    "amyloid beta": "Amyloid",
    "tau": "Tau",
    "inflammation": "Inflammation",
    "neurotransmitter receptors": "Symptomatic",
    "synaptic plasticity neuroprotection": "Neuroprotection",
    "metabolism and bioenergetics": "Metabolism",
    "multi target": None,  # spans several dashboard buckets at once — not a single-value mapping
    "undisclosed": None,
}


def map_cadro_to_target(cadro):
    return CADRO_TO_TARGET_MAP.get(normalize_text(cadro), None)


def infer_nih_target(cadro, purpose_class, purpose_detail):
    """
    Best-effort NIH -> dashboard `target` inference. CADRO alone maps
    "Neurotransmitter receptors" to the dashboard's broad "Symptomatic"
    bucket, but for STT (symptomatic-treatment) rows, purpose_detail
    carries a much more specific category — e.g. "neuropsychiatric
    (agitation)" — that matches the dashboard's own separate
    "Neuropsychiatric" bucket far better than "Symptomatic" does. Using
    purpose_detail first for STT rows avoids manufacturing a false
    target_conflict for drugs the dashboard already correctly labeled
    Neuropsychiatric (KarXT, ACP-204, Escitalopram, ...).
    """
    if purpose_class == "STT" and "neuropsychiatric" in purpose_detail.lower():
        return "Neuropsychiatric"
    return map_cadro_to_target(cadro)


def _extract_nih_modality(purpose_class, purpose_detail):
    # modality is only meaningful for DTT rows (see profile_nih_dataset's
    # note on purpose_detail) — STT rows' second segment is a symptomatic
    # CATEGORY, not a modality, so returning "" for STT avoids a false conflict
    if purpose_class != "DTT":
        return ""
    return purpose_detail.strip()


_MODALITY_TO_DASHBOARD_DRUG_TYPE = {
    "small molecule": "Small Molecule",
    "biologic": "Biologic",
}


def build_nih_conflict_audit(nih_df, resolved_drugs_df, match_audit_df):
    """
    outputs/nih_conflict_audit.csv source — one row per CONFIDENTLY
    matched pair (exact_canonical/exact_alias/normalized_exact only —
    fuzzy suggestions are excluded, since there's no confirmed pair yet
    to compare fields on), flagging disagreements in modality,
    target/pathway, sponsor, and phase for human review. This NEVER
    asserts NIH is correct and the dashboard is wrong (or vice versa) —
    see NIH_INTEGRATION_PLAN.md — it only surfaces where the two sources
    disagree.
    """
    resolved_by_name = {r["display_name"]: r for _, r in resolved_drugs_df.iterrows()}
    nih_by_canonical = {r["canonical_name"]: r for _, r in nih_df.iterrows()}

    matched = match_audit_df[
        (match_audit_df["record_type"] == "nih_record")
        & (match_audit_df["match_tier"].isin(["exact_canonical", "exact_alias", "normalized_exact"]))
    ]

    rows = []
    for _, m in matched.iterrows():
        nih_row = nih_by_canonical.get(m["name"])
        dash_row = resolved_by_name.get(m["matched_dashboard_name"])
        if nih_row is None or dash_row is None:
            continue

        nih_modality = _extract_nih_modality(nih_row["purpose_class"], nih_row["purpose_detail"])
        dashboard_drug_type = dash_row.get("drug_type", "")
        expected_drug_type = _MODALITY_TO_DASHBOARD_DRUG_TYPE.get(normalize_text(nih_modality))
        drug_type_conflict = bool(nih_modality) and expected_drug_type is not None and expected_drug_type != dashboard_drug_type

        mapped_target = infer_nih_target(nih_row["cadro"], nih_row["purpose_class"], nih_row["purpose_detail"])
        dashboard_target = dash_row.get("target", "")
        target_conflict = mapped_target is not None and mapped_target != dashboard_target

        dashboard_sponsors = [s.strip() for s in str(dash_row.get("sponsor", "")).split(";") if s.strip()]
        nih_sponsors = nih_row["lead_sponsors"]
        company_conflict = bool(dashboard_sponsors) and bool(nih_sponsors) and not any(
            _company_matches(normalize_text(ns), normalize_text(ds))
            for ns in nih_sponsors for ds in dashboard_sponsors
        )

        dashboard_phase = dash_row.get("phase_reached", "")
        nih_phase = nih_row["phase"]
        phase_conflict = bool(nih_phase) and bool(dashboard_phase) and nih_phase != dashboard_phase

        canonical_name_differs = m["matched_candidate_text"] != m["matched_dashboard_name"]

        rows.append({
            "dashboard_display_name": m["matched_dashboard_name"],
            "nih_agent": m["name"],
            "match_tier": m["match_tier"],
            "canonical_name_differs": canonical_name_differs,
            "dashboard_drug_type": dashboard_drug_type,
            "nih_modality": nih_modality,
            "drug_type_conflict": drug_type_conflict,
            "dashboard_target": dashboard_target,
            "nih_cadro": nih_row["cadro"],
            "nih_cadro_mapped_target": mapped_target if mapped_target is not None else "no_dashboard_equivalent",
            "target_conflict": target_conflict,
            "dashboard_sponsor": dash_row.get("sponsor", ""),
            "nih_sponsor": "; ".join(nih_sponsors),
            "company_conflict": company_conflict,
            "dashboard_phase": dashboard_phase,
            "nih_phase": nih_phase,
            "phase_conflict": phase_conflict,
            "nih_mechanism_of_action": nih_row["mechanism_of_action"],
        })

    columns = [
        "dashboard_display_name", "nih_agent", "match_tier", "canonical_name_differs",
        "dashboard_drug_type", "nih_modality", "drug_type_conflict",
        "dashboard_target", "nih_cadro", "nih_cadro_mapped_target", "target_conflict",
        "dashboard_sponsor", "nih_sponsor", "company_conflict",
        "dashboard_phase", "nih_phase", "phase_conflict",
        "nih_mechanism_of_action",
    ]
    return pd.DataFrame(rows, columns=columns)
