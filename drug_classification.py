# ============================================================
# DRUG CLASSIFICATION LIBRARY
#
# Pure helper functions for figuring out, from a ClinicalTrials.gov
# "Interventions" cell, which intervention is the sponsor's actual
# therapeutic candidate (as opposed to a placebo, a diagnostic
# imaging agent, a comparator drug, a device, etc).
#
# This file only DEFINES functions — it does not read trials.csv,
# does not write any output files, and does not print anything on
# import. That's deliberate: pipeline_viz.py runs its whole pipeline
# the moment it's imported, which makes it unsafe to import from a
# test file. Keeping this logic here means test_classification.py
# can import and test it in isolation, with no side effects.
#
# No network calls are made anywhere in this file.
# ============================================================

import re

import pandas as pd


def normalize_text(value):
    """
    Lowercase a string and strip punctuation so that two spellings of
    the same name compare equal — e.g. "Leqembi®", "leqembi", and
    "LEQEMBI" should all normalize to the same thing.

    Used for both drug-name matching and sponsor/company-name matching
    against data/official_pipeline.csv. Original (non-normalized) text
    is always preserved separately wherever it's displayed — this
    function is for INTERNAL comparison only.

    Returns "" for missing/NaN input, never None, so callers can always
    safely call .split() or do substring checks on the result.
    """
    if value is None or (pd.api.types.is_scalar(value) and pd.isna(value)):
        return ""
    text = str(value).lower()
    # collapse any run of non-alphanumeric characters (hyphens, slashes,
    # periods, parentheses, registered-trademark symbols, ...) into a
    # single space, so "BMS-708163" and "BMS 708163" normalize the same
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ClinicalTrials.gov's own intervention-type vocabulary — used only to
# detect a DUPLICATED type prefix left inside a parsed name (see
# _strip_duplicated_type_prefix below), e.g. a raw cell literally
# reading "DRUG: Drug: [18F]F-AraG (PET tracer)".
_KNOWN_INTERVENTION_TYPES = {
    "DRUG", "BIOLOGICAL", "DEVICE", "PROCEDURE", "RADIATION", "BEHAVIORAL",
    "GENETIC", "DIETARY_SUPPLEMENT", "COMBINATION_PRODUCT", "DIAGNOSTIC_TEST", "OTHER",
}


def _strip_duplicated_type_prefix(name):
    """
    Some real trials.csv rows contain a duplicated type prefix inside
    the name itself — e.g. "DRUG: Drug: [18F]F-AraG (PET tracer)"
    (NCT07611357). parse_interventions() splits on the FIRST colon, so
    the leftover name is still "Drug: [18F]F-AraG (PET tracer)". This
    strips any further leading "<TYPE>:" prefix, case-insensitively,
    but ONLY when that prefix is one of ct.gov's actual intervention
    types — never a generic colon-containing phrase that just happens
    to precede a real drug name (e.g. "Cohort A: SomeDrug" is left
    alone, since "COHORT_A" isn't a known intervention type).
    """
    stripped = name
    while ":" in stripped:
        prefix, rest = stripped.split(":", 1)
        if prefix.strip().upper().replace(" ", "_") in _KNOWN_INTERVENTION_TYPES:
            stripped = rest.strip()
        else:
            break
    return stripped or name


def parse_interventions(raw):
    """
    Split a ClinicalTrials.gov "Interventions" cell into a list of
    individual interventions, preserving both the type and the name.

    The raw format looks like:
        "DRUG: AR1001|OTHER: Placebo"
        "DRUG: SAR110894|DRUG: Donepezil|OTHER: Placebo"

    i.e. entries separated by "|", each as "TYPE: Name". This function
    returns EVERY entry (never just the first), as a list of dicts:
        [{"type": "DRUG", "name": "AR1001"}, {"type": "OTHER", "name": "Placebo"}]

    If an entry has no recognizable "TYPE:" prefix, "type" is set to
    None rather than the entry being silently dropped — nothing should
    disappear at the parsing stage; exclusion happens later, during
    classification, where the reason can be recorded. A duplicated
    leading type prefix inside the name (see _strip_duplicated_type_prefix)
    is stripped so the displayed name never starts with "Drug:".
    """
    if raw is None or (pd.api.types.is_scalar(raw) and pd.isna(raw)):
        return []

    parts = [p.strip() for p in str(raw).split("|") if p.strip()]
    interventions = []
    for part in parts:
        if ":" in part:
            itype, name = part.split(":", 1)
            itype = itype.strip().upper()
            name = _strip_duplicated_type_prefix(name.strip())
        else:
            itype, name = None, part
        if name:
            interventions.append({"type": itype, "name": name})
    return interventions


# ============================================================
# OFFICIAL PIPELINE REFERENCE (data/official_pipeline.csv)
# ============================================================

REQUIRED_PIPELINE_COLUMNS = ["company", "drug_name", "synonyms", "source_url", "notes"]

# Corporate-suffix words ignored when comparing a sponsor name (e.g. from
# trials.csv's "Sponsor" column, which often reads like "AriBio Co., Ltd.")
# against an official_pipeline.csv "company" value, so "AriBio" still
# matches "AriBio Co., Ltd." without requiring an exact string match.
COMPANY_STOPWORDS = {
    "inc", "co", "ltd", "corp", "corporation", "company", "pharmaceuticals",
    "pharmaceutical", "therapeutics", "biosciences", "laboratories", "labs",
    "group", "holdings", "llc", "plc",
}


def load_official_pipeline(path):
    """
    Read data/official_pipeline.csv into a list of record dicts, one per
    row, each carrying BOTH the original text (for display) and a
    normalized version of the same fields (for matching).

    Read-only: never writes to `path` or any other file. If the file is
    missing, returns [] rather than raising, so a missing reference file
    degrades to "no pipeline matches possible" instead of crashing the
    whole classification pipeline.

    Raises ValueError if the file exists but is missing one of the
    required columns (company, drug_name, synonyms, source_url, notes) —
    that's a real data problem the caller needs to know about, not
    something to silently paper over.
    """
    try:
        raw_df = pd.read_csv(path, dtype=str)
    except FileNotFoundError:
        return []

    missing = [c for c in REQUIRED_PIPELINE_COLUMNS if c not in raw_df.columns]
    if missing:
        raise ValueError(f"official_pipeline.csv at {path!r} is missing required column(s): {missing}")

    raw_df = raw_df.fillna("")

    records = []
    for _, row in raw_df.iterrows():
        company = str(row["company"]).strip()
        drug_name = str(row["drug_name"]).strip()
        synonyms_raw = str(row["synonyms"]).strip()
        source_url = str(row["source_url"]).strip()
        notes = str(row["notes"]).strip()

        if not company or not drug_name:
            # a row with no company or no drug name can't be matched
            # against anything meaningfully — skip it rather than let it
            # produce accidental matches on empty normalized strings
            continue

        synonyms = [s.strip() for s in synonyms_raw.split(";") if s.strip()]

        records.append({
            "company": company,
            "drug_name": drug_name,
            "synonyms": synonyms,
            "source_url": source_url,
            # notes is descriptive metadata ONLY — carried through for
            # display, never read by match_official_pipeline or
            # classify_intervention as evidence of anything
            "notes": notes,
            "company_normalized": normalize_text(company),
            "drug_name_normalized": normalize_text(drug_name),
            "synonyms_normalized": [normalize_text(s) for s in synonyms],
            # OPTIONAL columns (Phase 2 — scientific_classification.py's
            # tier-3 "official pipeline reference"): absent from
            # REQUIRED_PIPELINE_COLUMNS on purpose, so existing
            # official_pipeline.csv rows/tests without them keep working
            # unchanged. Blank string when the column doesn't exist or
            # the cell is empty for this row.
            "modality": str(row.get("modality", "")).strip() if "modality" in raw_df.columns else "",
            "target_pathways": str(row.get("target_pathways", "")).strip() if "target_pathways" in raw_df.columns else "",
        })
    return records


def _company_matches(company_normalized, sponsor_normalized):
    """
    True if a pipeline record's company and a trial's sponsor text
    plausibly refer to the same organization — after both are reduced
    to their non-generic words, one word-set must contain the other
    (e.g. {"eisai"} vs {"eisai", "inc"} matches; {"eisai"} vs
    {"biogen"} does not).
    """
    company_words = set(company_normalized.split())
    sponsor_words = set(sponsor_normalized.split())

    company_meaningful = company_words - COMPANY_STOPWORDS or company_words
    sponsor_meaningful = sponsor_words - COMPANY_STOPWORDS or sponsor_words

    if not company_meaningful or not sponsor_meaningful:
        return False
    return company_meaningful.issubset(sponsor_meaningful) or sponsor_meaningful.issubset(company_meaningful)


def match_official_pipeline(name, sponsor, pipeline_records):
    """
    Look up (name, sponsor) against the loaded official_pipeline.csv
    records. Sponsor/company must match — a drug-name match alone is
    never enough (a name appearing in the pipeline file for a DIFFERENT
    company does not count).

    Drug-name/synonym matching is EXACT on normalized text, not
    substring — a short name like "AR100" must not match "AR1001".

    If more than one distinct (company, drug_name) record matches,
    returns an ambiguous result rather than silently picking one.

    Returns a dict:
        matched: bool — True only for a single, confident match
        matched_company / matched_drug_name / matched_alias: original text
        source_url: original text (may be "")
        verification_status:
            "confirmed_official_match"          — matched, source_url present
            "pipeline_record_match_without_source" — matched, source_url blank
            "ambiguous_multiple_matches"         — more than one record matched
            "no_match"                           — nothing matched
        candidate_matches: populated only when ambiguous — list of the
            competing {"company", "drug_name"} pairs, for manual review
    """
    name_normalized = normalize_text(name)
    sponsor_normalized = normalize_text(sponsor)

    def no_match():
        return {
            "matched": False, "matched_company": "", "matched_drug_name": "",
            "matched_alias": "", "source_url": "", "verification_status": "no_match",
            "candidate_matches": [],
        }

    if not name_normalized or not sponsor_normalized:
        return no_match()

    raw_matches = []
    for record in pipeline_records:
        if not _company_matches(record["company_normalized"], sponsor_normalized):
            continue

        if name_normalized == record["drug_name_normalized"]:
            raw_matches.append({"record": record, "matched_alias": record["drug_name"]})
            continue

        for syn_original, syn_normalized in zip(record["synonyms"], record["synonyms_normalized"]):
            if name_normalized == syn_normalized:
                raw_matches.append({"record": record, "matched_alias": syn_original})
                break

    if not raw_matches:
        return no_match()

    # de-duplicate — the same (company, drug_name) row could appear twice
    # in raw_matches if both its drug_name AND one of its synonyms matched
    unique_by_record = {}
    for m in raw_matches:
        key = (m["record"]["company"], m["record"]["drug_name"])
        unique_by_record.setdefault(key, m)
    matches = list(unique_by_record.values())

    if len(matches) > 1:
        return {
            "matched": False, "matched_company": "", "matched_drug_name": "",
            "matched_alias": "", "source_url": "",
            "verification_status": "ambiguous_multiple_matches",
            "candidate_matches": [
                {"company": m["record"]["company"], "drug_name": m["record"]["drug_name"]}
                for m in matches
            ],
        }

    winner = matches[0]
    record = winner["record"]
    verification_status = (
        "confirmed_official_match" if record["source_url"] else "pipeline_record_match_without_source"
    )
    return {
        "matched": True,
        "matched_company": record["company"],
        "matched_drug_name": record["drug_name"],
        "matched_alias": winner["matched_alias"],
        "source_url": record["source_url"],
        "verification_status": verification_status,
        "candidate_matches": [],
    }


# ============================================================
# INTERVENTION CLASSIFICATION
# ============================================================

CLASSIFICATION_LABELS = [
    "sponsor_developed_therapeutic",
    "investigational_therapeutic_unverified",
    "placebo_or_sham",
    "comparator_or_background_therapy",
    "diagnostic_or_imaging_agent",
    "procedure",
    "device",
    "behavioral",
    "other",
    "uncertain",
]

def _contains_phrase(normalized_name, phrase):
    """
    True if `phrase` (e.g. "ct scan") appears as a contiguous sequence
    of WHOLE tokens in normalized_name — not merely as a raw substring.

    This matters: a raw `"ct scan" in "spect scan"` substring check is
    True, because the tail of "spe-CT" plus " SCAN" spells "ct scan" by
    coincidence, even though "SPECT scan" has nothing to do with a CT
    scan. Checking token sequences instead of characters avoids this
    entire class of false positive for every multi-word phrase list
    below (diagnostic tracers, procedures, behavioral activities).
    """
    tokens = normalized_name.split()
    phrase_tokens = phrase.split()
    n = len(phrase_tokens)
    if n == 0:
        return False
    return any(tokens[i:i + n] == phrase_tokens for i in range(len(tokens) - n + 1))


# --- Step 1: placebo/sham -----------------------------------
# "vehicle" alone is NOT treated as placebo — only "vehicle control" (a
# real diluent/vehicle used as an explicit control arm). A bare
# "Vehicle" entry could be part of an active formulation's name.
#
# "placebos" (plural) is included alongside "placebo" — a real
# trials.csv row ("DRUG: Bromocriptine Mesilate|DRUG: Placebos",
# NCT04413344) used the plural, which the singular-only token check
# missed, causing Bromocriptine Mesilate to be wrongly flagged
# "uncertain" (two unresolved candidates) instead of recognized as the
# trial's sole real candidate. Every other requested variant (matched
# placebo, matching placebo, placebo control, placebo comparator, sham
# control) already contains the bare token "placebo" or "sham" and
# needs no separate entry.
_PLACEBO_TOKENS = {"placebo", "placebos", "sham"}
_PLACEBO_PHRASES = ["vehicle control"]


def _is_placebo(normalized_name):
    tokens = set(normalized_name.split())
    if tokens & _PLACEBO_TOKENS:
        return True
    return any(_contains_phrase(normalized_name, phrase) for phrase in _PLACEBO_PHRASES)


# --- Step 1.5: non-therapeutic control arm (not literally a placebo
# substance, but still not a treatment) --------------------------------
# "No Intervention" / "Untreated" arms compare against nothing at all —
# there is no substance to call a "sham" or "placebo", but they are
# just as much a non-treatment control. Classified as "other" (not
# placebo_or_sham) so the distinction between "given an inactive
# substance" and "given nothing" is still visible in the data, per the
# requirement that these get their own reason text.
#
# "usual care" / "standard of care" are treated as non-therapeutic only
# when the ENTIRE (normalized) intervention name IS just that phrase —
# a conservative choice: if real data ever combines it with a named
# product (e.g. "Usual Care plus DrugX"), exact-phrase equality won't
# match, so it falls through to the therapeutic-candidate checks
# instead of risking a false exclusion of a real combination product.
_NON_THERAPEUTIC_CONTROL_TOKENS = {"untreated"}
_NON_THERAPEUTIC_CONTROL_PHRASES = ["no intervention"]
_ARM_LABEL_EXACT_PHRASES = {"usual care", "standard of care"}


def _is_non_therapeutic_control(normalized_name):
    tokens = set(normalized_name.split())
    if tokens & _NON_THERAPEUTIC_CONTROL_TOKENS:
        return True
    if any(_contains_phrase(normalized_name, phrase) for phrase in _NON_THERAPEUTIC_CONTROL_PHRASES):
        return True
    return normalized_name in _ARM_LABEL_EXACT_PHRASES


# --- Step 2: known diagnostic/imaging agent ------------------
# Deliberately does NOT include a generic "F18"/"F-18"/"C11" isotope
# pattern — an investigational therapeutic can legitimately carry
# isotope notation in its own name, so only specific, named tracers are
# excluded here.
#
# FDG: treating bare "FDG" as always diagnostic is a DATASET-SPECIFIC
# ASSUMPTION for this Alzheimer's-trials project — FDG-PET is
# overwhelmingly used here as a metabolic imaging tracer, not as a
# therapeutic. That assumption may not hold in other disease areas.
# The explicit multi-word forms ("FDG PET", "18F-FDG", "Fluorodeoxyglucose")
# are the more reliable signal and are matched preferentially below;
# bare "fdg" is kept as a fallback token specifically because this
# dataset's real Interventions column often just says "FDG" alone.
# Amyloid, Tau, and metabolic PET tracers used in AD trials, including
# brand names and both the fused ("mk6240") and hyphenated/spaced
# ("mk 6240", from "MK-6240") forms a name can normalize to. Single
# tokens are matched as a whole word (via set intersection below);
# multi-word forms are matched as a token SEQUENCE via _contains_phrase
# (not a raw substring — see _contains_phrase's docstring for why that
# distinction matters).
_DIAGNOSTIC_TOKENS = {
    # amyloid tracers
    "florbetapir", "amyvid", "av45", "florbetaben", "neuraceq",
    "flutemetamol", "vizamyl", "pib", "nav4694", "azd4694",
    # tau tracers
    "flortaucipir", "tauvid", "av1451", "t807", "mk6240", "pi2620",
    "ro948", "ro6958948", "gtp1", "pbb3", "apn1607",
    "thk5317", "thk5351", "thk5117",
    # metabolic imaging
    "fdg", "fluorodeoxyglucose",
    # generic imaging-context words that stand alone as a whole token
    "radiotracer",
}
_DIAGNOSTIC_PHRASES = [
    "pittsburgh compound b", "11c pib",
    "av 45", "av 1451", "mk 6240", "pi 2620", "apn 1607",
    "genentech tau probe 1", "pm pbb3",
    "thk 5317", "thk 5351", "thk 5117",
    "fdg pet", "18f fdg",
    # explicit imaging-context wording — real trials.csv text describes
    # some tracers only by role ("(PET tracer)"), not by name, e.g.
    # "[18F]F-AraG (PET tracer)" (NCT07611357). Matching the CONTEXT
    # phrase means we don't need to hardcode every isotope-labeled
    # compound name as inherently diagnostic — F-AraG itself is NOT
    # assumed diagnostic from isotope notation alone, only from this
    # explicit wording (see requirement: don't classify every F-ARA-G
    # use as diagnostic solely from "[18F]").
    "pet tracer", "pet imaging tracer", "imaging tracer",
    "pet ligand", "spect tracer", "spect ligand",
    "amyloid pet imaging", "tau pet imaging",
]

# "amyloid PET"/"tau PET"/"brain PET" (bare, no "scan") are diagnostic
# per the curated list above, but "amyloid PET scan"/"brain PET scan"
# must still fall through to the PROCEDURE check instead (tracer/agent/
# ligand -> diagnostic_or_imaging_agent; scan/procedure -> procedure).
# A plain _contains_phrase() match on "amyloid pet" would ALSO fire
# inside "amyloid pet scan" (it's a prefix of that longer sequence),
# so this checks the token immediately following the match and skips
# it when that token is "scan" — letting the procedure step (which
# runs after this one) correctly claim it instead.
_AMBIGUOUS_PET_PREFIXES = ["amyloid pet", "tau pet", "brain pet"]


def _has_bare_pet_context(normalized_name):
    tokens = normalized_name.split()
    for phrase in _AMBIGUOUS_PET_PREFIXES:
        phrase_tokens = phrase.split()
        n = len(phrase_tokens)
        for i in range(len(tokens) - n + 1):
            if tokens[i:i + n] == phrase_tokens:
                next_token = tokens[i + n] if i + n < len(tokens) else None
                if next_token != "scan":
                    return True
    return False


def _is_diagnostic_tracer(normalized_name):
    tokens = set(normalized_name.split())
    if tokens & _DIAGNOSTIC_TOKENS:
        return True
    if any(_contains_phrase(normalized_name, phrase) for phrase in _DIAGNOSTIC_PHRASES):
        return True
    return _has_bare_pet_context(normalized_name)


# --- Step 3: procedure/radiation -----------------------------
_PROCEDURE_TYPES = {"PROCEDURE", "RADIATION"}
# "spect" as a whole-token match covers both bare "SPECT" and "SPECT
# scan" ("spect" + "scan" as two separate tokens) in one entry — no
# separate "spect scan" phrase is needed. This must stay a whole-token
# check, not a substring check: a substring check for "ct scan" was
# previously (wrongly) matching inside "spect scan" by coincidence
# (the tail of "spe-CT" + " SCAN"); _contains_phrase's token-sequence
# matching (used for the phrases below) already prevents that class of
# false positive, and this token set uses the same whole-word principle.
_PROCEDURE_TOKENS = {"mri", "biopsy", "spect"}
_PROCEDURE_PHRASES = ["pet scan", "ct scan", "lumbar puncture", "blood draw", "radiation procedure"]


def _is_procedure(itype_upper, normalized_name):
    if itype_upper in _PROCEDURE_TYPES:
        return True
    tokens = set(normalized_name.split())
    if tokens & _PROCEDURE_TOKENS:
        return True
    return any(_contains_phrase(normalized_name, phrase) for phrase in _PROCEDURE_PHRASES)


# --- Step 5: behavioral ---------------------------------------
_BEHAVIORAL_TOKENS = {"exercise", "wii", "counseling", "counselling"}
_BEHAVIORAL_PHRASES = ["cognitive training", "virtual reality exercise", "game based rehabilitation", "nintendo wii"]


def _is_behavioral(itype_upper, normalized_name):
    if itype_upper == "BEHAVIORAL":
        return True
    tokens = set(normalized_name.split())
    if tokens & _BEHAVIORAL_TOKENS:
        return True
    return any(_contains_phrase(normalized_name, phrase) for phrase in _BEHAVIORAL_PHRASES)


def _passes_therapeutic_gate(itype, name):
    """
    True if this single intervention survives steps 1-5 (i.e. it is not
    placebo, not a non-therapeutic control arm, not a diagnostic
    tracer, not a procedure/radiation exam, not a device, not
    behavioral) and is therefore still a candidate therapeutic. Used
    both for the intervention being classified and for scanning its
    siblings — this is also what keeps "No Intervention"/"Untreated"
    from ever counting as a plausible therapeutic sibling.
    """
    normalized = normalize_text(name)
    itype_upper = (itype or "").strip().upper()
    if _is_placebo(normalized):
        return False
    if _is_non_therapeutic_control(normalized):
        return False
    if _is_diagnostic_tracer(normalized):
        return False
    if _is_procedure(itype_upper, normalized):
        return False
    if itype_upper == "DEVICE":
        return False
    if _is_behavioral(itype_upper, normalized):
        return False
    return True


# --- Step 6: candidate therapeutic ----------------------------

# FDA-approved Alzheimer's medicines that commonly appear as a
# comparator or background therapy in AD trials, but can also
# occasionally be the trial's actual studied treatment (e.g. a new
# dose, formulation, or repurposing study) — see classify_intervention
# for how that's disambiguated using sibling_interventions.
#
# Includes both generic and brand names, since trials.csv Interventions
# text uses either (a trial might list "Aricept" rather than
# "Donepezil"). The original intervention text is always preserved
# separately in the result dict — this set is only used internally to
# decide WHICH branch of logic applies, never to rewrite the name.
#   donepezil    <- Aricept
#   memantine    <- Namenda
#   rivastigmine <- Exelon
#   galantamine  <- Razadyne, Reminyl
#
# IMPORTANT ORDERING NOTE: this check must run BEFORE the
# KNOWN_COMPOUND_NAMES fallback below (see classify_intervention step 6).
# Several of these same names (donepezil, aricept, galantamine,
# rivastigmine, memantine) are ALSO present in KNOWN_COMPOUND_NAMES,
# since that list was copied wholesale from pipeline_viz.py's pathway
# lookup. If the known-compound check ran first, it would catch these
# names and return "investigational_therapeutic_unverified" before this
# module ever got a chance to apply the comparator/background-therapy
# disambiguation logic below — silently defeating rule 6d.
APPROVED_BACKGROUND_DRUGS = {
    "donepezil", "aricept",
    "memantine", "namenda",
    "rivastigmine", "exelon",
    "galantamine", "razadyne", "reminyl",
}

# Looks like a sponsor development code: 1-5 letters, optional hyphen,
# 3-7 digits, optional trailing letter — e.g. "AR1001", "SAR110894",
# "BMS-708163", "E2814". Deliberately requires digits, so ordinary
# words (Donepezil, Memantine, Wujia Yizhi granules) never match.
_DEV_CODE_PATTERN = re.compile(r"^[A-Za-z]{1,5}-?\d{3,7}[A-Za-z]?$")


def _looks_like_development_code(name):
    if not name:
        return False
    return bool(_DEV_CODE_PATTERN.match(name.strip()))


# TEMPORARY DUPLICATION — TODO once pipeline_viz.py has an
# `if __name__ == "__main__":` guard: move this shared compound-name
# data into its own side-effect-free module (or keep it here and have
# pipeline_viz.py import FROM here) so there is a single source of
# truth. Right now it is a copy of the compound names from
# pipeline_viz.py's KNOWN_COMPOUNDS dict (values/pathways dropped —
# only used here to answer "is this a name the public AD-pipeline
# literature already recognizes", not to look up a pathway), duplicated
# rather than imported because importing pipeline_viz.py directly would
# execute its entire read-trials.csv/write-output pipeline as a side
# effect of the import statement (see module docstring at the top of
# this file). Until that refactor happens, if you edit one list, edit
# the other to match.
_KNOWN_COMPOUND_KEYS = [
    "bms-708163", "avagacestat", "verubecestat", "mk-8931", "mk8931", "lanabecestat",
    "azd3293", "elenbecestat", "e2609", "atabecestat", "semagacestat", "ly450139",
    "begacestat", "gsi-953", "umibecestat", "cnp520", "ly2886721", "jnj-54861911",
    "bapineuzumab", "aab-003", "pf-05236812", "solanezumab", "gantenerumab",
    "crenezumab", "aducanumab", "lecanemab", "donanemab", "remternetug",
    "trontinemab", "ponezumab", "pf-04360365", "acc-001", "cad106", "abvac40",
    "tb006", "shr-1707", "khk6640", "qs-21", "ub-311", "affitope", "lu af20513",
    "pq912", "varoglutamstat", "ngp 555", "ngp555", "sar228810", "mabt5102a",
    "pf-04494700", "azeliragon", "bms-984923", "lx1001", "elnd005",
    "scyllo-inositol", "ct1812", "tramiprosate", "alz-801", "3aps", "chf 5074",
    "chf5074", "csp-1103", "mpc-7869", "tarenflurbil", "pbt2", "gv-971",
    "sodium oligomannate", "simufilam", "florbetapir", "florbetaben",
    "flutemetamol", "av-45", "av45", "azd4694", "nav4694",
    "gosuranemab", "biib092", "tilavonemab", "abbv-8e12", "semorinemab",
    "ro7105705", "zagotenemab", "ly3303560", "bepranemab", "ucb0107", "e2814",
    "jnj-63733657", "biib080", "nio752", "pnt001", "trx0014", "lmtm",
    "hydromethylthionine", "tideglusib", "spg302", "apn-1607", "thk-5351",
    "thk5351", "trx0037", "abbv-1758", "asn51", "flortaucipir", "av-1451",
    "av1451", "gtp1", "mk-6240", "mk6240", "pi-2620", "pi2620", "mni-187",
    "etanercept", "sargramostim", "minocycline", "al002", "al003", "pbr28",
    "fedaa1106", "dpa713", "xpro1595", "gsk2647544", "ntrx-07", "naproxen",
    "dimebon", "latrepirdine", "cerebrolysin", "t-817ma", "bryostatin",
    "posiphen", "buntanetap", "anavex2-73", "blarcamesine", "nilotinib",
    "bexarotene", "cilostazol", "tadalafil", "sildenafil", "mirodenafil",
    "ar1001", "bpn14770", "ath-1017", "fosgonimeton", "st101",
    "allopregnanolone", "dasatinib", "quercetin", "estrogen", "gv1001",
    "mem 1003", "xaliproden", "pf-04447943",
    "rosiglitazone", "pioglitazone", "metformin", "liraglutide", "semaglutide",
    "exenatide", "t3d-959", "simvastatin", "atorvastatin", "nicotinamide",
    "mib-626", "tricaprilin", "ac-1202",
    "donepezil", "aricept", "e2020", "galantamine", "rivastigmine",
    "memantine", "tacrine", "huperzine", "octohydroaminoacridine", "abt-089",
    "sam-531", "gsk239512", "abt-126", "azd3480", "bi 409306", "bi409306",
    "rasagiline", "jnj-39393406", "evp-6124", "sb-742457", "lecozotan",
    "talsaclidine", "htl0009936", "ac-3933", "pf-05212377",
    "karxt", "xanomeline", "trospium", "brexpiprazole", "pimavanserin",
    "risperidone", "daridorexant", "lemborexant", "suvorexant", "iti-007",
]
KNOWN_COMPOUND_NAMES = frozenset(normalize_text(k) for k in _KNOWN_COMPOUND_KEYS)


# ============================================================
# DOSE/FORMULATION NORMALIZATION (moved here from pipeline_viz.py's
# clean_drug_name() so both files share one implementation without
# pipeline_viz.py's import side effects — see module docstring).
# pipeline_viz.py now imports normalize_intervention_candidate_name and
# uses it wherever it used to define clean_drug_name locally.
# ============================================================

_DOSE_PATTERN = re.compile(
    r"\s*\(?\b\d+(\.\d+)?\s*(mg/kg|mcg/kg|mg/day|mg|mcg|g|iu|ml)\b\)?.*$", re.IGNORECASE
)
_MULTIPLIER_PATTERN = re.compile(r"\s*\d+x$", re.IGNORECASE)
_ROUTE_PATTERN = re.compile(
    r"\s*\((sc|iv|im|oral|multiple dose|part i and part ii)\)\s*$", re.IGNORECASE
)
_FORM_PATTERN = re.compile(
    r"\b(oral|transdermal|extended-release|extended release|\bER\b|\bXR\b|"
    r"tablets?|injection|solution|capsules?|controlled-release)\b",
    re.IGNORECASE,
)


def normalize_intervention_candidate_name(raw):
    """
    Strip common dose/route/formulation noise from an intervention name
    for COMPARISON purposes only — official pipeline matching,
    development-code detection, approved-background alias matching, and
    grouping dose-variant arms of the same drug into one candidate.

    Does NOT strip meaningful digits that are part of a development
    code: "AVP-786-18" has no recognized dose/unit suffix attached (no
    "mg"/"mcg"/"g"/"iu"/"ml" anywhere), so it passes through completely
    unchanged. Only text that actually looks like a dose annotation is
    removed:
        "TRx0237 150 mg/day"              -> "TRx0237"
        "AR1001 30 mg"                    -> "AR1001"
        "Donepezil 10 mg once daily"      -> "Donepezil"
        "Lecanemab 10 mg/kg IV every 2 weeks" -> "Lecanemab"
        "AVP-786-18"                      -> "AVP-786-18" (unchanged)

    The ORIGINAL intervention name must still be used for display
    (e.g. the intervention_name column in pipeline_interventions.csv) —
    this function's output, like normalize_text()'s, is for internal
    comparison only.
    """
    if raw is None or (pd.api.types.is_scalar(raw) and pd.isna(raw)):
        return ""
    name = str(raw).strip()
    name = _DOSE_PATTERN.sub("", name)
    name = _MULTIPLIER_PATTERN.sub("", name)
    name = _ROUTE_PATTERN.sub("", name)
    name = _FORM_PATTERN.sub("", name)
    name = re.sub(r"\s+", " ", name).strip(" -,")
    return name or str(raw).strip()


def _is_investigational_candidate_name(itype, name, sponsor, pipeline_records):
    """
    Used only to inspect a SIBLING intervention: does it look like a
    real (confirmed-or-likely) investigational drug, as opposed to
    placebo/non-therapeutic-control/diagnostic/procedure/device/
    behavioral, or an approved background medicine? Deliberately does
    not call classify_intervention (which would need to know ITS
    siblings too) — this is a narrower, non-recursive check.
    """
    if not _passes_therapeutic_gate(itype, name):
        return False

    candidate_name = normalize_intervention_candidate_name(name)
    match = match_official_pipeline(candidate_name, sponsor, pipeline_records)
    if match["verification_status"] in ("confirmed_official_match", "pipeline_record_match_without_source"):
        return True

    candidate_normalized = normalize_text(candidate_name)
    if set(candidate_normalized.split()) & APPROVED_BACKGROUND_DRUGS:
        return False

    if _looks_like_development_code(candidate_name):
        return True
    if any(compound in candidate_normalized for compound in KNOWN_COMPOUND_NAMES):
        return True
    return False


def _other_non_background_therapeutic_siblings(sibling_interventions):
    """
    Siblings that are still "in play" as a possible therapeutic
    candidate: they pass the placebo/non-therapeutic-control/
    diagnostic/procedure/device/behavioral gate, AND they are not an
    approved-background drug (those are handled by their own dedicated
    branch in classify_intervention, not by the sole-candidate rule
    below).
    """
    others = []
    for s in sibling_interventions:
        if not _passes_therapeutic_gate(s.get("type"), s.get("name")):
            continue
        candidate_normalized = normalize_text(normalize_intervention_candidate_name(s.get("name")))
        if set(candidate_normalized.split()) & APPROVED_BACKGROUND_DRUGS:
            continue
        others.append(s)
    return others


def classify_intervention(intervention_type, intervention_name, sponsor, sibling_interventions, pipeline_records):
    """
    Classify a single intervention from one trial into exactly one of
    CLASSIFICATION_LABELS, using (in priority order): placebo/sham,
    known diagnostic/imaging agent, procedure/radiation, device,
    behavioral, then — for whatever survives all of that — official
    pipeline match, approved-background-drug disambiguation (using
    sibling_interventions), development-code/known-compound pattern,
    and finally an "uncertain" fallback.

    sibling_interventions: the OTHER interventions in the same trial
    (list of {"type", "name"} dicts from parse_interventions, excluding
    this one) — needed only to tell an approved AD medicine used as a
    comparator apart from one that's the trial's actual studied
    treatment (see step 6d below).

    pipeline_records: the list returned by load_official_pipeline().
    """
    original_type = intervention_type
    original_name = intervention_name
    normalized_name = normalize_text(intervention_name)
    itype_upper = (intervention_type or "").strip().upper()
    # dose/formulation-stripped form, used ONLY for matching/grouping in
    # step 6 below (pipeline match, development-code detection, known-
    # compound/approved-background alias matching) — never for display
    candidate_name = normalize_intervention_candidate_name(intervention_name)
    candidate_normalized = normalize_text(candidate_name)

    def result(classification, reason, confidence, needs_manual_review=False,
               official_pipeline_match=False, matched_pipeline_drug=None,
               official_source_url="", verification_status="not_applicable"):
        return {
            "original_name": original_name,
            "original_type": original_type,
            "normalized_name": normalized_name,
            "candidate_name": candidate_name,
            "classification": classification,
            "reason": reason,
            "official_pipeline_match": official_pipeline_match,
            "matched_pipeline_drug": matched_pipeline_drug,
            "official_source_url": official_source_url,
            "verification_status": verification_status,
            "confidence": confidence,
            "needs_manual_review": needs_manual_review,
        }

    # Step 1: placebo/sham
    if _is_placebo(normalized_name):
        return result("placebo_or_sham", "name indicates a placebo/sham/vehicle-control arm", "high")

    # Step 1.5: non-therapeutic control arm — "No Intervention"/
    # "Untreated"/bare "Usual Care"/"Standard of Care". Not literally a
    # placebo substance, so kept as its own classification ("other")
    # rather than folded into placebo_or_sham.
    if _is_non_therapeutic_control(normalized_name):
        return result(
            "other",
            "non-treatment control arm (e.g. no intervention / untreated / usual care), not a therapeutic product",
            "high",
        )

    # Step 2: known diagnostic/imaging agent
    if _is_diagnostic_tracer(normalized_name):
        return result("diagnostic_or_imaging_agent", "matches a curated diagnostic/imaging tracer name", "high")

    # Step 3: procedure/radiation
    if _is_procedure(itype_upper, normalized_name):
        return result("procedure", "intervention type or name indicates a clinical procedure or imaging exam", "high")

    # Step 4: device
    if itype_upper == "DEVICE":
        return result("device", "intervention type is DEVICE", "high")

    # Step 5: behavioral
    if _is_behavioral(itype_upper, normalized_name):
        return result("behavioral", "intervention type or name indicates a behavioral/non-drug activity", "high")

    # Step 6: candidate therapeutic — official pipeline match takes
    # priority over everything below, for ANY drug-like name (even one
    # that happens to also be an approved background medicine's name).
    # Matching uses candidate_name (dose-stripped) rather than the raw
    # intervention_name, so "Lecanemab 10 mg/kg IV every 2 weeks" still
    # matches an official_pipeline.csv row for plain "Lecanemab".
    match = match_official_pipeline(candidate_name, sponsor, pipeline_records)

    if match["verification_status"] == "confirmed_official_match":
        return result(
            "sponsor_developed_therapeutic",
            f"matched official pipeline record for {match['matched_company']} ({match['matched_drug_name']}) with a source URL",
            "high",
            official_pipeline_match=True, matched_pipeline_drug=match["matched_drug_name"],
            official_source_url=match["source_url"], verification_status=match["verification_status"],
        )
    if match["verification_status"] == "pipeline_record_match_without_source":
        return result(
            "sponsor_developed_therapeutic",
            f"matched official pipeline record for {match['matched_company']} ({match['matched_drug_name']}) but source_url is blank — treat as unverified pending a citation",
            "medium", needs_manual_review=True,
            official_pipeline_match=True, matched_pipeline_drug=match["matched_drug_name"],
            official_source_url=match["source_url"], verification_status=match["verification_status"],
        )

    ambiguous_note = (
        " (matched multiple official pipeline records ambiguously — see candidate_matches)"
        if match["verification_status"] == "ambiguous_multiple_matches" else ""
    )

    is_approved_background = bool(set(candidate_normalized.split()) & APPROVED_BACKGROUND_DRUGS)
    if is_approved_background:
        has_investigational_sibling = any(
            _is_investigational_candidate_name(s.get("type"), s.get("name"), sponsor, pipeline_records)
            for s in sibling_interventions
        )
        if has_investigational_sibling:
            return result(
                "comparator_or_background_therapy",
                "approved Alzheimer's medicine used alongside a likely investigational candidate in the same trial" + ambiguous_note,
                "medium", verification_status=match["verification_status"],
            )

        other_drug_like_siblings = [
            s for s in sibling_interventions if _passes_therapeutic_gate(s.get("type"), s.get("name"))
        ]
        if not other_drug_like_siblings:
            return result(
                "investigational_therapeutic_unverified",
                "only non-placebo therapeutic in the trial; approved Alzheimer's medicine with no combination partner found, so it may be the treatment under study" + ambiguous_note,
                "medium", needs_manual_review=True, verification_status=match["verification_status"],
            )

        other_also_background = all(
            bool(set(normalize_text(normalize_intervention_candidate_name(s.get("name"))).split()) & APPROVED_BACKGROUND_DRUGS)
            for s in other_drug_like_siblings
        )
        if other_also_background:
            return result(
                "uncertain",
                "head-to-head comparison of approved Alzheimer's medicines; no investigational candidate identified" + ambiguous_note,
                "low", needs_manual_review=True, verification_status=match["verification_status"],
            )

        return result(
            "uncertain",
            "approved Alzheimer's medicine alongside another unresolved intervention; no primary candidate determined" + ambiguous_note,
            "low", needs_manual_review=True, verification_status=match["verification_status"],
        )

    if _looks_like_development_code(candidate_name) or any(
        compound in candidate_normalized for compound in KNOWN_COMPOUND_NAMES
    ):
        return result(
            "investigational_therapeutic_unverified",
            "no confirmed pipeline match; name resembles a sponsor development code or matches a known investigational compound list" + ambiguous_note,
            "medium", needs_manual_review=True, verification_status=match["verification_status"],
        )

    # General sole-plausible-therapeutic-candidate rule: this name isn't
    # code-shaped and isn't in the curated compound list (e.g. a named
    # herbal formulation like "Wujia Yizhi granules"), but if every OTHER
    # intervention in the trial is placebo/sham/diagnostic/procedural/
    # device/behavioral (or an approved-background drug, handled above),
    # this is the only real therapeutic candidate in the trial — it
    # should not be discarded just because it lacks a code name.
    # Two or more unresolved candidates in the same trial is exactly the
    # case this rule must NOT fire for — that's genuinely ambiguous and
    # falls through to "uncertain" below instead.
    if not _other_non_background_therapeutic_siblings(sibling_interventions):
        return result(
            "investigational_therapeutic_unverified",
            "sole plausible therapeutic candidate in this trial (all other interventions are placebo/diagnostic/procedural/device/behavioral), but lacks an official sponsor-pipeline match" + ambiguous_note,
            "medium", needs_manual_review=True, verification_status=match["verification_status"],
        )

    return result(
        "uncertain",
        "no pipeline match, not a recognized development-code pattern or known compound, and not an approved background medicine; multiple unresolved therapeutic candidates in this trial" + ambiguous_note,
        "low", needs_manual_review=True, verification_status=match["verification_status"],
    )


# ============================================================
# TRIAL-LEVEL AGGREGATION
# ============================================================

def build_interventions_dataframe(trials_df, pipeline_records, scope_overrides=None):
    """
    Build a long-format DataFrame — one row per (trial, intervention) —
    from a trials DataFrame.

    Expects trials_df to already have gone through pipeline_viz.py's
    column_map rename step, i.e. to have "nct_id", "sponsor", "title",
    and "interventions" columns (not the raw ClinicalTrials.gov export
    header names).

    For every trial, every one of its individual interventions is
    parsed (parse_interventions), classified (classify_intervention),
    and then scoped (classify_pipeline_scope — Phase 1A's intervention-
    scope gap closure), with that trial's OTHER interventions passed in
    as sibling_interventions. Nothing is dropped — every parsed
    intervention becomes exactly one output row, tagged with its
    trial's nct_id/sponsor/title for pipeline_interventions.csv.

    scope_overrides: dict from load_scope_overrides() (or {} / None for
    "no curated overrides") — passed straight through to
    classify_pipeline_scope() for every row.
    """
    scope_overrides = scope_overrides or {}
    rows = []
    for _, trial in trials_df.iterrows():
        nct_id = trial.get("nct_id")
        sponsor = trial.get("sponsor")
        title = trial.get("title")
        raw_interventions = trial.get("interventions")

        interventions = parse_interventions(raw_interventions)
        for i, interv in enumerate(interventions):
            siblings = interventions[:i] + interventions[i + 1:]
            classified = classify_intervention(
                interv["type"], interv["name"], sponsor, siblings, pipeline_records
            )
            scoped = classify_pipeline_scope(
                interv["type"], interv["name"], classified["classification"],
                classified["verification_status"], overrides=scope_overrides,
            )
            rows.append({
                "nct_id": nct_id,
                "sponsor": sponsor,
                "title": title,
                **classified,
                **scoped,
            })
    return pd.DataFrame(rows)


def resolve_developed_drug(classified_interventions):
    """
    Aggregate ONE trial's classified interventions (a list of the dicts
    classify_intervention() returns — everything with the same nct_id)
    into a single trial-level "what is the sponsor actually developing"
    summary.

    Resolution priority (see checkpoint discussion for why these are
    grouped into a "sponsor-developed tier" and an "unverified tier"
    rather than checked strictly top-to-bottom — a trial with 2
    confirmed pipeline matches plus 1 unverified candidate must be
    flagged as an ambiguous multiple-match case, not silently resolved
    by "exactly one unverified candidate" further down the list):

      1. Exactly one DISTINCT sponsor_developed_therapeutic candidate,
         confirmed_official_match -> choose it. confidence=high,
         needs_manual_review=False.
      2. Exactly one DISTINCT sponsor_developed_therapeutic candidate,
         pipeline_record_match_without_source -> choose it. confidence=medium,
         needs_manual_review=True.
      4. Two or more DISTINCT sponsor_developed_therapeutic candidates
         (any mix of confirmed/unsourced) -> do NOT silently choose one;
         developed_drug holds all candidate names joined with "; ".
         confidence=low, needs_manual_review=True. (checked here, before
         rule 3, so it always wins over a same-trial unverified candidate)
      3. Exactly one DISTINCT investigational_therapeutic_unverified
         candidate (and no sponsor-developed match at all) -> choose it
         as "possible". confidence=medium, needs_manual_review=True.
      5. Two or more DISTINCT investigational_therapeutic_unverified
         candidates (and no sponsor-developed match) -> developed_drug
         holds all candidate names joined with "; ". confidence=low,
         needs_manual_review=True.
      6. Nothing in either tier -> developed_drug="",
         drug_classification="no_therapeutic_candidate" (a trial-summary-only
         value, not one of CLASSIFICATION_LABELS). If the trial nonetheless
         contains "uncertain" intervention rows (e.g. two approved AD
         medicines head-to-head with no investigational candidate),
         needs_manual_review is still set True even though no drug name
         can be given.

    "DISTINCT" is the key word added this checkpoint: two intervention
    ROWS that are really the same drug at different doses (e.g.
    "TRx0237 150 mg/day" and "TRx0237 250 mg/day") must count as ONE
    candidate, not two — otherwise a dose-ranging trial of a single real
    investigational drug would always look "ambiguous" (rule 4/5) purely
    because ct.gov lists one row per dose arm. Grouping key: for
    sponsor_developed_therapeutic rows, the official pipeline's own
    matched_pipeline_drug (already canonical, dose-text-proof by
    construction); for investigational_therapeutic_unverified rows, the
    normalized, dose-stripped candidate_name classify_intervention()
    computed internally.
    """
    confirmed = [
        r for r in classified_interventions
        if r["classification"] == "sponsor_developed_therapeutic"
        and r["verification_status"] == "confirmed_official_match"
    ]
    unsourced = [
        r for r in classified_interventions
        if r["classification"] == "sponsor_developed_therapeutic"
        and r["verification_status"] == "pipeline_record_match_without_source"
    ]
    unverified = [r for r in classified_interventions if r["classification"] == "investigational_therapeutic_unverified"]

    def group_by(rows, key_fn):
        groups, order = {}, []
        for r in rows:
            k = key_fn(r)
            if k not in groups:
                groups[k] = []
                order.append(k)
            groups[k].append(r)
        return [groups[k] for k in order]

    confirmed_groups = group_by(confirmed, lambda r: r["matched_pipeline_drug"])
    unsourced_groups = group_by(unsourced, lambda r: r["matched_pipeline_drug"])
    unverified_groups = group_by(
        unverified, lambda r: r.get("candidate_name") and normalize_text(r["candidate_name"]) or normalize_text(r["original_name"])
    )
    sponsor_developed_groups = confirmed_groups + unsourced_groups

    def summary(developed_drug, drug_classification, reason, confidence, needs_manual_review,
                official_pipeline_match=False, official_source_url="", verification_status="",
                pipeline_scope="Needs Review", scope_reason="", scope_method="rule_classification",
                scope_confidence="low", manual_review_required=None):
        return {
            "developed_drug": developed_drug,
            "developed_drug_normalized": normalize_text(developed_drug),
            "drug_classification": drug_classification,
            "classification_reason": reason,
            "official_pipeline_match": official_pipeline_match,
            "official_source_url": official_source_url,
            "verification_status": verification_status,
            "classification_confidence": confidence,
            "needs_manual_review": needs_manual_review,
            # Phase 1A: pipeline_scope/scope_* carried forward from the
            # WINNING intervention row(s) below — resolve_developed_drug()
            # itself makes no new scope judgment, it just forwards
            # classify_pipeline_scope()'s per-intervention verdict to the
            # trial level so build_resolved_drugs_dataframe() can filter on it.
            "pipeline_scope": pipeline_scope,
            "scope_reason": scope_reason,
            "scope_method": scope_method,
            "scope_confidence": scope_confidence,
            "manual_review_required": manual_review_required if manual_review_required is not None else needs_manual_review,
        }

    def scope_fields(r):
        return {
            "pipeline_scope": r.get("pipeline_scope", "Needs Review"),
            "scope_reason": r.get("scope_reason", ""),
            "scope_method": r.get("scope_method", "rule_classification"),
            "scope_confidence": r.get("scope_confidence", "low"),
            "manual_review_required": r.get("manual_review_required", False),
        }

    # Rule 4 is checked ahead of rule 3 deliberately (see docstring).
    if len(sponsor_developed_groups) >= 2:
        names = "; ".join(group[0]["matched_pipeline_drug"] for group in sponsor_developed_groups)
        return summary(
            names, "sponsor_developed_therapeutic",
            f"multiple sponsor-developed matches found ({names}) — could not resolve to a single drug",
            "low", True, verification_status="multiple_candidates_unresolved",
            pipeline_scope="Needs Review",
            scope_reason="multiple distinct sponsor-developed candidates found in one trial — scope not determined at trial level",
            scope_method="rule_classification", scope_confidence="low", manual_review_required=True,
        )

    # Rule 1
    if len(confirmed_groups) == 1:
        r = confirmed_groups[0][0]
        return summary(
            r["matched_pipeline_drug"], "sponsor_developed_therapeutic",
            f"confirmed official pipeline match: {r['matched_pipeline_drug']}",
            "high", False,
            official_pipeline_match=True, official_source_url=r["official_source_url"],
            verification_status=r["verification_status"], **scope_fields(r),
        )

    # Rule 2
    if len(unsourced_groups) == 1:
        r = unsourced_groups[0][0]
        return summary(
            r["matched_pipeline_drug"], "sponsor_developed_therapeutic",
            f"official pipeline record matched ({r['matched_pipeline_drug']}) but source_url is blank — verify manually",
            "medium", True,
            official_pipeline_match=True, official_source_url=r["official_source_url"],
            verification_status=r["verification_status"], **scope_fields(r),
        )

    # Rule 3
    if len(unverified_groups) == 1:
        group = unverified_groups[0]
        r = group[0]
        display_name = r.get("candidate_name") or r["original_name"]
        return summary(
            display_name, "investigational_therapeutic_unverified",
            f"sole investigational candidate identified ({display_name}); not yet confirmed via official pipeline",
            "medium", True, verification_status=r["verification_status"], **scope_fields(r),
        )

    # Rule 5
    if len(unverified_groups) >= 2:
        names = "; ".join((group[0].get("candidate_name") or group[0]["original_name"]) for group in unverified_groups)
        return summary(
            names, "investigational_therapeutic_unverified",
            f"multiple unverified investigational candidates found ({names}) — could not resolve to a single drug",
            "low", True, verification_status="multiple_candidates_unresolved",
            pipeline_scope="Needs Review",
            scope_reason="multiple distinct unverified candidates found in one trial — scope not determined at trial level",
            scope_method="rule_classification", scope_confidence="low", manual_review_required=True,
        )

    # Rule 6: no therapeutic candidate in either tier
    has_uncertain_rows = any(r["classification"] == "uncertain" for r in classified_interventions)
    if has_uncertain_rows:
        return summary(
            "", "no_therapeutic_candidate",
            "no confirmed or investigational therapeutic candidate found, but this trial has unresolved/uncertain interventions worth a manual look",
            "low", True, verification_status="no_match",
            pipeline_scope="Needs Review", scope_reason="no therapeutic candidate; trial has unresolved/uncertain interventions",
            scope_method="rule_classification", scope_confidence="low", manual_review_required=True,
        )
    return summary(
        "", "no_therapeutic_candidate",
        "no drug/biological therapeutic candidate found among this trial's interventions",
        "high", False, verification_status="not_applicable",
        pipeline_scope="Exclude", scope_reason="no drug/biological therapeutic candidate found among this trial's interventions",
        scope_method="rule_classification", scope_confidence="high", manual_review_required=False,
    )


# ============================================================
# DRUG-LEVEL ROLLUP (pipeline_drugs.csv, new developed_drug-based source)
# ============================================================

_CONFIDENCE_RANK = {"high": 3, "medium": 2, "low": 1}
_DRUG_ROLLUP_STATUS_PRIORITY = ["FDA Approved", "Recruiting", "Active", "Completed", "Discontinued", "Unknown", "Other"]
# Every phase_clean value pipeline_viz.py's clean_phase() can produce
# must have a rank here — trials.csv carries far more than just Phase
# 1/2/3 (NA, Early Phase 1, Phase 4, and the combined Phase 1/Phase 2 /
# Phase 2/Phase 3 designations ct.gov itself uses), and every one of
# them is now a real contributor to a drug's rollup, not filtered out
# upstream. A phase_clean value missing from this dict would make
# g["phase_rank"] NaN for every one of that drug's rows, and if ALL of a
# drug's trials shared that value, .max() would be NaN too — leaving
# top_rows empty and raising an IndexError on the very next line.
_DRUG_ROLLUP_PHASE_RANK = {
    "Phase 4": 8, "Phase 3": 7, "Phase 2/Phase 3": 6, "Phase 2": 5,
    "Phase 1/Phase 2": 4, "Phase 1": 3, "Early Phase 1": 2, "NA": 1,
}


def _mode_or_first(series):
    m = series.mode()
    return m.iloc[0] if not m.empty else series.iloc[0]


def build_resolved_drugs_dataframe(trials_df):
    """
    Roll trials up into one row per drug, using resolve_developed_drug()'s
    trial-level fields as the source of truth — NOT
    primary_intervention_name()/drug_key. This is the new source for
    pipeline_drugs.csv (see pipeline_viz.py's legacy `drugs_df`, which
    is left untouched for the HTML/charts/spotlight until those are
    explicitly migrated).

    Only trials classified sponsor_developed_therapeutic or
    investigational_therapeutic_unverified, with a non-blank
    developed_drug, NOT flagged multiple_candidates_unresolved, AND whose
    Phase 1A pipeline_scope is one of RESOLVED_DRUGS_DF_ELIGIBLE_SCOPES
    (Therapeutic Drug / Diagnostic Agent / Non-Drug Intervention /
    Supportive Treatment / Needs Review) contribute a row. Everything
    else (placebo/diagnostic-tracer/procedure/device/behavioral/
    comparator/uncertain/no_therapeutic_candidate trials, trials with
    unresolved multiple candidates, AND — Phase 1A — anything whose scope
    resolved to "Exclude" or "Placebo or Comparator", e.g. a generic
    "Blood Test"/"CSF Biomarkers" description that classify_intervention()
    alone would have let through) produces NO drug row here — see
    build_unresolved_trials_dataframe() for where the drug-identity-
    ambiguous ones go instead, so nothing is silently dropped from the
    dataset as a WHOLE (pipeline_annotated.csv / pipeline_interventions.csv
    still carry every trial/intervention regardless).

    resolved_drugs_df is still the ONE drug-level source of truth for
    every dashboard component (Phase 0) — Phase 1A does not split it into
    a second dataframe. Instead every row carries pipeline_scope, and
    dashboard components that must show ONLY real therapeutic drugs
    filter resolved_drugs_df down to pipeline_scope == "Therapeutic Drug"
    themselves (pipeline_viz.py's therapeutic_drugs_df); components that
    intentionally offer an optional "reveal non-therapeutic records"
    view read the unfiltered resolved_drugs_df.

    Grouped by developed_drug_normalized — NOT by drug + sponsor. This
    preserves the pre-existing dashboard behavior (one row per drug,
    globally across all sponsors), which was already the case before
    this checkpoint (the old drug_key-based grouping did the same). If
    a drug has more than one distinct sponsor across its contributing
    trials, ALL of them are preserved in the `sponsor` field
    (semicolon-joined) rather than silently picking one via
    mode/first (the old behavior) — and needs_manual_review is forced
    True, since a shared drug name across multiple sponsors is exactly
    the kind of thing that needs a human to confirm actual ownership
    before this dashboard implies any one of them "develops" it.

    Expects trials_df to carry (per trial): nct_id, sponsor, phase_clean,
    status_clean, drug_type, target, enrollment, is_aribio,
    developed_drug, developed_drug_normalized, drug_classification,
    verification_status, classification_confidence, needs_manual_review.
    """
    eligible = trials_df[
        trials_df["drug_classification"].isin(["sponsor_developed_therapeutic", "investigational_therapeutic_unverified"])
        & trials_df["developed_drug"].fillna("").astype(str).str.strip().ne("")
        & (trials_df["verification_status"] != "multiple_candidates_unresolved")
        & trials_df["pipeline_scope"].isin(RESOLVED_DRUGS_DF_ELIGIBLE_SCOPES)
    ].copy()

    empty_columns = [
        "display_name", "phase_reached", "nct_id", "status_summary", "drug_type", "target",
        "sponsor", "trial_count", "max_enrollment", "is_aribio", "verification_status",
        "classification_confidence", "needs_manual_review", "confirmed_trial_count", "unverified_trial_count",
        "official_source_url", "classification_reason", "nct_ids",
        "pipeline_scope", "scope_reason", "scope_method", "scope_confidence", "manual_review_required",
    ]
    if eligible.empty:
        return pd.DataFrame(columns=empty_columns)

    eligible["drug_key"] = eligible["developed_drug_normalized"]

    def summarize(g):
        g = g.copy()
        g["phase_rank"] = g["phase_clean"].map(_DRUG_ROLLUP_PHASE_RANK)
        top_rows = g[g["phase_rank"] == g["phase_rank"].max()]
        statuses_at_top = top_rows["status_clean"].tolist()
        status = next((s for s in _DRUG_ROLLUP_STATUS_PRIORITY if s in statuses_at_top), statuses_at_top[0])

        confirmed_rows = g[g["drug_classification"] == "sponsor_developed_therapeutic"]
        unverified_rows = g[g["drug_classification"] == "investigational_therapeutic_unverified"]

        # canonical display name: prefer an official-pipeline-matched
        # name (already canonical — e.g. "Simufilam" — since
        # resolve_developed_drug() uses the pipeline record's own
        # drug_name for these) over a raw, possibly inconsistently
        # cased unverified candidate name
        if len(confirmed_rows):
            display_name = confirmed_rows["developed_drug"].iloc[0]
        else:
            display_name = min(g["developed_drug"], key=len)

        sponsors = sorted(set(g["sponsor"].dropna().astype(str)) - {""})
        sponsor_field = "; ".join(sponsors)
        sponsor_ambiguous = len(sponsors) > 1

        if len(confirmed_rows) and len(unverified_rows):
            verification_status = "mixed"
        else:
            statuses = set(g["verification_status"])
            verification_status = statuses.pop() if len(statuses) == 1 else "mixed"

        classification_confidence = min(
            (g["classification_confidence"].map(_CONFIDENCE_RANK).fillna(1)),
            default=1,
        )
        classification_confidence = {v: k for k, v in _CONFIDENCE_RANK.items()}[classification_confidence]

        needs_manual_review = bool(g["needs_manual_review"].any()) or sponsor_ambiguous

        # source URL / reason are only meaningful from a confirmed
        # official-pipeline match; fall back to the first unverified
        # row's reason so there's still something to show in row detail
        if len(confirmed_rows):
            official_source_url = next(
                (u for u in confirmed_rows.get("official_source_url", pd.Series(dtype=str)) if u), ""
            )
            classification_reason = confirmed_rows["classification_reason"].iloc[0] if "classification_reason" in confirmed_rows.columns else ""
        else:
            official_source_url = ""
            classification_reason = unverified_rows["classification_reason"].iloc[0] if "classification_reason" in unverified_rows.columns else ""

        nct_ids = "; ".join(sorted(set(g["nct_id"].dropna().astype(str))))

        # Phase 1A: pipeline_scope aggregation. In the overwhelming
        # majority of groups every contributing trial agrees on scope
        # (it's derived from the same winning intervention type/name) —
        # when they DON'T agree, this never silently promotes to
        # "Therapeutic Drug" just because that happens to be one of the
        # observed scopes; disagreement always routes to "Needs Review"
        # so a human resolves which trial's evidence is right.
        scopes_seen = set(g["pipeline_scope"].dropna())
        scope_disagreement = len(scopes_seen) > 1
        if scope_disagreement:
            pipeline_scope = "Needs Review"
            scope_reason = f"contributing trials disagree on pipeline scope ({'; '.join(sorted(scopes_seen))}) for this canonical drug name"
            scope_method = "aggregation_conflict"
            scope_confidence = "low"
        elif scopes_seen:
            pipeline_scope = scopes_seen.pop()
            scope_row = g[g["pipeline_scope"] == pipeline_scope].iloc[0]
            scope_reason = scope_row.get("scope_reason", "")
            scope_method = scope_row.get("scope_method", "")
            scope_confidence = scope_row.get("scope_confidence", "")
        else:
            pipeline_scope, scope_reason, scope_method, scope_confidence = "Needs Review", "no scope information available", "rule_classification", "low"

        manual_review_required = bool(g["manual_review_required"].any()) if "manual_review_required" in g.columns else False
        manual_review_required = manual_review_required or scope_disagreement

        return pd.Series({
            "display_name": display_name,
            "phase_reached": top_rows["phase_clean"].iloc[0],
            "nct_id": top_rows["nct_id"].iloc[0],
            "status_summary": status,
            "drug_type": _mode_or_first(g["drug_type"]),
            "target": _mode_or_first(g["target"]),
            "sponsor": sponsor_field,
            "trial_count": g["nct_id"].nunique(),
            "max_enrollment": g["enrollment"].max(),
            "is_aribio": bool(g["is_aribio"].any()) if "is_aribio" in g.columns else False,
            "verification_status": verification_status,
            "classification_confidence": classification_confidence,
            "needs_manual_review": needs_manual_review,
            "confirmed_trial_count": confirmed_rows["nct_id"].nunique(),
            "unverified_trial_count": unverified_rows["nct_id"].nunique(),
            "official_source_url": official_source_url,
            "classification_reason": classification_reason,
            "nct_ids": nct_ids,
            "pipeline_scope": pipeline_scope,
            "scope_reason": scope_reason,
            "scope_method": scope_method,
            "scope_confidence": scope_confidence,
            "manual_review_required": manual_review_required,
        })

    return (
        eligible.groupby("drug_key")
        .apply(summarize, include_groups=False)
        .reset_index(drop=True)
    )


def build_unresolved_trials_dataframe(trials_df):
    """
    Trials whose developed_drug resolution needs a human look — the
    audit trail for everything build_resolved_drugs_dataframe() leaves
    out of the drug rollup so nothing is silently dropped. A trial
    qualifies when:
      - verification_status == "multiple_candidates_unresolved" (2+
        distinct therapeutic candidates found, couldn't pick one), or
      - drug_classification == "no_therapeutic_candidate" AND
        needs_manual_review is True (no drug could be named, but the
        trial had unresolved "uncertain" interventions worth a look —
        as opposed to a cleanly non-therapeutic trial, e.g. a
        placebo-only or diagnostic-only trial, which needs no review
        at all and is correctly just absent from both CSVs), or
      - classification_confidence == "low"

    Expects trials_df to carry: nct_id, sponsor, title, interventions,
    developed_drug, drug_classification, classification_reason,
    classification_confidence, verification_status, needs_manual_review.
    """
    mask = (
        (trials_df["verification_status"] == "multiple_candidates_unresolved")
        | ((trials_df["drug_classification"] == "no_therapeutic_candidate") & (trials_df["needs_manual_review"] == True))  # noqa: E712
        | (trials_df["classification_confidence"] == "low")
    )
    unresolved = trials_df[mask].copy()
    return unresolved.rename(columns={
        "nct_id": "NCT Number",
        "sponsor": "Sponsor",
        "title": "Study Title",
        "interventions": "Interventions",
    })[[
        "NCT Number", "Sponsor", "Study Title", "Interventions",
        "developed_drug", "drug_classification", "classification_reason",
        "classification_confidence", "verification_status", "needs_manual_review",
    ]]


# ============================================================
# PHASE 0 — DASHBOARD DATA-SOURCE CONSOLIDATION
#
# These two functions exist so drug-level VISUALIZATIONS never need to
# recompute a target×phase cross-tab or a drug↔trial join inline — both
# are pure, unit-testable, and operate ONLY on resolved_drugs_df (the
# one drug-level source of truth), never on legacy_drugs_df or raw
# trial data.
# ============================================================

def build_target_phase_counts(resolved_drugs_df, targets, phases):
    """
    Cross-tab of (target, phase_reached) DRUG counts from resolved_drugs_df
    — the exact grid the target×phase heatmap renders. Extracted as a pure
    function (rather than left inline inside the Plotly-figure-building
    code) so it's directly unit-testable independent of the chart it feeds,
    and so it's unambiguous that the heatmap counts unique drugs, not trials.

    targets/phases are passed in (not hardcoded) so this can be reused for
    any subset/facet of resolved_drugs_df (e.g. the "Small Molecule only"
    heatmap tab) with the same target/phase ordering.
    """
    return [
        [len(resolved_drugs_df[(resolved_drugs_df["target"] == t) & (resolved_drugs_df["phase_reached"] == p)]) for p in phases]
        for t in targets
    ]


def build_resolved_drug_trial_links_df(resolved_drugs_df):
    """
    One row per (canonical drug, contributing trial) pair — the explicit
    join table between resolved_drugs_df (one row per drug) and the
    trial-level table (one row per NCT ID), built by exploding each drug's
    semicolon-joined nct_ids column.

    Exists so the drug<->trial many-to-many relationship is a first-class,
    directly queryable dataset (e.g. "which trials support this drug?",
    "how many distinct trials feed the drug rollup overall?") instead of
    an implicit string every caller has to re-split themselves.
    """
    rows = []
    for _, r in resolved_drugs_df.iterrows():
        nct_id_field = r.get("nct_ids", "")
        nct_ids = str(nct_id_field).split("; ") if nct_id_field else []
        for nct_id in nct_ids:
            nct_id = nct_id.strip()
            if nct_id:
                rows.append({"display_name": r["display_name"], "nct_id": nct_id})
    return pd.DataFrame(rows, columns=["display_name", "nct_id"])


def build_drug_date_rollup(resolved_drug_trial_links_df, trials_df):
    """
    One row per canonical drug: earliest_start_date (min across every
    contributing trial) and latest_primary_completion_date (max across
    every contributing trial) — the real date span this drug has been
    in clinical development on ClinicalTrials.gov, derived from its
    OWN contributing trials via resolved_drug_trial_links_df (never a
    single trial's dates alone, since a drug's earliest/most-recent
    activity often isn't on whichever trial happens to be its
    highest-phase one).

    Expects trials_df to carry nct_id, start_date_parsed, and
    primary_completion_date_parsed — already-parsed pandas Timestamps
    (NaT for missing/unparseable raw dates), computed once from ct.gov's
    raw "Start Date"/"Primary Completion Date" text before this is called.

    A drug with no contributing trial carrying a parseable date on
    either field gets NaT for that field here — never a fabricated date.
    """
    columns = ["display_name", "earliest_start_date", "latest_primary_completion_date"]
    if resolved_drug_trial_links_df.empty:
        return pd.DataFrame(columns=columns)

    dates_by_nct = trials_df.drop_duplicates(subset="nct_id").set_index("nct_id")[
        ["start_date_parsed", "primary_completion_date_parsed"]
    ]
    merged = resolved_drug_trial_links_df.join(dates_by_nct, on="nct_id")
    result = merged.groupby("display_name", as_index=False).agg(
        earliest_start_date=("start_date_parsed", "min"),
        latest_primary_completion_date=("primary_completion_date_parsed", "max"),
    )
    return result[columns]


# ============================================================
# PHASE 1A — INTERVENTION-SCOPE GAP CLOSURE
#
# classify_intervention() (above) answers "is this the sponsor's studied
# candidate, or a placebo/comparator/diagnostic/procedure/device/
# behavioral distractor?" — but it never checks the ClinicalTrials.gov
# intervention TYPE against categories that can slip through its
# therapeutic gate as a "candidate" (DIETARY_SUPPLEMENT, generic
# DIAGNOSTIC_TEST descriptions like "Blood Test", COMBINATION_PRODUCT,
# GENETIC). A dietary supplement with no dev-code-shaped name and no
# other intervention in its trial is, today, the "sole plausible
# candidate" per classify_intervention's step 6 — and becomes a
# canonical "drug" in resolved_drugs_df. That's the confirmed leakage
# this section closes.
#
# Deliberately a SEPARATE layer bolted on AFTER classify_intervention()
# rather than a rewrite of it: classify_intervention() and its 113
# existing tests stay byte-for-byte the same (it remains the intervention-
# type/therapeutic-candidate audit trail), and this layer adds the
# additional "is this actually eligible to be a dashboard drug record"
# judgment on top, without disturbing what's already verified correct.
# ============================================================

PIPELINE_SCOPE_LABELS = [
    "Therapeutic Drug",
    "Diagnostic Agent",
    "Non-Drug Intervention",
    "Supportive Treatment",
    "Placebo or Comparator",
    "Exclude",
    "Needs Review",
]

# Records with this pipeline_scope are eligible for the DEFAULT
# ("Therapeutic Drug") dashboard view.
THERAPEUTIC_SCOPE = "Therapeutic Drug"

# Records with these scopes still get a resolved_drugs_df row (so the
# dashboard's optional "reveal non-therapeutic records" filter has
# something real to show) — everything else (Exclude, Placebo or
# Comparator) never becomes a resolved_drugs_df row at all, per the
# requirement that placebo/comparator never appear in ordinary OR
# optional dashboard views, and that generic/junk descriptions never
# become a canonical "drug" of any kind.
RESOLVED_DRUGS_DF_ELIGIBLE_SCOPES = [
    "Therapeutic Drug", "Diagnostic Agent", "Non-Drug Intervention",
    "Supportive Treatment", "Needs Review",
]


# --- curated override file (data/reference/intervention_scope_overrides.csv) ---

REQUIRED_SCOPE_OVERRIDE_COLUMNS = [
    "normalized_intervention_name", "pipeline_scope", "canonical_name_override",
    "reason", "source", "reviewer", "verified_date",
]


def load_scope_overrides(path):
    """
    Read data/reference/intervention_scope_overrides.csv into a dict keyed
    by normalized intervention name, mirroring load_official_pipeline()'s
    read-only/missing-file-tolerant behavior: a missing file degrades to
    "no curated overrides available" (returns {}) rather than crashing the
    pipeline. Raises ValueError if the file exists but is missing a
    required column — a real data problem, not something to paper over.

    This exists specifically so exceptions to the general type/keyword
    rules below (e.g. "this one dietary supplement IS a genuine
    investigational therapeutic program") live in a reviewable CSV a
    non-engineer can maintain, rather than hardcoded in Python.
    """
    try:
        raw_df = pd.read_csv(path, dtype=str)
    except FileNotFoundError:
        return {}

    missing = [c for c in REQUIRED_SCOPE_OVERRIDE_COLUMNS if c not in raw_df.columns]
    if missing:
        raise ValueError(f"intervention_scope_overrides.csv at {path!r} is missing required column(s): {missing}")

    raw_df = raw_df.fillna("")

    overrides = {}
    for _, row in raw_df.iterrows():
        key = normalize_text(row["normalized_intervention_name"])
        if not key:
            continue
        overrides[key] = {
            "pipeline_scope": str(row["pipeline_scope"]).strip(),
            "canonical_name_override": str(row["canonical_name_override"]).strip(),
            "reason": str(row["reason"]).strip(),
            "source": str(row["source"]).strip(),
            "reviewer": str(row["reviewer"]).strip(),
            "verified_date": str(row["verified_date"]).strip(),
        }
    return overrides


# --- generic/junk description detection (type-agnostic keyword net) ---
# These names must NEVER become a canonical drug regardless of which
# CLASSIFICATION_LABELS bucket classify_intervention() put them in — they
# describe a procedure/measurement/generic-care-activity, not a product.
_GENERIC_DIAGNOSTIC_TEST_TOKENS = {"biomarker", "biomarkers", "genotyping", "genotype"}
_GENERIC_DIAGNOSTIC_TEST_PHRASES = [
    "blood test", "blood draw", "blood sample", "blood samples", "blood collection",
    "csf biomarkers", "csf collection", "csf sample", "csf samples",
    "cerebrospinal fluid biomarkers", "cerebrospinal fluid csf biomarkers",
    "biomarker analysis", "biomarker collection", "genetic testing", "apoe genotyping",
]


def _is_generic_diagnostic_description(normalized_name):
    tokens = set(normalized_name.split())
    if tokens & _GENERIC_DIAGNOSTIC_TEST_TOKENS:
        return True
    return any(_contains_phrase(normalized_name, phrase) for phrase in _GENERIC_DIAGNOSTIC_TEST_PHRASES)


# CBTi ("Cognitive Behavioral Therapy for insomnia") and similar digital/
# behavioral-therapy names sometimes arrive with a non-BEHAVIORAL
# ct.gov type (e.g. OTHER), so classify_intervention()'s type-based
# behavioral check (itype_upper == "BEHAVIORAL") can miss them — this is
# a type-agnostic keyword net that catches them by name regardless of
# the (possibly wrong) source type.
_GENERIC_NON_DRUG_TOKENS = {"cbti", "cbt"}
_GENERIC_NON_DRUG_PHRASES = [
    "cognitive behavioral therapy", "digital therapeutic application", "mobile application",
]


def _is_generic_non_drug_description(normalized_name):
    tokens = set(normalized_name.split())
    if tokens & _GENERIC_NON_DRUG_TOKENS:
        return True
    return any(_contains_phrase(normalized_name, phrase) for phrase in _GENERIC_NON_DRUG_PHRASES)


# --- GENETIC: gene therapy vs. genetic testing/genotyping/biomarker analysis ---
_GENE_THERAPY_TOKENS = {"aav", "aav2", "aav5", "aav9", "vector", "lentiviral", "lentivirus", "adenoviral", "crispr"}
_GENE_THERAPY_PHRASES = ["gene therapy", "viral vector", "gene transfer", "gene editing"]
_GENETIC_TESTING_TOKENS = {"genotyping", "genotype", "sequencing"}
_GENETIC_TESTING_PHRASES = [
    "genetic testing", "genetic screening", "genetic counseling", "genetic analysis",
    "apoe genotyping", "biomarker analysis",
]


def _looks_like_gene_therapy(normalized_name):
    tokens = set(normalized_name.split())
    if tokens & _GENE_THERAPY_TOKENS:
        return True
    return any(_contains_phrase(normalized_name, phrase) for phrase in _GENE_THERAPY_PHRASES)


def _looks_like_genetic_testing(normalized_name):
    tokens = set(normalized_name.split())
    if tokens & _GENETIC_TESTING_TOKENS:
        return True
    return any(_contains_phrase(normalized_name, phrase) for phrase in _GENETIC_TESTING_PHRASES)


def classify_pipeline_scope(intervention_type, name, classification, verification_status="", overrides=None):
    """
    Take ONE intervention's already-computed classify_intervention()
    output (classification, verification_status) plus its original
    ClinicalTrials.gov type/name, and decide whether it belongs in the
    dashboard's default therapeutic-drug population, an optional-filter
    category, or should never surface as a "drug" at all.

    Returns a dict: pipeline_scope (one of PIPELINE_SCOPE_LABELS),
    scope_reason, scope_method ("curated_override" | "rule_classification"
    | "rule_type" | "rule_keyword"), scope_confidence ("high"|"medium"|"low"),
    manual_review_required (bool), canonical_name_override (str, usually "").

    overrides: dict from load_scope_overrides() (or an equivalent dict
    built directly for tests) — keyed by normalize_text(name). Checked
    FIRST and wins over every rule below, so a curated correction always
    takes precedence over the automatic type/keyword rules.
    """
    itype_upper = (intervention_type or "").strip().upper()
    normalized_name = normalize_text(name)
    candidate_name = normalize_intervention_candidate_name(name)
    candidate_normalized = normalize_text(candidate_name)
    overrides = overrides or {}

    def result(scope, reason, method, confidence, manual_review_required=False, canonical_name_override=""):
        return {
            "pipeline_scope": scope,
            "scope_reason": reason,
            "scope_method": method,
            "scope_confidence": confidence,
            "manual_review_required": manual_review_required,
            "canonical_name_override": canonical_name_override,
        }

    # Step 0: curated override always wins, whatever the automatic rules
    # below would otherwise decide — this is how a genuine investigational
    # program that happens to be typed DIETARY_SUPPLEMENT (or any other
    # default-excluded type) gets promoted, or a wrongly-slipped-through
    # name gets corrected, without editing this file's Python.
    override = overrides.get(candidate_normalized) or overrides.get(normalized_name)
    if override and override.get("pipeline_scope") in PIPELINE_SCOPE_LABELS:
        return result(
            override["pipeline_scope"],
            override.get("reason") or "curated override (data/reference/intervention_scope_overrides.csv)",
            "curated_override", "high",
            manual_review_required=(override["pipeline_scope"] == "Needs Review"),
            canonical_name_override=override.get("canonical_name_override", ""),
        )

    # Step 1: classify_intervention()'s existing non-therapeutic labels
    # map directly — these are ALREADY excluded from developed_drug
    # resolution (resolve_developed_drug only ever picks a
    # sponsor_developed_therapeutic/investigational_therapeutic_unverified
    # winner), so this mapping mainly exists for completeness/auditing
    # (classification_gap_audit.csv covers every intervention, not just
    # therapeutic candidates).
    if classification == "placebo_or_sham":
        return result("Placebo or Comparator", "placebo/sham/vehicle-control arm", "rule_classification", "high")
    if classification == "comparator_or_background_therapy":
        return result(
            "Placebo or Comparator",
            "approved background/comparator therapy, not the trial's studied candidate",
            "rule_classification", "medium",
        )
    if classification == "diagnostic_or_imaging_agent":
        return result("Diagnostic Agent", "matches a curated diagnostic/imaging tracer name", "rule_classification", "high")
    if classification == "device":
        return result("Non-Drug Intervention", "intervention type is DEVICE", "rule_type", "high")
    if classification == "behavioral":
        return result("Non-Drug Intervention", "behavioral/non-drug activity", "rule_type", "high")
    if classification == "procedure":
        # RADIATION is ambiguous by type alone — it can be a diagnostic
        # imaging exam OR a non-drug therapeutic procedure; PROCEDURE
        # itself is always non-drug (an exam/collection, not a product).
        if itype_upper == "RADIATION":
            if _is_diagnostic_tracer(normalized_name) or "scan" in normalized_name.split() or _contains_phrase(normalized_name, "imaging"):
                return result("Diagnostic Agent", "RADIATION-type intervention with imaging/scan wording", "rule_keyword", "medium")
            return result(
                "Non-Drug Intervention",
                "RADIATION-type intervention with no imaging evidence in its name — confirm manually",
                "rule_type", "low", manual_review_required=True,
            )
        return result("Non-Drug Intervention", "clinical procedure or imaging exam", "rule_type", "high")
    if classification == "other":
        return result(
            "Non-Drug Intervention",
            "non-treatment control arm (e.g. no intervention / untreated / usual care), not a product",
            "rule_classification", "high",
        )

    # From here, classification is "uncertain", "sponsor_developed_therapeutic",
    # or "investigational_therapeutic_unverified" — i.e. classify_intervention()
    # did NOT already rule this out as placebo/diagnostic/procedure/device/
    # behavioral/comparator/non-treatment-control. Everything below is the
    # NEW type/keyword gating that step never applied — including for
    # "uncertain" rows: real data shows generic descriptions like "Blood
    # Test" or "CSF Biomarkers" often land as "uncertain" (because
    # classify_intervention()'s multi-candidate-ambiguity rule fires
    # before it ever gets a chance to look at the ct.gov type), so these
    # type/keyword checks must run for "uncertain" too, not just the two
    # therapeutic-candidate labels, or the confirmed leakage would still
    # only get the generic "Needs Review" fallback instead of the more
    # specific Exclude/Diagnostic Agent/Non-Drug Intervention verdict the
    # requirement calls for.
    # Type-agnostic generic-description net runs FIRST, ahead of the
    # per-type gates below — a name that plainly reads as "Blood Test" or
    # "CSF Biomarkers" (regardless of whether ct.gov typed it
    # DIAGNOSTIC_TEST, COMBINATION_PRODUCT, or something else) is the
    # single most specific evidence available and must win over any
    # type-based fallback; same for "CBTi with Application", which needs
    # to resolve as a non-drug supportive activity even when its ct.gov
    # type is COMBINATION_PRODUCT rather than BEHAVIORAL.
    if _is_generic_diagnostic_description(normalized_name):
        return result(
            "Exclude",
            "generic diagnostic/biomarker test description (e.g. Blood Test, CSF Biomarkers) — never a canonical drug name",
            "rule_keyword", "high",
        )
    if _is_generic_non_drug_description(normalized_name):
        return result(
            "Non-Drug Intervention",
            "name describes a non-drug (digital/behavioral) supportive activity, not a pharmaceutical",
            "rule_keyword", "high",
        )

    if itype_upper == "DIETARY_SUPPLEMENT":
        return result(
            "Supportive Treatment" if classification == "sponsor_developed_therapeutic" else "Non-Drug Intervention",
            "dietary supplement / nutraceutical — excluded from the default therapeutic view unless curated as a "
            "genuine investigational program (data/reference/intervention_scope_overrides.csv)",
            "rule_type", "medium",
            manual_review_required=(classification == "sponsor_developed_therapeutic"),
        )

    if itype_upper == "DIAGNOSTIC_TEST":
        return result(
            "Diagnostic Agent", "intervention type is DIAGNOSTIC_TEST", "rule_type", "medium", manual_review_required=True,
        )

    if itype_upper == "COMBINATION_PRODUCT":
        # Best-effort component check for Phase 1A: if the descriptive
        # phrase contains a recognized investigational/known compound or
        # dev-code, the genuine therapeutic component is preserved
        # (kept eligible) rather than discarded outright — but still
        # flagged for manual review, since the display name is still the
        # whole combination phrase, not the isolated component (full
        # component-level name extraction is out of Phase 1A's scope).
        if _looks_like_development_code(candidate_name) or any(
            compound in candidate_normalized for compound in KNOWN_COMPOUND_NAMES
        ):
            return result(
                "Therapeutic Drug",
                "combination product name contains a recognized investigational/known compound — "
                "therapeutic component preserved, but the combined name still needs a human check",
                "rule_keyword", "medium", manual_review_required=True,
            )
        return result(
            "Needs Review",
            "combination product — components must be inspected individually before treating the whole phrase "
            "as one canonical drug name",
            "rule_type", "low", manual_review_required=True,
        )

    if itype_upper == "GENETIC":
        if _looks_like_genetic_testing(normalized_name):
            return result(
                "Exclude", "genetic testing/genotyping/biomarker analysis, not a gene-therapy product", "rule_keyword", "high",
            )
        if _looks_like_gene_therapy(normalized_name):
            return result(
                "Therapeutic Drug", "name indicates an actual gene-therapy product (vector/AAV/gene-therapy wording)",
                "rule_keyword", "medium", manual_review_required=True,
            )
        return result(
            "Needs Review",
            "GENETIC intervention type with no clear evidence of gene therapy vs. genetic testing/genotyping",
            "rule_type", "low", manual_review_required=True,
        )

    if classification == "uncertain":
        return result(
            "Needs Review",
            "classify_intervention() could not resolve a confident therapeutic candidate for this intervention",
            "rule_classification", "low", manual_review_required=True,
        )

    # DRUG / BIOLOGICAL (or blank/unrecognized type) survivors — the
    # existing placebo/imaging/comparator/generic-description exclusions
    # above already ran, so this really is the therapeutic population.
    if classification == "sponsor_developed_therapeutic":
        return result(
            "Therapeutic Drug",
            "confirmed or unsourced official sponsor-pipeline match",
            "rule_classification", "high" if verification_status == "confirmed_official_match" else "medium",
        )
    return result(
        "Therapeutic Drug",
        "investigational therapeutic candidate; drug/biological intervention type with no disqualifying evidence found",
        "rule_classification", "medium", manual_review_required=True,
    )


def build_scope_audit_dataframe(interventions_df):
    """
    outputs/classification_gap_audit.csv source: one row per DISTINCT
    (normalized_name, ClinicalTrials.gov intervention type) combination
    seen anywhere in interventions_df — not one row per raw occurrence —
    so a name repeated across many trials produces one auditable row with
    every contributing NCT ID listed, rather than a wall of duplicates.

    Expects interventions_df to already carry classify_intervention()'s
    columns (normalized_name, original_name, original_type, classification)
    AND classify_pipeline_scope()'s columns (pipeline_scope, scope_reason,
    scope_method, scope_confidence, manual_review_required) — i.e. the
    dataframe build_interventions_dataframe() returns.
    """
    columns = [
        "raw_intervention_name", "normalized_name", "nct_ids",
        "clinicaltrials_intervention_type", "previous_drug_type",
        "new_pipeline_scope", "scope_reason", "scope_method",
        "scope_confidence", "dashboard_eligible", "manual_review_required",
    ]
    if interventions_df.empty:
        return pd.DataFrame(columns=columns)

    def summarize(g):
        return pd.Series({
            "raw_intervention_name": g["original_name"].iloc[0],
            "nct_ids": "; ".join(sorted(set(g["nct_id"].dropna().astype(str)))),
            "previous_drug_type": g["classification"].iloc[0],
            "new_pipeline_scope": g["pipeline_scope"].iloc[0],
            "scope_reason": g["scope_reason"].iloc[0],
            "scope_method": g["scope_method"].iloc[0],
            "scope_confidence": g["scope_confidence"].iloc[0],
            "dashboard_eligible": bool(g["pipeline_scope"].iloc[0] == THERAPEUTIC_SCOPE),
            "manual_review_required": bool(g["manual_review_required"].any()),
        })

    grouped = (
        interventions_df.groupby(["normalized_name", "original_type"], sort=False, dropna=False)
        .apply(summarize, include_groups=False)
        .reset_index()
        .rename(columns={"original_type": "clinicaltrials_intervention_type"})
    )
    return grouped[columns]
