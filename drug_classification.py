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

import hashlib
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
_PROCEDURE_PHRASES = [
    "pet scan", "ct scan", "lumbar puncture", "blood draw", "blood withdrawal", "radiation procedure",
]


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


# --- Step 5.5: expanded non-drug activity/device net -----------------
# Broader than the checks above: catches neuromodulation/electrical-
# stimulation devices, digital/app-based interventions, and additional
# cognitive/educational/exercise/observational-monitoring activities
# that arrive with a non-DEVICE/non-BEHAVIORAL/non-PROCEDURE ct.gov
# type (most commonly "OTHER") and so slip past every check above —
# e.g. "Active transcutaneous vagus nerve stimulation..." (NCT04908358,
# ct.gov type OTHER) was landing as an unverified therapeutic candidate
# before this net existed.
#
# Deliberately consulted ONLY when the candidate name does NOT also
# carry known-compound/dev-code evidence (see _has_known_therapeutic_
# evidence and its call sites below) — protects a real drug that
# happens to be described alongside a procedure/device word in the
# same intervention string, e.g. "etanercept and repeated contrast
# ultrasound" (etanercept is a real, KNOWN_COMPOUND_NAMES-listed
# biologic; the trailing "ultrasound" must not exclude it).
_EXTENDED_NON_DRUG_TOKENS = {
    # neuromodulation / electrical stimulation devices
    "tms", "rtms", "tdcs", "tacs", "dbs", "tvns", "neurostimulation", "neuromodulation",
    # digital / app-based
    "app", "smartphone", "ipad",
    # observational / monitoring
    "questionnaire", "questionnaires", "actigraphy", "sensor",
    # extended cognitive/educational/exercise
    "psychoeducation", "psychoeducational", "prehabilitation", "rehabilitation", "exercises",
    # generic device
    "device",
    # recruitment/informational material or simple equipment name, not a
    # drug — confirmed via real-data audit: "Flyer" (NCT07334392) is the
    # sole surviving "candidate" in a trial whose only other
    # interventions are named exercise programs, itself ct.gov-typed
    # OTHER, with no known-compound/dev-code match of its own.
    "flyer",
}
_EXTENDED_NON_DRUG_PHRASES = [
    # neuromodulation / electrical stimulation
    "transcranial magnetic stimulation", "repetitive transcranial magnetic stimulation",
    "transcranial direct current stimulation", "transcranial alternating current stimulation",
    "transcranial pulse stimulation", "transcranial ultrasound stimulation",
    "vagus nerve stimulation", "vagal nerve stimulation", "deep brain stimulation",
    "electrical stimulation", "auditory stimulation", "acoustic stimulation",
    "sensory stimulation", "photobiomodulation",
    # digital / app-based
    "mobile application", "digital therapeutic", "digital platform", "digital health",
    "mobile health", "digital table",
    # extended cognitive/educational/exercise/therapy-modality (specific
    # named non-drug modalities only — never the bare word "therapy" on
    # its own, which would wrongly exclude real modalities like "Stem
    # Cell Therapy" or "Gene Therapy")
    "cognitive stimulation therapy", "cognitive rehabilitation", "memory training",
    "brain training", "spaced retrieval training", "reminiscence therapy", "music therapy",
    "laughter therapy", "art therapy", "occupational therapy", "physical therapy",
    "speech therapy", "water immersion heat therapy", "heat therapy",
    "patient education", "health education", "dementia education",
    "educational materials", "educational intervention", "educational program",
    "educational session", "psychoeducational intervention", "psychoeducational messages",
    "physical activity", "adapted physical activity", "aerobic exercises",
    "physical training", "strength training", "interval strength training",
    "language adaptation rehabilitation", "neurofeedback training",
    # observational / monitoring
    "data collection", "activity monitoring", "remote monitoring", "video recording",
    "physical exam", "monitoring system", "sensor system", "ambient sensor",
    "supportive care",
]


def _is_extended_non_drug_activity(normalized_name):
    tokens = set(normalized_name.split())
    if tokens & _EXTENDED_NON_DRUG_TOKENS:
        return True
    return any(_contains_phrase(normalized_name, phrase) for phrase in _EXTENDED_NON_DRUG_PHRASES)


# Shared with classify_intervention()'s Step 5.5 return AND
# build_resolved_drugs_exclusion_audit_dataframe()'s filter below — a
# single source of truth for "this record was caught by the expanded
# non-drug net specifically" rather than duplicating the literal string.
EXTENDED_NON_DRUG_REASON = (
    "name indicates a neuromodulation/digital/educational/exercise/observational-monitoring "
    "non-drug activity or device"
)


def _has_known_therapeutic_evidence(candidate_normalized):
    """True if candidate_normalized contains a recognized investigational
    compound name anywhere in it — the override guard for
    _is_extended_non_drug_activity (see its docstring)."""
    return bool(candidate_normalized) and any(
        compound in candidate_normalized for compound in KNOWN_COMPOUND_NAMES
    )


def _passes_therapeutic_gate(itype, name):
    """
    True if this single intervention survives steps 1-5.5 (i.e. it is
    not placebo, not a non-therapeutic control arm, not a diagnostic
    tracer, not a procedure/radiation exam, not a device, not
    behavioral, and not an expanded-net non-drug activity/device) and
    is therefore still a candidate therapeutic. Used both for the
    intervention being classified and for scanning its siblings — this
    is also what keeps "No Intervention"/"Untreated" from ever counting
    as a plausible therapeutic sibling.
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
    candidate_normalized = normalize_text(normalize_intervention_candidate_name(name))
    if not _has_known_therapeutic_evidence(candidate_normalized) and _is_extended_non_drug_activity(normalized):
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
    "etanercept", "sargramostim", "minocycline", "al002", "al003", "aln-app", "pbr28",
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

    # Step 5.5: expanded non-drug activity/device net (neuromodulation,
    # digital/app, extended cognitive/educational/exercise/observational-
    # monitoring) — see _is_extended_non_drug_activity's docstring.
    # Skipped when the name also carries known-compound evidence, so a
    # real drug incidentally described alongside a device/procedure word
    # (e.g. "etanercept and repeated contrast ultrasound") is preserved.
    if (
        not _has_known_therapeutic_evidence(candidate_normalized)
        and _is_extended_non_drug_activity(normalized_name)
    ):
        return result(
            "behavioral",
            EXTENDED_NON_DRUG_REASON,
            "high",
        )

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
    header names). "Brief Summary" and "Study Design" are read under
    their ORIGINAL raw trials.csv column names — column_map never
    renames them, so they pass through unchanged — and are used only by
    classify_pipeline_scope()'s isotope-labeled-tracer check; missing
    either column degrades to "" (that check simply never fires, same
    as any other optional evidence in this pipeline).

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
        brief_summary = trial.get("Brief Summary", "")
        study_design = trial.get("Study Design", "")

        interventions = parse_interventions(raw_interventions)
        for i, interv in enumerate(interventions):
            siblings = interventions[:i] + interventions[i + 1:]
            classified = classify_intervention(
                interv["type"], interv["name"], sponsor, siblings, pipeline_records
            )
            scoped = classify_pipeline_scope(
                interv["type"], interv["name"], classified["classification"],
                classified["verification_status"], overrides=scope_overrides,
                brief_summary=brief_summary, study_title=title, study_design=study_design,
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
                scope_confidence="low", manual_review_required=None, diagnostic_subtype=""):
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
            "diagnostic_subtype": diagnostic_subtype,
        }

    def scope_fields(r):
        return {
            "pipeline_scope": r.get("pipeline_scope", "Needs Review"),
            "scope_reason": r.get("scope_reason", ""),
            "scope_method": r.get("scope_method", "rule_classification"),
            "scope_confidence": r.get("scope_confidence", "low"),
            "manual_review_required": r.get("manual_review_required", False),
            "diagnostic_subtype": r.get("diagnostic_subtype", ""),
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
    pipeline_scope is one of RESOLVED_DRUGS_DF_ELIGIBLE_SCOPES
    (currently just "Therapeutic Drug" — a record only enters
    resolved_drugs_df at all if its primary investigational
    intervention resolved to a real drug/biologic) contribute a row.
    Everything else (placebo/diagnostic-tracer/procedure/device/
    behavioral/comparator/uncertain/no_therapeutic_candidate trials,
    trials with unresolved multiple candidates, and anything whose
    scope resolved to Diagnostic Agent/Non-Drug Intervention/Supportive
    Treatment/Needs Review/Exclude/Placebo or Comparator) produces NO
    drug row here — see build_unresolved_trials_dataframe() for where
    the drug-identity-ambiguous ones go instead, so nothing is silently
    dropped from the dataset as a WHOLE (pipeline_annotated.csv /
    pipeline_interventions.csv / outputs/non_drug_exclusion_audit.csv
    still carry every trial/intervention regardless).

    resolved_drugs_df is the ONE drug-level source of truth every
    dashboard component reads (main table, KPI counts, filters, charts,
    competitive-attention scoring, Upcoming Milestones — everything).
    Since it now only ever contains Therapeutic Drug scope rows, the
    dashboard's "reveal non-therapeutic records" toggle still exists in
    the UI but has nothing left to reveal — that's an intentional
    consequence of this narrowing, not a bug.

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
        "diagnostic_subtype",
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
            diagnostic_subtype = ""
        elif scopes_seen:
            pipeline_scope = scopes_seen.pop()
            scope_row = g[g["pipeline_scope"] == pipeline_scope].iloc[0]
            scope_reason = scope_row.get("scope_reason", "")
            scope_method = scope_row.get("scope_method", "")
            scope_confidence = scope_row.get("scope_confidence", "")
            diagnostic_subtype = scope_row.get("diagnostic_subtype", "") if pipeline_scope == "Diagnostic Agent" else ""
        else:
            pipeline_scope, scope_reason, scope_method, scope_confidence = "Needs Review", "no scope information available", "rule_classification", "low"
            diagnostic_subtype = ""

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
            "diagnostic_subtype": diagnostic_subtype,
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


# Ascending clinical-progression order for the competitive-matrix chart's
# x-axis (development maturity). Deliberately narrower than PHASE_ORDER --
# "NA"/unresolved-phase drugs have no meaningful position on a maturity
# axis, so build_relevance_matrix() drops them rather than placing them
# at some arbitrary spot.
MATURITY_PHASE_ORDER = ["Early Phase 1", "Phase 1", "Phase 1/Phase 2", "Phase 2", "Phase 2/Phase 3", "Phase 3", "Phase 4"]


def _deterministic_unit_jitter(seed_text):
    """
    Deterministic pseudo-random float in [-1, 1] derived from seed_text
    via a stable hash — NOT Python's built-in hash(), which is
    randomized per-process (PYTHONHASHSEED) and would give a different
    jitter on every dashboard regeneration. The same seed always
    produces the same output, so the same drug lands in the same
    jittered spot on the competitive-matrix chart every time.
    """
    digest = hashlib.md5(seed_text.encode("utf-8")).hexdigest()
    as_int = int(digest[:8], 16)
    return (as_int / 0xFFFFFFFF) * 2 - 1


def build_relevance_matrix(resolved_drugs_df, top_n=40):
    """
    Top-N competitor drugs (by aribio_relevance_score, AR1001 itself
    excluded and rows with an unresolved phase dropped), with a numeric
    maturity_x position and small deterministic jitter_x/jitter_y
    offsets appended — the exact data the "AR1001 Competitive
    Landscape" chart plots as one bubble per drug (x=maturity_x/
    development stage, y=aribio_relevance_score, color=target).
    Extracted as a pure function for the same reason as
    build_target_phase_counts(): unit-testable independent of the
    Plotly figure it feeds.

    The jitter columns are visual-only nudges — display code adds them
    to maturity_x/aribio_relevance_score when plotting so overlapping
    same-phase/same-score drugs are distinguishable, but the true
    phase_reached/aribio_relevance_score values returned here are never
    altered, and hover text should always read from those, not the
    jittered plot position.

    AR1001 itself (is_aribio == True) is excluded — its relevance score
    is scored against itself, which isn't a meaningful point among its
    own competitors; the chart shows it separately as a fixed reference
    marker instead.
    """
    columns = ["display_name", "sponsor", "phase_reached", "target", "modality",
               "aribio_relevance_score", "aribio_relevance_reasons", "maturity_x",
               "jitter_x", "jitter_y"]
    if resolved_drugs_df is None or resolved_drugs_df.empty:
        return pd.DataFrame(columns=columns)
    competitors = resolved_drugs_df[
        (~resolved_drugs_df["is_aribio"]) & (resolved_drugs_df["phase_reached"].isin(MATURITY_PHASE_ORDER))
    ].copy()
    if "aribio_relevance_reasons" not in competitors.columns:
        competitors["aribio_relevance_reasons"] = ""
    maturity_index = {phase: i for i, phase in enumerate(MATURITY_PHASE_ORDER)}
    competitors["maturity_x"] = competitors["phase_reached"].map(maturity_index)
    # magnitudes kept small relative to the axes' own scale (maturity
    # steps are 1 apart, relevance runs 0-100) -- enough to separate
    # overlapping dots without visually implying a different phase or
    # a materially different score
    competitors["jitter_x"] = competitors["display_name"].apply(
        lambda name: _deterministic_unit_jitter(f"{name}|x") * 0.28)
    competitors["jitter_y"] = competitors["display_name"].apply(
        lambda name: _deterministic_unit_jitter(f"{name}|y") * 3.5)
    competitors = competitors.sort_values(["aribio_relevance_score", "display_name"], ascending=[False, True])
    return competitors[columns].head(top_n).reset_index(drop=True)


def summarize_relevance_scores(scores):
    """
    Distribution stats for a collection of AR1001 relevance scores
    (e.g. the Top-40 competitor set) -- min/max/median/unique count/
    per-score frequency. Exists so the deterministic point-sum
    scoring's discreteness (how many competitors tie on the same
    score) can be REPORTED rather than silently papered over by the
    chart's visual jitter.
    """
    clean = pd.Series(scores).dropna()
    if clean.empty:
        return {"min": None, "max": None, "median": None, "n_unique": 0, "n_total": 0, "score_counts": {}}
    counts = clean.value_counts().sort_index(ascending=False)
    return {
        "min": int(clean.min()),
        "max": int(clean.max()),
        "median": float(clean.median()),
        "n_unique": int(clean.nunique()),
        "n_total": int(len(clean)),
        "score_counts": {int(score): int(count) for score, count in counts.items()},
    }


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

# resolved_drugs_df is the ONE source every dashboard component reads
# (main table, KPI counts, filters, charts, competitive-attention
# scoring, Upcoming Milestones — everything). A record enters it ONLY
# if its primary investigational intervention resolved to "Therapeutic
# Drug" scope, i.e. a real DRUG/BIOLOGICAL therapeutic candidate.
#
# Previously this also admitted Diagnostic Agent / Non-Drug
# Intervention / Supportive Treatment / Needs Review, specifically so
# the dashboard's optional "reveal non-therapeutic records" table
# filter had real data to show. That toggle still exists in the UI
# (unchanged, per requirement) but now has nothing extra to reveal —
# every non-"Therapeutic Drug" record is excluded at the source instead
# of merely hidden client-side. This is a deliberate, explicit
# narrowing: excluded records remain fully auditable in
# pipeline_annotated.csv / pipeline_interventions.csv / pipeline_
# unresolved_trials.csv / outputs/non_drug_exclusion_audit.csv — they
# just never become a resolved_drugs_df ("drug") row.
RESOLVED_DRUGS_DF_ELIGIBLE_SCOPES = [
    "Therapeutic Drug",
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


# --- isotope-labeled PET/SPECT tracer leakage (uncurated names) -------
#
# _is_diagnostic_tracer() above only catches NAMED tracers on a curated
# list — it can never be complete, since sponsors/investigators mint new
# alphanumeric tracer codes (e.g. "[18F]MNI-1126", "11C-JNJ-63779586")
# constantly, and a static list can't anticipate every one. This is a
# SEPARATE, narrower net: an intervention whose own NAME carries
# radiochemistry isotope-label notation (18F/F-18/11C/C-11/123I/I-123
# immediately preceding a compound code, bracketed or not) is a
# CANDIDATE — never
# reclassified on that basis alone (per requirement: don't assume
# diagnostic solely from an unusual/isotope-looking name — a genuine
# therapeutic radiopharmaceutical, or a compound code that merely
# CONTAINS "11c"/"18f" as a coincidental substring — e.g. "SSR180711C" —
# must not be swept in). The isotope-name match only fires this rule
# when COMBINED with real study-level evidence: ct.gov's own Primary
# Purpose field reading DIAGNOSTIC, or explicit PET/SPECT/radioligand/
# imaging wording in the trial's own title or brief summary — i.e. the
# same "intervention type, description, study purpose/title" evidence
# classes named in the audit requirement, not name-guessing.
#
# The lookbehind requires the isotope token to start at a genuine word
# boundary (string start, or right after a non-alphanumeric character —
# space/hyphen/bracket/paren/slash), which is exactly what excludes
# "SSR180711C" (its "11C" is embedded mid-code, immediately preceded by
# the digit "0", not a boundary) while still matching "18FAV45" (no
# lookahead requirement, so an isotope prefix fused directly onto a
# compound code like "18F"+"AV45" still matches).
_ISOTOPE_LABELED_NAME_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:\[\s*)?(?:18[\s-]?F|F[\s-]?18|11[\s-]?C|C[\s-]?11|123[\s-]?I|I[\s-]?123)",
    re.IGNORECASE,
)


def _is_isotope_labeled_name(raw_name):
    return bool(_ISOTOPE_LABELED_NAME_RE.search(str(raw_name or "")))


# Study-level (not name-level) evidence that a trial's purpose is
# diagnostic/imaging — checked against the trial's Brief Summary and
# Study Title (both free text) plus its Study Design's "Primary
# Purpose" field (ct.gov's own structured DESIGNATION, stored as plain
# text in the "Study Design" column — see ctgov_normalize.py's
# _format_study_design(), which always ends that string with
# "Primary Purpose: <VALUE>"). ct.gov's Primary Purpose is a strong
# signal WHEN present but is unreliable on its own — real tracer-
# validation trials in this dataset are inconsistently tagged
# TREATMENT/BASIC_SCIENCE/OTHER/blank instead of DIAGNOSTIC — so
# explicit imaging/tracer wording in the title or summary is checked
# as an equally-sufficient alternative, not a fallback of last resort.
_DIAGNOSTIC_STUDY_CONTEXT_PHRASES = [
    "pet imaging", "pet tracer", "pet ligand", "pet radioligand", "pet study",
    "pet scan", "pet/ct", "spect imaging", "spect tracer", "spect ligand",
    "radioligand", "radiotracer", "radiopharmaceutical", "imaging marker",
    "imaging agent", "positron emission tomography", "biodistribution and dosimetry",
    "biodistribution and radiation dosimetry",
]
# "biodistribution" alone (not just the longer phrases above) is checked
# as a standalone token too — early-phase ("Phase 0") tracer-validation
# studies are routinely titled/summarized around biodistribution and
# absorbed-dose estimation without ever using the word "PET"/"imaging"
# explicitly. Safe as a bare-token match ONLY because this function is
# exclusively reached from a call site that already gated on the
# intervention's own name carrying isotope-label notation — a
# non-isotope-named therapeutic's unrelated PK/biodistribution study
# never reaches this check at all.
_DIAGNOSTIC_STUDY_CONTEXT_TOKENS = {"biodistribution"}


def _has_diagnostic_study_context(brief_summary, study_title, study_design):
    design_text = str(study_design or "")
    purpose_match = re.search(r"Primary Purpose:\s*([A-Za-z_]+)", design_text)
    if purpose_match and purpose_match.group(1).strip().upper() == "DIAGNOSTIC":
        return True
    combined = normalize_text(str(brief_summary or "") + " " + str(study_title or ""))
    if set(combined.split()) & _DIAGNOSTIC_STUDY_CONTEXT_TOKENS:
        return True
    return any(_contains_phrase(combined, phrase) for phrase in _DIAGNOSTIC_STUDY_CONTEXT_PHRASES)


# diagnostic_subtype: best-effort, evidence-based categorization for
# CONFIRMED diagnostic agents only (never used to decide IF something
# is diagnostic — only to label what KIND, once already confirmed).
# Falls back to "" when no target-pathway keyword is found (a real,
# honest "unknown subtype" rather than a forced guess) — the caller
# still records "PET tracer" for anything confirmed diagnostic via the
# isotope-name path even without a specific pathway match, since that
# path only ever fires on PET/SPECT-imaging evidence.
# Each set also includes the well-known BRAND/compound names for that
# pathway (mirrors drug_classification.py's own _DIAGNOSTIC_TOKENS
# categorization) — a name like "florbetapir" never literally contains
# the word "amyloid", so the name-first priority in determine_
# diagnostic_subtype() below would otherwise have no way to resolve it
# without falling back to the (possibly sibling-contaminated) shared
# trial summary text.
_AMYLOID_SUBTYPE_TOKENS = {
    "amyloid", "florbetapir", "amyvid", "av45", "florbetaben", "neuraceq",
    "flutemetamol", "vizamyl", "pib", "nav4694", "azd4694",
}
_AMYLOID_SUBTYPE_PHRASES = ["beta amyloid", "a beta", "amyloid imaging", "amyloid deposition", "pittsburgh compound b"]
_TAU_SUBTYPE_TOKENS = {
    "tau", "tauopathy", "tauopathies", "flortaucipir", "tauvid", "av1451", "t807",
    "mk6240", "pi2620", "ro948", "ro6958948", "gtp1", "pbb3", "apn1607",
    "thk5317", "thk5351", "thk5117",
}
_TAU_SUBTYPE_PHRASES = ["tau imaging", "tau protein", "neurofibrillary tangle"]
_TSPO_SUBTYPE_TOKENS = {
    "tspo", "pbr", "pbr28", "pbr06", "pbr111", "microglia", "microglial",
    "neuroinflammation", "dpa714", "feppa", "er176",
}
_TSPO_SUBTYPE_PHRASES = ["translocator protein", "microglial activation"]


def _match_subtype(text):
    tokens = set(text.split())
    if tokens & _TAU_SUBTYPE_TOKENS or any(_contains_phrase(text, p) for p in _TAU_SUBTYPE_PHRASES):
        return "Tau PET tracer"
    if tokens & _TSPO_SUBTYPE_TOKENS or any(_contains_phrase(text, p) for p in _TSPO_SUBTYPE_PHRASES):
        return "TSPO/neuroinflammation PET tracer"
    if tokens & _AMYLOID_SUBTYPE_TOKENS or any(_contains_phrase(text, p) for p in _AMYLOID_SUBTYPE_PHRASES):
        return "Amyloid PET tracer"
    return ""


def determine_diagnostic_subtype(name, brief_summary, study_title):
    # The intervention's OWN name is checked first and, if it gives any
    # pathway signal, wins outright — a trial can carry sibling
    # interventions for different pathways (e.g. a tau tracer dosed
    # alongside an amyloid tracer in the same protocol), and both would
    # share the same trial-level Brief Summary/Title text, which could
    # otherwise mislabel one tracer's subtype using wording that
    # actually describes its sibling. The shared summary/title text is
    # only consulted as a fallback for names that give no signal at all
    # (common for opaque internal codes like "MNI-1126"/"W372").
    name_match = _match_subtype(normalize_text(str(name or "")))
    if name_match:
        return name_match
    combined = normalize_text(str(name or "") + " " + str(brief_summary or "") + " " + str(study_title or ""))
    return _match_subtype(combined) or "PET tracer"


# ============================================================
# NON-THERAPEUTIC DRUG PURPOSE — a REAL drug/biologic (survives every
# check above) can still not belong in the therapeutic-drug population
# if the TRIAL is using it for a non-treatment purpose: as a
# pharmacological challenge/probe agent, a diagnostic-tool-development
# substance, a contrast/imaging agent (non-isotope-labeled — the
# isotope-name path above already covers PET/SPECT tracers), or a
# deprescribing/withdrawal target.
#
# Real-data audit finding: "safety, tolerability, [and] pharmacokinetics"
# language is NOT a reliable protective signal on its own — genuine
# diagnostic-agent Phase 1 studies (e.g. a novel imaging contrast agent)
# routinely discuss "safety" too. What actually separates the two
# classes in this dataset is whether the text describes the SUBSTANCE's
# own role as diagnostic/imaging/challenge/probe (e.g. "diagnostic
# potential of X", "developed for use in contrast-enabled imaging",
# "PET imaging with X ... challenge") versus ordinary investigational-
# drug development framing (safety/PK/dose-escalation/immune response
# in AD patients, with no diagnostic-role language at all). So this
# check requires ct.gov's own Primary Purpose reading DIAGNOSTIC AND a
# SPECIFIC phrase describing the substance's diagnostic/imaging/
# challenge/probe role — never a bare "safety" or "biomarker" mention,
# which are far too common in genuine therapeutic Phase 1 trials to be
# useful signals (a real AD candidate's Phase 1 trial routinely reports
# effects on a CSF/plasma biomarker as an OUTCOME, without the drug
# itself being a diagnostic tool).
#
# Verified against real data during this fix: correctly excludes
# Pramlintide ("challenge test" + Primary Purpose DIAGNOSTIC),
# Scopolamine ("develop a diagnostic tool for AD"), BAY1006578
# ("diagnostic potential of ... radiation dosimetry"), Aftobetin-HCl
# ("fluorescence detection"), DSPE-DOTA-Gd Liposomal ("contrast-enabled
# ... imaging"), and LPS ("Lipopolysaccharide Challenge") — while
# correctly PRESERVING LY450139 dihydrate/semagacestat, TC-5619,
# AMDX-2011P, and V950 (all genuine Phase 1 safety/PK/immunogenicity
# studies of real investigational AD candidates, none of which mention
# any diagnostic/imaging/challenge role for the drug itself).
_DIAGNOSTIC_CHALLENGE_PROBE_NAME_PHRASES = [
    "challenge test", "pharmacological challenge", "provocation test",
]
_DIAGNOSTIC_CHALLENGE_PROBE_CONTEXT_PHRASES = [
    "challenge test", "pharmacological challenge", "provocation test", "challenge for the study",
    "diagnostic potential", "diagnostic tool", "diagnostic biomarker", "novel diagnostic",
    "imaging diagnostic", "blood-based test",
    "contrast-enabled", "contrast agent", "for use in contrast",
    "fluorescence detection", "radiation dosimetry",
    "molecular probe", "research probe", "diagnostic probe", "imaging probe",
]
# Bare-word fallback for the context check — deliberately only ever
# consulted AFTER the Primary Purpose: DIAGNOSTIC gate already passed
# (see the caller), which is what makes a single distinctive word safe
# here: a genuinely diagnostic-purpose trial mentioning "challenge" or
# "probe" anywhere in its title/summary is real signal, not noise, in a
# way it would NOT be if checked ungated against every trial.
_DIAGNOSTIC_CHALLENGE_PROBE_CONTEXT_TOKENS = {"challenge", "probe"}


def _is_diagnostic_challenge_or_probe_purpose(name, brief_summary, study_title, study_design):
    normalized_name = normalize_text(name)
    # The intervention's own name can be unambiguous on its own (e.g.
    # "Pramlintide challenge test") without needing trial-context
    # corroboration — "challenge test"/"pharmacological challenge"/
    # "provocation test" as part of a drug's OWN listed name is not
    # something a genuine therapeutic candidate's name would ever say.
    if any(_contains_phrase(normalized_name, phrase) for phrase in _DIAGNOSTIC_CHALLENGE_PROBE_NAME_PHRASES):
        return True

    design_text = str(study_design or "")
    purpose_match = re.search(r"Primary Purpose:\s*([A-Za-z_]+)", design_text)
    is_diagnostic_purpose = bool(purpose_match and purpose_match.group(1).strip().upper() == "DIAGNOSTIC")
    if not is_diagnostic_purpose:
        return False

    combined = normalize_text(str(name or "") + " " + str(brief_summary or "") + " " + str(study_title or ""))
    if set(combined.split()) & _DIAGNOSTIC_CHALLENGE_PROBE_CONTEXT_TOKENS:
        return True
    return any(_contains_phrase(combined, phrase) for phrase in _DIAGNOSTIC_CHALLENGE_PROBE_CONTEXT_PHRASES)


# --- deprescribing/medication withdrawal + procedural support -------
# Name-level only (no trial-context lookup needed) — these read
# unambiguously from the intervention's own name in every real example
# found in this dataset, e.g. "Deprescribing of target anticholinergics"
# (NCT04270474, ct.gov-typed OTHER — not even a DRUG-typed row; the
# WITHDRAWAL PROCESS is the intervention, not an administered drug).
_DEPRESCRIBING_OR_PROCEDURAL_SUPPORT_TOKENS = {"deprescribing"}
_DEPRESCRIBING_OR_PROCEDURAL_SUPPORT_PHRASES = [
    "medication withdrawal", "drug withdrawal", "drug discontinuation",
    "medication discontinuation", "dose tapering", "deprescribing intervention",
    "procedural sedation", "conscious sedation", "sedation for imaging", "anesthesia for imaging",
]


def _is_deprescribing_or_procedural_support(normalized_name):
    tokens = set(normalized_name.split())
    if tokens & _DEPRESCRIBING_OR_PROCEDURAL_SUPPORT_TOKENS:
        return True
    return any(_contains_phrase(normalized_name, phrase) for phrase in _DEPRESCRIBING_OR_PROCEDURAL_SUPPORT_PHRASES)


def classify_pipeline_scope(intervention_type, name, classification, verification_status="", overrides=None,
                             brief_summary="", study_title="", study_design=""):
    """
    Take ONE intervention's already-computed classify_intervention()
    output (classification, verification_status) plus its original
    ClinicalTrials.gov type/name, and decide whether it belongs in the
    dashboard's default therapeutic-drug population, an optional-filter
    category, or should never surface as a "drug" at all.

    Returns a dict: pipeline_scope (one of PIPELINE_SCOPE_LABELS),
    scope_reason, scope_method ("curated_override" | "rule_classification"
    | "rule_type" | "rule_keyword"), scope_confidence ("high"|"medium"|"low"),
    manual_review_required (bool), canonical_name_override (str, usually ""),
    diagnostic_subtype (str, "" unless pipeline_scope == "Diagnostic Agent"
    and evidence supports a specific PET-tracer category — see
    determine_diagnostic_subtype()).

    overrides: dict from load_scope_overrides() (or an equivalent dict
    built directly for tests) — keyed by normalize_text(name). Checked
    FIRST and wins over every rule below, so a curated correction always
    takes precedence over the automatic type/keyword rules.

    brief_summary/study_title/study_design: the intervention's trial's
    own text fields, used ONLY by the isotope-labeled-tracer check below
    (see _has_diagnostic_study_context) — optional, default "" so every
    existing call site/test keeps working unchanged.
    """
    itype_upper = (intervention_type or "").strip().upper()
    normalized_name = normalize_text(name)
    candidate_name = normalize_intervention_candidate_name(name)
    candidate_normalized = normalize_text(candidate_name)
    overrides = overrides or {}

    def result(scope, reason, method, confidence, manual_review_required=False, canonical_name_override="",
               diagnostic_subtype=""):
        return {
            "pipeline_scope": scope,
            "scope_reason": reason,
            "scope_method": method,
            "scope_confidence": confidence,
            "manual_review_required": manual_review_required,
            "canonical_name_override": canonical_name_override,
            "diagnostic_subtype": diagnostic_subtype,
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
        return result(
            "Diagnostic Agent", "matches a curated diagnostic/imaging tracer name", "rule_classification", "high",
            diagnostic_subtype=determine_diagnostic_subtype(name, brief_summary, study_title),
        )
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
                return result(
                    "Diagnostic Agent", "RADIATION-type intervention with imaging/scan wording", "rule_keyword", "medium",
                    diagnostic_subtype=determine_diagnostic_subtype(name, brief_summary, study_title),
                )
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

    # Uncurated isotope-labeled PET/SPECT tracer leakage (see the block
    # above _is_isotope_labeled_name for the full rationale). Runs after
    # every itype-specific gate above and BEFORE the "uncertain"/
    # therapeutic-fallback resolution below, since those are exactly the
    # two outcomes this was silently falling into. Never applied to
    # sponsor_developed_therapeutic — a confirmed official-pipeline match
    # is real, verified evidence of a genuine therapeutic asset and is
    # never second-guessed by a name/summary heuristic.
    if (
        classification != "sponsor_developed_therapeutic"
        and _is_isotope_labeled_name(name)
        and _has_diagnostic_study_context(brief_summary, study_title, study_design)
    ):
        return result(
            "Diagnostic Agent",
            "isotope-labeled tracer name (18F/11C-style notation) combined with diagnostic study context "
            "(ClinicalTrials.gov Primary Purpose: DIAGNOSTIC and/or explicit PET/SPECT/radioligand/imaging "
            "wording in the trial's title or brief summary)",
            "rule_keyword", "high",
            diagnostic_subtype=determine_diagnostic_subtype(name, brief_summary, study_title),
        )

    # A real drug/biologic used for a non-treatment purpose within THIS
    # trial (pharmacological challenge/testing, diagnostic-tool
    # development, contrast/imaging agent use, experimental probe) —
    # see _is_diagnostic_challenge_or_probe_purpose's docstring for the
    # evidence bar and the real examples that calibrated it. Same
    # sponsor_developed_therapeutic protection as the isotope check above.
    if (
        classification != "sponsor_developed_therapeutic"
        and _is_diagnostic_challenge_or_probe_purpose(name, brief_summary, study_title, study_design)
    ):
        return result(
            "Diagnostic Agent",
            "drug/biologic used as a pharmacological challenge/probe agent or for diagnostic-tool/"
            "contrast-imaging development in this trial, not as an investigational AD treatment",
            "rule_keyword", "high",
            diagnostic_subtype=determine_diagnostic_subtype(name, brief_summary, study_title),
        )

    # Deprescribing/medication-withdrawal interventions and drugs whose
    # role in this trial is procedural support (e.g. sedation to enable
    # an imaging exam) — not an administered investigational treatment
    # candidate. Name-level only; see _is_deprescribing_or_procedural_
    # support's docstring.
    if (
        classification != "sponsor_developed_therapeutic"
        and _is_deprescribing_or_procedural_support(normalized_name)
    ):
        return result(
            "Non-Drug Intervention",
            "name indicates a deprescribing/medication-withdrawal intervention or a drug used only for "
            "procedural support, not an investigational AD treatment",
            "rule_keyword", "high",
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


def build_diagnostic_agent_audit_dataframe(interventions_df):
    """
    outputs/diagnostic_agent_audit.csv source: every intervention name
    that is EITHER already classified pipeline_scope == "Diagnostic
    Agent" (confirmed, via any of the diagnostic detection paths in
    classify_pipeline_scope() — curated name list, isotope-labeled name
    + study context, RADIATION-type imaging wording, or DIAGNOSTIC_TEST
    type) OR carries isotope-labeled name notation (18F/11C-style) but
    was NOT reclassified because supporting study context couldn't be
    confirmed — the "uncertain, needs manual review" bucket. One row
    per distinct normalized_name, so a name repeated across many trials
    produces one auditable row with every contributing NCT ID.

    previously_leaked_into_therapeutic_dashboard is True exactly when
    scope_method/scope_reason indicate the NEW isotope-name-plus-context
    rule is what classified this row — i.e. before that rule existed,
    this intervention fell through to the generic "Therapeutic Drug"
    fallback (see classify_pipeline_scope()'s final fallback) and would
    have counted toward Therapeutic Drug KPIs/heatmaps/charts.
    Pre-existing correctly-classified diagnostic agents (curated list,
    RADIATION-imaging, DIAGNOSTIC_TEST type) were already excluded
    before this phase, so that flag is False for them.

    Expects interventions_df to carry classify_intervention()'s and
    classify_pipeline_scope()'s columns, i.e. the dataframe
    build_interventions_dataframe() returns.
    """
    columns = [
        "name", "nct_ids", "trial_count", "evidence_used",
        "previous_pipeline_scope", "new_pipeline_scope", "diagnostic_subtype",
        "confidence", "previously_leaked_into_therapeutic_dashboard",
    ]
    if interventions_df.empty:
        return pd.DataFrame(columns=columns)

    df = interventions_df.copy()
    df["_isotope_named"] = df["original_name"].apply(_is_isotope_labeled_name)
    is_diagnostic = df["pipeline_scope"] == "Diagnostic Agent"
    is_uncertain_candidate = df["_isotope_named"] & ~is_diagnostic
    audit_rows = df[is_diagnostic | is_uncertain_candidate]
    if audit_rows.empty:
        return pd.DataFrame(columns=columns)

    def summarize(g):
        row = g.iloc[0]
        current_scope = row["pipeline_scope"]
        newly_caught = row["scope_method"] == "rule_keyword" and "isotope-labeled" in str(row["scope_reason"])

        if current_scope == "Diagnostic Agent":
            previous_scope = "Therapeutic Drug" if newly_caught else "Diagnostic Agent"
            evidence_used = row["scope_reason"]
            confidence = row["scope_confidence"]
        else:
            # isotope-named but not (yet) reclassified — flagged for a
            # human to review, scope is left exactly as classify_
            # pipeline_scope() already decided it (no silent change).
            previous_scope = current_scope
            evidence_used = (
                "isotope-labeled name pattern (18F/11C-style notation) present, but no confirmed "
                "diagnostic study context (ClinicalTrials.gov Primary Purpose, or PET/imaging/"
                "radioligand wording in title or brief summary) was found — needs manual review"
            )
            confidence = "low"

        return pd.Series({
            "name": row["original_name"],
            "nct_ids": "; ".join(sorted(set(g["nct_id"].dropna().astype(str)))),
            "trial_count": g["nct_id"].nunique(),
            "evidence_used": evidence_used,
            "previous_pipeline_scope": previous_scope,
            "new_pipeline_scope": current_scope,
            "diagnostic_subtype": row.get("diagnostic_subtype", "") or "",
            "confidence": confidence,
            "previously_leaked_into_therapeutic_dashboard": bool(newly_caught or (current_scope == "Therapeutic Drug")),
        })

    result = (
        audit_rows.groupby("normalized_name", sort=False)
        .apply(summarize, include_groups=False)
        .reset_index(drop=True)
    )
    return result.sort_values(
        ["previously_leaked_into_therapeutic_dashboard", "name"], ascending=[False, True]
    ).reset_index(drop=True)[columns]


def build_resolved_drugs_exclusion_audit_dataframe(interventions_df):
    """
    outputs/non_drug_exclusion_audit.csv source: every intervention name
    excluded from resolved_drugs_df that's worth a human's attention —
    the union of two populations:

      1. Interventions classify_intervention() marked
         sponsor_developed_therapeutic/investigational_therapeutic_
         unverified (i.e. eligible to be resolved as a trial's
         developed_drug) but whose pipeline_scope ended up NOT
         "Therapeutic Drug" — diagnostic agents, dietary supplements,
         DIAGNOSTIC_TEST-typed records, generic descriptions, etc. This
         is the population that used to leak into resolved_drugs_df
         under the old, broader RESOLVED_DRUGS_DF_ELIGIBLE_SCOPES.

      2. Interventions caught DIRECTLY by the expanded non-drug net
         (_is_extended_non_drug_activity — neuromodulation/electrical
         stimulation, digital/apps, extended cognitive/educational/
         exercise/observational-monitoring), identified by
         reason == EXTENDED_NON_DRUG_REASON. These are classified
         "behavioral" from the start (never investigational_
         therapeutic_unverified), so population 1's filter alone
         would miss them entirely — e.g. a trial whose only two arms
         are "Active transcutaneous vagus nerve stimulation..." and
         "Sham transcutaneous vagus nerve stimulation..." now resolves
         to NO developed_drug candidate at all (both arms are
         correctly excluded), so it never reaches population 1's
         classification filter, but is still exactly the kind of
         record this audit exists to report.

    Deliberately excludes unambiguous non-candidates (placebo_or_sham,
    "other"/non-treatment-control arms) that were never going to be
    mistaken for a drug — auditing every placebo arm in the dataset
    would bury the genuinely useful rows in noise.

    One row per distinct normalized_name, with every contributing NCT
    ID. exclusion_reason prefers classify_intervention()'s own reason
    for population 2 (more specific than the generic "behavioral/
    non-drug activity" scope_reason it maps to) and scope_reason
    otherwise.
    """
    columns = [
        "name", "clinicaltrials_intervention_type", "nct_ids", "trial_count",
        "classification", "pipeline_scope", "exclusion_reason", "confidence",
    ]
    if interventions_df.empty:
        return pd.DataFrame(columns=columns)

    was_therapeutic_candidate = interventions_df["classification"].isin(
        ["sponsor_developed_therapeutic", "investigational_therapeutic_unverified"]
    ) & (interventions_df["pipeline_scope"] != THERAPEUTIC_SCOPE)
    caught_by_extended_net = interventions_df["reason"] == EXTENDED_NON_DRUG_REASON
    candidate_rows = interventions_df[was_therapeutic_candidate | caught_by_extended_net]
    if candidate_rows.empty:
        return pd.DataFrame(columns=columns)

    def summarize(g):
        row = g.iloc[0]
        exclusion_reason = row["reason"] if row["reason"] == EXTENDED_NON_DRUG_REASON else row["scope_reason"]
        return pd.Series({
            "name": row["original_name"],
            "clinicaltrials_intervention_type": row["original_type"],
            "nct_ids": "; ".join(sorted(set(g["nct_id"].dropna().astype(str)))),
            "trial_count": g["nct_id"].nunique(),
            "classification": row["classification"],
            "pipeline_scope": row["pipeline_scope"],
            "exclusion_reason": exclusion_reason,
            "confidence": row["scope_confidence"],
        })

    result = (
        candidate_rows.groupby("normalized_name", sort=False)
        .apply(summarize, include_groups=False)
        .reset_index(drop=True)
    )
    return result.sort_values(["pipeline_scope", "name"]).reset_index(drop=True)[columns]


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
