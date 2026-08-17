# ============================================================
# COMPETITIVE INTELLIGENCE — AriBio relevance scoring
#
# A deterministic, rule-based similarity score (0-100) between a
# resolved drug's already-computed profile and a reference drug's
# (AR1001 by default) — explicitly NOT an AI/LLM output. Every point
# awarded is tied to one plain-language reason, so the score is fully
# auditable: nothing here is a black box, unlike an "AI-curated"
# feed would be.
#
# This is deliberately scoped to fields the rest of this pipeline
# already resolves with real evidence (target_pathways, modality,
# therapeutic_purpose_class, phase_reached) — no new, unverified data
# source is introduced, and nothing about drug identity/classification
# is changed by importing or using this module.
#
# Like drug_classification.py/nih_reference.py/scientific_classification.py,
# this module only DEFINES functions — no file I/O or printing at
# import time.
# ============================================================

# Ranks phase_reached for PROXIMITY comparison (how close is this drug's
# stage to the reference drug's), not "which is more advanced" — so the
# direction doesn't matter, only the rank distance. Deliberately a
# separate, local table from drug_classification.py's
# _DRUG_ROLLUP_PHASE_RANK (that one answers "which of this drug's OWN
# trials is furthest along"; this one answers "how far apart are two
# DIFFERENT drugs' phases").
PHASE_RANK_FOR_SCORING = {
    "NA": 0, "Early Phase 1": 1, "Phase 1": 2, "Phase 1/Phase 2": 3,
    "Phase 2": 4, "Phase 2/Phase 3": 5, "Phase 3": 6, "Phase 4": 7,
}

# Point weights — sum to 100 when every dimension matches exactly.
# Target pathway (shared mechanism of action) is weighted well above
# modality -- what a competitor's drug DOES to the disease matters far
# more to AR1001's competitive relevance than how it's delivered (pill
# vs. infusion vs. injection). Modality still earns a small number of
# points (it's a real, auditable signal, just a minor one) rather than
# being dropped from scoring entirely. Sponsor type was added later
# (see _SPONSOR_TYPE_POINTS below) by taking 10 points proportionally
# from these four (they summed to 100 before; 60/5/25/10 -> 54/5/22/9).
_TARGET_PATHWAY_POINTS = 54
_MODALITY_POINTS = 5
_PURPOSE_CLASS_POINTS = 22
_PHASE_PROXIMITY_POINTS = 9  # same phase OR one rank step away ("adjacent")

# A company-sponsored competitor is a more serious, better-funded
# competitive threat than a purely academic/institutional one -- same
# real-world reasoning as prioritizing industry sponsorship anywhere
# else in competitive intelligence. Scored the same way every other
# dimension here is: matches AR1001's OWN sponsor type (AriBio Co.,
# Ltd. -- a company), not an absolute "is this drug company-sponsored"
# check -- consistent with modality/purpose-class/phase all being
# "same as reference" comparisons, not standalone judgments.
_SPONSOR_TYPE_POINTS = 10

_PURPOSE_CLASS_LABELS = {"DTT": "Disease-targeted", "STT": "Symptomatic"}

# "Low relevance" ceiling applied when a competitor's OWN trial history
# raises doubt about whether it's still a live competitive threat, no
# matter how well its mechanism/modality/phase otherwise line up with
# AR1001's. These drugs are still shown (not filtered out of the
# dashboard) so the underlying match reasons stay visible/auditable —
# just capped low rather than removed.
LOW_RELEVANCE_CAP = 20
LOW_RELEVANCE_DISCONTINUED_STATUS = "Discontinued"
LOW_RELEVANCE_YEAR_CUTOFF = 2016      # no trial activity since before this year
LOW_RELEVANCE_ENROLLMENT_CUTOFF = 210  # largest trial under this many participants


def compute_relevance_score(target_pathways, modality, purpose_class, phase_reached,
                             reference_target_pathways, reference_modality,
                             reference_purpose_class, reference_phase_reached,
                             sponsor_type=None, reference_sponsor_type=None,
                             status=None, latest_activity_year=None, max_enrollment=None):
    """
    Returns (score: int 0-100, reasons: list[str]).

    target_pathways / reference_target_pathways: lists (a drug can have
        more than one — see scientific_classification.py's multi-target
        support). Any overlap at all earns the full target-pathway
        points; this intentionally does not partial-credit a 1-of-3
        overlap lower than a 3-of-3 one — sharing ANY real mechanism
        with the reference drug is the meaningful signal for
        competitive relevance, not exact pathway-set equality.
    modality / reference_modality: e.g. "Small Molecule", "Biologic".
    purpose_class / reference_purpose_class: "DTT" or "STT" (blank if
        this drug has no NIH-sourced therapeutic-purpose classification
        — contributes 0 points either way, never guessed).
    phase_reached / reference_phase_reached: one of PHASE_RANK_FOR_SCORING's
        keys.
    sponsor_type / reference_sponsor_type: "Company" or
        "University/Institution" (see drug_classification.py's
        classify_sponsor_type() — ct.gov's own lead-sponsor
        classification, "any industry-funded trial counts the whole
        drug as Company"). Blank/None never matches, same as
        purpose_class.
    status / latest_activity_year / max_enrollment: optional, describe
        THIS drug (not the reference) — its drug-level status_summary,
        the year of its MOST RECENT trial activity (e.g. latest primary
        completion date across all its trials — deliberately not the
        earliest trial start, which would misflag long-running,
        currently-active programs as stale just because their very
        first trial predates the cutoff), and its largest trial's
        enrollment. When given, they can only push the score DOWN (cap
        it at LOW_RELEVANCE_CAP), never up, and never change which
        reasons above were earned — see module-level LOW_RELEVANCE_*
        constants.
    """
    score = 0
    reasons = []

    shared_pathways = sorted(set(target_pathways or []) & set(reference_target_pathways or []))
    if shared_pathways:
        score += _TARGET_PATHWAY_POINTS
        reasons.append(f"Shares target pathway(s): {', '.join(shared_pathways)}")

    if modality and modality == reference_modality:
        score += _MODALITY_POINTS
        reasons.append(f"Same modality: {modality}")

    if purpose_class and purpose_class == reference_purpose_class:
        score += _PURPOSE_CLASS_POINTS
        reasons.append(f"Same treatment approach: {_PURPOSE_CLASS_LABELS.get(purpose_class, purpose_class)}")

    this_rank = PHASE_RANK_FOR_SCORING.get(phase_reached)
    ref_rank = PHASE_RANK_FOR_SCORING.get(reference_phase_reached)
    if this_rank is not None and ref_rank is not None:
        distance = abs(this_rank - ref_rank)
        if distance == 0:
            score += _PHASE_PROXIMITY_POINTS
            reasons.append(f"Same phase: {phase_reached}")
        elif distance == 1:
            score += _PHASE_PROXIMITY_POINTS
            reasons.append(f"Adjacent phase: {phase_reached}")

    if sponsor_type and sponsor_type == reference_sponsor_type:
        score += _SPONSOR_TYPE_POINTS
        reasons.append(f"Same sponsor type: {sponsor_type}")

    low_relevance_flags = []
    if status == LOW_RELEVANCE_DISCONTINUED_STATUS:
        low_relevance_flags.append("discontinued/withdrawn")
    if latest_activity_year is not None and latest_activity_year < LOW_RELEVANCE_YEAR_CUTOFF:
        low_relevance_flags.append(f"no trial activity since before {LOW_RELEVANCE_YEAR_CUTOFF}")
    if max_enrollment is not None and max_enrollment < LOW_RELEVANCE_ENROLLMENT_CUTOFF:
        low_relevance_flags.append(f"largest trial under {LOW_RELEVANCE_ENROLLMENT_CUTOFF} participants")

    if low_relevance_flags and score > LOW_RELEVANCE_CAP:
        score = LOW_RELEVANCE_CAP
        reasons.append(f"Capped at {LOW_RELEVANCE_CAP}/100 (low relevance): {'; '.join(low_relevance_flags)}")

    return score, reasons
