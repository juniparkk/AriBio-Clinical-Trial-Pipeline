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
_TARGET_PATHWAY_POINTS = 40
_MODALITY_POINTS = 25
_PURPOSE_CLASS_POINTS = 15
_SAME_PHASE_POINTS = 20
_ADJACENT_PHASE_POINTS = 10

_PURPOSE_CLASS_LABELS = {"DTT": "Disease-targeted", "STT": "Symptomatic"}


def compute_relevance_score(target_pathways, modality, purpose_class, phase_reached,
                             reference_target_pathways, reference_modality,
                             reference_purpose_class, reference_phase_reached):
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
            score += _SAME_PHASE_POINTS
            reasons.append(f"Same phase: {phase_reached}")
        elif distance == 1:
            score += _ADJACENT_PHASE_POINTS
            reasons.append(f"Adjacent phase: {phase_reached}")

    return score, reasons
