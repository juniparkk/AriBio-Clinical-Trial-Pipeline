# ============================================================
# COMPETITIVE INTELLIGENCE — AriBio relevance scoring
#
# A deterministic, rule-based relevance score (0-100) for a resolved
# drug's already-computed profile, benchmarked against a reference drug
# (AR1001 by default) — explicitly NOT an AI/LLM output. Every point
# awarded is tied to one plain-language reason, so the score is fully
# auditable: nothing here is a black box, unlike an "AI-curated"
# feed would be.
#
# Mixed relative/absolute design: modality and sponsor type score a
# competitor's SIMILARITY to AR1001 (same value as the reference).
# Target pathway, phase, and trial size score the competitor's OWN
# profile on an absolute scale instead -- a drug pursuing a disease-
# modifying mechanism, already in a late clinical phase, or running a
# large trial is a serious competitive signal regardless of what
# phase/pathway AR1001 itself happens to be at.
#
# This is deliberately scoped to fields the rest of this pipeline
# already resolves with real evidence (target_pathways, modality,
# therapeutic_purpose_class, phase_reached, max_enrollment) — no new,
# unverified data source is introduced, and nothing about drug
# identity/classification is changed by importing or using this module.
#
# Like drug_classification.py/nih_reference.py/scientific_classification.py,
# this module only DEFINES functions — no file I/O or printing at
# import time.
# ============================================================

from drug_classification import STALE_PHASE3_DISCONTINUED_LABEL

# Ranks phase_reached for PROXIMITY comparison -- used elsewhere in the
# pipeline (ctgov_changes.py's phase-change detection, competitive_
# attention.py's _is_advancement) to tell "which of two phases is
# further along," not by this module's own scoring anymore (see
# _PHASE_3_POINTS/_PHASE_2_POINTS below, which score phase_reached on
# an absolute scale rather than comparing it to a reference phase).
# Deliberately a separate, local table from drug_classification.py's
# _DRUG_ROLLUP_PHASE_RANK (that one answers "which of this drug's OWN
# trials is furthest along"; this one answers "how far apart are two
# DIFFERENT drugs' phases").
PHASE_RANK_FOR_SCORING = {
    "NA": 0, "Early Phase 1": 1, "Phase 1": 2, "Phase 1/Phase 2": 3,
    "Phase 2": 4, "Phase 2/Phase 3": 5, "Phase 3": 6, "Phase 4": 7,
}

# Point weights — sum to 100 when a competitor earns every dimension.
_MODALITY_POINTS = 10

# Phase is scored on an absolute scale, not proximity-to-reference: a
# competitor already in Phase 3 (AR1001's own phase) is the strongest
# signal, Phase 2 a weaker but real one, and every earlier/other phase
# (Phase 1, Phase 1/Phase 2, Phase 2/Phase 3, Phase 4, Early Phase 1,
# NA) earns no phase points at all -- no partial credit for
# "adjacent" phases the way the previous formula gave.
_PHASE_3_POINTS = 15
_PHASE_2_POINTS = 5

_SPONSOR_TYPE_POINTS = 20

# A large trial is itself a competitive signal (real recruitment
# capacity, real investment) independent of anything else about the
# drug -- absolute threshold, not a comparison to AR1001's own
# enrollment.
_LARGE_TRIAL_POINTS = 30
_LARGE_TRIAL_ENROLLMENT_THRESHOLD = 550

# Target pathway is also scored on an absolute scale: does THIS drug
# pursue one of the field's disease-modifying mechanisms, at all --
# not "does it happen to share AR1001's specific pathway." The five
# below are every target_pathways category that targets the disease
# process itself; Symptomatic and Neuropsychiatric (also real
# categories elsewhere in this pipeline) manage symptoms rather than
# modify the underlying disease, so they earn no points here.
_TARGET_PATHWAY_POINTS = 25
_DISEASE_MODIFYING_TARGET_PATHWAYS = {"Amyloid", "Tau", "Inflammation", "Neuroprotection", "Metabolism"}

# "Low relevance" ceiling applied when a competitor's OWN trial history
# raises doubt about whether it's still a live competitive threat, no
# matter how well its mechanism/modality/phase otherwise line up with
# AR1001's. These drugs are still shown (not filtered out of the
# dashboard) so the underlying match reasons stay visible/auditable —
# just capped low rather than removed.
LOW_RELEVANCE_CAP = 20
LOW_RELEVANCE_DISCONTINUED_STATUS = "Discontinued"
# See drug_classification.STALE_PHASE3_DISCONTINUED_LABEL -- a Phase 3
# trial past its completion date by years with no FDA approval, never
# formally closed by ct.gov, is just as much a non-live competitive
# signal as an explicit Discontinued status, so it earns the same cap.
LOW_RELEVANCE_STALE_PHASE3_STATUS = STALE_PHASE3_DISCONTINUED_LABEL
LOW_RELEVANCE_YEAR_CUTOFF = 2016      # no trial activity since before this year
LOW_RELEVANCE_ENROLLMENT_CUTOFF = 210  # largest trial under this many participants


def compute_relevance_score(target_pathways, modality, phase_reached, reference_modality,
                             sponsor_type=None, reference_sponsor_type=None,
                             status=None, latest_activity_year=None, max_enrollment=None):
    """
    Returns (score: int 0-100, reasons: list[str]).

    target_pathways: this drug's target pathway(s) — a list (a drug can
        have more than one; see scientific_classification.py's
        multi-target support). Scored absolutely against
        _DISEASE_MODIFYING_TARGET_PATHWAYS, not compared to the
        reference drug's own pathway(s).
    modality / reference_modality: e.g. "Small Molecule", "Biologic".
    phase_reached: one of PHASE_RANK_FOR_SCORING's keys, scored
        absolutely (see _PHASE_3_POINTS/_PHASE_2_POINTS above) — not
        compared to a reference phase.
    sponsor_type / reference_sponsor_type: "Company" or
        "University/Institution" (see drug_classification.py's
        classify_sponsor_type() — ct.gov's own lead-sponsor
        classification, "any industry-funded trial counts the whole
        drug as Company"). Blank/None never matches.
    status / latest_activity_year / max_enrollment: describe THIS
        drug. max_enrollment also earns _LARGE_TRIAL_POINTS on its own
        when above _LARGE_TRIAL_ENROLLMENT_THRESHOLD. All three can
        additionally push the score DOWN via the low-relevance cap
        (cap it at LOW_RELEVANCE_CAP, never up, and never change which
        reasons above were earned) — see module-level LOW_RELEVANCE_*
        constants.
    """
    score = 0
    reasons = []

    matched_pathways = sorted(set(target_pathways or []) & _DISEASE_MODIFYING_TARGET_PATHWAYS)
    if matched_pathways:
        score += _TARGET_PATHWAY_POINTS
        reasons.append(f"Disease-modifying target pathway: {', '.join(matched_pathways)}")

    if modality and modality == reference_modality:
        score += _MODALITY_POINTS
        reasons.append(f"Same modality: {modality}")

    if phase_reached == "Phase 3":
        score += _PHASE_3_POINTS
        reasons.append("Reached Phase 3")
    elif phase_reached == "Phase 2":
        score += _PHASE_2_POINTS
        reasons.append("Reached Phase 2")

    if sponsor_type and sponsor_type == reference_sponsor_type:
        score += _SPONSOR_TYPE_POINTS
        reasons.append(f"Same sponsor type: {sponsor_type}")

    if max_enrollment is not None and max_enrollment > _LARGE_TRIAL_ENROLLMENT_THRESHOLD:
        score += _LARGE_TRIAL_POINTS
        reasons.append(f"Large trial: {int(max_enrollment)} participants (over {_LARGE_TRIAL_ENROLLMENT_THRESHOLD})")

    low_relevance_flags = []
    if status in (LOW_RELEVANCE_DISCONTINUED_STATUS, LOW_RELEVANCE_STALE_PHASE3_STATUS):
        low_relevance_flags.append("discontinued/withdrawn")
    if latest_activity_year is not None and latest_activity_year < LOW_RELEVANCE_YEAR_CUTOFF:
        low_relevance_flags.append(f"no trial activity since before {LOW_RELEVANCE_YEAR_CUTOFF}")
    if max_enrollment is not None and max_enrollment < LOW_RELEVANCE_ENROLLMENT_CUTOFF:
        low_relevance_flags.append(f"largest trial under {LOW_RELEVANCE_ENROLLMENT_CUTOFF} participants")

    if low_relevance_flags and score > LOW_RELEVANCE_CAP:
        score = LOW_RELEVANCE_CAP
        reasons.append(f"Capped at {LOW_RELEVANCE_CAP}/100 (low relevance): {'; '.join(low_relevance_flags)}")

    return score, reasons
