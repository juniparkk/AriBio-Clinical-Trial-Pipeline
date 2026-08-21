# ============================================================
# COMPETITIVE_ATTENTION — deterministic AriBio competitive-priority scoring
#
# Ranks already-detected pipeline changes (outputs/pipeline_changes.csv)
# by how much competitive attention they deserve at AriBio. This is
# NOT a scientific-accuracy review system — it never judges whether a
# competitor's drug works, whether a trial will succeed, or what its
# regulatory status is. It only combines already-verified, already-
# computed facts (change type, phase, timing, AR1001 similarity) into
# one deterministic 0-100 score via a fixed, documented point table.
#
# Hard rules (enforced by construction, not by prompting):
#   - The score is a plain sum of integer point constants below, run
#     through pandas/Python arithmetic — no text generation is ever
#     part of the score. why_it_matters/relevance_factors are built
#     FROM the same factor list the score was computed from (see
#     _describe_factors), never independently "written."
#   - Never infers clinical success from a "Completed" status, failure
#     from "Terminated"/"Withdrawn" alone, or FDA regulatory status
#     from any field (this module has no FDA-status factor at all).
#   - Never invents route, mechanism, biomarker, or population
#     similarity: route/endpoint/biomarker points are 0 unless the
#     watchlist config explicitly lists a route/endpoint/biomarker
#     AND that exact (case-insensitive) value is present in real data
#     — no fuzzy matching, no guessing from drug names.
#   - Non-therapeutic records (pipeline_scope != "Therapeutic Drug")
#     are excluded entirely, not just scored low.
#
# ------------------------------------------------------------
# SCORING FORMULA (all constants below; theoretical max ~103,
# clipped to 100; floor clipped to 0)
#
#   change_type_points          (see _change_type_points)
#     + results_posted................................ +30
#     + phase_change / highest_drug_phase_change,
#       ADVANCING (rank increases)..................... +25
#     + new_trial, phase includes Phase 2 or 3......... +25
#     + new_trial, any other phase..................... +10
#     + new_drug........................................ +10
#     + sponsor_change.................................. +8
#     + status_change, landing on Recruiting/Active..... +8
#     + status_change, any other status................. +3
#     + completion_date_change / primary_completion_date_change,
#       "substantial" (>=30-day shift, either direction). +8
#     + completion_date_change / primary_completion_date_change,
#       not substantial................................. +2
#     + enrollment_change, "substantial" (>=15% relative) +5
#     + trial_disappeared................................ +5
#     + phase_change / highest_drug_phase_change,
#       NOT advancing (lateral or regression)............ +3 / +5
#
#   + absolute_phase_points      (see PHASE_POINTS)
#     Phase 3 +20, Phase 2 +12, Phase 1/Phase 2 +6, Phase 1 +3,
#     Early Phase 1 +1, Phase 4 / NA +0
#
#   + timing_points               (trial-level rows only)
#     primary completion within priority_completion_imminent_days
#       (default 30)................................... +15
#     primary completion within priority_completion_soon_days
#       (default 90, and not already "imminent")......... +10
#     study completion within the last recently_completed_days
#       (default 30)..................................... +8
#
#   + aribio_similarity_points
#     round(aribio_relevance_score * 0.25)  -- reuses the ALREADY
#     computed, evidence-based 0-100 similarity-to-AR1001 score from
#     pipeline_drugs.csv (competitive_intelligence.py) -- max +25
#
#   + modality_match_points        (modality in watchlist priority_modalities) +5
#   + route_match_points           (route in watchlist priority_routes -- always
#                                    0 today; no upstream route data source exists)
#   + endpoint_match_points        (a configured priority_endpoints string is a
#                                    literal case-insensitive substring of this
#                                    trial's outcome-measure text)              +5
#   + biomarker_match_points       (same mechanism, priority_biomarkers)        +5
#   + population_match_points      (trial's Sex AND Age exactly match the
#                                    primary AriBio asset's reference_sex/age)  +3
#   + named_competitor_points      (the drug's/trial's sponsor word-matches a
#                                    company on watchlist competitor_companies,
#                                    e.g. "Eli Lilly" matching "Eli Lilly and
#                                    Company" -- same word-overlap matcher
#                                    drug_classification.py already uses for
#                                    official-pipeline sponsor matching)       +20
#
#   - low_confidence_penalty       (classification_confidence == "low")  -10
#
#   relevance_score = clip(sum of the above, 0, 100)
# ------------------------------------------------------------
# ============================================================

import pandas as pd

import aribio_watchlist
from drug_classification import normalize_text, _company_matches, STALE_PHASE3_DISCONTINUED_LABEL

# --- change-type point constants ---
POINTS_RESULTS_POSTED = 30
POINTS_PHASE_ADVANCEMENT = 25
POINTS_NEW_TRIAL_PHASE23 = 25
POINTS_NEW_TRIAL_OTHER = 10
POINTS_NEW_DRUG = 10
POINTS_SPONSOR_CHANGE = 8
POINTS_STATUS_ACTIVE = 8
POINTS_STATUS_OTHER = 3
POINTS_DATE_CHANGE_SUBSTANTIAL = 8
POINTS_DATE_CHANGE_MINOR = 2
POINTS_ENROLLMENT_SUBSTANTIAL = 5
POINTS_TRIAL_DISAPPEARED = 5
POINTS_PHASE_CHANGE_NONADVANCING = 3
POINTS_DRUG_PHASE_CHANGE_NONADVANCING = 5

# --- absolute-phase priority (independent of what changed) ---
PHASE_POINTS = {
    "Phase 3": 20, "Phase 2": 12, "Phase 1/Phase 2": 6, "Phase 1": 3,
    "Early Phase 1": 1, "Phase 4": 0, "NA": 0,
}

# --- timing (trial-level only) ---
POINTS_PRIMARY_COMPLETION_IMMINENT = 15
POINTS_PRIMARY_COMPLETION_SOON = 10
POINTS_RECENTLY_COMPLETED = 8

# --- similarity / config-driven factors ---
ARIBIO_SIMILARITY_SCALE = 0.25
POINTS_MODALITY_MATCH = 5
POINTS_ROUTE_MATCH = 5
POINTS_ENDPOINT_MATCH = 5
POINTS_BIOMARKER_MATCH = 5
POINTS_POPULATION_MATCH = 3
# A change from a company on the watchlist's competitor_companies list
# (config/aribio_watchlist.yaml) is treated as inherently worth a
# second look, independent of what else about the change scores high
# or low -- comparable in size to the two biggest change-type factors
# (results_posted/phase_advancement, +30/+25) so a named competitor's
# change reliably lands in the High/Critical tier on its own. See
# render_needs_attention_section() in competitive_attention_viz.py for
# how this ALSO guarantees display, not just a higher rank.
POINTS_NAMED_COMPETITOR = 20

ACTIVE_STATUSES = {"Recruiting", "Active"}

ATTENTION_COLUMNS = [
    "priority_rank", "priority_level", "relevance_score", "aribio_relevance_score",
    "canonical_drug_name", "nct_id", "company_or_sponsor", "change_type", "old_value", "new_value",
    "highest_phase", "modality", "target_pathways", "trial_status",
    "primary_completion_date", "completion_date", "why_it_matters",
    "relevance_factors", "source", "needs_human_review", "is_named_competitor",
]

_CHANGE_TYPE_DESCRIPTIONS = {
    "new_trial": "A new trial was registered{drug_clause}.",
    "trial_disappeared": "A previously tracked trial{drug_clause} no longer matches the current query scope.",
    "status_change": "Trial status changed from {old_value} to {new_value}.",
    "phase_change": "Trial phase changed from {old_value} to {new_value}.",
    "highest_drug_phase_change": "{canonical_drug_name}'s highest reported phase changed from {old_value} to {new_value}.",
    "enrollment_change": "Enrollment changed from {old_value} to {new_value}.",
    "primary_completion_date_change": "Primary completion date changed from {old_value} to {new_value}.",
    "completion_date_change": "Study/Overall completion date changed from {old_value} to {new_value}.",
    "sponsor_change": "Sponsor changed from {old_value} to {new_value}.",
    "results_posted": "Results were newly posted on ClinicalTrials.gov.",
    "new_drug": "A new candidate, {canonical_drug_name}, was identified in the pipeline.",
}


def _is_advancement(old_phase, new_phase):
    from competitive_intelligence import PHASE_RANK_FOR_SCORING
    old_rank, new_rank = PHASE_RANK_FOR_SCORING.get(old_phase), PHASE_RANK_FOR_SCORING.get(new_phase)
    return old_rank is not None and new_rank is not None and new_rank > old_rank


def build_drug_lookup(drugs_df):
    if drugs_df is None or drugs_df.empty:
        return {}
    return {r["display_name"]: r.to_dict() for _, r in drugs_df.iterrows()}


def build_trial_lookup(annotated_df, trials_df):
    lookup = {}
    if trials_df is not None and not trials_df.empty:
        for _, r in trials_df.iterrows():
            lookup[r["NCT Number"]] = {
                "sponsor": r.get("Sponsor", ""),
                "study_status": r.get("Study Status", ""),
                "primary_completion_date": r.get("Primary Completion Date", ""),
                "completion_date": r.get("Completion Date", ""),
                "sex": r.get("Sex", ""),
                "age": r.get("Age", ""),
                "outcomes_text": " ".join(str(r.get(c, "")) for c in
                                           ("Primary Outcome Measures", "Secondary Outcome Measures", "Other Outcome Measures")),
            }
    if annotated_df is not None and not annotated_df.empty:
        for _, r in annotated_df.iterrows():
            entry = lookup.setdefault(r["nct_id"], {})
            entry["pipeline_scope"] = r.get("pipeline_scope", "")
            entry["verification_status"] = r.get("verification_status", "")
            entry["classification_confidence"] = r.get("classification_confidence", "")
            entry["phase_clean"] = r.get("phase_clean", "")
            entry["status_clean"] = r.get("status_clean", "")
            entry["developed_drug"] = r.get("developed_drug", "")
    return lookup


def is_therapeutic(change_row, drug_lookup, trial_lookup):
    """Non-therapeutic records are excluded entirely, per project rule."""
    if change_row["entity_type"] == "drug":
        info = drug_lookup.get(change_row["canonical_drug_name"])
        return bool(info) and info.get("pipeline_scope") == "Therapeutic Drug"

    info = trial_lookup.get(change_row.get("nct_id"))
    if info:
        return info.get("pipeline_scope") == "Therapeutic Drug"
    # Trial no longer present in the current lookup (e.g. trial_disappeared) —
    # fall back to whether change detection itself found an associated drug.
    return bool(str(change_row.get("canonical_drug_name") or "").strip())


def _change_type_points(change_row):
    """Returns (points: int, factor_label: str or None)."""
    ct = change_row["change_type"]
    old_v, new_v = change_row.get("old_value", ""), change_row.get("new_value", "")

    if ct == "results_posted":
        return POINTS_RESULTS_POSTED, "results newly posted"

    if ct == "phase_change":
        if _is_advancement(old_v, new_v):
            return POINTS_PHASE_ADVANCEMENT, "phase advancement"
        return POINTS_PHASE_CHANGE_NONADVANCING, "phase change"

    if ct == "highest_drug_phase_change":
        if _is_advancement(old_v, new_v):
            return POINTS_PHASE_ADVANCEMENT, "highest drug phase advancement"
        return POINTS_DRUG_PHASE_CHANGE_NONADVANCING, "highest drug phase change"

    if ct == "new_trial":
        new_v_str = str(new_v or "")
        if "Phase 2" in new_v_str or "Phase 3" in new_v_str:
            return POINTS_NEW_TRIAL_PHASE23, "new Phase 2/3 trial"
        return POINTS_NEW_TRIAL_OTHER, "new trial"

    if ct == "new_drug":
        return POINTS_NEW_DRUG, "newly identified drug"

    if ct == "sponsor_change":
        return POINTS_SPONSOR_CHANGE, "sponsor change"

    if ct == "status_change":
        if new_v in ACTIVE_STATUSES:
            return POINTS_STATUS_ACTIVE, "status change to active/recruiting"
        return POINTS_STATUS_OTHER, "status change"

    if ct in ("primary_completion_date_change", "completion_date_change"):
        if change_row.get("importance") == "Medium":
            return POINTS_DATE_CHANGE_SUBSTANTIAL, "substantial completion-date shift"
        return POINTS_DATE_CHANGE_MINOR, "completion-date shift"

    if ct == "enrollment_change":
        if change_row.get("importance") == "Medium":
            return POINTS_ENROLLMENT_SUBSTANTIAL, "substantial enrollment change"
        return 0, None

    if ct == "trial_disappeared":
        return POINTS_TRIAL_DISAPPEARED, "trial disappeared from query scope"

    return 0, None


def _timing_points(trial_info, today, thresholds):
    if trial_info is None:
        return []
    factors = []
    pcd = pd.to_datetime(trial_info.get("primary_completion_date", ""), errors="coerce")
    cd = pd.to_datetime(trial_info.get("completion_date", ""), errors="coerce")

    if pd.notna(pcd):
        days_out = (pcd - today).days
        if 0 <= days_out <= thresholds["primary_completion_imminent_days"]:
            factors.append((POINTS_PRIMARY_COMPLETION_IMMINENT, "primary completion within 30 days"))
        elif 0 <= days_out <= thresholds["primary_completion_soon_days"]:
            factors.append((POINTS_PRIMARY_COMPLETION_SOON, "primary completion within 90 days"))

    if pd.notna(cd):
        days_since = (today - cd).days
        if 0 <= days_since <= thresholds["recently_completed_days"]:
            factors.append((POINTS_RECENTLY_COMPLETED, "recently completed"))

    return factors


def _named_competitor_match(drug_info, trial_info, watchlist):
    """Returns the FIRST watchlist competitor_companies entry whose
    normalized words are a subset of (or contain) this trial's/drug's
    own sponsor's normalized words -- e.g. watchlist entry "Eli Lilly"
    matches ct.gov sponsor string "Eli Lilly and Company". Reuses
    drug_classification.py's own word-overlap matcher (the same one
    official-pipeline sponsor matching uses) rather than a naive
    substring check, which would false-positive on e.g. "Roche" inside
    "University of Rochester" or "Biogen" inside "NeuroBiogen Co., Ltd".

    A drug can have multiple semicolon-joined sponsors (see
    resolved_drugs_df's own sponsor field) -- checks all of them, not
    just the first.
    """
    companies = watchlist.get("competitor_companies") or []
    if not companies:
        return None
    sponsor_text = (trial_info or {}).get("sponsor") or (drug_info or {}).get("sponsor") or ""
    sponsor_parts = [normalize_text(p) for p in str(sponsor_text).split(";") if p.strip()]
    sponsor_parts = [p for p in sponsor_parts if p]
    if not sponsor_parts:
        return None
    for company in companies:
        company_norm = normalize_text(company)
        if not company_norm:
            continue
        for sponsor_norm in sponsor_parts:
            if _company_matches(company_norm, sponsor_norm):
                return company
    return None


def _similarity_points(drug_info, trial_info, watchlist):
    factors = []

    matched_company = _named_competitor_match(drug_info, trial_info, watchlist)
    if matched_company:
        factors.append((POINTS_NAMED_COMPETITOR, f"named competitor on watchlist ({matched_company})"))

    if drug_info is not None:
        score = drug_info.get("aribio_relevance_score")
        if pd.notna(score) and score:
            pts = round(float(score) * ARIBIO_SIMILARITY_SCALE)
            if pts:
                factors.append((pts, "similar profile to AR1001"))

    modality = (drug_info or {}).get("modality", "")
    if modality and modality in (watchlist.get("priority_modalities") or []):
        factors.append((POINTS_MODALITY_MATCH, f"priority modality ({modality})"))

    primary_asset = aribio_watchlist.get_primary_asset(watchlist)
    route = primary_asset.get("route")
    if route and route in (watchlist.get("priority_routes") or []):
        # No upstream classification step currently produces a route
        # field for competitor drugs/trials -- this factor is a
        # documented no-op until one exists. Never inferred/guessed.
        pass

    outcomes_text = ((trial_info or {}).get("outcomes_text") or "").lower()
    for endpoint in (watchlist.get("priority_endpoints") or []):
        if endpoint and endpoint.lower() in outcomes_text:
            factors.append((POINTS_ENDPOINT_MATCH, f"overlapping endpoint ({endpoint})"))
            break
    for biomarker in (watchlist.get("priority_biomarkers") or []):
        if biomarker and biomarker.lower() in outcomes_text:
            factors.append((POINTS_BIOMARKER_MATCH, f"overlapping biomarker ({biomarker})"))
            break

    if trial_info is not None:
        ref_sex = primary_asset.get("reference_sex")
        ref_age = primary_asset.get("reference_age")
        if ref_sex and ref_age and trial_info.get("sex") == ref_sex and trial_info.get("age") == ref_age:
            factors.append((POINTS_POPULATION_MATCH, "similar eligibility population"))

    return factors


def _describe_factors(change_type_factor, phase_points, phase_label, timing_factors,
                       similarity_factors, penalty):
    factors = []
    if change_type_factor[1]:
        factors.append((change_type_factor[1], change_type_factor[0]))
    if phase_points:
        factors.append((f"reached {phase_label}", phase_points))
    for pts, label in timing_factors:
        factors.append((label, pts))
    for pts, label in similarity_factors:
        factors.append((label, pts))
    if penalty:
        factors.append(("low classification confidence", -penalty))
    return factors


def _why_it_matters(change_row, factors):
    template = _CHANGE_TYPE_DESCRIPTIONS.get(change_row["change_type"], "A change was detected.")
    # A blank cell that round-tripped through CSV (e.g. via
    # update_changes_history()'s accumulated history file) reads back
    # as pandas' float NaN, which is truthy in Python -- guard with
    # pd.isna() first, or `nan or ""` silently keeps NaN and this
    # renders the literal text "nan" instead of omitting the clause.
    raw_drug = change_row.get("canonical_drug_name")
    drug = "" if pd.isna(raw_drug) else str(raw_drug).strip()
    drug_clause = f" for {drug}" if drug else ""
    base = template.format(
        old_value=change_row.get("old_value", ""), new_value=change_row.get("new_value", ""),
        canonical_drug_name=drug, drug_clause=drug_clause,
    )
    supporting = [label for label, pts in factors if pts > 0 and label not in base][:2]
    if supporting:
        base += " " + "; ".join(s[0].upper() + s[1:] for s in supporting) + "."
    return base


def priority_level_for_score(score, thresholds):
    if score >= thresholds["critical_score"]:
        return "Critical"
    if score >= thresholds["high_score"]:
        return "High"
    if score >= thresholds["medium_score"]:
        return "Medium"
    return "Low"


def score_row(change_row, drug_lookup, trial_lookup, watchlist, today):
    thresholds = watchlist["alert_thresholds"]

    drug_info = drug_lookup.get(change_row.get("canonical_drug_name"))
    trial_info = trial_lookup.get(change_row.get("nct_id")) if change_row["entity_type"] == "trial" else None

    change_type_factor = _change_type_points(change_row)

    phase_label = ""
    if change_row["entity_type"] == "drug":
        phase_label = change_row.get("new_value", "") or (drug_info or {}).get("phase_reached", "")
    else:
        phase_label = (drug_info or {}).get("phase_reached") or (trial_info or {}).get("phase_clean", "")
        if not phase_label and change_row["change_type"] == "phase_change":
            phase_label = change_row.get("new_value", "")
    phase_points = PHASE_POINTS.get(phase_label, 0)

    timing_factors = _timing_points(trial_info, today, thresholds) if change_row["entity_type"] == "trial" else []
    similarity_factors = _similarity_points(drug_info, trial_info, watchlist)

    confidence = (drug_info or trial_info or {}).get("classification_confidence", "")
    penalty = thresholds["low_confidence_score_penalty"] if confidence == "low" else 0

    raw_score = (
        change_type_factor[0] + phase_points
        + sum(p for p, _ in timing_factors)
        + sum(p for p, _ in similarity_factors)
        - penalty
    )
    score = max(0, min(100, raw_score))

    factors = _describe_factors(change_type_factor, phase_points, phase_label, timing_factors, similarity_factors, penalty)
    is_named_competitor = _named_competitor_match(drug_info, trial_info, watchlist) is not None

    return score, factors, drug_info, trial_info, is_named_competitor


def compute_attention(changes_df, drugs_df, annotated_df, trials_df, watchlist, today=None):
    """Returns a DataFrame shaped exactly like outputs/competitive_attention.csv,
    ranked by relevance_score descending. Non-therapeutic records are excluded.

    A drug-level change row (e.g. "new_drug") carries nct_id == "" by
    construction (see ctgov_changes.detect_drug_level_changes) -- its
    nct_id is backfilled here from annotated_df via build_drug_to_nct_lookup()/
    _backfilled_nct_id(), the same mechanism prepare_recent_changes() already
    uses, so Needs Attention links to a real trial instead of always
    showing "no linked trial" when one demonstrably exists.
    """
    today = pd.Timestamp(today) if today is not None else pd.Timestamp.now(tz=None).normalize()

    if changes_df is None or changes_df.empty:
        return pd.DataFrame(columns=ATTENTION_COLUMNS)

    drug_lookup = build_drug_lookup(drugs_df)
    trial_lookup = build_trial_lookup(annotated_df, trials_df)
    drug_nct_lookup = build_drug_to_nct_lookup(annotated_df)

    rows = []
    for _, change_row in changes_df.iterrows():
        if not is_therapeutic(change_row, drug_lookup, trial_lookup):
            continue

        score, factors, drug_info, trial_info, is_named_competitor = score_row(change_row, drug_lookup, trial_lookup, watchlist, today)

        highest_phase = ""
        if change_row["entity_type"] == "drug":
            highest_phase = (drug_info or {}).get("phase_reached") or change_row.get("new_value", "")
        else:
            highest_phase = (drug_info or {}).get("phase_reached") or (trial_info or {}).get("phase_clean", "")

        modality = (drug_info or {}).get("modality", "")
        target_pathways = (drug_info or {}).get("target_pathways", "")
        trial_status = (trial_info or {}).get("status_clean", "") if trial_info else (drug_info or {}).get("status_summary", "")
        company = (trial_info or {}).get("sponsor", "") or (drug_info or {}).get("sponsor", "") or change_row.get("sponsor_or_company", "")
        pcd = (trial_info or {}).get("primary_completion_date", "") if trial_info else ""
        cd = (trial_info or {}).get("completion_date", "") if trial_info else ""

        raw_relevance = (drug_info or {}).get("aribio_relevance_score")
        aribio_relevance_score = int(raw_relevance) if raw_relevance is not None and pd.notna(raw_relevance) else None

        confidence = (drug_info or trial_info or {}).get("classification_confidence", "")
        needs_human_review = bool(
            score >= watchlist["alert_thresholds"]["high_score"]
            or change_row.get("needs_review")
            or confidence == "low"
        )

        rows.append({
            "priority_rank": None,  # filled in after sorting
            "priority_level": priority_level_for_score(score, watchlist["alert_thresholds"]),
            "relevance_score": score,
            "aribio_relevance_score": aribio_relevance_score,
            "canonical_drug_name": change_row.get("canonical_drug_name", ""),
            "nct_id": _backfilled_nct_id(change_row.get("nct_id"), change_row.get("canonical_drug_name"), drug_nct_lookup),
            "company_or_sponsor": company,
            "change_type": change_row["change_type"],
            "old_value": change_row.get("old_value", ""),
            "new_value": change_row.get("new_value", ""),
            "highest_phase": highest_phase,
            "modality": modality,
            "target_pathways": target_pathways,
            "trial_status": trial_status,
            "primary_completion_date": pcd,
            "completion_date": cd,
            "why_it_matters": _why_it_matters(change_row, factors),
            "relevance_factors": "; ".join(f"{label} ({'+' if pts >= 0 else ''}{pts})" for label, pts in factors),
            "source": change_row.get("source", ""),
            "needs_human_review": needs_human_review,
            "is_named_competitor": is_named_competitor,
        })

    result = pd.DataFrame(rows, columns=ATTENTION_COLUMNS)
    if result.empty:
        return result

    result = result.sort_values(
        ["relevance_score", "nct_id", "canonical_drug_name"], ascending=[False, True, True]
    ).reset_index(drop=True)
    result["priority_rank"] = result.index + 1
    return result


def describe_change(change_row):
    """Plain factual one-line description of a single pipeline_changes.csv
    row, with NO scoring involved — used by the unscored "Recent
    Changes" feed. Reuses _why_it_matters() with an empty factors list
    so the same change_type phrase templates apply without any
    scored "supporting factors" clause being appended."""
    return _why_it_matters(change_row, [])


RECENT_CHANGES_WINDOW_DAYS = 30


def build_drug_to_nct_lookup(annotated_df):
    """Maps a resolved drug name (annotated_df's own `developed_drug`)
    to ONE representative NCT ID from the full annotated trial set.

    Used to backfill a real, clickable trial link onto DRUG-LEVEL
    change rows (change_type == "new_drug", etc.) -- those rows carry
    nct_id == "" by construction (see ctgov_changes.detect_drug_level_
    changes: a drug-level change isn't tied to any single trial event),
    even though the drug itself is almost always associated with at
    least one real trial elsewhere in the dataset. When a drug maps to
    more than one trial, the first one encountered is used -- an
    arbitrary but real, clickable link beats "no linked trial" when a
    trial demonstrably exists.
    """
    if annotated_df is None or annotated_df.empty:
        return {}
    lookup = {}
    for _, r in annotated_df.iterrows():
        drug = r.get("developed_drug")
        if drug is None or (isinstance(drug, float) and pd.isna(drug)) or not str(drug).strip():
            continue
        lookup.setdefault(str(drug).strip(), r.get("nct_id", ""))
    return lookup


def _backfilled_nct_id(nct, drug, drug_nct_lookup):
    """Shared by compute_attention() and prepare_recent_changes(): a
    row's own nct_id wins whenever it's set; only a genuinely blank one
    (drug-level changes like "new_drug" carry nct_id == "" by
    construction) falls back to build_drug_to_nct_lookup()'s
    drug-name -> representative-NCT mapping."""
    if nct is not None and not (isinstance(nct, float) and pd.isna(nct)) and str(nct).strip():
        return nct
    if drug is not None and not (isinstance(drug, float) and pd.isna(drug)) and str(drug).strip():
        return drug_nct_lookup.get(str(drug).strip(), nct)
    return nct


def prepare_recent_changes(changes_df, today=None, window_days=RECENT_CHANGES_WINDOW_DAYS, drug_nct_lookup=None):
    """Returns rows detected within the trailing `window_days` days of
    `today` (default: now; 30-day window by default), sorted purely by
    detected_date -- most recent first -- with a "description" column
    added via describe_change(). Not truncated here —
    competitive_attention_viz.py's renderer owns the "top N of total"
    slicing, same as render_needs_attention_section. Kept separate
    from compute_attention()/Needs Attention: this is the raw,
    UNSCORED "what changed" feed — the renderer consumes "description"
    directly and performs no computation of its own.

    `changes_df` is expected to already span the intended history
    window (e.g. update_changes_history()'s accumulated, pruned output)
    — this function's own date filter is a second, defensive guarantee
    that nothing older than `window_days` is ever shown, regardless of
    what the caller passed in.

    `drug_nct_lookup` (see build_drug_to_nct_lookup()): when given, a
    row whose own nct_id is blank gets backfilled from this lookup by
    its canonical_drug_name, so genuinely drug-level changes (like
    "new_drug") still link to a real trial when one is resolvable,
    instead of always showing "no linked trial".
    """
    fallback_columns = ["detected_date", "entity_type", "nct_id", "canonical_drug_name",
                         "sponsor_or_company", "change_type", "old_value", "new_value",
                         "importance", "source", "needs_review", "description"]
    if changes_df is None or changes_df.empty:
        return pd.DataFrame(columns=fallback_columns)

    today_ts = pd.Timestamp(today) if today is not None else pd.Timestamp.now(tz=None).normalize()
    cutoff = today_ts - pd.Timedelta(days=window_days)
    detected = pd.to_datetime(changes_df["detected_date"], errors="coerce")
    windowed = changes_df.loc[(detected >= cutoff) & (detected <= today_ts)].copy()
    if windowed.empty:
        return pd.DataFrame(columns=fallback_columns)

    # kind="stable" keeps same-date rows in a deterministic (original)
    # order across runs, rather than an unstable-sort coin flip.
    windowed = windowed.sort_values("detected_date", ascending=False, kind="stable")
    windowed["description"] = windowed.apply(describe_change, axis=1)

    if drug_nct_lookup:
        windowed["nct_id"] = windowed.apply(
            lambda row: _backfilled_nct_id(row.get("nct_id"), row.get("canonical_drug_name"), drug_nct_lookup),
            axis=1,
        )

    return windowed.reset_index(drop=True)


def update_changes_history(changes_df, history_path, today=None, window_days=RECENT_CHANGES_WINDOW_DAYS):
    """Appends this run's freshly-detected changes into the accumulated
    change-history file at `history_path` (creating it if it doesn't
    exist yet), then prunes anything older than `window_days` days from
    `today` (default: now). This file is intentionally NOT an unbounded
    log — it exists solely to give prepare_recent_changes() a real
    trailing window to draw from; nothing outside that window is ever
    displayed, so nothing outside it is kept on disk either.

    Deduplicates on (detected_date, entity_type, nct_id, change_type,
    old_value, new_value) so re-running the pipeline more than once on
    the same day never double-counts a change already recorded.

    Returns the pruned, combined DataFrame (already written to
    `history_path`) so the caller can hand it directly to
    prepare_recent_changes() without a redundant re-read.

    pipeline_changes.csv itself is untouched by this function — it
    keeps its existing, separate meaning as "just this run's diff"
    (see run_pipeline.py). This history file is additive, not a
    replacement.
    """
    import os
    existing = pd.read_csv(history_path, low_memory=False) if os.path.exists(history_path) else None
    frames = [df for df in (existing, changes_df) if df is not None and not df.empty]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if combined.empty:
        os.makedirs(os.path.dirname(history_path) or ".", exist_ok=True)
        combined.to_csv(history_path, index=False)
        return combined

    dedupe_cols = [c for c in ("detected_date", "entity_type", "nct_id", "change_type", "old_value", "new_value")
                   if c in combined.columns]
    if dedupe_cols:
        combined = combined.drop_duplicates(subset=dedupe_cols, keep="last")

    today_ts = pd.Timestamp(today) if today is not None else pd.Timestamp.now(tz=None).normalize()
    cutoff = today_ts - pd.Timedelta(days=window_days)
    detected = pd.to_datetime(combined["detected_date"], errors="coerce")
    combined = combined.loc[(detected >= cutoff) & (detected <= today_ts)].reset_index(drop=True)

    os.makedirs(os.path.dirname(history_path) or ".", exist_ok=True)
    combined.to_csv(history_path, index=False)
    return combined


def build_milestones(annotated_df, trials_df, changes_df, watchlist, today=None, drugs_df=None):
    """Buckets currently-therapeutic trials by completion-date timing for
    the "Upcoming Competitive Milestones" dashboard section.

    Returns a dict with four lists of row dicts:
      next_30_days, next_90_days (31-90 days out, mutually exclusive
      with next_30_days), recently_completed, materially_delayed.

    Never states or implies a result is expected — only that a
    completion-date milestone is near/passed/moved. materially_delayed
    is sourced from pipeline_changes.csv's own substantial,
    later-direction completion-date changes (a real detected change,
    not a static inference from a single snapshot).

    `drugs_df` (pipeline_drugs.csv's per-drug rollup, optional): when
    given, each item's `drug_type` is looked up from it by drug name --
    without it, drug_type is simply blank rather than an error, since
    every other field here is already resolvable from annotated_df/
    trials_df alone.
    """
    today = pd.Timestamp(today) if today is not None else pd.Timestamp.now(tz=None).normalize()
    thresholds = watchlist["alert_thresholds"]

    trial_lookup = build_trial_lookup(annotated_df, trials_df)
    drug_lookup = build_drug_lookup(drugs_df)
    therapeutic_ids = {nct_id for nct_id, info in trial_lookup.items() if info.get("pipeline_scope") == "Therapeutic Drug"}

    next_30, next_90, recently_completed = [], [], []
    for nct_id in sorted(therapeutic_ids):
        info = trial_lookup[nct_id]
        pcd = pd.to_datetime(info.get("primary_completion_date", ""), errors="coerce")
        cd = pd.to_datetime(info.get("completion_date", ""), errors="coerce")
        drug_name = info.get("developed_drug", "") or nct_id

        item = {
            "nct_id": nct_id,
            "drug_name": drug_name,
            "sponsor": info.get("sponsor", ""),
            "status": info.get("status_clean", ""),
            "phase": info.get("phase_clean", ""),
            "drug_type": drug_lookup.get(drug_name, {}).get("drug_type", ""),
            "primary_completion_date": info.get("primary_completion_date", ""),
            "completion_date": info.get("completion_date", ""),
        }

        # A discontinued trial (terminated/withdrawn/suspended) is not
        # actually progressing toward its on-file completion date, so
        # it's excluded from the "upcoming" buckets — it can still
        # appear in recently_completed, where a past date is a real,
        # already-happened fact regardless of why the trial ended.
        if pd.notna(pcd) and info.get("status_clean") not in ("Discontinued", STALE_PHASE3_DISCONTINUED_LABEL):
            days_out = (pcd - today).days
            if 0 <= days_out <= thresholds["primary_completion_imminent_days"]:
                next_30.append(item)
            elif thresholds["primary_completion_imminent_days"] < days_out <= thresholds["primary_completion_soon_days"]:
                next_90.append(item)

        if pd.notna(cd):
            days_since = (today - cd).days
            if 0 <= days_since <= thresholds["recently_completed_days"]:
                recently_completed.append(item)

    materially_delayed = []
    if changes_df is not None and not changes_df.empty:
        date_changes = changes_df[
            changes_df["change_type"].isin(["primary_completion_date_change", "completion_date_change"])
            & (changes_df["importance"] == "Medium")
        ]
        for _, row in date_changes.iterrows():
            if row["nct_id"] not in therapeutic_ids:
                continue
            old_d = pd.to_datetime(row.get("old_value", ""), errors="coerce")
            new_d = pd.to_datetime(row.get("new_value", ""), errors="coerce")
            if pd.isna(old_d) or pd.isna(new_d):
                continue
            if (new_d - old_d).days >= thresholds["major_delay_days"]:
                info = trial_lookup.get(row["nct_id"], {})
                drug_name = row.get("canonical_drug_name") or info.get("developed_drug", "") or row["nct_id"]
                materially_delayed.append({
                    "nct_id": row["nct_id"],
                    "drug_name": drug_name,
                    "sponsor": info.get("sponsor", row.get("sponsor_or_company", "")),
                    "phase": info.get("phase_clean", ""),
                    "drug_type": drug_lookup.get(drug_name, {}).get("drug_type", ""),
                    "change_type": row["change_type"],
                    "old_value": row.get("old_value", ""),
                    "new_value": row.get("new_value", ""),
                    "days_delayed": (new_d - old_d).days,
                })

    return {
        "next_30_days": next_30,
        "next_90_days": next_90,
        "recently_completed": recently_completed,
        "materially_delayed": materially_delayed,
    }


def load_and_compute(changes_path="outputs/pipeline_changes.csv", drugs_path="pipeline_drugs.csv",
                      annotated_path="pipeline_annotated.csv", trials_path="trials.csv",
                      watchlist_path=None, today=None):
    import os

    def _read(path):
        return pd.read_csv(path, low_memory=False) if os.path.exists(path) else None

    changes_df = _read(changes_path)
    drugs_df = _read(drugs_path)
    annotated_df = _read(annotated_path)
    trials_df = _read(trials_path)
    watchlist = aribio_watchlist.load_watchlist(watchlist_path) if watchlist_path else aribio_watchlist.load_watchlist()

    return compute_attention(changes_df, drugs_df, annotated_df, trials_df, watchlist, today=today)
