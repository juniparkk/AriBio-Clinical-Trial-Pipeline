# ============================================================
# CTGOV_CHANGES — snapshot-based pipeline change detection
#
# Compares the trials.csv / pipeline_drugs.csv / pipeline_annotated.csv
# state from BEFORE a refresh (i.e. whatever was already committed to
# git — the output of the LAST successful refresh) against the freshly
# fetched + rebuilt state produced by THIS refresh, and emits one row
# per meaningful change.
#
# Deliberately uses the git-tracked trials.csv/pipeline_drugs.csv/
# pipeline_annotated.csv as the "previous" side of the comparison —
# NOT data/snapshots/ (which is .gitignore'd and does not persist
# across GitHub Actions runs; a fresh checkout only restores
# git-tracked files). This makes the comparison work identically
# whether run_pipeline.py runs locally or in CI.
#
# Deliberately factual, never interpretive:
#   - never infers clinical success from a "Completed" status
#   - never infers failure from "Terminated"/"Withdrawn" alone
#   - never infers or fabricates FDA regulatory status
#   old_value/new_value always carry the plain ct.gov field values
#   (cleaned the same way pipeline_viz.py's dashboard already does —
#   see PHASE_LABELS/STATUS_MAP below — never editorialized).
#
# "Ignore harmless formatting-only changes": string fields are
# compared after whitespace normalization (Sponsor also
# case-insensitively — ct.gov capitalization cleanups are not a real
# sponsor change). A date or enrollment value that differs numerically
# is still recorded (it IS real information), but only graded Medium
# importance when the shift is "substantial" (see the threshold
# constants below) — small numeric drift is recorded at Low
# importance rather than silently dropped, so nothing is hidden from
# the audit trail.
# ============================================================

import pandas as pd

from competitive_intelligence import PHASE_RANK_FOR_SCORING

# Mirrors pipeline_viz.py's STEP 2 PHASE_LABELS exactly — an
# intentional, narrowly-scoped duplication (NOT a re-derivation; see
# that file's own comment: "an EXACT map against ct.gov's own phase
# enum"). Kept here so this module never has to import pipeline_viz.py
# itself, which would execute its entire top-level pipeline as a side
# effect (it has no __main__ guard). If ct.gov ever adds a new phase
# enum value, update both copies together.
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

# Mirrors pipeline_viz.py's STEP 2 STATUS_MAP exactly — same rationale.
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

CHANGES_COLUMNS = [
    "detected_date", "entity_type", "nct_id", "canonical_drug_name",
    "sponsor_or_company", "change_type", "old_value", "new_value",
    "importance", "source", "needs_review",
]

# "Substantial" thresholds for Medium-vs-Low grading — explicit,
# documented constants (same spirit as ctgov_snapshot.py's
# MIN_RATIO/MAX_RATIO) rather than unstated magic numbers.
ENROLLMENT_SUBSTANTIAL_RELATIVE = 0.15  # >=15% relative change
DATE_SUBSTANTIAL_DAYS = 30              # >=30-day shift

# Importance rules. The three explicit rules the project specified are
# marked (*); everything else is a documented, reasoned default for a
# change_type that detect list requires but the importance rules
# didn't explicitly cover:
#   High:
#     - new_trial, when the trial's phase includes Phase 2 or 3 (*)
#     - phase_change / highest_drug_phase_change, when it's an
#       ADVANCEMENT (rank increases) (*)
#     - results_posted (*)
#   Medium:
#     - status_change, any direction — including into
#       Terminated/Withdrawn, WITHOUT implying failure (*)
#     - new_trial, any other phase
#     - phase_change / highest_drug_phase_change, non-advancing
#       (lateral or a rank decrease — e.g. a data correction)
#     - enrollment_change / primary_completion_date_change /
#       completion_date_change, when "substantial" (*)
#     - new_drug (a new candidate identified — significant, but not
#       one of the three explicit High triggers)
#     - sponsor_change (ownership/licensing changes are operationally
#       significant; always also needs_review)
#     - trial_disappeared (no longer matches the query scope — could
#       be a real change or a metadata edit; always needs_review)
#   Low:
#     - enrollment_change / date changes, when not "substantial"


def clean_phase(raw):
    if pd.isna(raw) or not str(raw).strip():
        return "NA"
    return PHASE_LABELS.get(str(raw).strip().upper(), "NA")


def clean_status(raw):
    if pd.isna(raw) or not str(raw).strip():
        return "Unknown"
    key = str(raw).strip().upper().replace(" ", "_")
    return STATUS_MAP.get(key, "Other")


def _norm(x):
    if pd.isna(x):
        return ""
    return " ".join(str(x).strip().split())


def _num_or_none(x):
    try:
        if pd.isna(x):
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def _num_to_str(v):
    if v is None:
        return ""
    return str(int(v)) if v == int(v) else str(v)


def _enrollment_change(old_raw, new_raw):
    """Returns (old_str, new_str, substantial: bool) or None if unchanged."""
    old_v, new_v = _num_or_none(old_raw), _num_or_none(new_raw)
    if old_v == new_v:
        return None
    if old_v is None or new_v is None:
        return (_num_to_str(old_v), _num_to_str(new_v), True)
    relative = abs(new_v - old_v) / max(abs(old_v), abs(new_v), 1)
    return (_num_to_str(old_v), _num_to_str(new_v), relative >= ENROLLMENT_SUBSTANTIAL_RELATIVE)


def _date_change(old_raw, new_raw):
    """Returns (old_str, new_str, substantial: bool) or None if unchanged."""
    old_s, new_s = _norm(old_raw), _norm(new_raw)
    if old_s == new_s:
        return None
    if not old_s or not new_s:
        return (old_s, new_s, True)
    old_d, new_d = pd.to_datetime(old_s, errors="coerce"), pd.to_datetime(new_s, errors="coerce")
    if pd.isna(old_d) or pd.isna(new_d):
        return (old_s, new_s, True)
    return (old_s, new_s, abs((new_d - old_d).days) >= DATE_SUBSTANTIAL_DAYS)


def _phase_rank(clean_label):
    return PHASE_RANK_FOR_SCORING.get(clean_label)


def _phase_contains_2_or_3(raw_phases):
    parts = {p.strip().upper() for p in str(raw_phases or "").split("|") if p.strip()}
    return bool(parts & {"PHASE2", "PHASE3"})


def _row(detected_date, entity_type, nct_id, canonical_drug_name, sponsor_or_company,
         change_type, old_value, new_value, importance, source, needs_review):
    return {
        "detected_date": detected_date,
        "entity_type": entity_type,
        "nct_id": nct_id or "",
        "canonical_drug_name": canonical_drug_name or "",
        "sponsor_or_company": sponsor_or_company or "",
        "change_type": change_type,
        "old_value": "" if old_value is None else old_value,
        "new_value": "" if new_value is None else new_value,
        "importance": importance,
        "source": source,
        "needs_review": bool(needs_review),
    }


def build_drug_lookup(annotated_df):
    """nct_id -> developed_drug, from a pipeline_annotated.csv-shaped DataFrame."""
    if annotated_df is None or annotated_df.empty:
        return {}
    lookup = {}
    for _, r in annotated_df.iterrows():
        drug = r.get("developed_drug")
        if pd.notna(drug) and str(drug).strip():
            lookup[r["nct_id"]] = str(drug).strip()
    return lookup


def detect_trial_level_changes(old_trials_df, new_trials_df, drug_lookup_new, drug_lookup_old,
                                detected_date, source):
    rows = []
    if old_trials_df is None or old_trials_df.empty:
        return rows  # nothing to compare a first-ever refresh against

    old_by_id = old_trials_df.set_index("NCT Number", drop=False)
    new_by_id = new_trials_df.set_index("NCT Number", drop=False)
    old_ids, new_ids = set(old_by_id.index), set(new_by_id.index)

    for nct_id in sorted(new_ids - old_ids):
        new_row = new_by_id.loc[nct_id]
        raw_phases = new_row.get("Phases", "")
        importance = "High" if _phase_contains_2_or_3(raw_phases) else "Medium"
        rows.append(_row(
            detected_date, "trial", nct_id, drug_lookup_new.get(nct_id, ""), new_row.get("Sponsor", ""),
            "new_trial", "", clean_phase(raw_phases), importance, source, importance == "High",
        ))

    for nct_id in sorted(old_ids - new_ids):
        old_row = old_by_id.loc[nct_id]
        rows.append(_row(
            detected_date, "trial", nct_id, drug_lookup_old.get(nct_id, ""), old_row.get("Sponsor", ""),
            "trial_disappeared", clean_status(old_row.get("Study Status", "")), "", "Medium", source, True,
        ))

    for nct_id in sorted(old_ids & new_ids):
        old_row, new_row = old_by_id.loc[nct_id], new_by_id.loc[nct_id]
        drug_name = drug_lookup_new.get(nct_id, drug_lookup_old.get(nct_id, ""))
        sponsor_new = new_row.get("Sponsor", "")

        old_status, new_status = clean_status(old_row.get("Study Status", "")), clean_status(new_row.get("Study Status", ""))
        if old_status != new_status:
            rows.append(_row(detected_date, "trial", nct_id, drug_name, sponsor_new,
                              "status_change", old_status, new_status, "Medium", source, False))

        old_phase, new_phase = clean_phase(old_row.get("Phases", "")), clean_phase(new_row.get("Phases", ""))
        if old_phase != new_phase:
            old_rank, new_rank = _phase_rank(old_phase), _phase_rank(new_phase)
            advancement = old_rank is not None and new_rank is not None and new_rank > old_rank
            rows.append(_row(detected_date, "trial", nct_id, drug_name, sponsor_new,
                              "phase_change", old_phase, new_phase,
                              "High" if advancement else "Medium", source, advancement))

        enr = _enrollment_change(old_row.get("Enrollment"), new_row.get("Enrollment"))
        if enr is not None:
            old_v, new_v, substantial = enr
            rows.append(_row(detected_date, "trial", nct_id, drug_name, sponsor_new,
                              "enrollment_change", old_v, new_v, "Medium" if substantial else "Low", source, False))

        pcd = _date_change(old_row.get("Primary Completion Date"), new_row.get("Primary Completion Date"))
        if pcd is not None:
            old_v, new_v, substantial = pcd
            rows.append(_row(detected_date, "trial", nct_id, drug_name, sponsor_new,
                              "primary_completion_date_change", old_v, new_v,
                              "Medium" if substantial else "Low", source, False))

        cd = _date_change(old_row.get("Completion Date"), new_row.get("Completion Date"))
        if cd is not None:
            old_v, new_v, substantial = cd
            rows.append(_row(detected_date, "trial", nct_id, drug_name, sponsor_new,
                              "completion_date_change", old_v, new_v,
                              "Medium" if substantial else "Low", source, False))

        old_sponsor, new_sponsor = _norm(old_row.get("Sponsor")), _norm(new_row.get("Sponsor"))
        if old_sponsor.casefold() != new_sponsor.casefold():
            rows.append(_row(detected_date, "trial", nct_id, drug_name, new_sponsor,
                              "sponsor_change", old_sponsor, new_sponsor, "Medium", source, True))

        old_results, new_results = _norm(old_row.get("Study Results")), _norm(new_row.get("Study Results"))
        if old_results.upper() != "YES" and new_results.upper() == "YES":
            rows.append(_row(detected_date, "trial", nct_id, drug_name, sponsor_new,
                              "results_posted", old_results, new_results, "High", source, True))

    return rows


def detect_drug_level_changes(old_drugs_df, new_drugs_df, detected_date, source):
    rows = []
    if old_drugs_df is None or old_drugs_df.empty or "display_name" not in old_drugs_df.columns:
        return rows
    if new_drugs_df is None or new_drugs_df.empty or "display_name" not in new_drugs_df.columns:
        return rows

    old_by_name = old_drugs_df.set_index("display_name", drop=False)
    new_by_name = new_drugs_df.set_index("display_name", drop=False)
    old_names, new_names = set(old_by_name.index), set(new_by_name.index)

    for name in sorted(new_names - old_names):
        row = new_by_name.loc[name]
        rows.append(_row(detected_date, "drug", "", name, row.get("sponsor", ""),
                          "new_drug", "", row.get("phase_reached", ""), "Medium", source, False))

    for name in sorted(old_names & new_names):
        old_row, new_row = old_by_name.loc[name], new_by_name.loc[name]
        old_phase, new_phase = _norm(old_row.get("phase_reached")), _norm(new_row.get("phase_reached"))
        if old_phase != new_phase:
            old_rank, new_rank = _phase_rank(old_phase), _phase_rank(new_phase)
            advancement = old_rank is not None and new_rank is not None and new_rank > old_rank
            rows.append(_row(detected_date, "drug", "", name, new_row.get("sponsor", ""),
                              "highest_drug_phase_change", old_phase, new_phase,
                              "High" if advancement else "Medium", source, advancement))

    return rows


def detect_changes(old_trials_df, new_trials_df, old_drugs_df, new_drugs_df,
                    old_annotated_df, new_annotated_df, detected_date, source):
    """Returns a DataFrame shaped exactly like outputs/pipeline_changes.csv."""
    drug_lookup_new = build_drug_lookup(new_annotated_df)
    drug_lookup_old = build_drug_lookup(old_annotated_df)

    rows = []
    rows.extend(detect_trial_level_changes(
        old_trials_df, new_trials_df, drug_lookup_new, drug_lookup_old, detected_date, source
    ))
    rows.extend(detect_drug_level_changes(old_drugs_df, new_drugs_df, detected_date, source))

    return pd.DataFrame(rows, columns=CHANGES_COLUMNS)
