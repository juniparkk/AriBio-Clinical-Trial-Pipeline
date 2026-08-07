# ============================================================
# CTGOV_SNAPSHOT — validation + immutable snapshot storage
#
# Two responsibilities, kept together because they share the
# "previous snapshot" concept:
#
#   1. validate_dataframe() — the accept/reject gate a freshly
#      normalized dataset must pass BEFORE it's allowed to become a
#      new snapshot or touch trials.csv. Mirrors this project's
#      long-standing rule (see Phase 0/1A/1B audits): never silently
#      accept or fabricate data — abort loudly and preserve whatever
#      was already known-good.
#
#   2. Snapshot read/write — data/snapshots/trials_<timestamp>.csv,
#      NEVER overwritten (timestamped to the second, not just the
#      date, so two refreshes on the same day can't collide — a
#      stricter guarantee than the suggested date-only naming), plus
#      a small latest_snapshot.json pointer used both to find the
#      most recent good snapshot and to feed the next run's
#      row-count sanity check.
#
#      data/raw/clinicaltrials/<run_id>/ stores the raw API JSON +
#      fetch metadata for EVERY attempt (including ones that later
#      fail validation) — an audit trail, not a "known-good" store.
# ============================================================

import json
import os
import re
from datetime import datetime, timezone

RAW_DIR = os.path.join("data", "raw", "clinicaltrials")
SNAPSHOT_DIR = os.path.join("data", "snapshots")
LATEST_POINTER_PATH = os.path.join(SNAPSHOT_DIR, "latest_snapshot.json")

NCT_ID_RE = re.compile(r"^NCT\d{8}$")

# A new snapshot's row count must fall within [MIN_RATIO, MAX_RATIO]
# of the previous good snapshot's row count. MIN_RATIO guards against
# a truncated/partial fetch silently passing as "the new normal".
# MAX_RATIO guards against a scope bug (e.g. the condition filter
# being dropped) ballooning the dataset. Real day-to-day trial counts
# move by a handful of studies, not double digits of percent, so this
# is deliberately loose — it's a sanity backstop, not a tight bound.
MIN_RATIO = 0.85
MAX_RATIO = 1.5

REQUIRED_COLUMNS = [
    "NCT Number", "Study Title", "Study URL", "Acronym", "Study Status",
    "Brief Summary", "Study Results", "Conditions", "Interventions",
    "Primary Outcome Measures", "Secondary Outcome Measures",
    "Other Outcome Measures", "Sponsor", "Collaborators", "Sex", "Age",
    "Phases", "Enrollment", "Funder Type", "Study Type", "Study Design",
    "Other IDs", "Start Date", "Primary Completion Date",
    "Completion Date", "First Posted", "Results First Posted",
    "Last Update Posted", "Locations", "Study Documents",
]


def validate_dataframe(df, previous_row_count=None):
    """Returns (ok: bool, errors: list[str]).

    Every check runs (doesn't short-circuit) so a single failed
    refresh reports everything wrong at once, not one error at a time
    across repeated runs.
    """
    errors = []

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")

    if len(df) == 0:
        errors.append("Normalized dataset is empty (0 rows)")

    if "NCT Number" in df.columns:
        nct_ids = df["NCT Number"].astype(str)

        invalid = nct_ids[~nct_ids.str.match(NCT_ID_RE)]
        if len(invalid) > 0:
            sample = invalid.head(5).tolist()
            errors.append(f"{len(invalid)} invalid NCT ID(s), e.g. {sample}")

        dupes = nct_ids[nct_ids.duplicated()]
        if len(dupes) > 0:
            sample = sorted(set(dupes.head(5).tolist()))
            errors.append(f"{len(dupes)} duplicate NCT ID(s), e.g. {sample}")

    if previous_row_count is not None and previous_row_count > 0 and len(df) > 0:
        ratio = len(df) / previous_row_count
        if ratio < MIN_RATIO or ratio > MAX_RATIO:
            errors.append(
                f"Row count {len(df)} is outside the expected range "
                f"[{MIN_RATIO * previous_row_count:.0f}, {MAX_RATIO * previous_row_count:.0f}] "
                f"given the previous snapshot's {previous_row_count} rows"
            )

    return (len(errors) == 0, errors)


def _utc_now_compact():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")


def write_raw_snapshot(studies, fetch_meta, run_id=None):
    """Writes raw API JSON + fetch metadata for one fetch attempt.

    Written UNCONDITIONALLY, before validation — this is the audit
    trail requirement (item 6), not the "known-good" snapshot store.
    Returns the run directory path.
    """
    run_id = run_id or _utc_now_compact()
    run_dir = os.path.join(RAW_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "studies.json"), "w") as f:
        json.dump(studies, f)

    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(fetch_meta, f, indent=2, default=str)

    return run_dir


def get_latest_snapshot_info():
    """Reads the latest_snapshot.json pointer, or None if no snapshot exists yet."""
    if not os.path.exists(LATEST_POINTER_PATH):
        return None
    with open(LATEST_POINTER_PATH) as f:
        return json.load(f)


def write_validated_snapshot(df, snapshot_meta, run_id=None):
    """Writes an immutable, VALIDATED snapshot CSV + updates the latest pointer.

    Only ever called after validate_dataframe() has passed — the
    caller (run_pipeline.py) is responsible for not calling this on
    invalid data. Filename is timestamped to the second so repeated
    same-day refreshes never collide/overwrite.
    """
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)

    run_id = run_id or _utc_now_compact()
    filename = f"trials_{run_id}.csv"
    csv_path = os.path.join(SNAPSHOT_DIR, filename)

    if os.path.exists(csv_path):
        raise FileExistsError(
            f"Refusing to overwrite existing snapshot: {csv_path}"
        )

    df.to_csv(csv_path, index=False)

    meta = dict(snapshot_meta)
    meta["snapshot_filename"] = filename
    meta["normalized_row_count"] = len(df)
    meta_path = os.path.join(SNAPSHOT_DIR, f"trials_{run_id}.meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    with open(LATEST_POINTER_PATH, "w") as f:
        json.dump(
            {
                "snapshot_filename": filename,
                "csv_path": csv_path,
                "row_count": len(df),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            f,
            indent=2,
        )

    return csv_path
