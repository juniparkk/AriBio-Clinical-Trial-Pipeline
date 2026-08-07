# ============================================================
# RUN_PIPELINE — safe, automatic ClinicalTrials.gov data refresh
#
#     python3 run_pipeline.py --refresh
#
# Fetches fresh Alzheimer's-disease interventional trial data from
# the official ClinicalTrials.gov API v2 (no scraping), normalizes it
# into the exact schema trials.csv already uses, validates it, and
# only THEN rebuilds the dashboard (pipeline_drugs.csv,
# pipeline_overview.html, etc.) by re-running the existing
# pipeline_viz.py unchanged. Detects meaningful changes against the
# previous refresh (outputs/pipeline_changes.csv) and runs the full
# test suite afterward.
#
# Scope preserved exactly as-is (see ctgov_client.py): Alzheimer
# Disease condition, interventional studies only, every status
# (recruiting/completed/terminated/withdrawn/...), every phase
# (NA/Early Phase 1/.../Phase 4/combined) — no silent narrowing.
#
# Safety contract:
#   - A fetch failure (network/timeout/malformed JSON) aborts before
#     anything on disk changes.
#   - A validation failure (bad schema, invalid/duplicate NCT IDs,
#     empty response, implausible row-count swing) aborts BEFORE
#     trials.csv or any dashboard output is touched — the previous
#     good snapshot and the previous trials.csv are left exactly as
#     they were.
#   - trials.csv is only overwritten after validation passes, and
#     only from a snapshot that has itself already been written
#     immutably to data/snapshots/ first.
#   - If pipeline_viz.py itself fails on the new data (a rebuild bug,
#     not a data problem), trials.csv is rolled back to what it was
#     before this run, and pipeline_changes.csv is left untouched
#     (never regenerated from a failed/partial rebuild).
#
# Change detection (ctgov_changes.py) compares trials.csv /
# pipeline_drugs.csv / pipeline_annotated.csv as they existed on disk
# BEFORE this refresh (i.e. whatever the LAST successful refresh
# committed to git) against the freshly rebuilt versions. This is
# deliberately NOT based on data/snapshots/ (.gitignore'd, does not
# persist across GitHub Actions runs) — the git-tracked files are the
# only state guaranteed to survive between CI runs.
#
# No AI "Needs Attention" summary yet — this phase is change
# DETECTION only, per project scope.
# ============================================================

import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone

import pandas as pd

import ctgov_changes
import ctgov_client
import ctgov_normalize
import ctgov_snapshot

TRIALS_CSV = "trials.csv"
TRIALS_CSV_BACKUP = "trials.csv.pre_refresh_backup"
DRUGS_CSV = "pipeline_drugs.csv"
ANNOTATED_CSV = "pipeline_annotated.csv"
CHANGES_CSV = os.path.join("outputs", "pipeline_changes.csv")

TEST_FILES = [
    "test_classification.py",
    "test_competitive_intelligence.py",
    "test_dashboard_table.py",
    "test_nih_reference.py",
    "test_scientific_classification.py",
    "test_ctgov_pipeline.py",
    "test_ctgov_changes.py",
]


def _print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def _utc_now_compact():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")


def _read_csv_if_exists(path):
    return pd.read_csv(path, low_memory=False) if os.path.exists(path) else None


def run_refresh():
    run_id = _utc_now_compact()
    fetch_timestamp = datetime.now(timezone.utc).isoformat()

    _print_header("STEP 1/7: Fetching ClinicalTrials.gov data (API v2)")
    ctgov_data_timestamp = ctgov_client.fetch_data_version()
    print(f"ct.gov data timestamp (from /api/v2/version): {ctgov_data_timestamp}")

    try:
        studies, fetch_meta = ctgov_client.fetch_all_studies()
    except ctgov_client.CtGovFetchError as e:
        print(f"\nABORTED: fetch failed — {e}")
        print("Nothing on disk was changed. Previous trials.csv and snapshots are untouched.")
        return 1

    print(f"Pages fetched: {fetch_meta['pages_fetched']}")
    print(f"API records retrieved: {fetch_meta['api_records_retrieved']}")
    print(f"ct.gov reported totalCount: {fetch_meta['total_count_reported']}")
    print(f"Query URL (page 1): {fetch_meta['query_url_example']}")

    full_fetch_meta = {
        "run_id": run_id,
        "fetch_timestamp": fetch_timestamp,
        "ctgov_data_timestamp": ctgov_data_timestamp,
        **fetch_meta,
    }

    _print_header("STEP 2/7: Saving raw snapshot (audit trail)")
    raw_dir = ctgov_snapshot.write_raw_snapshot(studies, full_fetch_meta, run_id=run_id)
    print(f"Raw API response saved to: {raw_dir}")

    _print_header("STEP 3/7: Normalizing to trials.csv schema")
    df = ctgov_normalize.normalize_studies(studies)
    print(f"Normalized row count: {len(df)}")

    # The "previous" state for BOTH the row-count sanity check below and
    # change detection later is whatever trials.csv/pipeline_drugs.csv/
    # pipeline_annotated.csv already contain on disk right now — i.e.
    # the git-tracked output of the last successful refresh. This is
    # read ONCE, up front, before anything overwrites those files.
    old_trials_df = _read_csv_if_exists(TRIALS_CSV)
    old_drugs_df = _read_csv_if_exists(DRUGS_CSV)
    old_annotated_df = _read_csv_if_exists(ANNOTATED_CSV)

    previous_row_count = len(old_trials_df) if old_trials_df is not None else None
    if old_trials_df is not None:
        print(f"Previous trials.csv on disk: {previous_row_count} rows")
    else:
        print("No previous trials.csv found — this is the first run.")

    _print_header("STEP 4/7: Validating")
    ok, errors = ctgov_snapshot.validate_dataframe(df, previous_row_count=previous_row_count)
    full_fetch_meta["validation_status"] = "passed" if ok else "failed"
    full_fetch_meta["validation_errors"] = errors

    if not ok:
        # Re-write metadata.json in the raw run dir with the failure
        # recorded, but do NOT create a validated snapshot and do NOT
        # touch trials.csv — the previous good snapshot stays current.
        ctgov_snapshot.write_raw_snapshot(studies, full_fetch_meta, run_id=run_id)
        print("VALIDATION FAILED:")
        for err in errors:
            print(f"  - {err}")
        print("\nABORTED: previous snapshot and trials.csv are preserved unchanged.")
        print(f"Raw (rejected) response kept for inspection at: {raw_dir}")
        return 1

    print("Validation passed:")
    print(f"  - Required columns present: yes")
    print(f"  - NCT IDs valid and unique: yes")
    print(f"  - Row count within expected range of previous trials.csv: yes")

    snapshot_csv_path = ctgov_snapshot.write_validated_snapshot(df, full_fetch_meta, run_id=run_id)
    print(f"Validated snapshot written to: {snapshot_csv_path}")

    _print_header("STEP 5/7: Rebuilding dashboard")
    had_previous_trials_csv = old_trials_df is not None
    try:
        if had_previous_trials_csv:
            shutil.copy2(TRIALS_CSV, TRIALS_CSV_BACKUP)

        shutil.copy2(snapshot_csv_path, TRIALS_CSV)
        print(f"{TRIALS_CSV} updated from new snapshot.")

        result = subprocess.run(
            [sys.executable, "pipeline_viz.py"],
            capture_output=True,
            text=True,
        )
        print(result.stdout[-4000:])
        if result.returncode != 0:
            print(result.stderr[-4000:])
            raise RuntimeError(f"pipeline_viz.py exited with code {result.returncode}")

        print("pipeline_viz.py completed successfully (pipeline_drugs.csv, pipeline_overview.html regenerated).")
    except Exception as e:
        print(f"\nDASHBOARD REBUILD FAILED: {e}")
        if had_previous_trials_csv:
            shutil.copy2(TRIALS_CSV_BACKUP, TRIALS_CSV)
            print(f"{TRIALS_CSV} rolled back to its pre-refresh state.")
        print("The new snapshot remains saved in data/snapshots/ for inspection, "
              "but was NOT promoted to production trials.csv. outputs/pipeline_changes.csv "
              "was NOT regenerated.")
        return 1
    finally:
        if os.path.exists(TRIALS_CSV_BACKUP):
            os.remove(TRIALS_CSV_BACKUP)

    _print_header("STEP 6/7: Detecting changes vs. previous refresh")
    new_drugs_df = _read_csv_if_exists(DRUGS_CSV)
    new_annotated_df = _read_csv_if_exists(ANNOTATED_CSV)
    detected_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    change_source = f"ClinicalTrials.gov API v2 (snapshot {run_id})"

    changes_df = ctgov_changes.detect_changes(
        old_trials_df, df, old_drugs_df, new_drugs_df, old_annotated_df, new_annotated_df,
        detected_date=detected_date, source=change_source,
    )
    os.makedirs(os.path.dirname(CHANGES_CSV) or ".", exist_ok=True)
    changes_df.to_csv(CHANGES_CSV, index=False)

    if old_trials_df is None:
        print("No previous trials.csv — change detection skipped (nothing to compare against).")
    print(f"{len(changes_df)} change(s) detected, written to {CHANGES_CSV}")
    if len(changes_df):
        counts = changes_df["change_type"].value_counts()
        for change_type, count in counts.items():
            print(f"  - {change_type}: {count}")
        importance_counts = changes_df["importance"].value_counts()
        print(f"  Importance: " + ", ".join(f"{k}={v}" for k, v in importance_counts.items()))

    _print_header("STEP 7/7: Running tests")
    all_passed = True
    for test_file in TEST_FILES:
        if not os.path.exists(test_file):
            continue
        result = subprocess.run([sys.executable, test_file], capture_output=True, text=True)
        status = "PASS" if result.returncode == 0 else "FAIL"
        if result.returncode != 0:
            all_passed = False
        print(f"[{status}] {test_file}")
        if result.returncode != 0:
            print(result.stdout[-2000:])
            print(result.stderr[-2000:])

    _print_header("REFRESH SUMMARY")
    print(f"Fetch timestamp:            {fetch_timestamp}")
    print(f"ct.gov data timestamp:      {ctgov_data_timestamp}")
    print(f"API query (page 1):         {fetch_meta['query_url_example']}")
    print(f"API records retrieved:      {fetch_meta['api_records_retrieved']}")
    print(f"Normalized row count:       {len(df)}")
    print(f"Snapshot filename:          {snapshot_csv_path}")
    print(f"Changes detected:           {len(changes_df)} ({CHANGES_CSV})")
    print(f"All tests passed:           {all_passed}")

    return 0 if all_passed else 1


def main():
    if "--refresh" not in sys.argv:
        print(__doc__ if __doc__ else "")
        print("Usage: python3 run_pipeline.py --refresh")
        return 0
    return run_refresh()


if __name__ == "__main__":
    sys.exit(main())
