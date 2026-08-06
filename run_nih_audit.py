# ============================================================
# PHASE 1B — NIH REFERENCE DATASET AUDIT (audit-only, run separately
# from pipeline_viz.py)
#
# Deliberately does NOT import/run pipeline_viz.py — that would
# regenerate pipeline_overview.html and every pipeline_*.csv as a side
# effect of import (pipeline_viz.py has no __main__ guard), which Phase
# 1B explicitly must not do. Instead this reads the ALREADY-generated
# pipeline_drugs.csv as the resolved_drugs_df snapshot to audit against.
#
# Run with:
#     .venv/bin/python run_nih_audit.py
# ============================================================

import os

import pandas as pd

from nih_reference import (
    parse_nih_dataset,
    profile_nih_dataset,
    summarize_nih_dataset_shape,
    build_nih_match_audit,
    build_nih_conflict_audit,
)

NIH_CSV_PATH = "nih_data.csv"
RESOLVED_DRUGS_CSV_PATH = "pipeline_drugs.csv"

nih_df = parse_nih_dataset(NIH_CSV_PATH)
resolved_drugs_df = pd.read_csv(RESOLVED_DRUGS_CSV_PATH)

print("=== NIH REFERENCE DATASET AUDIT (Phase 1B) ===")
print(f"nih_data.csv: {len(nih_df)} agent-entry rows parsed")
print(f"pipeline_drugs.csv (resolved_drugs_df snapshot): {len(resolved_drugs_df)} drug rows")
print()

os.makedirs("outputs", exist_ok=True)

profile_df = profile_nih_dataset(nih_df)
profile_df.to_csv("outputs/nih_dataset_profile.csv", index=False)
print(f"=== SAVED: outputs/nih_dataset_profile.csv ({len(profile_df)} field rows) ===")

shape_summary = summarize_nih_dataset_shape(nih_df)
print("dataset shape summary (see NIH_INTEGRATION_PLAN.md for narrative):")
for k, v in shape_summary.items():
    print(f"  {k}: {v}")
print()

match_audit_df = build_nih_match_audit(nih_df, resolved_drugs_df)
match_audit_df.to_csv("outputs/nih_match_audit.csv", index=False)
print(f"=== SAVED: outputs/nih_match_audit.csv ({len(match_audit_df)} rows) ===")

nih_side = match_audit_df[match_audit_df["record_type"] == "nih_record"]
dash_side = match_audit_df[match_audit_df["record_type"] == "dashboard_only"]

print("NIH-side match tiers:")
print(nih_side["match_tier"].value_counts().to_string())
print()

n_matched_confident = nih_side["match_tier"].isin(["exact_canonical", "exact_alias", "normalized_exact"]).sum()
print(f"NIH agent rows with a confident dashboard match: {n_matched_confident} / {len(nih_side)} "
      f"({100 * n_matched_confident / len(nih_side):.1f}%)")
print()

print("Dashboard-side unmatched buckets:")
print(dash_side["dashboard_bucket"].value_counts().to_string())
print()

conflict_audit_df = build_nih_conflict_audit(nih_df, resolved_drugs_df, match_audit_df)
conflict_audit_df.to_csv("outputs/nih_conflict_audit.csv", index=False)
print(f"=== SAVED: outputs/nih_conflict_audit.csv ({len(conflict_audit_df)} matched-pair rows) ===")
print("conflict counts:")
for col in ["drug_type_conflict", "target_conflict", "company_conflict", "phase_conflict", "canonical_name_differs"]:
    n = int(conflict_audit_df[col].sum())
    print(f"  {col}: {n} / {len(conflict_audit_df)} ({100 * n / len(conflict_audit_df):.1f}%)" if len(conflict_audit_df) else f"  {col}: 0")
print()

print("=== DONE — no dashboard files, HTML, or classification logic were touched ===")
