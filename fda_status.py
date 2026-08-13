# ============================================================
# FDA STATUS REFERENCE
#
# Adds a genuinely separate "FDA status" signal to resolved_drugs_df —
# never derived from ClinicalTrials.gov trial status (see
# pipeline_viz.py's STATUS_MAP, where "APPROVED_FOR_MARKETING" already
# gets relabeled "FDA Approved" at the TRIAL level; that value rolling
# up into a drug's status_summary is exactly the trial-status/FDA-status
# conflation this module exists to stop repeating for anything new).
#
# Sourced entirely from a small, hand-curated reference file
# (data/reference/fda_status_reference.csv) — the same tier-1
# curated-override pattern scientific_classification.py already uses
# for drug_classification_overrides.csv — chosen deliberately over a
# live openFDA API integration: there are only a handful of
# FDA-approved-or-withdrawn Alzheimer's drugs that will ever appear in
# this dataset, so a small reviewed file is both less engineering and
# more trustworthy than automated name-matching against Drugs@FDA,
# which would still need per-row manual verification anyway.
#
# Absence from the reference file means "Unknown", never "Not FDA
# Approved" — silence in a hand-curated file is not evidence of
# non-approval, it just means nobody has looked yet.
# ============================================================

import pandas as pd

from drug_classification import normalize_text

REQUIRED_FDA_STATUS_COLUMNS = [
    "canonical_drug_name", "indication", "fda_status", "approval_type",
    "approval_date", "withdrawal_date", "application_status",
    "source_title", "source_url", "verified_date", "notes",
]

# Matches the vocabulary the FDA-status workflow is built around —
# any other value in the reference CSV is a data-entry error, not a
# new category to silently accept.
FDA_STATUS_VALUES = [
    "FDA Approved", "Not FDA Approved", "Under Review",
    "Approval Withdrawn", "Not Applicable", "Unknown",
]

_UNKNOWN_FDA_STATUS = {
    "canonical_drug_name": "", "fda_status": "Unknown", "indication": "",
    "approval_type": "", "approval_date": "", "withdrawal_date": "",
    "application_status": "", "source_title": "", "source_url": "",
    "verified_date": "", "notes": "",
}


def load_fda_status_reference(path):
    """
    Read data/reference/fda_status_reference.csv into a dict keyed by
    normalized canonical_drug_name — missing-file-tolerant, same as
    scientific_classification.load_drug_classification_overrides(): a
    missing file degrades to "no FDA data" ({}), never a crash.
    """
    try:
        raw_df = pd.read_csv(path, dtype=str)
    except FileNotFoundError:
        return {}

    missing = [c for c in REQUIRED_FDA_STATUS_COLUMNS if c not in raw_df.columns]
    if missing:
        raise ValueError(f"fda_status_reference.csv at {path!r} is missing required column(s): {missing}")

    raw_df = raw_df.fillna("")

    reference = {}
    for _, row in raw_df.iterrows():
        key = normalize_text(row["canonical_drug_name"])
        if not key:
            continue
        status = str(row["fda_status"]).strip()
        if status not in FDA_STATUS_VALUES:
            raise ValueError(
                f"fda_status_reference.csv row {row['canonical_drug_name']!r} has "
                f"fda_status={status!r}, which is not one of {FDA_STATUS_VALUES}"
            )
        reference[key] = {col: str(row[col]).strip() for col in REQUIRED_FDA_STATUS_COLUMNS}
    return reference


def match_drug_to_fda_status(display_name, synonyms, fda_reference):
    """
    Try the dashboard drug's own display_name, then each known synonym,
    against the FDA reference — normalized-exact only, same
    no-fuzzy-matching-for-automatic-decisions rule as
    scientific_classification.match_drug_to_nih(). A drug with no
    match gets Unknown, never a guess.
    """
    for name in [display_name] + list(synonyms or []):
        key = normalize_text(name)
        if key in fda_reference:
            return fda_reference[key]
    return dict(_UNKNOWN_FDA_STATUS)
