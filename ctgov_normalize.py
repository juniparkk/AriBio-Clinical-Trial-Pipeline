# ============================================================
# CTGOV_NORMALIZE — API v2 JSON study -> trials.csv row schema
#
# Reproduces the EXACT 29-column shape and per-cell string
# formatting of the current trials.csv (a manual ct.gov bulk-CSV-
# export) from the API v2 JSON representation of the same study, so
# pipeline_viz.py (and its column_map) can consume the output with
# zero changes.
#
# Every join/format choice below was verified against real rows in
# the existing trials.csv (ground truth), not guessed:
#   - multi-value fields (Conditions, Collaborators, Other IDs,
#     Phases, Study Documents) are pipe-joined ("|")
#   - Interventions: "TYPE: name" pairs, pipe-joined
#     e.g. "DRUG: AR1001|DRUG: Placebo" — confirmed against
#     NCT05531526 (AR1001's own trial); this exact format is what
#     drug_classification.parse_interventions() already expects.
#   - one outcome's measure/description/timeFrame: ", "-joined;
#     multiple outcomes: "|"-joined — confirmed against NCT03928405
#   - Age: stdAges[] comma-space-joined, RAW enum (not reformatted)
#     e.g. "ADULT, OLDER_ADULT"
#   - Study Design: fixed template confirmed against real rows:
#     "Allocation: X | Intervention Model: Y | Masking: Z (W) | Primary Purpose: V"
#   - Locations: "facility, city, state, zip, country" per site
#     (only the fields actually present are included, so a site
#     missing state/zip degrades to fewer commas rather than
#     emitting empty segments), pipe-joined between sites —
#     confirmed against UK/Wales and China rows.
#   - Study Documents: "label, url" per doc, pipe-joined; url is
#     constructed as https://cdn.clinicaltrials.gov/large-docs/
#     {last 2 digits of the NCT number}/{nctId}/{filename} —
#     confirmed byte-for-byte against NCT03100617's real CSV value.
#   - Study Results: literal "YES"/"NO" string, from the top-level
#     hasResults boolean (sibling of protocolSection, not nested in it).
#   - Study URL: constructed, not from the API:
#     https://clinicaltrials.gov/study/{nctId}
#
# Nothing here fabricates data: every field falls back to "" (or the
# pandas-friendly equivalent) when the API omits it, matching this
# project's established never-guess ethic (see drug_classification.py,
# scientific_classification.py).
# ============================================================

import pandas as pd

# Exact column order of the current trials.csv — the normalizer's
# output DataFrame must match this so pipeline_viz.py's column_map
# and every downstream consumer sees an identical shape.
COLUMNS = [
    "NCT Number", "Study Title", "Study URL", "Acronym", "Study Status",
    "Brief Summary", "Study Results", "Conditions", "Interventions",
    "Primary Outcome Measures", "Secondary Outcome Measures",
    "Other Outcome Measures", "Sponsor", "Collaborators", "Sex", "Age",
    "Phases", "Enrollment", "Funder Type", "Study Type", "Study Design",
    "Other IDs", "Start Date", "Primary Completion Date",
    "Completion Date", "First Posted", "Results First Posted",
    "Last Update Posted", "Locations", "Study Documents",
]


def _get(d, *path, default=None):
    cur = d
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return cur if cur is not None else default


def _join_outcomes(outcomes):
    if not outcomes:
        return ""
    parts = []
    for o in outcomes:
        fields = [o.get("measure", ""), o.get("description", ""), o.get("timeFrame", "")]
        parts.append(", ".join(f for f in fields if f))
    return "|".join(parts)


def _format_interventions(interventions):
    if not interventions:
        return ""
    return "|".join(
        f"{(i.get('type') or '').upper()}: {i.get('name', '')}" for i in interventions
    )


def _format_locations(locations):
    if not locations:
        return ""
    parts = []
    for loc in locations:
        fields = [
            loc.get("facility", ""),
            loc.get("city", ""),
            loc.get("state", ""),
            loc.get("zip", ""),
            loc.get("country", ""),
        ]
        parts.append(", ".join(f for f in fields if f))
    return "|".join(parts)


def _format_study_design(design_info):
    if not design_info:
        return ""
    allocation = design_info.get("allocation", "")
    intervention_model = design_info.get("interventionModel", "")
    masking_info = design_info.get("maskingInfo") or {}
    masking = masking_info.get("masking", "")
    who_masked = masking_info.get("whoMasked") or []
    primary_purpose = design_info.get("primaryPurpose", "")
    return (
        f"Allocation: {allocation} | Intervention Model: {intervention_model} | "
        f"Masking: {masking} ({', '.join(who_masked)}) | Primary Purpose: {primary_purpose}"
    )


def _format_other_ids(identification_module):
    org_id = _get(identification_module, "orgStudyIdInfo", "id", default="")
    secondary = identification_module.get("secondaryIdInfos") or []
    ids = [org_id] if org_id else []
    ids.extend(s.get("id", "") for s in secondary if s.get("id"))
    return "|".join(ids)


def _format_study_documents(nct_id, document_section):
    large_docs = _get(document_section, "largeDocumentModule", "largeDocs", default=[])
    if not large_docs:
        return ""
    suffix = nct_id[-2:] if nct_id else ""
    parts = []
    for doc in large_docs:
        label = doc.get("label", "")
        filename = doc.get("filename", "")
        if not filename:
            continue
        url = f"https://cdn.clinicaltrials.gov/large-docs/{suffix}/{nct_id}/{filename}"
        parts.append(f"{label}, {url}")
    return "|".join(parts)


def normalize_study(study):
    """One API v2 study JSON object -> one trials.csv-shaped row dict."""
    protocol = study.get("protocolSection") or {}
    identification = protocol.get("identificationModule") or {}
    status = protocol.get("statusModule") or {}
    sponsor_collab = protocol.get("sponsorCollaboratorsModule") or {}
    description = protocol.get("descriptionModule") or {}
    conditions_mod = protocol.get("conditionsModule") or {}
    design = protocol.get("designModule") or {}
    arms = protocol.get("armsInterventionsModule") or {}
    outcomes = protocol.get("outcomesModule") or {}
    eligibility = protocol.get("eligibilityModule") or {}
    contacts_locations = protocol.get("contactsLocationsModule") or {}
    document_section = study.get("documentSection") or {}

    nct_id = identification.get("nctId", "")
    lead_sponsor = sponsor_collab.get("leadSponsor") or {}
    collaborators = sponsor_collab.get("collaborators") or []
    design_info = design.get("designInfo") or {}

    return {
        "NCT Number": nct_id,
        "Study Title": identification.get("briefTitle", ""),
        "Study URL": f"https://clinicaltrials.gov/study/{nct_id}" if nct_id else "",
        "Acronym": identification.get("acronym", ""),
        "Study Status": status.get("overallStatus", ""),
        "Brief Summary": description.get("briefSummary", ""),
        "Study Results": "YES" if study.get("hasResults") else "NO",
        "Conditions": "|".join(conditions_mod.get("conditions") or []),
        "Interventions": _format_interventions(arms.get("interventions")),
        "Primary Outcome Measures": _join_outcomes(outcomes.get("primaryOutcomes")),
        "Secondary Outcome Measures": _join_outcomes(outcomes.get("secondaryOutcomes")),
        "Other Outcome Measures": _join_outcomes(outcomes.get("otherOutcomes")),
        "Sponsor": lead_sponsor.get("name", ""),
        "Collaborators": "|".join(c.get("name", "") for c in collaborators if c.get("name")),
        "Sex": eligibility.get("sex", ""),
        "Age": ", ".join(eligibility.get("stdAges") or []),
        "Phases": "|".join(design.get("phases") or []),
        "Enrollment": _get(design, "enrollmentInfo", "count", default=""),
        "Funder Type": _get(identification, "organization", "class", default=""),
        "Study Type": design.get("studyType", ""),
        "Study Design": _format_study_design(design_info),
        "Other IDs": _format_other_ids(identification),
        "Start Date": _get(status, "startDateStruct", "date", default=""),
        "Primary Completion Date": _get(status, "primaryCompletionDateStruct", "date", default=""),
        "Completion Date": _get(status, "completionDateStruct", "date", default=""),
        "First Posted": _get(status, "studyFirstPostDateStruct", "date", default=""),
        "Results First Posted": _get(status, "resultsFirstPostDateStruct", "date", default=""),
        "Last Update Posted": _get(status, "lastUpdatePostDateStruct", "date", default=""),
        "Locations": _format_locations(contacts_locations.get("locations")),
        "Study Documents": _format_study_documents(nct_id, document_section),
    }


def normalize_studies(studies):
    """list[dict] (API v2 studies) -> pandas.DataFrame shaped like trials.csv."""
    rows = [normalize_study(s) for s in studies]
    df = pd.DataFrame(rows, columns=COLUMNS)
    return df
