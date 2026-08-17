# ============================================================
# ARIBIO_WATCHLIST — loader for config/aribio_watchlist.yaml
#
# Deliberately defensive: a missing or malformed watchlist file must
# never crash the refresh pipeline. Falling back to DEFAULT_WATCHLIST
# (which mirrors the checked-in YAML's AR1001 seed) keeps
# competitive_attention.py's scoring behavior unchanged if the config
# file is ever temporarily absent — this is a config file, not a
# secret, so "missing" is treated as "use the built-in default,"
# not an error.
# ============================================================

import os

import yaml

DEFAULT_PATH = os.path.join("config", "aribio_watchlist.yaml")

DEFAULT_WATCHLIST = {
    "aribio_assets": [
        {
            "name": "AR1001",
            "is_primary": True,
            "sponsor": "AriBio Co., Ltd.",
            "modality": "Small Molecule",
            "target_pathways": ["Amyloid", "Tau", "Neuroprotection"],
            "therapeutic_purpose_class": "DTT",
            "phase_reached": "Phase 3",
            "route": "Oral",
            "reference_sex": "ALL",
            "reference_age": "ADULT, OLDER_ADULT",
        }
    ],
    "priority_phases": ["Phase 3", "Phase 2"],
    "priority_modalities": ["Small Molecule"],
    "priority_pathways": [],
    "priority_routes": ["Oral"],
    "priority_endpoints": [],
    "priority_biomarkers": [],
    "competitor_companies": [
        "Annovis", "Alzheon", "Anavex", "AB Science", "Cognition Therapeutics",
        "Eli Lilly", "Roche", "Biogen", "Eisai", "Janssen", "TauRx Therapeutics",
    ],
    "alert_thresholds": {
        "critical_score": 80,
        "high_score": 60,
        "medium_score": 35,
        "primary_completion_imminent_days": 30,
        "primary_completion_soon_days": 90,
        "recently_completed_days": 30,
        "major_delay_days": 90,
        "low_confidence_score_penalty": 10,
    },
}


def load_watchlist(path=DEFAULT_PATH):
    """Loads and validates config/aribio_watchlist.yaml.

    Returns a dict with every DEFAULT_WATCHLIST key guaranteed present
    (missing keys/sections in the file fall back to the default for
    just that section, so a partially-filled-in watchlist still works).
    """
    watchlist = {k: v for k, v in DEFAULT_WATCHLIST.items()}

    if not os.path.exists(path):
        return watchlist

    try:
        with open(path) as f:
            loaded = yaml.safe_load(f)
    except (yaml.YAMLError, OSError):
        return watchlist

    if not isinstance(loaded, dict):
        return watchlist

    for key in DEFAULT_WATCHLIST:
        if key in loaded and loaded[key] is not None:
            watchlist[key] = loaded[key]

    return watchlist


def get_primary_asset(watchlist):
    """Returns the aribio_assets entry with is_primary: true, or the
    first entry if none is marked, or an empty dict if the list itself
    is empty (scoring degrades gracefully — see competitive_attention.py)."""
    assets = watchlist.get("aribio_assets") or []
    for asset in assets:
        if asset.get("is_primary"):
            return asset
    return assets[0] if assets else {}
