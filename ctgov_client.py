# ============================================================
# CTGOV_CLIENT — ClinicalTrials.gov API v2 HTTP layer
#
# Pure network/pagination code: build the query, page through
# /api/v2/studies with the ct.gov cursor (pageToken) protocol, and
# hand back the raw list of study JSON objects. No normalization,
# no file I/O, no pandas — kept separate so ctgov_normalize.py and
# the tests can work with plain dicts/lists.
#
# Uses stdlib urllib.request rather than adding `requests` as a new
# dependency — this project currently only requires pandas/plotly,
# and urllib is trivially mockable in tests via the `fetch_fn`
# injection point on every function here (no network access needed
# to test pagination/error handling).
#
# SCOPE (must exactly match the existing trials.csv export — see
# run_pipeline.py's docstring for the full requirement):
#   - condition: Alzheimer Disease
#   - study type: INTERVENTIONAL only
#   - NO overallStatus filter (recruiting AND completed AND
#     terminated AND withdrawn ... every status, same as today)
#   - NO phase filter (NA / Early Phase 1 / Phase 4 / combined
#     phases all included, same as today)
# ============================================================

import json
import urllib.error
import urllib.parse
import urllib.request

API_BASE = "https://clinicaltrials.gov/api/v2"

# Condition + study-type scope. This is the ONLY scope restriction —
# no status/phase filters — matching the current dataset's inclusion
# criteria exactly (reverse-engineered from trials.csv: 100%
# INTERVENTIONAL, all 9 ct.gov status values present).
QUERY_COND = "Alzheimer Disease"
FILTER_ADVANCED = "AREA[StudyType]INTERVENTIONAL"

# Module-level field restriction: pulls back only what the normalizer
# needs (see ctgov_normalize.py's FIELD_MAP docstring), which keeps
# the raw JSON payload much smaller than a full study record — this
# matters both for fetch time and for the "don't commit huge raw API
# responses" git-hygiene requirement.
FIELDS = [
    "protocolSection.identificationModule",
    "protocolSection.statusModule",
    "protocolSection.sponsorCollaboratorsModule",
    "protocolSection.descriptionModule",
    "protocolSection.conditionsModule",
    "protocolSection.designModule",
    "protocolSection.armsInterventionsModule",
    "protocolSection.outcomesModule",
    "protocolSection.eligibilityModule",
    "protocolSection.contactsLocationsModule",
    "documentSection",
    "hasResults",
]

PAGE_SIZE = 1000
DEFAULT_TIMEOUT = 30


class CtGovFetchError(Exception):
    """Raised for any network, timeout, HTTP, or malformed-JSON failure.

    run_pipeline.py catches this at the top level and aborts the
    refresh WITHOUT touching trials.csv or any existing snapshot —
    a failed fetch must never partially rebuild the dashboard.
    """


def build_query_params(page_token=None):
    """The exact, fixed query parameters sent on every page request.

    Returned as a dict (not a URL) so run_pipeline.py can print it
    verbatim in the "exact API query" section of its report, and so
    tests can assert on it directly without string-parsing a URL.
    """
    params = {
        "query.cond": QUERY_COND,
        "filter.advanced": FILTER_ADVANCED,
        "fields": ",".join(FIELDS),
        "pageSize": str(PAGE_SIZE),
        "countTotal": "true",
    }
    if page_token:
        params["pageToken"] = page_token
    return params


def build_query_url(page_token=None):
    params = build_query_params(page_token)
    return f"{API_BASE}/studies?{urllib.parse.urlencode(params)}"


def _default_fetch(url, timeout=DEFAULT_TIMEOUT):
    """Real HTTP GET via stdlib urllib. Returns the raw response bytes."""
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as e:
        raise CtGovFetchError(f"HTTP {e.code} from {url}: {e.reason}") from e
    except urllib.error.URLError as e:
        raise CtGovFetchError(f"Network error fetching {url}: {e.reason}") from e
    except TimeoutError as e:
        raise CtGovFetchError(f"Timeout fetching {url}") from e


def fetch_all_studies(fetch_fn=None, max_pages=1000):
    """Pages through /api/v2/studies until nextPageToken is absent.

    fetch_fn(url) -> bytes : injection point for tests. Defaults to a
    real HTTP GET. Raising CtGovFetchError from fetch_fn simulates a
    network/timeout failure; returning malformed bytes simulates a
    malformed response — both are exercised directly in
    test_ctgov_pipeline.py without touching the network.

    Returns (studies: list[dict], meta: dict) where meta carries
    pages_fetched, total_count_reported (ct.gov's own countTotal, may
    differ slightly from len(studies) if the count shifts mid-fetch),
    and the first page's query params (for reporting).
    """
    fetch = fetch_fn or _default_fetch

    studies = []
    page_token = None
    pages_fetched = 0
    total_count_reported = None
    first_query_params = None

    while True:
        params = build_query_params(page_token)
        if first_query_params is None:
            first_query_params = params
        url = f"{API_BASE}/studies?{urllib.parse.urlencode(params)}"

        try:
            raw = fetch(url)
        except CtGovFetchError:
            raise
        except Exception as e:
            raise CtGovFetchError(f"Fetch failed for {url}: {e}") from e

        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, TypeError) as e:
            raise CtGovFetchError(f"Malformed JSON response from {url}: {e}") from e

        if not isinstance(payload, dict) or "studies" not in payload:
            raise CtGovFetchError(
                f"Unexpected response shape from {url}: missing 'studies' key"
            )

        page_studies = payload["studies"]
        if not isinstance(page_studies, list):
            raise CtGovFetchError(
                f"Unexpected response shape from {url}: 'studies' is not a list"
            )

        studies.extend(page_studies)
        pages_fetched += 1

        if total_count_reported is None and "totalCount" in payload:
            total_count_reported = payload["totalCount"]

        page_token = payload.get("nextPageToken")
        if not page_token:
            break

        if pages_fetched >= max_pages:
            raise CtGovFetchError(
                f"Exceeded max_pages={max_pages} while paginating {url} — "
                "aborting to avoid an unbounded fetch loop"
            )

    meta = {
        "pages_fetched": pages_fetched,
        "total_count_reported": total_count_reported,
        "api_records_retrieved": len(studies),
        "query_params": first_query_params,
        "query_url_example": f"{API_BASE}/studies?{urllib.parse.urlencode(first_query_params)}",
    }
    return studies, meta


def fetch_data_version(fetch_fn=None):
    """Best-effort call to /api/v2/version for ct.gov's own dataTimestamp.

    Non-fatal: returns None on any failure rather than aborting the
    whole refresh — this is metadata for the report, not something
    the pipeline depends on for correctness.
    """
    fetch = fetch_fn or _default_fetch
    url = f"{API_BASE}/version"
    try:
        raw = fetch(url)
        payload = json.loads(raw)
        return payload.get("dataTimestamp")
    except Exception:
        return None
