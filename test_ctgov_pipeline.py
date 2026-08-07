# ============================================================
# TESTS for the ClinicalTrials.gov automatic-refresh pipeline
# (ctgov_client.py, ctgov_normalize.py, ctgov_snapshot.py,
# run_pipeline.py)
#
# Plain-Python tests (no pytest install needed, matching this
# project's existing test files) — run with:
#     .venv/bin/python test_ctgov_pipeline.py
#
# No real network access anywhere in this file: every ct.gov call is
# replaced with an injected fetch_fn, and every filesystem write is
# redirected to a temp directory via monkeypatched module globals.
# ============================================================

import json
import os
import shutil
import tempfile

import pandas as pd

import competitive_attention_viz
import ctgov_client
import ctgov_normalize
import ctgov_snapshot
import run_pipeline


# ------------------------------------------------------------
# ctgov_client: pagination, timeout/failure, malformed, empty
# ------------------------------------------------------------

def test_pagination_follows_next_page_token_across_multiple_pages():
    page1 = {
        "studies": [{"protocolSection": {"identificationModule": {"nctId": "NCT00000001"}}}],
        "nextPageToken": "TOKEN_2",
    }
    page2 = {
        "studies": [{"protocolSection": {"identificationModule": {"nctId": "NCT00000002"}}}],
        "nextPageToken": "TOKEN_3",
    }
    page3 = {
        "studies": [{"protocolSection": {"identificationModule": {"nctId": "NCT00000003"}}}],
        "totalCount": 3,
    }

    def fake_fetch(url):
        if "pageToken=TOKEN_2" in url:
            return json.dumps(page2).encode()
        if "pageToken=TOKEN_3" in url:
            return json.dumps(page3).encode()
        return json.dumps(page1).encode()

    studies, meta = ctgov_client.fetch_all_studies(fetch_fn=fake_fetch)
    assert len(studies) == 3
    assert meta["pages_fetched"] == 3
    assert meta["api_records_retrieved"] == 3
    nct_ids = [s["protocolSection"]["identificationModule"]["nctId"] for s in studies]
    assert nct_ids == ["NCT00000001", "NCT00000002", "NCT00000003"]


def test_fetch_timeout_or_network_error_raises_ctgov_fetch_error():
    def fake_fetch(url):
        raise TimeoutError("simulated timeout")

    try:
        ctgov_client.fetch_all_studies(fetch_fn=fake_fetch)
        assert False, "expected CtGovFetchError"
    except ctgov_client.CtGovFetchError:
        pass


def test_malformed_json_response_raises_ctgov_fetch_error():
    def fake_fetch(url):
        return b"this is not valid json {{{"

    try:
        ctgov_client.fetch_all_studies(fetch_fn=fake_fetch)
        assert False, "expected CtGovFetchError"
    except ctgov_client.CtGovFetchError:
        pass


def test_response_missing_studies_key_raises_ctgov_fetch_error():
    def fake_fetch(url):
        return json.dumps({"unexpected": "shape"}).encode()

    try:
        ctgov_client.fetch_all_studies(fetch_fn=fake_fetch)
        assert False, "expected CtGovFetchError"
    except ctgov_client.CtGovFetchError:
        pass


def test_empty_studies_list_is_a_valid_fetch_but_zero_records():
    def fake_fetch(url):
        return json.dumps({"studies": [], "totalCount": 0}).encode()

    studies, meta = ctgov_client.fetch_all_studies(fetch_fn=fake_fetch)
    assert studies == []
    assert meta["api_records_retrieved"] == 0


def test_runaway_pagination_is_bounded_by_max_pages():
    def fake_fetch(url):
        return json.dumps(
            {"studies": [{"protocolSection": {}}], "nextPageToken": "AGAIN"}
        ).encode()

    try:
        ctgov_client.fetch_all_studies(fetch_fn=fake_fetch, max_pages=3)
        assert False, "expected CtGovFetchError from exceeding max_pages"
    except ctgov_client.CtGovFetchError:
        pass


# ------------------------------------------------------------
# ctgov_normalize: schema compatibility + real-field formatting
# ------------------------------------------------------------

def test_normalize_produces_exact_trials_csv_column_set():
    if not os.path.exists("trials.csv"):
        return  # nothing to compare against in this environment
    existing_columns = pd.read_csv("trials.csv", nrows=0).columns.tolist()
    assert ctgov_normalize.COLUMNS == existing_columns


def test_normalize_study_formats_interventions_like_existing_pipeline_expects():
    study = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT05531526"},
            "armsInterventionsModule": {
                "interventions": [
                    {"type": "DRUG", "name": "AR1001"},
                    {"type": "DRUG", "name": "Placebo"},
                ]
            },
        },
        "hasResults": False,
    }
    row = ctgov_normalize.normalize_study(study)
    assert row["Interventions"] == "DRUG: AR1001|DRUG: Placebo"
    assert row["Study URL"] == "https://clinicaltrials.gov/study/NCT05531526"
    assert row["Study Results"] == "NO"


def test_normalize_study_hasresults_true_maps_to_yes():
    study = {"protocolSection": {"identificationModule": {"nctId": "NCT00000001"}}, "hasResults": True}
    row = ctgov_normalize.normalize_study(study)
    assert row["Study Results"] == "YES"


def test_normalize_study_missing_optional_fields_degrades_to_blank_not_crash():
    study = {"protocolSection": {"identificationModule": {"nctId": "NCT00000001"}}}
    row = ctgov_normalize.normalize_study(study)
    assert row["Acronym"] == ""
    assert row["Collaborators"] == ""
    assert row["Study Documents"] == ""
    assert row["Locations"] == ""


def test_normalize_study_locations_omits_missing_state_zip_without_double_commas():
    study = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT00000001"},
            "contactsLocationsModule": {
                "locations": [
                    {"facility": "Some Hospital", "city": "London", "country": "United Kingdom"}
                ]
            },
        }
    }
    row = ctgov_normalize.normalize_study(study)
    assert row["Locations"] == "Some Hospital, London, United Kingdom"


def test_normalize_study_documents_url_matches_cdn_pattern():
    study = {
        "protocolSection": {"identificationModule": {"nctId": "NCT03100617"}},
        "documentSection": {
            "largeDocumentModule": {
                "largeDocs": [
                    {"label": "Study Protocol", "filename": "Prot_000.pdf"}
                ]
            }
        },
    }
    row = ctgov_normalize.normalize_study(study)
    assert row["Study Documents"] == (
        "Study Protocol, https://cdn.clinicaltrials.gov/large-docs/17/NCT03100617/Prot_000.pdf"
    )


def test_normalize_studies_returns_dataframe_with_columns_in_order():
    df = ctgov_normalize.normalize_studies([])
    assert list(df.columns) == ctgov_normalize.COLUMNS
    assert len(df) == 0


# ------------------------------------------------------------
# ctgov_snapshot: validation
# ------------------------------------------------------------

def _make_valid_df(n=5):
    rows = []
    for i in range(n):
        row = {col: "" for col in ctgov_normalize.COLUMNS}
        row["NCT Number"] = f"NCT{i:08d}"
        row["Study Type"] = "INTERVENTIONAL"
        rows.append(row)
    return pd.DataFrame(rows, columns=ctgov_normalize.COLUMNS)


def test_validate_accepts_well_formed_dataframe():
    df = _make_valid_df()
    ok, errors = ctgov_snapshot.validate_dataframe(df)
    assert ok is True
    assert errors == []


def test_validate_rejects_missing_required_columns():
    df = _make_valid_df().drop(columns=["Sponsor"])
    ok, errors = ctgov_snapshot.validate_dataframe(df)
    assert ok is False
    assert any("Missing required columns" in e for e in errors)


def test_validate_rejects_empty_dataframe():
    df = pd.DataFrame(columns=ctgov_normalize.COLUMNS)
    ok, errors = ctgov_snapshot.validate_dataframe(df)
    assert ok is False
    assert any("empty" in e for e in errors)


def test_validate_rejects_invalid_nct_id_format():
    df = _make_valid_df()
    df.loc[0, "NCT Number"] = "NOT-A-VALID-ID"
    ok, errors = ctgov_snapshot.validate_dataframe(df)
    assert ok is False
    assert any("invalid NCT ID" in e for e in errors)


def test_validate_rejects_duplicate_nct_ids():
    df = _make_valid_df()
    df.loc[1, "NCT Number"] = df.loc[0, "NCT Number"]
    ok, errors = ctgov_snapshot.validate_dataframe(df)
    assert ok is False
    assert any("duplicate NCT ID" in e for e in errors)


def test_validate_rejects_implausible_row_count_drop_vs_previous():
    df = _make_valid_df(n=5)
    ok, errors = ctgov_snapshot.validate_dataframe(df, previous_row_count=1000)
    assert ok is False
    assert any("outside the expected range" in e for e in errors)


def test_validate_accepts_reasonable_growth_vs_previous():
    df = _make_valid_df(n=10)
    ok, errors = ctgov_snapshot.validate_dataframe(df, previous_row_count=9)
    assert ok is True


def test_validate_skips_row_count_check_when_no_previous_snapshot():
    df = _make_valid_df(n=3)
    ok, errors = ctgov_snapshot.validate_dataframe(df, previous_row_count=None)
    assert ok is True


# ------------------------------------------------------------
# ctgov_snapshot: immutable snapshot writing + preservation
#
# Redirects the module's directory constants to a temp sandbox so
# nothing here touches the real data/snapshots or data/raw dirs.
# ------------------------------------------------------------

class _SnapshotSandbox:
    def __enter__(self):
        self.tmp = tempfile.mkdtemp(prefix="ctgov_snapshot_test_")
        self._orig_raw_dir = ctgov_snapshot.RAW_DIR
        self._orig_snapshot_dir = ctgov_snapshot.SNAPSHOT_DIR
        self._orig_pointer = ctgov_snapshot.LATEST_POINTER_PATH
        ctgov_snapshot.RAW_DIR = os.path.join(self.tmp, "raw")
        ctgov_snapshot.SNAPSHOT_DIR = os.path.join(self.tmp, "snapshots")
        ctgov_snapshot.LATEST_POINTER_PATH = os.path.join(ctgov_snapshot.SNAPSHOT_DIR, "latest_snapshot.json")
        return self.tmp

    def __exit__(self, *exc):
        ctgov_snapshot.RAW_DIR = self._orig_raw_dir
        ctgov_snapshot.SNAPSHOT_DIR = self._orig_snapshot_dir
        ctgov_snapshot.LATEST_POINTER_PATH = self._orig_pointer
        shutil.rmtree(self.tmp, ignore_errors=True)


def test_write_validated_snapshot_creates_csv_and_updates_pointer():
    with _SnapshotSandbox():
        df = _make_valid_df(n=4)
        path = ctgov_snapshot.write_validated_snapshot(df, {"note": "test"}, run_id="2026-01-01T000000Z")
        assert os.path.exists(path)
        assert path.endswith("trials_2026-01-01T000000Z.csv")

        pointer = ctgov_snapshot.get_latest_snapshot_info()
        assert pointer["row_count"] == 4
        assert pointer["snapshot_filename"] == "trials_2026-01-01T000000Z.csv"

        reloaded = pd.read_csv(path)
        assert len(reloaded) == 4


def test_write_raw_snapshot_saves_studies_and_metadata():
    with _SnapshotSandbox():
        studies = [{"protocolSection": {"identificationModule": {"nctId": "NCT00000001"}}}]
        run_dir = ctgov_snapshot.write_raw_snapshot(studies, {"fetch_timestamp": "now"}, run_id="run1")
        assert os.path.exists(os.path.join(run_dir, "studies.json"))
        assert os.path.exists(os.path.join(run_dir, "metadata.json"))
        with open(os.path.join(run_dir, "studies.json")) as f:
            assert json.load(f) == studies


def test_previous_good_snapshot_is_preserved_after_a_later_validation_failure():
    with _SnapshotSandbox():
        good_df = _make_valid_df(n=5)
        good_path = ctgov_snapshot.write_validated_snapshot(good_df, {}, run_id="good_run")
        with open(good_path) as f:
            good_content_before = f.read()

        # Simulate a second, bad fetch: caller must check validate_dataframe
        # BEFORE calling write_validated_snapshot — mirror that contract here.
        bad_df = pd.DataFrame(columns=ctgov_normalize.COLUMNS)  # empty -> invalid
        ok, errors = ctgov_snapshot.validate_dataframe(bad_df, previous_row_count=5)
        assert ok is False

        # A correct caller does NOT write a snapshot when validation fails.
        # Confirm the earlier good snapshot file and pointer are untouched.
        with open(good_path) as f:
            good_content_after = f.read()
        assert good_content_after == good_content_before

        pointer = ctgov_snapshot.get_latest_snapshot_info()
        assert pointer["snapshot_filename"] == "trials_good_run.csv"


def test_write_validated_snapshot_refuses_to_overwrite_existing_file():
    with _SnapshotSandbox():
        df = _make_valid_df(n=2)
        ctgov_snapshot.write_validated_snapshot(df, {}, run_id="dup_run")
        try:
            ctgov_snapshot.write_validated_snapshot(df, {}, run_id="dup_run")
            assert False, "expected FileExistsError on duplicate run_id"
        except FileExistsError:
            pass


# ------------------------------------------------------------
# run_pipeline: full --refresh wiring, mocked network + subprocess
# ------------------------------------------------------------

class _RefreshSandbox:
    """Redirects run_pipeline + ctgov_snapshot's file targets into a
    temp sandbox, and stubs the network/subprocess boundary, so a
    "successful refresh" can be exercised end-to-end with no real
    network access and no writes to the real project files."""

    def __enter__(self):
        self.tmp = tempfile.mkdtemp(prefix="ctgov_refresh_test_")
        self._orig_raw_dir = ctgov_snapshot.RAW_DIR
        self._orig_snapshot_dir = ctgov_snapshot.SNAPSHOT_DIR
        self._orig_pointer = ctgov_snapshot.LATEST_POINTER_PATH
        ctgov_snapshot.RAW_DIR = os.path.join(self.tmp, "raw")
        ctgov_snapshot.SNAPSHOT_DIR = os.path.join(self.tmp, "snapshots")
        ctgov_snapshot.LATEST_POINTER_PATH = os.path.join(ctgov_snapshot.SNAPSHOT_DIR, "latest_snapshot.json")

        self._orig_trials_csv = run_pipeline.TRIALS_CSV
        self._orig_backup = run_pipeline.TRIALS_CSV_BACKUP
        self._orig_drugs_csv = run_pipeline.DRUGS_CSV
        self._orig_annotated_csv = run_pipeline.ANNOTATED_CSV
        self._orig_changes_csv = run_pipeline.CHANGES_CSV
        self._orig_attention_csv = run_pipeline.ATTENTION_CSV
        self._orig_overview_html = run_pipeline.OVERVIEW_HTML
        run_pipeline.TRIALS_CSV = os.path.join(self.tmp, "trials.csv")
        run_pipeline.TRIALS_CSV_BACKUP = os.path.join(self.tmp, "trials.csv.bak")
        run_pipeline.DRUGS_CSV = os.path.join(self.tmp, "pipeline_drugs.csv")
        run_pipeline.ANNOTATED_CSV = os.path.join(self.tmp, "pipeline_annotated.csv")
        run_pipeline.CHANGES_CSV = os.path.join(self.tmp, "outputs", "pipeline_changes.csv")
        run_pipeline.ATTENTION_CSV = os.path.join(self.tmp, "outputs", "competitive_attention.csv")
        run_pipeline.OVERVIEW_HTML = os.path.join(self.tmp, "pipeline_overview.html")

        self._orig_fetch_all = ctgov_client.fetch_all_studies
        self._orig_fetch_version = ctgov_client.fetch_data_version
        self._orig_subprocess_run = run_pipeline.subprocess.run

        return self

    def __exit__(self, *exc):
        ctgov_snapshot.RAW_DIR = self._orig_raw_dir
        ctgov_snapshot.SNAPSHOT_DIR = self._orig_snapshot_dir
        ctgov_snapshot.LATEST_POINTER_PATH = self._orig_pointer
        run_pipeline.TRIALS_CSV = self._orig_trials_csv
        run_pipeline.TRIALS_CSV_BACKUP = self._orig_backup
        run_pipeline.DRUGS_CSV = self._orig_drugs_csv
        run_pipeline.ANNOTATED_CSV = self._orig_annotated_csv
        run_pipeline.CHANGES_CSV = self._orig_changes_csv
        run_pipeline.ATTENTION_CSV = self._orig_attention_csv
        run_pipeline.OVERVIEW_HTML = self._orig_overview_html
        ctgov_client.fetch_all_studies = self._orig_fetch_all
        ctgov_client.fetch_data_version = self._orig_fetch_version
        run_pipeline.subprocess.run = self._orig_subprocess_run
        shutil.rmtree(self.tmp, ignore_errors=True)


def _canned_studies(n=5):
    return [
        {
            "protocolSection": {"identificationModule": {"nctId": f"NCT{i:08d}"}},
            "hasResults": False,
        }
        for i in range(n)
    ]


def test_successful_refresh_writes_snapshot_and_updates_trials_csv():
    with _RefreshSandbox() as sb:
        studies = _canned_studies(5)
        ctgov_client.fetch_all_studies = lambda: (
            studies,
            {
                "pages_fetched": 1,
                "total_count_reported": 5,
                "api_records_retrieved": 5,
                "query_params": {"query.cond": "Alzheimer Disease"},
                "query_url_example": "https://clinicaltrials.gov/api/v2/studies?query.cond=Alzheimer+Disease",
            },
        )
        ctgov_client.fetch_data_version = lambda: "2026-08-07T09:00:05"

        class _FakeCompleted:
            returncode = 0
            stdout = "ok"
            stderr = ""

        run_pipeline.subprocess.run = lambda *a, **k: _FakeCompleted()

        # The mocked subprocess call stands in for pipeline_viz.py, which
        # normally writes this file (with the placeholder token) itself —
        # write a minimal stand-in so STEP 7's splice step has something
        # real to operate on.
        with open(run_pipeline.OVERVIEW_HTML, "w") as f:
            f.write(f"<html><body>{competitive_attention_viz.PLACEHOLDER}</body></html>")

        exit_code = run_pipeline.run_refresh()

        assert exit_code == 0
        assert os.path.exists(run_pipeline.TRIALS_CSV)
        rebuilt = pd.read_csv(run_pipeline.TRIALS_CSV)
        assert len(rebuilt) == 5

        pointer = ctgov_snapshot.get_latest_snapshot_info()
        assert pointer["row_count"] == 5

        with open(run_pipeline.OVERVIEW_HTML) as f:
            overview_content = f.read()
        assert competitive_attention_viz.PLACEHOLDER not in overview_content
        assert "Needs Attention" in overview_content
        assert "Upcoming Competitive Milestones" in overview_content
        assert os.path.exists(run_pipeline.ATTENTION_CSV)
        assert not os.path.exists(run_pipeline.TRIALS_CSV_BACKUP)


def test_refresh_aborts_and_preserves_previous_trials_csv_on_fetch_failure():
    with _RefreshSandbox() as sb:
        with open(run_pipeline.TRIALS_CSV, "w") as f:
            f.write("NCT Number\nNCT00000001\n")

        def failing_fetch():
            raise ctgov_client.CtGovFetchError("simulated network failure")

        ctgov_client.fetch_all_studies = failing_fetch
        ctgov_client.fetch_data_version = lambda: None

        exit_code = run_pipeline.run_refresh()

        assert exit_code == 1
        with open(run_pipeline.TRIALS_CSV) as f:
            assert f.read() == "NCT Number\nNCT00000001\n"


def test_refresh_aborts_before_rebuild_on_validation_failure():
    with _RefreshSandbox() as sb:
        with open(run_pipeline.TRIALS_CSV, "w") as f:
            f.write("NCT Number\nNCT00000001\n")

        # Empty study list -> normalizes to an empty (invalid) dataframe
        ctgov_client.fetch_all_studies = lambda: (
            [],
            {
                "pages_fetched": 1,
                "total_count_reported": 0,
                "api_records_retrieved": 0,
                "query_params": {},
                "query_url_example": "https://clinicaltrials.gov/api/v2/studies?",
            },
        )
        ctgov_client.fetch_data_version = lambda: None

        rebuild_was_called = []
        run_pipeline.subprocess.run = lambda *a, **k: rebuild_was_called.append(1)

        exit_code = run_pipeline.run_refresh()

        assert exit_code == 1
        assert rebuild_was_called == []  # pipeline_viz.py must never run on invalid data
        with open(run_pipeline.TRIALS_CSV) as f:
            assert f.read() == "NCT Number\nNCT00000001\n"
        assert ctgov_snapshot.get_latest_snapshot_info() is None


def test_refresh_rolls_back_trials_csv_if_dashboard_rebuild_fails():
    with _RefreshSandbox() as sb:
        # 5 old rows so the 5-study canned fetch below passes the
        # row-count sanity check and this test can reach the dashboard
        # rebuild stage it's actually testing.
        with open(run_pipeline.TRIALS_CSV, "w") as f:
            f.write("NCT Number\n" + "\n".join(f"NCT{i:08d}" for i in range(5)) + "\n")

        studies = _canned_studies(5)
        ctgov_client.fetch_all_studies = lambda: (
            studies,
            {
                "pages_fetched": 1,
                "total_count_reported": 5,
                "api_records_retrieved": 5,
                "query_params": {},
                "query_url_example": "https://clinicaltrials.gov/api/v2/studies?",
            },
        )
        ctgov_client.fetch_data_version = lambda: None

        class _FailingCompleted:
            returncode = 1
            stdout = ""
            stderr = "boom: pipeline_viz.py crashed"

        run_pipeline.subprocess.run = lambda *a, **k: _FailingCompleted()

        original_content = "NCT Number\n" + "\n".join(f"NCT{i:08d}" for i in range(5)) + "\n"
        exit_code = run_pipeline.run_refresh()

        assert exit_code == 1
        with open(run_pipeline.TRIALS_CSV) as f:
            assert f.read() == original_content
        # the new snapshot must still exist for inspection even though
        # it wasn't promoted to trials.csv
        assert ctgov_snapshot.get_latest_snapshot_info()["row_count"] == 5


ALL_TESTS = [
    test_pagination_follows_next_page_token_across_multiple_pages,
    test_fetch_timeout_or_network_error_raises_ctgov_fetch_error,
    test_malformed_json_response_raises_ctgov_fetch_error,
    test_response_missing_studies_key_raises_ctgov_fetch_error,
    test_empty_studies_list_is_a_valid_fetch_but_zero_records,
    test_runaway_pagination_is_bounded_by_max_pages,
    test_normalize_produces_exact_trials_csv_column_set,
    test_normalize_study_formats_interventions_like_existing_pipeline_expects,
    test_normalize_study_hasresults_true_maps_to_yes,
    test_normalize_study_missing_optional_fields_degrades_to_blank_not_crash,
    test_normalize_study_locations_omits_missing_state_zip_without_double_commas,
    test_normalize_study_documents_url_matches_cdn_pattern,
    test_normalize_studies_returns_dataframe_with_columns_in_order,
    test_validate_accepts_well_formed_dataframe,
    test_validate_rejects_missing_required_columns,
    test_validate_rejects_empty_dataframe,
    test_validate_rejects_invalid_nct_id_format,
    test_validate_rejects_duplicate_nct_ids,
    test_validate_rejects_implausible_row_count_drop_vs_previous,
    test_validate_accepts_reasonable_growth_vs_previous,
    test_validate_skips_row_count_check_when_no_previous_snapshot,
    test_write_validated_snapshot_creates_csv_and_updates_pointer,
    test_write_raw_snapshot_saves_studies_and_metadata,
    test_previous_good_snapshot_is_preserved_after_a_later_validation_failure,
    test_write_validated_snapshot_refuses_to_overwrite_existing_file,
    test_successful_refresh_writes_snapshot_and_updates_trials_csv,
    test_refresh_aborts_and_preserves_previous_trials_csv_on_fetch_failure,
    test_refresh_aborts_before_rebuild_on_validation_failure,
    test_refresh_rolls_back_trials_csv_if_dashboard_rebuild_fails,
]


def run_test(test_fn):
    try:
        test_fn()
        print(f"PASS  {test_fn.__name__}")
        return True
    except AssertionError as e:
        print(f"FAIL  {test_fn.__name__}  -- {e}")
        return False
    except Exception as e:
        print(f"ERROR {test_fn.__name__}  -- {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    results = [run_test(t) for t in ALL_TESTS]
    passed = sum(results)
    total = len(results)
    print()
    print(f"{passed}/{total} tests passed")
    if passed != total:
        raise SystemExit(1)
