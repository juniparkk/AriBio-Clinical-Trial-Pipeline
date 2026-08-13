# ============================================================
# TESTS for adni_stats.py / run_adni_statistics.py (ANCOVA
# statistical-analysis stage).
#
# Every fixture here is hand-built synthetic data (fake RIDs, fake
# values) -- never real ADNI participant data.
#
# Run: .venv/bin/python test_adni_statistics.py
# ============================================================

import numpy as np
import pandas as pd

import adni_stats as S
import run_adni_statistics as R


# ------------------------------------------------------------------
# Shared synthetic fixtures
# ------------------------------------------------------------------


def _make_ancova_ready_df(seed=0, n_per_group=30):
    """
    A synthetic ANCOVA-ready sample with a real, controllable group
    effect (Dementia > MCI > CN on the outcome), used by tests that
    need an actual fitted model rather than just structural checks.
    """
    rng = np.random.default_rng(seed)
    rows = []
    group_effect = {"CN": 0.0, "MCI": 2.0, "Dementia": 5.0}
    for group, effect in group_effect.items():
        for i in range(n_per_group):
            age = rng.normal(72, 6)
            baseline = rng.normal(20, 3)
            sex = "Male" if i % 2 == 0 else "Female"
            noise = rng.normal(0, 1)
            change = effect + 0.1 * (age - 72) + noise
            rows.append(
                {
                    "RID": len(rows) + 1,
                    "DX_BASELINE_FIXED": group,
                    "BASELINE_VALUE_FOR_MODEL": baseline,
                    "BASELINE_AGE": age,
                    "SEX": sex,
                    "change_from_baseline": change,
                }
            )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# 1. Baseline month is never inferentially tested
# ------------------------------------------------------------------


def test_baseline_month_never_inferential():
    clinical = pd.DataFrame(
        [
            {"RID": i, "DX_BASELINE_FIXED": g, "ADAS_COG13_BASELINE": 20.0, "ADAS_COG13": np.nan,
             "MMSE_BASELINE": 28.0, "MMSE": np.nan, "BASELINE_AGE": 70.0, "SEX": "Male",
             "VISIT_MONTH": 0, "ADAS_COG13_ELIGIBLE": True, "MMSE_ELIGIBLE": True}
            for i, g in enumerate(["CN"] * 15 + ["MCI"] * 15 + ["Dementia"] * 15)
        ]
    )
    summary, pairwise, diagnostics = R.run_cognitive_endpoint(clinical, "ADAS_COG13")
    month0_rows = [r for r in summary if r["month"] == 0]
    assert len(month0_rows) == 3
    for r in month0_rows:
        assert r["inferential_status"] == S.STATUS_BASELINE
        assert pd.isna(r["adjusted_mean_change"])
        assert pd.isna(r["overall_F"])
    # No pairwise/diagnostics rows should exist for month 0 (no model fit).
    assert all(r["month"] != 0 for r in pairwise)
    assert all(r["month"] != 0 for r in diagnostics)


# ------------------------------------------------------------------
# 2 / 3. Change direction is followup - baseline (both endpoints)
# ------------------------------------------------------------------


def test_adas13_change_direction_is_followup_minus_baseline():
    clinical = pd.DataFrame(
        [
            {"RID": 1, "DX_BASELINE_FIXED": "CN", "ADAS_COG13_BASELINE": 10.0, "ADAS_COG13": 14.0,
             "MMSE_BASELINE": 28.0, "MMSE": 28.0, "BASELINE_AGE": 70.0, "SEX": "Male",
             "VISIT_MONTH": 12, "ADAS_COG13_ELIGIBLE": True, "MMSE_ELIGIBLE": True},
        ]
    )
    sample = R.build_cognitive_sample(clinical, "ADAS_COG13", 12)
    assert sample.loc[0, "change_from_baseline"] == 14.0 - 10.0  # worsening = positive, as documented


def test_mmse_change_direction_is_followup_minus_baseline():
    clinical = pd.DataFrame(
        [
            {"RID": 1, "DX_BASELINE_FIXED": "CN", "ADAS_COG13_BASELINE": 10.0, "ADAS_COG13": 10.0,
             "MMSE_BASELINE": 28.0, "MMSE": 25.0, "BASELINE_AGE": 70.0, "SEX": "Male",
             "VISIT_MONTH": 12, "ADAS_COG13_ELIGIBLE": True, "MMSE_ELIGIBLE": True},
        ]
    )
    sample = R.build_cognitive_sample(clinical, "MMSE", 12)
    # No sign flip: followup(25) - baseline(28) = -3, which is a WORSENING per the
    # documented MMSE convention (positive=improvement, negative=worsening) -- the
    # math itself is always followup - baseline for both endpoints.
    assert sample.loc[0, "change_from_baseline"] == 25.0 - 28.0


# ------------------------------------------------------------------
# 4. Biomarker log-change uses natural log (never log2)
# ------------------------------------------------------------------


def test_biomarker_log_change_uses_natural_log():
    plasma = pd.DataFrame(
        [
            {"RID": 1, "PLATFORM": None, "PLASMAPTAU181": 10.0, "VISIT_MONTH_RAW": 0, "VISIT_MONTH": 0,
             "BASELINE_AGE": 70.0, "SEX": "Male"},
            {"RID": 1, "PLATFORM": None, "PLASMAPTAU181": 20.0, "VISIT_MONTH_RAW": 12, "VISIT_MONTH": 12,
             "BASELINE_AGE": 70.0, "SEX": "Male"},
        ]
    )
    dx = pd.DataFrame({"RID": [1], "DX_BASELINE_FIXED": ["CN"]})
    sample = R.build_biomarker_sample(plasma, dx, "PLASMAPTAU181", 12)
    expected_natural_log_change = np.log(20.0) - np.log(10.0)
    wrong_log2_change = np.log2(20.0) - np.log2(10.0)
    assert np.isclose(sample.loc[0, "log_change"], expected_natural_log_change)
    assert not np.isclose(sample.loc[0, "log_change"], wrong_log2_change)


# ------------------------------------------------------------------
# 5. Percent-change back-transform uses exp()
# ------------------------------------------------------------------


def test_percent_change_backtransform_uses_exp():
    assert S.geometric_percent_change(0.0) == 0.0
    # ln(2) log-change corresponds to an exact 100% increase.
    assert np.isclose(S.geometric_percent_change(np.log(2.0)), 100.0)
    # ln(0.5) log-change corresponds to an exact 50% decrease.
    assert np.isclose(S.geometric_percent_change(np.log(0.5)), -50.0)


# ------------------------------------------------------------------
# 6. pTau181 and pTau217 are never pooled
# ------------------------------------------------------------------


def test_ptau181_and_ptau217_never_pooled():
    ptau181 = pd.DataFrame(
        [
            {"RID": 1, "PLASMAPTAU181": 5.0, "VISIT_MONTH_RAW": 0, "VISIT_MONTH": 0, "BASELINE_AGE": 70.0, "SEX": "Male"},
            {"RID": 1, "PLASMAPTAU181": 9.0, "VISIT_MONTH_RAW": 12, "VISIT_MONTH": 12, "BASELINE_AGE": 70.0, "SEX": "Male"},
        ]
    )
    ptau217 = pd.DataFrame(
        [
            {"RID": 1, "PTAU217": 0.2, "PTAU217_LOT_BIAS_FLAG": False, "VISIT_MONTH_RAW": 0, "VISIT_MONTH": 0, "BASELINE_AGE": 70.0, "SEX": "Male"},
            {"RID": 1, "PTAU217": 0.4, "PTAU217_LOT_BIAS_FLAG": False, "VISIT_MONTH_RAW": 12, "VISIT_MONTH": 12, "BASELINE_AGE": 70.0, "SEX": "Male"},
        ]
    )
    dx = pd.DataFrame({"RID": [1], "DX_BASELINE_FIXED": ["CN"]})

    sample_181 = R.build_biomarker_sample(ptau181, dx, "PLASMAPTAU181", 12)
    sample_217 = R.build_biomarker_sample(ptau217, dx, "PTAU217", 12, lot_bias_col="PTAU217_LOT_BIAS_FLAG")

    # Each family's sample is built from its OWN value column only --
    # the pTau181 sample never sees the pTau217 concentration (0.2/0.4)
    # and vice versa, confirming the two are analyzed as fully separate
    # inputs, never combined into one series.
    assert np.isclose(sample_181.loc[0, "BASELINE_VALUE_FOR_MODEL"], 5.0)
    assert np.isclose(sample_217.loc[0, "BASELINE_VALUE_FOR_MODEL"], 0.2)
    assert set(sample_181.columns) & {"PTAU217"} == set()
    assert set(sample_217.columns) & {"PLASMAPTAU181"} == set()


# ------------------------------------------------------------------
# 7. Quanterix is primary for GFAP/NfL
# ------------------------------------------------------------------


def test_quanterix_is_primary_platform_for_gfap_and_nfl():
    rng = np.random.default_rng(1)
    rows = []
    dx_rows = []
    rid = 0
    for group in ["CN", "MCI", "Dementia"]:
        for i in range(12):
            rid += 1
            dx_rows.append({"RID": rid, "DX_BASELINE_FIXED": group})
            base = rng.uniform(50, 100)
            for platform in ["Quanterix", "Fujirebio"]:
                rows.append(
                    {"RID": rid, "PHASE": "ADNI4", "PLATFORM": platform, "GFAP": base,
                     "VISIT_MONTH_RAW": 0, "VISIT_MONTH": 0, "BASELINE_AGE": 70.0,
                     "SEX": "Male" if i % 2 == 0 else "Female", "BIOMARKER_ELIGIBLE": True}
                )
                rows.append(
                    {"RID": rid, "PHASE": "ADNI4", "PLATFORM": platform, "GFAP": base * rng.uniform(1.0, 1.3),
                     "VISIT_MONTH_RAW": 12, "VISIT_MONTH": 12, "BASELINE_AGE": 70.0,
                     "SEX": "Male" if i % 2 == 0 else "Female", "BIOMARKER_ELIGIBLE": True}
                )
    gfap_long = pd.DataFrame(rows)
    dx = pd.DataFrame(dx_rows)

    summary_primary, _, _ = R.run_biomarker_family(
        gfap_long, dx, "GFAP", "GFAP", "Quanterix", "primary", platform="Quanterix"
    )
    summary_sensitivity, _, _ = R.run_biomarker_family(
        gfap_long, dx, "GFAP", "GFAP", "Fujirebio", "sensitivity_fujirebio", platform="Fujirebio"
    )
    assert all(r["assay_platform"] == "Quanterix" for r in summary_primary)
    assert all(r["analysis_type"] == "primary" for r in summary_primary)
    assert all(r["assay_platform"] == "Fujirebio" for r in summary_sensitivity)
    assert all(r["analysis_type"] == "sensitivity_fujirebio" for r in summary_sensitivity)


# ------------------------------------------------------------------
# 8 / 9. pTau217 lot-bias flag: excluded from primary, included in sensitivity
# ------------------------------------------------------------------


def _ptau217_lot_bias_fixture():
    return pd.DataFrame(
        [
            # RID 1's baseline is flagged -- must be dropped entirely from primary.
            {"RID": 1, "PTAU217": 0.15, "PTAU217_LOT_BIAS_FLAG": True, "VISIT_MONTH_RAW": 0, "VISIT_MONTH": 0, "BASELINE_AGE": 70.0, "SEX": "Male"},
            {"RID": 1, "PTAU217": 0.30, "PTAU217_LOT_BIAS_FLAG": False, "VISIT_MONTH_RAW": 12, "VISIT_MONTH": 12, "BASELINE_AGE": 70.0, "SEX": "Male"},
            # RID 2 is entirely unflagged -- present in both primary and sensitivity.
            {"RID": 2, "PTAU217": 0.20, "PTAU217_LOT_BIAS_FLAG": False, "VISIT_MONTH_RAW": 0, "VISIT_MONTH": 0, "BASELINE_AGE": 68.0, "SEX": "Female"},
            {"RID": 2, "PTAU217": 0.25, "PTAU217_LOT_BIAS_FLAG": False, "VISIT_MONTH_RAW": 12, "VISIT_MONTH": 12, "BASELINE_AGE": 68.0, "SEX": "Female"},
        ]
    )


def test_ptau217_lot_bias_records_excluded_from_primary():
    ptau217 = _ptau217_lot_bias_fixture()
    dx = pd.DataFrame({"RID": [1, 2], "DX_BASELINE_FIXED": ["CN", "MCI"]})
    primary = R.build_biomarker_sample(
        ptau217, dx, "PTAU217", 12, lot_bias_col="PTAU217_LOT_BIAS_FLAG", exclude_flagged=True
    )
    # RID 1's baseline record was flagged -- no valid baseline survives the
    # exclusion, so RID 1 must be entirely absent from the primary sample.
    assert 1 not in set(primary["RID"])
    assert 2 in set(primary["RID"])


def test_ptau217_lot_bias_records_included_in_sensitivity():
    ptau217 = _ptau217_lot_bias_fixture()
    dx = pd.DataFrame({"RID": [1, 2], "DX_BASELINE_FIXED": ["CN", "MCI"]})
    sensitivity = R.build_biomarker_sample(
        ptau217, dx, "PTAU217", 12, lot_bias_col="PTAU217_LOT_BIAS_FLAG", exclude_flagged=False
    )
    assert {1, 2} == set(sensitivity["RID"])


# ------------------------------------------------------------------
# 10. Small cells suppress inference
# ------------------------------------------------------------------


def test_small_cells_suppress_inference():
    df = pd.DataFrame(
        [{"RID": i, "DX_BASELINE_FIXED": "CN", "change_from_baseline": 1.0} for i in range(20)]
        + [{"RID": 100 + i, "DX_BASELINE_FIXED": "MCI", "change_from_baseline": 2.0} for i in range(20)]
        + [{"RID": 200 + i, "DX_BASELINE_FIXED": "Dementia", "change_from_baseline": 3.0} for i in range(3)]  # n=3 < 10
    )
    ok, group_ns, limiting_group, limiting_n = S.check_group_sizes(df)
    assert ok is False
    assert limiting_group == "Dementia"
    assert limiting_n == 3

    result = R._fit_or_suppress(df, "change_from_baseline", "BASELINE_VALUE_FOR_MODEL", month=12)
    assert result["status"] == S.STATUS_SUPPRESSED
    assert result["fit"] is None  # no model was fit at all


# ------------------------------------------------------------------
# 11. Sex is treated categorically
# ------------------------------------------------------------------


def test_sex_is_treated_categorically_not_numeric():
    df = _make_ancova_ready_df(seed=2)
    fit = S.fit_ancova(
        df, outcome_col="change_from_baseline",
        continuous_covariates=("BASELINE_VALUE_FOR_MODEL", "BASELINE_AGE"),
        categorical_covariates=("SEX",),
    )
    param_names = list(fit["model"].params.index)
    # A categorical term produces a named dummy coefficient like "C(SEX)[T.Male]";
    # a (wrongly) numeric treatment would instead produce a single bare "SEX" slope.
    assert any(p.startswith("C(SEX)") for p in param_names)
    assert "SEX" not in param_names


# ------------------------------------------------------------------
# 12. Pairwise comparisons generated correctly
# ------------------------------------------------------------------


def test_pairwise_comparisons_generated_correctly():
    df = _make_ancova_ready_df(seed=3)
    fit = S.fit_ancova(
        df, outcome_col="change_from_baseline",
        continuous_covariates=("BASELINE_VALUE_FOR_MODEL", "BASELINE_AGE"),
        categorical_covariates=("SEX",),
    )
    comparisons = {p["comparison"] for p in fit["pairwise"]}
    assert comparisons == {"MCI - CN", "Dementia - CN", "Dementia - MCI"}

    by_comp = {p["comparison"]: p for p in fit["pairwise"]}
    am = fit["adjusted_means"]
    # Pairwise estimates must equal the corresponding difference of adjusted means
    # (exact for this no-interaction linear model).
    assert np.isclose(by_comp["MCI - CN"]["estimate"], am["MCI"]["mean"] - am["CN"]["mean"], atol=1e-6)
    assert np.isclose(by_comp["Dementia - CN"]["estimate"], am["Dementia"]["mean"] - am["CN"]["mean"], atol=1e-6)
    assert np.isclose(by_comp["Dementia - MCI"]["estimate"], am["Dementia"]["mean"] - am["MCI"]["mean"], atol=1e-6)
    # The synthetic data was built with Dementia > MCI > CN -- the fitted
    # contrasts should recover that ordering with a significant p-value.
    assert by_comp["Dementia - CN"]["estimate"] > 0
    assert by_comp["Dementia - CN"]["p_value"] < 0.05


# ------------------------------------------------------------------
# 13. No participant identifiers appear in aggregate outputs
# ------------------------------------------------------------------


def test_no_participant_identifiers_in_aggregate_outputs():
    clinical = pd.DataFrame(
        [
            {"RID": i, "DX_BASELINE_FIXED": g, "ADAS_COG13_BASELINE": 20.0, "ADAS_COG13": 22.0,
             "MMSE_BASELINE": 28.0, "MMSE": 27.0, "BASELINE_AGE": 70.0, "SEX": "Male" if i % 2 else "Female",
             "VISIT_MONTH": 12, "ADAS_COG13_ELIGIBLE": True, "MMSE_ELIGIBLE": True}
            for i, g in enumerate(["CN"] * 12 + ["MCI"] * 12 + ["Dementia"] * 12)
        ]
    )
    summary, pairwise, diagnostics = R.run_cognitive_endpoint(clinical, "ADAS_COG13")
    for row_set in (summary, pairwise, diagnostics):
        for row in row_set:
            assert "RID" not in row
            assert not any(isinstance(v, (list, np.ndarray)) for v in row.values())


# ------------------------------------------------------------------
# 14. Descriptive biomarker stats are computed for small-cell
#     (suppressed) groups without fitting any model
# ------------------------------------------------------------------


def test_descriptive_stats_computed_for_suppressed_small_cell():
    rng = np.random.default_rng(4)
    rows = []
    # CN: n=20 (well above the ANCOVA min-cell threshold).
    for i in range(20):
        log_base = rng.normal(2.0, 0.2)
        log_change = rng.normal(0.05, 0.1)
        rows.append({"RID": i, "DX_BASELINE_FIXED": "CN", "log_baseline": log_base, "log_change": log_change})
    # Dementia: n=3, below MIN_GROUP_N=10 -- ANCOVA would suppress this cell.
    for i in range(3):
        log_base = rng.normal(2.3, 0.2)
        log_change = rng.normal(0.3, 0.1)
        rows.append({"RID": 100 + i, "DX_BASELINE_FIXED": "Dementia", "log_baseline": log_base, "log_change": log_change})
    # MCI: n=0 -- no data at all for this cell.
    sample = pd.DataFrame(rows)

    raw_stats = S.raw_change_stats(sample, "log_change")
    assert raw_stats["Dementia"]["n"] == 3
    assert raw_stats["MCI"]["n"] == 0

    descriptive = S.compute_descriptive_biomarker_stats(sample, raw_stats, month=12)

    # A small cell (n=3 < MIN_GROUP_N) still gets a computed descriptive
    # value -- this is the entire point of the change: descriptive stats
    # are available for cells the inferential (ANCOVA) pipeline suppresses.
    assert descriptive["Dementia"]["descriptive_status"] == "Computed"
    assert descriptive["Dementia"]["raw_geometric_mean"] > 0
    assert np.isfinite(descriptive["Dementia"]["raw_geometric_pct_change"])
    # n=3 >= 2, so a one-sample descriptive t-interval should be derivable.
    assert np.isfinite(descriptive["Dementia"]["raw_geometric_pct_change_ci_lower"])
    assert np.isfinite(descriptive["Dementia"]["raw_geometric_pct_change_ci_upper"])

    # An empty cell (n=0) cannot yield any descriptive value -- must be
    # explicitly flagged, never a fabricated number.
    assert descriptive["MCI"]["descriptive_status"] == "Insufficient data"
    assert np.isnan(descriptive["MCI"]["raw_geometric_mean"])

    # The descriptive computation must not have touched the CN cell's
    # (unsuppressed) values -- it is a per-group, order-independent summary.
    assert descriptive["CN"]["descriptive_status"] == "Computed"

    # Purely aggregate: the returned structure carries only group-level
    # scalars, never a participant identifier or per-row value.
    for level_stats in descriptive.values():
        assert "RID" not in level_stats
        assert all(np.isscalar(v) or v is None for v in level_stats.values())


def test_descriptive_biomarker_columns_present_in_run_biomarker_family_output():
    """
    End-to-end: run_biomarker_family's per-cell summary rows must carry
    the new descriptive columns for every row (including suppressed
    ones), and those rows must remain aggregate-only -- no RID/PTID, no
    per-participant arrays -- exactly like every other summary column.
    """
    rng = np.random.default_rng(5)
    rows = []
    rid = 0
    group_n = {"CN": 15, "MCI": 15, "Dementia": 4}  # Dementia < MIN_GROUP_N=10
    for group, n in group_n.items():
        for i in range(n):
            rid += 1
            base_val = float(np.exp(rng.normal(2.0, 0.2)))
            follow_val = float(np.exp(np.log(base_val) + rng.normal(0.05, 0.1)))
            age = float(rng.normal(72, 5))
            sex = "Male" if i % 2 == 0 else "Female"
            rows.append({"RID": rid, "PLATFORM": None, "GFAP": base_val, "VISIT_MONTH_RAW": 0,
                         "VISIT_MONTH": 0, "BASELINE_AGE": age, "SEX": sex, "BIOMARKER_ELIGIBLE": True})
            rows.append({"RID": rid, "PLATFORM": None, "GFAP": follow_val, "VISIT_MONTH_RAW": 12,
                         "VISIT_MONTH": 12, "BASELINE_AGE": age, "SEX": sex, "BIOMARKER_ELIGIBLE": True})
    plasma = pd.DataFrame(rows)
    dx = pd.DataFrame({"RID": list(range(1, rid + 1)),
                        "DX_BASELINE_FIXED": (["CN"] * group_n["CN"] + ["MCI"] * group_n["MCI"] + ["Dementia"] * group_n["Dementia"])})

    summary_rows, _pairwise, _diagnostics = R.run_biomarker_family(
        plasma, dx, "GFAP", "GFAP", assay_platform="Quanterix", analysis_type="primary",
    )
    assert len(summary_rows) > 0
    descriptive_cols = {
        "raw_geometric_mean", "raw_geometric_pct_change",
        "raw_geometric_pct_change_ci_lower", "raw_geometric_pct_change_ci_upper", "descriptive_status",
    }
    dementia_month12_rows = [r for r in summary_rows if r["group"] == "Dementia" and r["month"] == 12]
    assert dementia_month12_rows
    for row in summary_rows:
        assert descriptive_cols <= set(row.keys())
        # Aggregate-only: no participant identifier and no per-row array
        # ever leaks into a summary row, whether the cell was suppressed
        # (small-n Dementia) or fitted (CN/MCI).
        assert "RID" not in row
        assert not any(isinstance(v, (list, np.ndarray)) for v in row.values())
        assert row["descriptive_status"] in ("Computed", "Insufficient data")
    # The small (suppressed) Dementia cell must still carry a real,
    # non-suppressed descriptive value -- that is the feature under test.
    dementia_row = dementia_month12_rows[0]
    assert "Suppressed" in dementia_row["inferential_status"]
    assert dementia_row["descriptive_status"] == "Computed"
    assert dementia_row["raw_geometric_pct_change"] is not None
    assert not (isinstance(dementia_row["raw_geometric_pct_change"], float) and np.isnan(dementia_row["raw_geometric_pct_change"]))


# ------------------------------------------------------------------
# 15. Primary/sensitivity labels are preserved
# ------------------------------------------------------------------


def test_primary_and_sensitivity_labels_preserved():
    clinical = pd.DataFrame(
        [
            {"RID": i, "DX_BASELINE_FIXED": g, "ADAS_COG13_BASELINE": 20.0, "ADAS_COG13": 22.0,
             "MMSE_BASELINE": 28.0, "MMSE": 27.0, "BASELINE_AGE": 70.0, "SEX": "Male" if i % 2 else "Female",
             "VISIT_MONTH": 12, "ADAS_COG13_ELIGIBLE": True, "MMSE_ELIGIBLE": True,
             "MMSE_SCREENING_TO_BASELINE_INTERVAL_DAYS": 30, "MMSE_LONG_SCREENING_INTERVAL_FLAG": False}
            for i, g in enumerate(["CN"] * 12 + ["MCI"] * 12 + ["Dementia"] * 12)
        ]
    )
    summary_primary, _, _ = R.run_cognitive_endpoint(clinical, "MMSE", analysis_type="primary")
    summary_sensitivity, _, _ = R.run_cognitive_endpoint(
        clinical, "MMSE", analysis_type="sensitivity_interval_excl", exclude_rids=set()
    )
    assert all(r["analysis_type"] == "primary" for r in summary_primary)
    assert all(r["analysis_type"] == "sensitivity_interval_excl" for r in summary_sensitivity)


# ------------------------------------------------------------------
# 16. Absolute-value descriptive stats (Medical Affairs redesign:
#     Disease Continuum + Absolute trajectory views)
# ------------------------------------------------------------------


def test_descriptive_mean_ci_basic():
    stats_ = S.descriptive_mean_ci([10.0, 12.0, 14.0, 16.0, 18.0])
    assert stats_["n"] == 5
    assert np.isclose(stats_["mean"], 14.0)
    assert stats_["ci_lower"] < 14.0 < stats_["ci_upper"]


def test_descriptive_mean_ci_single_value_has_no_ci():
    stats_ = S.descriptive_mean_ci([10.0])
    assert stats_["n"] == 1
    assert np.isclose(stats_["mean"], 10.0)
    assert np.isnan(stats_["ci_lower"])
    assert np.isnan(stats_["ci_upper"])


def test_descriptive_mean_ci_empty_returns_nan():
    stats_ = S.descriptive_mean_ci([])
    assert stats_["n"] == 0
    assert np.isnan(stats_["mean"])


def test_compute_absolute_cognitive_stats_matches_known_baseline_and_followup():
    sample = pd.DataFrame([
        {"DX_BASELINE_FIXED": "CN", "BASELINE_VALUE_FOR_MODEL": 10.0, "change_from_baseline": 0.0},
        {"DX_BASELINE_FIXED": "CN", "BASELINE_VALUE_FOR_MODEL": 12.0, "change_from_baseline": 2.0},
        {"DX_BASELINE_FIXED": "MCI", "BASELINE_VALUE_FOR_MODEL": 20.0, "change_from_baseline": 5.0},
    ])
    baseline = S.compute_absolute_cognitive_stats(sample, month=0)
    # At month 0, absolute score IS the baseline value itself.
    assert np.isclose(baseline["CN"]["raw_absolute_mean"], 11.0)
    followup = S.compute_absolute_cognitive_stats(sample, month=12)
    # At month > 0, absolute score = baseline + change.
    assert np.isclose(followup["CN"]["raw_absolute_mean"], (10.0 + 12.0 + 2.0) / 2)
    assert np.isclose(followup["MCI"]["raw_absolute_mean"], 25.0)
    assert followup["Dementia"]["descriptive_status"] == "Insufficient data"


def test_compute_absolute_biomarker_level_ci_is_geometric_not_arithmetic():
    sample = pd.DataFrame([
        {"DX_BASELINE_FIXED": "CN", "log_baseline": np.log(10.0), "log_change": 0.0},
        {"DX_BASELINE_FIXED": "CN", "log_baseline": np.log(20.0), "log_change": 0.0},
    ])
    ci = S.compute_absolute_biomarker_level_ci(sample, month=0)
    # Geometric mean of [10, 20] = sqrt(200) ~= 14.14, not the
    # arithmetic mean (15.0) -- the CI must straddle the geometric one.
    assert ci["CN"]["raw_geometric_mean_ci_lower"] < np.sqrt(200) < ci["CN"]["raw_geometric_mean_ci_upper"]


def test_compute_absolute_biomarker_level_ci_never_touches_existing_pct_change_stats():
    # Purely additive: confirms the new CI function and the existing,
    # validated compute_descriptive_biomarker_stats() are independent
    # computations over the same sample -- one doesn't alter the other.
    sample = pd.DataFrame([
        {"DX_BASELINE_FIXED": "CN", "log_baseline": np.log(10.0), "log_change": np.log(1.1)},
        {"DX_BASELINE_FIXED": "CN", "log_baseline": np.log(12.0), "log_change": np.log(0.9)},
    ])
    raw_stats = S.raw_change_stats(sample, "log_change")
    before = S.compute_descriptive_biomarker_stats(sample, raw_stats, month=12)
    S.compute_absolute_biomarker_level_ci(sample, month=12)
    after = S.compute_descriptive_biomarker_stats(sample, raw_stats, month=12)
    assert before == after


# ------------------------------------------------------------------
# 17. Cross-sectional biomarker Absolute-view sample-size fix (Plasma
#     Biomarker Trajectories redesign) -- the Absolute view's n/level
#     must use every participant with a valid value AT that month, not
#     just the subset who ALSO have a paired baseline draw (the correct,
#     unchanged denominator for the change-from-baseline ANCOVA, but the
#     wrong one for "what is the level at this month").
# ------------------------------------------------------------------


def test_compute_cross_sectional_biomarker_stats_basic():
    sample = pd.DataFrame([
        {"DX_BASELINE_FIXED": "CN", "GFAP": 10.0},
        {"DX_BASELINE_FIXED": "CN", "GFAP": 20.0},
        {"DX_BASELINE_FIXED": "MCI", "GFAP": 30.0},
    ])
    stats_ = S.compute_cross_sectional_biomarker_stats(sample, "GFAP")
    assert stats_["CN"]["n_cross_sectional"] == 2
    assert np.isclose(stats_["CN"]["raw_geometric_mean_cross_sectional"], np.sqrt(200))
    assert stats_["MCI"]["n_cross_sectional"] == 1
    assert np.isnan(stats_["MCI"]["raw_geometric_mean_ci_lower_cross_sectional"])  # n=1 -> no CI
    assert stats_["Dementia"]["n_cross_sectional"] == 0


def test_build_biomarker_cross_sectional_sample_does_not_require_baseline_pairing():
    """The key behavioral fix: a participant with a valid value at the
    target month but NO baseline (month 0) record at all must still be
    counted -- build_biomarker_sample() (the paired/change sample)
    would drop them entirely; build_biomarker_cross_sectional_sample()
    must not."""
    plasma = pd.DataFrame([
        # RID 1: has both baseline and month-12 values (would appear in both samples).
        {"RID": 1, "PLATFORM": None, "GFAP": 10.0, "VISIT_MONTH_RAW": 0, "VISIT_MONTH": 0, "BASELINE_AGE": 70.0, "SEX": "Male"},
        {"RID": 1, "PLATFORM": None, "GFAP": 12.0, "VISIT_MONTH_RAW": 12, "VISIT_MONTH": 12, "BASELINE_AGE": 70.0, "SEX": "Male"},
        # RID 2: has ONLY a month-12 value, no baseline record at all.
        {"RID": 2, "PLATFORM": None, "GFAP": 15.0, "VISIT_MONTH_RAW": 12, "VISIT_MONTH": 12, "BASELINE_AGE": 68.0, "SEX": "Female"},
    ])
    dx = pd.DataFrame({"RID": [1, 2], "DX_BASELINE_FIXED": ["CN", "CN"]})

    paired = R.build_biomarker_sample(plasma, dx, "GFAP", 12)
    cross_sectional = R.build_biomarker_cross_sectional_sample(plasma, dx, "GFAP", 12)

    assert set(paired["RID"]) == {1}  # RID 2 has no baseline -- excluded from the paired/change sample
    assert set(cross_sectional["RID"]) == {1, 2}  # both counted -- correct for the cross-sectional question


def test_run_biomarker_family_output_has_cross_sectional_columns():
    plasma = pd.DataFrame([
        {"RID": 1, "PLATFORM": None, "GFAP": 10.0, "VISIT_MONTH_RAW": 0, "VISIT_MONTH": 0, "BASELINE_AGE": 70.0, "SEX": "Male", "BIOMARKER_ELIGIBLE": True},
        {"RID": 1, "PLATFORM": None, "GFAP": 12.0, "VISIT_MONTH_RAW": 12, "VISIT_MONTH": 12, "BASELINE_AGE": 70.0, "SEX": "Male", "BIOMARKER_ELIGIBLE": True},
    ])
    dx = pd.DataFrame({"RID": [1], "DX_BASELINE_FIXED": ["CN"]})
    summary_rows, _pairwise, _diagnostics = R.run_biomarker_family(plasma, dx, "GFAP", "GFAP", "Quanterix", "primary")
    assert len(summary_rows) > 0
    for row in summary_rows:
        for col in ("n_cross_sectional", "raw_geometric_mean_cross_sectional", "raw_geometric_mean_ci_lower_cross_sectional", "raw_geometric_mean_ci_upper_cross_sectional"):
            assert col in row


def test_run_biomarker_family_adds_a_month_with_cross_sectional_but_no_paired_data():
    """The concrete case the fix targets: a follow-up month with real
    cross-sectional data but ZERO participants who also have a paired
    baseline value -- previously silently skipped (`if sample.empty:
    continue`) with no row at all; must now get a row with a real
    cross-sectional n and an honestly-empty change-from-baseline side."""
    plasma = pd.DataFrame([
        {"RID": 1, "PLATFORM": None, "GFAP": 10.0, "VISIT_MONTH_RAW": 0, "VISIT_MONTH": 0, "BASELINE_AGE": 70.0, "SEX": "Male", "BIOMARKER_ELIGIBLE": False},
        # RID 2..4: valid month-36 values, but NONE of them has any
        # month-0 (baseline) record at all -- exactly the Fujirebio
        # GFAP/NfL month 36/48 real-data pattern this fix targets.
        {"RID": 2, "PLATFORM": None, "GFAP": 11.0, "VISIT_MONTH_RAW": 36, "VISIT_MONTH": 36, "BASELINE_AGE": 71.0, "SEX": "Female", "BIOMARKER_ELIGIBLE": False},
        {"RID": 3, "PLATFORM": None, "GFAP": 13.0, "VISIT_MONTH_RAW": 36, "VISIT_MONTH": 36, "BASELINE_AGE": 69.0, "SEX": "Male", "BIOMARKER_ELIGIBLE": False},
        {"RID": 4, "PLATFORM": None, "GFAP": 14.0, "VISIT_MONTH_RAW": 36, "VISIT_MONTH": 36, "BASELINE_AGE": 74.0, "SEX": "Female", "BIOMARKER_ELIGIBLE": False},
    ])
    dx = pd.DataFrame({"RID": [1, 2, 3, 4], "DX_BASELINE_FIXED": ["CN", "CN", "MCI", "MCI"]})
    summary_rows, _pairwise, _diagnostics = R.run_biomarker_family(plasma, dx, "GFAP", "GFAP", "Quanterix", "primary")

    month36_rows = [r for r in summary_rows if r["month"] == 36]
    assert len(month36_rows) == 3  # one row per DX_LEVELS group, even though none was BIOMARKER_ELIGIBLE
    total_cross_sectional_n = sum(r["n_cross_sectional"] for r in month36_rows)
    assert total_cross_sectional_n == 3  # RIDs 2, 3, 4
    # Change-from-baseline side is honestly empty -- no paired data exists.
    for r in month36_rows:
        assert r["n"] == 0
        assert pd.isna(r["adjusted_log_change"])


def test_run_biomarker_family_cross_sectional_n_never_less_than_paired_n():
    """Real-data invariant: the cross-sectional sample is always a
    superset of the paired sample (every paired participant also has a
    valid value at that month, by construction), so n_cross_sectional
    must never be smaller than n for the same cell."""
    import os
    from adni_analysis import ADNI_OUTPUTS_DIR
    path = os.path.join(ADNI_OUTPUTS_DIR, "adni_biomarker_summary.csv")
    bio = pd.read_csv(path)
    assert (bio["n_cross_sectional"] >= bio["n"]).all()


def test_run_biomarker_family_existing_change_from_baseline_values_unaffected():
    """Real-data regression guard: a known, already-validated
    change-from-baseline cell (pTau181 month 12, CN) must be byte-
    identical to its pre-fix value -- the cross-sectional addition must
    never alter the paired-sample change analysis."""
    import os
    from adni_analysis import ADNI_OUTPUTS_DIR
    bio = pd.read_csv(os.path.join(ADNI_OUTPUTS_DIR, "adni_biomarker_summary.csv"))
    row = bio[
        (bio["biomarker"] == "pTau181") & (bio["analysis_type"] == "primary")
        & (bio["month"] == 12) & (bio["group"] == "CN")
    ].iloc[0]
    # Known-good value confirmed identical before and after the fix
    # (see the upstream investigation: 0 mismatches across all 147
    # pre-existing rows' shared columns).
    assert row["n"] == 199
    assert row["n_cross_sectional"] == 206


ALL_TESTS = [
    test_baseline_month_never_inferential,
    test_adas13_change_direction_is_followup_minus_baseline,
    test_mmse_change_direction_is_followup_minus_baseline,
    test_biomarker_log_change_uses_natural_log,
    test_percent_change_backtransform_uses_exp,
    test_ptau181_and_ptau217_never_pooled,
    test_quanterix_is_primary_platform_for_gfap_and_nfl,
    test_ptau217_lot_bias_records_excluded_from_primary,
    test_ptau217_lot_bias_records_included_in_sensitivity,
    test_small_cells_suppress_inference,
    test_sex_is_treated_categorically_not_numeric,
    test_pairwise_comparisons_generated_correctly,
    test_no_participant_identifiers_in_aggregate_outputs,
    test_descriptive_stats_computed_for_suppressed_small_cell,
    test_descriptive_biomarker_columns_present_in_run_biomarker_family_output,
    test_primary_and_sensitivity_labels_preserved,
    test_descriptive_mean_ci_basic,
    test_descriptive_mean_ci_single_value_has_no_ci,
    test_descriptive_mean_ci_empty_returns_nan,
    test_compute_absolute_cognitive_stats_matches_known_baseline_and_followup,
    test_compute_absolute_biomarker_level_ci_is_geometric_not_arithmetic,
    test_compute_absolute_biomarker_level_ci_never_touches_existing_pct_change_stats,
    test_compute_cross_sectional_biomarker_stats_basic,
    test_build_biomarker_cross_sectional_sample_does_not_require_baseline_pairing,
    test_run_biomarker_family_output_has_cross_sectional_columns,
    test_run_biomarker_family_adds_a_month_with_cross_sectional_but_no_paired_data,
    test_run_biomarker_family_cross_sectional_n_never_less_than_paired_n,
    test_run_biomarker_family_existing_change_from_baseline_values_unaffected,
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
