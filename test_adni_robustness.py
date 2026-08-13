# ============================================================
# TESTS for adni_robustness.py / run_adni_robustness.py (final
# robustness pass: HC3 covariance, influence sensitivity, dashboard
# eligibility classification).
#
# Every fixture here is hand-built synthetic data (fake RIDs, fake
# values) -- never real ADNI participant data.
#
# Run: .venv/bin/python test_adni_robustness.py
# ============================================================

import numpy as np
import pandas as pd

import adni_robustness as RB
import adni_stats as S
import run_adni_robustness as RR


# ------------------------------------------------------------------
# Shared synthetic fixtures
# ------------------------------------------------------------------


def _make_ancova_ready_df(seed=0, n_per_group=30, outlier=False):
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
    if outlier:
        # One deliberately extreme, high-leverage observation in CN --
        # enough to noticeably shift the CN adjusted mean when excluded,
        # without collapsing any group below MIN_GROUP_N.
        rows.append(
            {
                "RID": 9999,
                "DX_BASELINE_FIXED": "CN",
                "BASELINE_VALUE_FOR_MODEL": 20.0,
                "BASELINE_AGE": 72.0,
                "SEX": "Male",
                "change_from_baseline": 80.0,
            }
        )
    return pd.DataFrame(rows)


def _fit(df):
    return S.fit_ancova(
        df, outcome_col="change_from_baseline",
        continuous_covariates=("BASELINE_VALUE_FOR_MODEL", "BASELINE_AGE"),
        categorical_covariates=("SEX",),
    )


# ------------------------------------------------------------------
# 1. HC3 uses the identical model specification
# ------------------------------------------------------------------


def test_hc3_uses_identical_model_specification():
    df = _make_ancova_ready_df(seed=1)
    fit = _fit(df)
    hc3_model = RB.fit_hc3(fit["model"])

    # Same underlying model object -- HC3 re-covariances, never re-specifies.
    assert hc3_model.model is fit["model"].model
    # Identical point estimates (order-matched via param_names_of, since
    # HC3's .params loses its pandas index -- see adni_stats.param_names_of).
    conv_names = S.param_names_of(fit["model"])
    hc3_names = S.param_names_of(hc3_model)
    assert conv_names == hc3_names
    assert np.allclose(np.asarray(fit["model"].params), np.asarray(hc3_model.params))
    # But the covariance matrix (and therefore SE) must differ -- that's
    # the entire point of refitting with HC3.
    assert not np.allclose(fit["model"].bse, np.ravel(hc3_model.bse))


# ------------------------------------------------------------------
# 2. Suppressed models remain suppressed (never refit)
# ------------------------------------------------------------------


def test_suppressed_models_remain_suppressed_no_refit():
    small_df = pd.DataFrame(
        [{"RID": i, "DX_BASELINE_FIXED": "CN", "BASELINE_VALUE_FOR_MODEL": 20.0,
          "BASELINE_AGE": 70.0, "SEX": "Male", "change_from_baseline": 1.0} for i in range(20)]
        + [{"RID": 100 + i, "DX_BASELINE_FIXED": "MCI", "BASELINE_VALUE_FOR_MODEL": 20.0,
            "BASELINE_AGE": 70.0, "SEX": "Male", "change_from_baseline": 2.0} for i in range(20)]
        + [{"RID": 200 + i, "DX_BASELINE_FIXED": "Dementia", "BASELINE_VALUE_FOR_MODEL": 20.0,
            "BASELINE_AGE": 70.0, "SEX": "Male", "change_from_baseline": 3.0} for i in range(3)]  # n=3 < 10
    )
    robustness_rows, eligibility_row = RR._process_one(("ADAS_COG13", "", "primary"), small_df, month=12)

    assert robustness_rows == []  # no HC3 rows, no influence rows -- nothing was refit
    assert eligibility_row["classification"] == RB.CLASS_DESCRIPTIVE
    assert "Suppressed" in eligibility_row["reason"]


# ------------------------------------------------------------------
# 3. Influential observations are removed only in sensitivity analysis
# ------------------------------------------------------------------


def test_influential_observations_removed_only_in_sensitivity_analysis():
    df = _make_ancova_ready_df(seed=2, n_per_group=40, outlier=True)
    fit = _fit(df)

    original_n = len(fit["_df"])
    rows, n_excluded = RB.run_influence_sensitivity(
        fit, fit["group_term"], fit["_group_levels"], fit["_reference_level"]
    )
    assert n_excluded >= 1
    # The primary fit's own data must be completely untouched by the
    # sensitivity computation -- same row count, outlier RID still present.
    assert len(fit["_df"]) == original_n
    assert 9999 in set(fit["_df"]["RID"])
    # A comparison was actually produced (not skipped as "not assessable").
    assert len(rows) > 0
    cn_row = next(r for r in rows if r["level"] == "adjusted_mean" and r["group_or_comparison"] == "CN")
    # Excluding one extreme +80 outlier from CN should pull CN's adjusted
    # mean down relative to the conventional (outlier-included) estimate.
    assert cn_row["alternative_estimate"] < cn_row["conventional_estimate"]


# ------------------------------------------------------------------
# 4. Significance changes are detected correctly
# ------------------------------------------------------------------


def test_significance_change_detection():
    # p crosses 0.05 in one direction only.
    sig, _ = RB._is_significant(0.03, None, None)
    not_sig, _ = RB._is_significant(0.08, None, None)
    assert sig is True
    assert not_sig is False

    # CI-exclusion-of-zero logic.
    _, excludes_zero = RB._is_significant(np.nan, 1.0, 2.0)
    _, includes_zero = RB._is_significant(np.nan, -1.0, 2.0)
    assert excludes_zero is True
    assert includes_zero is False

    # End-to-end: a pairwise contrast whose CI straddles zero under one
    # covariance treatment and excludes it under the other must flag.
    fake_pairwise_conventional = {"p_value": 0.20, "ci_lower": -0.5, "ci_upper": 1.5}
    fake_pairwise_hc3 = {"p_value": 0.01, "ci_lower": 0.2, "ci_upper": 1.1}
    conv_sig, conv_excl = RB._is_significant(**{
        "p": fake_pairwise_conventional["p_value"],
        "ci_lower": fake_pairwise_conventional["ci_lower"],
        "ci_upper": fake_pairwise_conventional["ci_upper"],
    })
    hc3_sig, hc3_excl = RB._is_significant(**{
        "p": fake_pairwise_hc3["p_value"],
        "ci_lower": fake_pairwise_hc3["ci_lower"],
        "ci_upper": fake_pairwise_hc3["ci_upper"],
    })
    would_flag = (conv_sig != hc3_sig) or (conv_excl != hc3_excl)
    assert would_flag is True


# ------------------------------------------------------------------
# 5. GFAP/NfL inference is not fabricated
# ------------------------------------------------------------------


def test_gfap_nfl_inference_not_fabricated_when_cells_are_small():
    # Every group well below MIN_GROUP_N=10 -- must never be pushed into
    # "Adjusted analysis" or "Sensitivity concern" just because it's GFAP/NfL.
    tiny_gfap = pd.DataFrame(
        [{"RID": i, "DX_BASELINE_FIXED": g, "BASELINE_VALUE_FOR_MODEL": 100.0,
          "BASELINE_AGE": 70.0, "SEX": "Male", "log_change": 0.1}
         for i, g in enumerate(["CN"] * 3 + ["MCI"] * 3 + ["Dementia"] * 3)]
    )
    _, eligibility_row = RR._process_one(("GFAP", "Quanterix", "primary"), tiny_gfap, month=12)
    assert eligibility_row["classification"] not in (RB.CLASS_ADJUSTED, RB.CLASS_SENSITIVITY_CONCERN)
    assert eligibility_row["classification"] == RB.CLASS_DESCRIPTIVE


# ------------------------------------------------------------------
# 6. Participant identifiers never enter outputs
# ------------------------------------------------------------------


def test_no_participant_identifiers_in_outputs():
    df = _make_ancova_ready_df(seed=4, n_per_group=25)
    # Distinctive, large RIDs -- so a coincidental match against an
    # ordinary small integer elsewhere in a row (month, n_excluded, ...)
    # can't produce a false positive here.
    df["RID"] = df["RID"] + 900000
    robustness_rows, eligibility_row = RR._process_one(("ADAS_COG13", "", "primary"), df, month=12)

    known_rids = set(df["RID"])
    for row in robustness_rows + [eligibility_row]:
        assert "RID" not in row
        for value in row.values():
            if isinstance(value, (int, np.integer)):
                assert value not in known_rids


ALL_TESTS = [
    test_hc3_uses_identical_model_specification,
    test_suppressed_models_remain_suppressed_no_refit,
    test_influential_observations_removed_only_in_sensitivity_analysis,
    test_significance_change_detection,
    test_gfap_nfl_inference_not_fabricated_when_cells_are_small,
    test_no_participant_identifiers_in_outputs,
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
