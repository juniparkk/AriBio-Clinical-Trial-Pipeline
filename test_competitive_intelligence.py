# ============================================================
# TESTS for competitive_intelligence.py (AriBio relevance score)
#
# Plain-Python tests (no pytest install needed) — run with:
#     .venv/bin/python test_competitive_intelligence.py
# ============================================================

from competitive_intelligence import compute_relevance_score, PHASE_RANK_FOR_SCORING


def test_max_profile_scores_100():
    score, reasons = compute_relevance_score(
        ["Amyloid"], "Small Molecule", "Phase 3", "Small Molecule",
        sponsor_type="Company", reference_sponsor_type="Company",
        max_enrollment=600,
    )
    assert score == 100
    assert len(reasons) == 5


def test_completely_unrelated_profile_scores_0():
    score, reasons = compute_relevance_score(
        ["Symptomatic"], "Biologic", "NA", "Small Molecule",
    )
    assert score == 0
    assert reasons == []


def test_target_pathway_multiple_disease_modifying_matches_earn_the_points_only_once():
    # target pathway is scored absolutely now (is ANY of this drug's own
    # pathways disease-modifying), not compared to the reference's
    # pathway(s) -- two matching pathways still earn the flat 25 points
    # once, not per-pathway credit.
    score, reasons = compute_relevance_score(
        ["Amyloid", "Tau"], "", "NA", "",
    )
    assert score == 25
    assert "Amyloid" in reasons[0] and "Tau" in reasons[0]


def test_target_pathway_symptomatic_and_neuropsychiatric_earn_no_points():
    # Real target_pathways categories elsewhere in this pipeline, but
    # they manage symptoms rather than the disease process itself, so
    # neither counts as "disease-modifying" here.
    score, reasons = compute_relevance_score(
        ["Symptomatic", "Neuropsychiatric"], "", "NA", "",
    )
    assert score == 0
    assert reasons == []


def test_target_pathway_all_five_disease_modifying_categories_earn_points():
    for pathway in ["Amyloid", "Tau", "Inflammation", "Neuroprotection", "Metabolism"]:
        score, _ = compute_relevance_score([pathway], "", "NA", "")
        assert score == 25, pathway


def test_modality_match_contributes_10_points():
    score, _ = compute_relevance_score(
        [], "Small Molecule", "Phase 1", "Small Molecule",
    )
    assert score == 10


def test_phase_3_contributes_15_points():
    score, reasons = compute_relevance_score([], "", "Phase 3", "")
    assert score == 15
    assert "Phase 3" in reasons[0]


def test_phase_2_contributes_5_points():
    score, reasons = compute_relevance_score([], "", "Phase 2", "")
    assert score == 5
    assert "Phase 2" in reasons[0]


def test_other_phases_contribute_no_phase_points():
    # Absolute scale now, not proximity-to-reference -- no partial
    # credit for phases "adjacent" to Phase 3, and an unrecognized
    # phase string must not crash.
    for phase in ["NA", "Early Phase 1", "Phase 1", "Phase 1/Phase 2",
                  "Phase 2/Phase 3", "Phase 4", "SomeUnrecognizedPhase"]:
        score, reasons = compute_relevance_score([], "", phase, "")
        assert score == 0, phase
        assert reasons == [], phase


def test_phase_rank_covers_every_pipeline_viz_phase_order_value():
    # PHASE_RANK_FOR_SCORING is still used elsewhere (ctgov_changes.py's
    # phase-change detection, competitive_attention.py's
    # _is_advancement) even though compute_relevance_score() itself no
    # longer scores phase by proximity/rank -- regression guard that
    # every phase_clean value pipeline_viz.py's clean_phase() can
    # produce stays rankable there.
    expected = {"NA", "Early Phase 1", "Phase 1", "Phase 1/Phase 2", "Phase 2",
                "Phase 2/Phase 3", "Phase 3", "Phase 4"}
    assert set(PHASE_RANK_FOR_SCORING.keys()) == expected


def test_score_never_exceeds_100_or_goes_negative():
    score, _ = compute_relevance_score(
        ["Amyloid", "Tau"], "Small Molecule", "Phase 3", "Small Molecule",
        sponsor_type="Company", reference_sponsor_type="Company",
        max_enrollment=10000,
    )
    assert 0 <= score <= 100


def test_reasons_are_human_readable_and_ordered():
    score, reasons = compute_relevance_score(
        ["Amyloid"], "Small Molecule", "Phase 3", "Small Molecule",
        sponsor_type="Company", reference_sponsor_type="Company",
        max_enrollment=600,
    )
    assert reasons[0].startswith("Disease-modifying target pathway")
    assert reasons[1].startswith("Same modality")
    assert reasons[2].startswith("Reached Phase 3")
    assert reasons[3].startswith("Same sponsor type")
    assert reasons[4].startswith("Large trial")
    assert all(isinstance(r, str) and r for r in reasons)


def test_sponsor_type_match_contributes_20_points():
    score, reasons = compute_relevance_score(
        [], "", "NA", "",
        sponsor_type="Company", reference_sponsor_type="Company",
    )
    assert score == 20
    assert "Same sponsor type: Company" in reasons[0]


def test_sponsor_type_mismatch_contributes_no_points():
    score, reasons = compute_relevance_score(
        [], "", "NA", "",
        sponsor_type="University/Institution", reference_sponsor_type="Company",
    )
    assert score == 0
    assert reasons == []


def test_blank_sponsor_type_never_matches_and_never_scores():
    score, reasons = compute_relevance_score(
        [], "", "NA", "",
        sponsor_type=None, reference_sponsor_type=None,
    )
    assert score == 0
    assert reasons == []


def test_large_trial_over_550_contributes_30_points():
    score, reasons = compute_relevance_score([], "", "NA", "", max_enrollment=551)
    assert score == 30
    assert "Large trial" in reasons[0]


def test_trial_of_exactly_550_does_not_earn_large_trial_points():
    score, reasons = compute_relevance_score([], "", "NA", "", max_enrollment=550)
    assert score == 0
    assert reasons == []


def test_small_trial_does_not_earn_large_trial_points():
    score, reasons = compute_relevance_score([], "", "NA", "", max_enrollment=300)
    assert score == 0
    assert reasons == []


def test_donanemab_real_profile_scores_75():
    # Regression guard for a real-world profile: Donanemab is a
    # Biologic (different modality from AR1001's Small Molecule) that
    # shares the Amyloid disease-modifying pathway, sits at Phase 4 (no
    # phase points under the absolute Phase 3/Phase 2 scale), shares
    # AR1001's Company sponsor type, and runs a large (>550-participant)
    # trial:
    # 25 (pathway) + 0 (modality) + 0 (phase) + 20 (sponsor) + 30 (trial size) = 75.
    score, _ = compute_relevance_score(
        ["Amyloid"], "Biologic", "Phase 4", "Small Molecule",
        sponsor_type="Company", reference_sponsor_type="Company",
        status="Recruiting", latest_activity_year=2030, max_enrollment=2996,
    )
    assert score == 75


def test_discontinued_status_caps_score_at_20():
    score, reasons = compute_relevance_score(
        ["Amyloid"], "Small Molecule", "Phase 3", "Small Molecule",
        status="Discontinued",
    )
    assert score == 20
    assert "Capped at 20/100" in reasons[-1]
    assert "discontinued/withdrawn" in reasons[-1]


def test_no_recent_trial_activity_caps_score_at_20():
    score, reasons = compute_relevance_score(
        ["Amyloid"], "Small Molecule", "Phase 3", "Small Molecule",
        latest_activity_year=2015,
    )
    assert score == 20
    assert "no trial activity since before 2016" in reasons[-1]


def test_small_trial_caps_score_at_20():
    score, reasons = compute_relevance_score(
        ["Amyloid"], "Small Molecule", "Phase 3", "Small Molecule",
        max_enrollment=209,
    )
    assert score == 20
    assert "largest trial under 210 participants" in reasons[-1]


def test_enrollment_of_exactly_210_does_not_trigger_the_low_relevance_cap():
    # 210 participants is under the >550 large-trial bonus threshold
    # (so no trial-size points either), but it doesn't trip the <210
    # low-relevance cap: 25 (pathway) + 10 (modality) + 15 (phase 3)
    # + 20 (sponsor) = 70, uncapped.
    score, _ = compute_relevance_score(
        ["Amyloid"], "Small Molecule", "Phase 3", "Small Molecule",
        sponsor_type="Company", reference_sponsor_type="Company",
        max_enrollment=210,
    )
    assert score == 70


def test_low_relevance_cap_never_raises_a_score_that_is_already_lower():
    score, reasons = compute_relevance_score(
        [], "", "Phase 1", "Small Molecule",
        status="Discontinued", latest_activity_year=2010, max_enrollment=10,
    )
    assert score == 0
    assert reasons == []


def test_low_relevance_flags_are_ignored_when_not_provided():
    # status/latest_activity_year/max_enrollment default to None -- must
    # never be treated as disqualifying just because they're absent
    score, reasons = compute_relevance_score(
        ["Amyloid"], "Small Molecule", "Phase 3", "Small Molecule",
        sponsor_type="Company", reference_sponsor_type="Company",
        max_enrollment=600,
    )
    assert score == 100
    assert not any("Capped" in r for r in reasons)


ALL_TESTS = [
    test_max_profile_scores_100,
    test_completely_unrelated_profile_scores_0,
    test_target_pathway_multiple_disease_modifying_matches_earn_the_points_only_once,
    test_target_pathway_symptomatic_and_neuropsychiatric_earn_no_points,
    test_target_pathway_all_five_disease_modifying_categories_earn_points,
    test_modality_match_contributes_10_points,
    test_phase_3_contributes_15_points,
    test_phase_2_contributes_5_points,
    test_other_phases_contribute_no_phase_points,
    test_phase_rank_covers_every_pipeline_viz_phase_order_value,
    test_score_never_exceeds_100_or_goes_negative,
    test_reasons_are_human_readable_and_ordered,
    test_sponsor_type_match_contributes_20_points,
    test_sponsor_type_mismatch_contributes_no_points,
    test_blank_sponsor_type_never_matches_and_never_scores,
    test_large_trial_over_550_contributes_30_points,
    test_trial_of_exactly_550_does_not_earn_large_trial_points,
    test_small_trial_does_not_earn_large_trial_points,
    test_donanemab_real_profile_scores_75,
    test_discontinued_status_caps_score_at_20,
    test_no_recent_trial_activity_caps_score_at_20,
    test_small_trial_caps_score_at_20,
    test_enrollment_of_exactly_210_does_not_trigger_the_low_relevance_cap,
    test_low_relevance_cap_never_raises_a_score_that_is_already_lower,
    test_low_relevance_flags_are_ignored_when_not_provided,
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
