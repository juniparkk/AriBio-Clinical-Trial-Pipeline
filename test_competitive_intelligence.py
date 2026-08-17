# ============================================================
# TESTS for competitive_intelligence.py (AriBio relevance score)
#
# Plain-Python tests (no pytest install needed) — run with:
#     .venv/bin/python test_competitive_intelligence.py
# ============================================================

from competitive_intelligence import compute_relevance_score, PHASE_RANK_FOR_SCORING


def test_identical_profile_scores_100():
    score, reasons = compute_relevance_score(
        ["Amyloid", "Tau", "Neuroprotection"], "Small Molecule", "DTT", "Phase 3",
        ["Amyloid", "Tau", "Neuroprotection"], "Small Molecule", "DTT", "Phase 3",
        sponsor_type="Company", reference_sponsor_type="Company",
    )
    assert score == 100
    assert len(reasons) == 5


def test_completely_unrelated_profile_scores_0():
    score, reasons = compute_relevance_score(
        ["Metabolism"], "Biologic", "STT", "NA",
        ["Amyloid"], "Small Molecule", "DTT", "Phase 3",
    )
    assert score == 0
    assert reasons == []


def test_partial_overlap_any_shared_pathway_gives_full_pathway_credit():
    # sharing just ONE of three reference pathways still earns the full
    # 54 points — any real mechanistic overlap is the meaningful signal,
    # not exact pathway-set equality (see module docstring)
    score, reasons = compute_relevance_score(
        ["Amyloid"], "Biologic", "STT", "NA",
        ["Amyloid", "Tau", "Neuroprotection"], "Small Molecule", "DTT", "Phase 3",
    )
    assert score == 54
    assert "Amyloid" in reasons[0]


def test_modality_match_contributes_5_points():
    # Phase 1 vs. Phase 4 is 5 ranks apart (see PHASE_RANK_FOR_SCORING)
    # -> deliberately contributes 0 phase points, isolating modality's
    # own contribution. Weighted well below target pathway -- see
    # module-level comment on _MODALITY_POINTS.
    score, _ = compute_relevance_score(
        [], "Small Molecule", "", "Phase 1",
        [], "Small Molecule", "", "Phase 4",
    )
    assert score == 5


def test_purpose_class_match_contributes_22_points():
    score, _ = compute_relevance_score(
        [], "", "DTT", "Phase 1",
        [], "", "DTT", "Phase 4",
    )
    assert score == 22


def test_blank_purpose_class_never_matches_and_never_scores():
    # a drug with no NIH-sourced purpose classification must not
    # silently "match" another blank one — blank means "unknown," not
    # "confirmed same as reference". Phases deliberately far apart too,
    # so this isolates the purpose_class/modality blank-vs-blank cases.
    score, reasons = compute_relevance_score(
        [], "", "", "Phase 1",
        [], "", "", "Phase 4",
    )
    assert score == 0
    assert reasons == []


def test_same_phase_contributes_9_points():
    score, reasons = compute_relevance_score(
        [], "", "", "Phase 2",
        [], "", "", "Phase 2",
    )
    assert score == 9
    assert "Same phase" in reasons[0]


def test_adjacent_phase_contributes_9_points():
    # "Phase 1/Phase 2" sits between "Phase 1" and "Phase 2" in
    # PHASE_RANK_FOR_SCORING (it's a real, distinct ct.gov designation,
    # not just a midpoint label) — one rank step from "Phase 1".
    # Adjacent and same-phase now share one "phase proximity" tier/point
    # value (see _PHASE_PROXIMITY_POINTS) — only the reason text
    # distinguishes them, not the score.
    score, reasons = compute_relevance_score(
        [], "", "", "Phase 1",
        [], "", "", "Phase 1/Phase 2",
    )
    assert score == 9
    assert "Adjacent phase" in reasons[0]


def test_two_phases_apart_contributes_no_points():
    # "Phase 1" -> "Phase 2" is TWO rank steps apart ("Phase 1/Phase 2"
    # sits between them), not adjacent
    score, reasons = compute_relevance_score(
        [], "", "", "Phase 1",
        [], "", "", "Phase 2",
    )
    assert score == 0
    assert reasons == []


def test_unrecognized_phase_value_does_not_crash_and_scores_no_phase_points():
    score, reasons = compute_relevance_score(
        [], "", "", "SomeUnrecognizedPhase",
        [], "", "", "Phase 3",
    )
    assert score == 0
    assert reasons == []


def test_phase_rank_covers_every_pipeline_viz_phase_order_value():
    # regression guard: every phase_clean value pipeline_viz.py's
    # clean_phase() can produce must be rankable here, or a same-phase/
    # adjacent-phase comparison would silently score 0 for a drug at
    # that phase, even against an identical reference
    expected = {"NA", "Early Phase 1", "Phase 1", "Phase 1/Phase 2", "Phase 2",
                "Phase 2/Phase 3", "Phase 3", "Phase 4"}
    assert set(PHASE_RANK_FOR_SCORING.keys()) == expected


def test_score_never_exceeds_100_or_goes_negative():
    score, _ = compute_relevance_score(
        ["Amyloid", "Tau"], "Small Molecule", "DTT", "Phase 3",
        ["Amyloid", "Tau", "Neuroprotection"], "Small Molecule", "DTT", "Phase 3",
    )
    assert 0 <= score <= 100


def test_reasons_are_human_readable_and_ordered():
    score, reasons = compute_relevance_score(
        ["Amyloid"], "Small Molecule", "DTT", "Phase 3",
        ["Amyloid"], "Small Molecule", "DTT", "Phase 3",
        sponsor_type="Company", reference_sponsor_type="Company",
    )
    assert reasons[0].startswith("Shares target pathway")
    assert reasons[1].startswith("Same modality")
    assert reasons[2].startswith("Same treatment approach")
    assert reasons[3].startswith("Same phase")
    assert reasons[4].startswith("Same sponsor type")
    assert all(isinstance(r, str) and r for r in reasons)


def test_sponsor_type_match_contributes_10_points():
    # "NA" vs. "Phase 4" (7 ranks apart, see PHASE_RANK_FOR_SCORING) so
    # phase proximity contributes 0, isolating sponsor type's own
    # contribution -- unlike an unrecognized string, "NA" IS a real,
    # rankable phase value, so it would otherwise match itself.
    score, reasons = compute_relevance_score(
        [], "", "", "NA",
        [], "", "", "Phase 4",
        sponsor_type="Company", reference_sponsor_type="Company",
    )
    assert score == 10
    assert "Same sponsor type: Company" in reasons[0]


def test_sponsor_type_mismatch_contributes_no_points():
    score, reasons = compute_relevance_score(
        [], "", "", "NA",
        [], "", "", "Phase 4",
        sponsor_type="University/Institution", reference_sponsor_type="Company",
    )
    assert score == 0
    assert reasons == []


def test_blank_sponsor_type_never_matches_and_never_scores():
    score, reasons = compute_relevance_score(
        [], "", "", "NA",
        [], "", "", "Phase 4",
        sponsor_type=None, reference_sponsor_type=None,
    )
    assert score == 0
    assert reasons == []


def test_donanemab_real_profile_scores_95():
    # Regression guard for AriBio's real-world calibration target:
    # Donanemab (Biologic, shares Amyloid pathway with AR1001, same DTT
    # approach, one phase step from AR1001's Phase 3, same sponsor type
    # as AR1001 -- both Company-sponsored (Eli Lilly / AriBio) -- and
    # actively recruiting with recent trial activity and a large
    # enrollment) must land at 95/100.
    score, _ = compute_relevance_score(
        ["Amyloid"], "Biologic", "DTT", "Phase 4",
        ["Amyloid", "Tau", "Neuroprotection"], "Small Molecule", "DTT", "Phase 3",
        sponsor_type="Company", reference_sponsor_type="Company",
        status="Recruiting", latest_activity_year=2030, max_enrollment=2996,
    )
    assert score == 95


def test_discontinued_status_caps_score_at_20():
    score, reasons = compute_relevance_score(
        ["Amyloid"], "Small Molecule", "DTT", "Phase 3",
        ["Amyloid"], "Small Molecule", "DTT", "Phase 3",
        status="Discontinued",
    )
    assert score == 20
    assert "Capped at 20/100" in reasons[-1]
    assert "discontinued/withdrawn" in reasons[-1]


def test_no_recent_trial_activity_caps_score_at_20():
    score, reasons = compute_relevance_score(
        ["Amyloid"], "Small Molecule", "DTT", "Phase 3",
        ["Amyloid"], "Small Molecule", "DTT", "Phase 3",
        latest_activity_year=2015,
    )
    assert score == 20
    assert "no trial activity since before 2016" in reasons[-1]


def test_small_trial_caps_score_at_20():
    score, reasons = compute_relevance_score(
        ["Amyloid"], "Small Molecule", "DTT", "Phase 3",
        ["Amyloid"], "Small Molecule", "DTT", "Phase 3",
        max_enrollment=209,
    )
    assert score == 20
    assert "largest trial under 210 participants" in reasons[-1]


def test_enrollment_of_exactly_210_does_not_trigger_cap():
    score, _ = compute_relevance_score(
        ["Amyloid"], "Small Molecule", "DTT", "Phase 3",
        ["Amyloid"], "Small Molecule", "DTT", "Phase 3",
        sponsor_type="Company", reference_sponsor_type="Company",
        max_enrollment=210,
    )
    assert score == 100


def test_low_relevance_cap_never_raises_a_score_that_is_already_lower():
    score, reasons = compute_relevance_score(
        [], "", "", "Phase 1",
        ["Amyloid"], "Small Molecule", "DTT", "Phase 3",
        status="Discontinued", latest_activity_year=2010, max_enrollment=10,
    )
    assert score == 0
    assert reasons == []


def test_low_relevance_flags_are_ignored_when_not_provided():
    # status/latest_activity_year/max_enrollment default to None -- must
    # never be treated as disqualifying just because they're absent
    score, reasons = compute_relevance_score(
        ["Amyloid"], "Small Molecule", "DTT", "Phase 3",
        ["Amyloid"], "Small Molecule", "DTT", "Phase 3",
        sponsor_type="Company", reference_sponsor_type="Company",
    )
    assert score == 100
    assert not any("Capped" in r for r in reasons)


ALL_TESTS = [
    test_identical_profile_scores_100,
    test_completely_unrelated_profile_scores_0,
    test_partial_overlap_any_shared_pathway_gives_full_pathway_credit,
    test_modality_match_contributes_5_points,
    test_purpose_class_match_contributes_22_points,
    test_blank_purpose_class_never_matches_and_never_scores,
    test_same_phase_contributes_9_points,
    test_adjacent_phase_contributes_9_points,
    test_two_phases_apart_contributes_no_points,
    test_unrecognized_phase_value_does_not_crash_and_scores_no_phase_points,
    test_phase_rank_covers_every_pipeline_viz_phase_order_value,
    test_score_never_exceeds_100_or_goes_negative,
    test_reasons_are_human_readable_and_ordered,
    test_sponsor_type_match_contributes_10_points,
    test_sponsor_type_mismatch_contributes_no_points,
    test_blank_sponsor_type_never_matches_and_never_scores,
    test_donanemab_real_profile_scores_95,
    test_discontinued_status_caps_score_at_20,
    test_no_recent_trial_activity_caps_score_at_20,
    test_small_trial_caps_score_at_20,
    test_enrollment_of_exactly_210_does_not_trigger_cap,
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
