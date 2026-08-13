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
    )
    assert score == 100
    assert len(reasons) == 4


def test_completely_unrelated_profile_scores_0():
    score, reasons = compute_relevance_score(
        ["Metabolism"], "Biologic", "STT", "NA",
        ["Amyloid"], "Small Molecule", "DTT", "Phase 3",
    )
    assert score == 0
    assert reasons == []


def test_partial_overlap_any_shared_pathway_gives_full_pathway_credit():
    # sharing just ONE of three reference pathways still earns the full
    # 40 points — any real mechanistic overlap is the meaningful signal,
    # not exact pathway-set equality (see module docstring)
    score, reasons = compute_relevance_score(
        ["Amyloid"], "Biologic", "STT", "NA",
        ["Amyloid", "Tau", "Neuroprotection"], "Small Molecule", "DTT", "Phase 3",
    )
    assert score == 40
    assert "Amyloid" in reasons[0]


def test_modality_match_contributes_35_points():
    # Phase 1 vs. Phase 4 is 5 ranks apart (see PHASE_RANK_FOR_SCORING)
    # -> deliberately contributes 0 phase points, isolating modality's
    # own contribution. Weighted above phase/purpose-class -- see
    # module-level comment on _MODALITY_POINTS.
    score, _ = compute_relevance_score(
        [], "Small Molecule", "", "Phase 1",
        [], "Small Molecule", "", "Phase 4",
    )
    assert score == 35


def test_purpose_class_match_contributes_15_points():
    score, _ = compute_relevance_score(
        [], "", "DTT", "Phase 1",
        [], "", "DTT", "Phase 4",
    )
    assert score == 15


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


def test_same_phase_contributes_10_points():
    score, reasons = compute_relevance_score(
        [], "", "", "Phase 2",
        [], "", "", "Phase 2",
    )
    assert score == 10
    assert "Same phase" in reasons[0]


def test_adjacent_phase_contributes_5_points():
    # "Phase 1/Phase 2" sits between "Phase 1" and "Phase 2" in
    # PHASE_RANK_FOR_SCORING (it's a real, distinct ct.gov designation,
    # not just a midpoint label) — one rank step from "Phase 1"
    score, reasons = compute_relevance_score(
        [], "", "", "Phase 1",
        [], "", "", "Phase 1/Phase 2",
    )
    assert score == 5
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
    )
    assert reasons[0].startswith("Shares target pathway")
    assert reasons[1].startswith("Same modality")
    assert reasons[2].startswith("Same treatment approach")
    assert reasons[3].startswith("Same phase")
    assert all(isinstance(r, str) and r for r in reasons)


ALL_TESTS = [
    test_identical_profile_scores_100,
    test_completely_unrelated_profile_scores_0,
    test_partial_overlap_any_shared_pathway_gives_full_pathway_credit,
    test_modality_match_contributes_35_points,
    test_purpose_class_match_contributes_15_points,
    test_blank_purpose_class_never_matches_and_never_scores,
    test_same_phase_contributes_10_points,
    test_adjacent_phase_contributes_5_points,
    test_two_phases_apart_contributes_no_points,
    test_unrecognized_phase_value_does_not_crash_and_scores_no_phase_points,
    test_phase_rank_covers_every_pipeline_viz_phase_order_value,
    test_score_never_exceeds_100_or_goes_negative,
    test_reasons_are_human_readable_and_ordered,
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
