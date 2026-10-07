"""Fixture tests for selection.py. Run: python -m pytest test_selection.py -q"""
from selection import round_means, pareto_front, single_best

# The real July Sports run: rounds 1,2,4,7,8 tie at 0.76 on validation while
# costing very different amounts. Round 1 is the untouched seed.
SPORTS = [
    {"round": 1, "score": 0.76, "avg_cost": 3.27e-06},
    {"round": 2, "score": 0.76, "avg_cost": 9.95e-05},
    {"round": 3, "score": 0.44, "avg_cost": 9.12e-04},
    {"round": 4, "score": 0.76, "avg_cost": 3.39e-04},
    {"round": 5, "score": 0.74, "avg_cost": 1.11e-03},
    {"round": 6, "score": 0.56, "avg_cost": 1.00e-05},
    {"round": 7, "score": 0.76, "avg_cost": 1.14e-04},
    {"round": 8, "score": 0.76, "avg_cost": 6.84e-04},
    {"round": 9, "score": 0.54, "avg_cost": 1.07e-04},
]


def test_repeats_are_averaged_not_counted_twice():
    entries = [{"round": 1, "score": 0.8, "avg_cost": 1.0},
               {"round": 1, "score": 0.6, "avg_cost": 3.0}]
    m = round_means(entries)
    assert m[1]["score"] == 0.7 and m[1]["cost"] == 2.0 and m[1]["repeats"] == 2


def test_rounds_with_none_score_are_dropped():
    entries = [{"round": 1, "score": 0.5, "avg_cost": 1.0},
               {"round": 2, "score": None, "avg_cost": 0.0}]
    assert set(round_means(entries)) == {1}


def test_single_best_breaks_ties_by_cost_not_position():
    m = round_means(SPORTS)
    # Five rounds tie at 0.76; round 1 is cheapest so it wins on the stated rule.
    assert single_best(m) == 1
    # But it must win for the cost reason, not because it came first. Make an
    # equally-scoring later round cheaper and it has to take the slot.
    cheaper = [dict(e) for e in SPORTS]
    cheaper[7]["avg_cost"] = 1e-09          # round 8
    assert single_best(round_means(cheaper)) == 8


def test_earliest_tie_rule_is_not_reproduced():
    """Guards the specific defect: max(sorted(means), key=...) returned round 1
    purely because it sorted first. Here a later round is both cheaper and
    higher scoring, so any position-based rule would pick the wrong one."""
    entries = [{"round": 1, "score": 0.76, "avg_cost": 5.0},
               {"round": 2, "score": 0.80, "avg_cost": 1.0}]
    assert single_best(round_means(entries)) == 2


def test_pareto_front_keeps_cheap_and_accurate_extremes():
    front = pareto_front(round_means(SPORTS))
    assert 1 in front           # cheapest at the top score
    assert 3 not in front       # worse score and higher cost than 1
    assert 5 not in front       # 0.74 at the highest cost, dominated by round 1
    for r in front:
        assert r in range(1, 10)


def test_pareto_front_drops_exact_duplicates():
    entries = [{"round": 1, "score": 0.5, "avg_cost": 2.0},
               {"round": 2, "score": 0.5, "avg_cost": 2.0}]
    assert pareto_front(round_means(entries)) == [1]


def test_pareto_front_is_never_empty():
    entries = [{"round": 1, "score": 0.5, "avg_cost": 2.0}]
    assert pareto_front(round_means(entries)) == [1]


def test_missing_cost_is_rejected_not_zeroed():
    import pytest
    for bad in (None, float("nan"), float("inf"), -1.0):
        entries = [{"round": 1, "score": 0.9, "avg_cost": bad}]
        with pytest.raises(ValueError):
            round_means(entries)


def test_absent_cost_key_is_rejected():
    import pytest
    with pytest.raises(ValueError):
        round_means([{"round": 1, "score": 0.9}])
