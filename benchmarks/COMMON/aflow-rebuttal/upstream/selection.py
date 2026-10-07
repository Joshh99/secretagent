"""Round selection for the AFlow arm.

Implements PREREGISTRATION.md D2. AFlow ships three different definitions of
"best round" and they disagree:

  test_pass.py:best_round     highest mean validation, ties to the EARLIEST round
  Optimizer.test              hardcodes rounds = [1]
  interface.load_best_round   returns top_rounds[1], the second entry

The earliest-tie rule is what made the July run report AFlow's own untouched seed
as its result on Sports, where rounds 1, 2, 4, 7 and 8 all tied at 0.76.

Primary selection here is the validation Pareto front over (score, cost), since the
comparison is bi-objective. Single-point selection is available for table cells and
breaks ties by lowest validation cost, which is a real criterion rather than an
arbitrary position in the round order.
"""
import math
from collections import defaultdict


def round_means(entries):
    """Mean validation score and cost per round from a results.json list.

    A missing or non-finite cost used to become 0.0, which put the candidate at
    the cheap end of the front and let it win selection on an absent number.
    Costs must be present and finite or the entry is rejected.
    """
    scores, costs = defaultdict(list), defaultdict(list)
    for e in entries:
        if e.get("score") is None:
            # A round whose optimize step raised is logged with score None.
            continue
        cost = e.get("avg_cost")
        if cost is None or not math.isfinite(cost) or cost < 0:
            raise ValueError(
                f"round {e.get('round')} has avg_cost={cost!r}. Cost drives Pareto "
                "selection, so it cannot be missing, negative or non-finite.")
        scores[e["round"]].append(e["score"])
        costs[e["round"]].append(cost)
    return {
        r: {
            "score": sum(v) / len(v),
            "cost": sum(costs[r]) / len(costs[r]),
            "repeats": len(v),
        }
        for r, v in scores.items()
    }


def pareto_front(means):
    """Rounds not dominated on (higher score, lower cost).

    A round is dominated when another is at least as good on both axes and
    strictly better on one. Exact ties on both axes keep the earliest round only,
    so an unchanged candidate does not occupy two slots on the front.
    """
    rounds = sorted(means)
    front = []
    for r in rounds:
        a = means[r]
        dominated = False
        for other in rounds:
            if other == r:
                continue
            b = means[other]
            better_or_equal = b["score"] >= a["score"] and b["cost"] <= a["cost"]
            strictly_better = b["score"] > a["score"] or b["cost"] < a["cost"]
            if better_or_equal and strictly_better:
                dominated = True
                break
            if b["score"] == a["score"] and b["cost"] == a["cost"] and other < r:
                dominated = True
                break
        if not dominated:
            front.append(r)
    return front


def single_best(means):
    """Highest mean validation score, ties broken by lowest mean validation cost.

    The final round-number term only breaks a tie where score and cost are both
    exactly equal, which keeps the result deterministic.
    """
    if not means:
        raise ValueError("no scored rounds")
    return min(means, key=lambda r: (-means[r]["score"], means[r]["cost"], r))
