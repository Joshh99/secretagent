"""Checks that the new AFlow adapters score exactly like the secretagent harness.

The strongest available check costs nothing: replay each saved Table 2/3 column's
own `predicted_output` values through the adapter that will score the AFlow side,
and require the adapter to reproduce that column's saved `correct` values. If the
two ever disagree, the AFlow side would be graded by a different rule than the
column it is being compared against.

The adapters live in the AFlow checkout at run time, so the AFlow modules they
import are stubbed here. No network or model call is made.
"""

import ast
import csv
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "benchmarks" / "COMMON" / "aflow-rebuttal" / "upstream"
RESULTS = ROOT / "benchmarks" / "COMMON" / "results"


def _load_adapter(filename: str):
    """Import one archived adapter with the AFlow modules it expects stubbed."""
    saved = {name: sys.modules.get(name) for name in
             ("benchmarks", "benchmarks.benchmark", "benchmarks.scorers_natural_plan",
              "benchmarks.scorers_medcalc", "scripts", "scripts.logs")}

    benchmarks_pkg = types.ModuleType("benchmarks")
    benchmarks_pkg.__path__ = []
    benchmark_mod = types.ModuleType("benchmarks.benchmark")

    class BaseBenchmark:
        def __init__(self, name, file_path, log_path):
            self.name, self.file_path, self.log_path = name, file_path, log_path

    benchmark_mod.BaseBenchmark = BaseBenchmark

    scripts_pkg = types.ModuleType("scripts")
    scripts_pkg.__path__ = []
    logs_mod = types.ModuleType("scripts.logs")
    logs_mod.logger = types.SimpleNamespace(info=lambda *a, **k: None)

    sys.modules.update({
        "benchmarks": benchmarks_pkg,
        "benchmarks.benchmark": benchmark_mod,
        "scripts": scripts_pkg,
        "scripts.logs": logs_mod,
    })
    # The adapters import the scorers as AFlow will see them, from verbatim
    # copies of the secretagent originals.
    for archived, attr in [("scorers_natural_plan.py", "benchmarks.scorers_natural_plan"),
                           ("scorers_medcalc.py", "benchmarks.scorers_medcalc")]:
        spec = importlib.util.spec_from_file_location(attr, UPSTREAM / archived)
        module = importlib.util.module_from_spec(spec)
        sys.modules[attr] = module
        spec.loader.exec_module(module)

    try:
        name = f"_adapter_{filename.replace('.py', '')}"
        spec = importlib.util.spec_from_file_location(name, UPSTREAM / filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


def _baseline_rows(relative: str) -> list[dict]:
    with (RESULTS / relative / "results.csv").open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _as_correct(value) -> float:
    text = str(value).strip()
    if text.lower() in {"true", "false"}:
        return 1.0 if text.lower() == "true" else 0.0
    return float(text)


# --- the archived scorers must stay identical to the originals ---------------

@pytest.mark.parametrize("archived,original", [
    ("scorers_natural_plan.py", "natural_plan/eval_utils.py"),
    ("scorers_medcalc.py", "medcalc/accuracy.py"),
])
def test_archived_scorer_is_a_verbatim_copy(archived, original):
    """A drifted copy would score the AFlow side by a different rule."""
    mine = hashlib.sha256((UPSTREAM / archived).read_bytes()).hexdigest()
    theirs = hashlib.sha256((ROOT / "benchmarks" / original).read_bytes()).hexdigest()
    assert mine == theirs, f"{archived} has drifted from benchmarks/{original}"


def test_medcalc_number_parsing_matches_the_harness():
    """The adapter copies _extract_number, so it must behave the same."""
    adapter = _load_adapter("benchmarks_medcalc.py")
    source = (ROOT / "benchmarks" / "medcalc" / "expt.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    original = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == "_extract_number")
    import typing
    namespace = {"re": __import__("re"), "Any": typing.Any, "Optional": typing.Optional}
    exec(compile(ast.Module([original], []), "<original>", "exec"), namespace)
    harness = namespace["_extract_number"]

    for value in [None, 12, 3.5, "7", "7.25", "ANSWER: 42", "<answer>8.5</answer>",
                  "the result is 19.2", "**exception raised**: boom", "no digits here",
                  "1e3", "-4", "Step 1: add 2 and 3. Final value 5"]:
        assert adapter.extract_number(value) == harness(value), value


# --- each adapter must reproduce its column's saved scores -------------------

def test_naturalplan_meeting_adapter_reproduces_the_saved_column():
    adapter = _load_adapter("benchmarks_naturalplan_meeting.py")
    benchmark = adapter.NaturalPlanMeetingBenchmark("m", "f", "l")
    rows = _baseline_rows("natural_plan/meeting/20260504.065725.workflow")
    assert len(rows) == 100

    disagreements = []
    for row in rows:
        instance = ast.literal_eval(row["expected_output"])
        score, _ = benchmark.calculate_score(json.dumps(instance, default=str),
                                             row["predicted_output"])
        if score != _as_correct(row["correct"]):
            disagreements.append(row["case_name"])
    assert not disagreements, f"{len(disagreements)} cases scored differently"


def test_naturalplan_trip_adapter_reproduces_the_saved_column():
    adapter = _load_adapter("benchmarks_naturalplan_trip.py")
    benchmark = adapter.NaturalPlanTripBenchmark("t", "f", "l")
    rows = _baseline_rows("natural_plan/trip/20260504.074537.workflow")
    assert len(rows) == 100

    disagreements = []
    for row in rows:
        instance = ast.literal_eval(row["expected_output"])
        score, _ = benchmark.calculate_score(json.dumps(instance, default=str),
                                             row["predicted_output"])
        if score != _as_correct(row["correct"]):
            disagreements.append(row["case_name"])
    assert not disagreements, f"{len(disagreements)} cases scored differently"


# The MedCalc Rules column was scored on 2026-04-25. Commit 32f797fa, on
# 2026-05-01, changed the rule-category test from `category == "rule-based"` to
# a lookup in RULE_CATEGORY_LABELS. The run passed fine-grained labels (risk,
# diagnosis, severity), so under the old code the exact-match branch never
# fired and every rule case was given the 5% tolerance meant for formulas.
# These two cases are the ones that tolerance let through.
MEDCALC_RULES_SCORED_BEFORE_THE_FIX = {"test.0921", "test.0900"}


def test_medcalc_adapter_reproduces_the_formulas_column():
    adapter = _load_adapter("benchmarks_medcalc.py")
    benchmark = adapter.MedCalcBenchmark("m", "f", "l")
    rows = _baseline_rows("medcalc/formulas/20260425.233811.workflow")
    assert len(rows) == 660

    disagreements = []
    for row in rows:
        problem = {"target": row["expected_output"],
                   "output_type": row["output_type"],
                   "category": row["category"],
                   "lower_limit": None, "upper_limit": None}
        score, _ = benchmark.score_case(problem, row["predicted_output"])
        if score != _as_correct(row["correct"]):
            disagreements.append(row["case_name"])
    assert not disagreements, f"{len(disagreements)} of 660 cases scored differently"


def test_medcalc_rules_column_was_scored_before_the_rule_category_fix():
    """The saved Rules column is 0.53 points high, and this pins which cases.

    The adapter is correct; the saved column is not. Comparing an AFlow number
    scored by today's rule against this column would grade AFlow more strictly
    than the column it sits next to, so the column needs rescoring from its own
    saved rows, which costs nothing.
    """
    adapter = _load_adapter("benchmarks_medcalc.py")
    benchmark = adapter.MedCalcBenchmark("m", "f", "l")
    rows = _baseline_rows("medcalc/rules/20260425.233811.workflow")
    assert len(rows) == 380

    disagreements, rescored = [], 0.0
    for row in rows:
        problem = {"target": row["expected_output"],
                   "output_type": row["output_type"],
                   "category": row["category"],
                   "lower_limit": None, "upper_limit": None}
        score, _ = benchmark.score_case(problem, row["predicted_output"])
        rescored += score
        if score != _as_correct(row["correct"]):
            disagreements.append(row["case_name"])

    assert set(disagreements) == MEDCALC_RULES_SCORED_BEFORE_THE_FIX
    # Each disagreement is a non-exact answer inside 5% of its gold.
    for row in rows:
        if row["case_name"] in MEDCALC_RULES_SCORED_BEFORE_THE_FIX:
            assert _as_correct(row["correct"]) == 1.0
            assert float(row["exact_match"]) == 0.0
            error = abs(float(row["predicted_numeric"]) - float(row["expected_output"]))
            assert 0 < error / abs(float(row["expected_output"])) <= 0.05

    saved = sum(_as_correct(row["correct"]) for row in rows) / len(rows)
    assert saved == pytest.approx(0.4974, abs=1e-4)
    assert rescored / len(rows) == pytest.approx(0.4921, abs=1e-4)


def test_medcalc_rejects_scoring_without_the_per_case_limits():
    adapter = _load_adapter("benchmarks_medcalc.py")
    benchmark = adapter.MedCalcBenchmark("m", "f", "l")
    with pytest.raises(NotImplementedError):
        benchmark.calculate_score("1.0", "1.0")


def test_rulearena_nba_adapter_reads_a_yes_or_no():
    adapter = _load_adapter("benchmarks_rulearena_nba.py")
    benchmark = adapter.RuleArenaNBABenchmark("n", "f", "l")

    assert benchmark.extract_boolean("True") is True
    assert benchmark.extract_boolean("no") is False
    assert benchmark.extract_boolean("Final answer: yes") is True
    assert benchmark.extract_boolean("The move is illegal") is False
    # A reply that says neither, or both, is not an answer.
    assert benchmark.extract_boolean("it depends") is None
    assert benchmark.extract_boolean("this is true and false") is None
    assert benchmark.extract_boolean("") is None
    assert benchmark.extract_boolean(None) is None

    # bool() of a non-empty string is True, which is the trap this avoids.
    assert benchmark.calculate_score("False", "no")[0] == 1.0
    assert benchmark.calculate_score("False", "yes")[0] == 0.0
    assert benchmark.calculate_score("True", "something unparseable")[0] == 0.0


def test_rulearena_nba_gold_is_always_a_boolean_in_the_saved_column():
    adapter = _load_adapter("benchmarks_rulearena_nba.py")
    benchmark = adapter.RuleArenaNBABenchmark("n", "f", "l")
    rows = _baseline_rows("rulearena/nba/20260430.023413.workflow")
    assert len(rows) == 46

    golds = [benchmark.extract_boolean(
        "True" if float(row["expected_output"]) == 1.0 else "False") for row in rows]
    assert sum(golds) == 34 and len(golds) - sum(golds) == 12
    with pytest.raises(ValueError):
        benchmark.calculate_score("not an answer", "yes")


def test_rulearena_nba_reads_a_numeric_yes_or_no():
    """The seed prompt asks for a single number, so replies arrive as 1.0/0.0.

    Reading those as text found both "1" and "0" inside "1.0" and called the
    reply ambiguous, which scored a smoke run at 0.119 against a 0.609
    baseline. A numeric reply is the common case and is unambiguous.
    """
    adapter = _load_adapter("benchmarks_rulearena_nba.py")
    benchmark = adapter.RuleArenaNBABenchmark("n", "f", "l")

    assert benchmark.extract_boolean("1.0") is True
    assert benchmark.extract_boolean("0.0") is False
    assert benchmark.extract_boolean("1") is True
    assert benchmark.extract_boolean("0") is False
    assert benchmark.extract_boolean("The answer is:\n1.0") is True
    # A number that is neither is not an answer.
    assert benchmark.extract_boolean("2.5") is None
    assert benchmark.extract_boolean("-1") is None
    # Words still work.
    assert benchmark.extract_boolean("yes") is True
    assert benchmark.extract_boolean("it depends") is None

    assert benchmark.calculate_score("True", "1.0")[0] == 1.0
    assert benchmark.calculate_score("True", "0.0")[0] == 0.0
    assert benchmark.calculate_score("False", "0.0")[0] == 1.0


def test_rulearena_nba_adapter_reproduces_the_saved_column():
    """Replay the column's own predictions, the check NBA was missing.

    Meeting, Trip and MedCalc were each verified by replaying their saved
    predicted_output values. NBA was not: it only had gold parsing and
    hand-written replies. Its baseline predictions are stored as "1.0" and
    "0.0", which is exactly the form the adapter mis-read as ambiguous, so the
    gap in the test was precisely where the bug was.
    """
    adapter = _load_adapter("benchmarks_rulearena_nba.py")
    benchmark = adapter.RuleArenaNBABenchmark("n", "f", "l")
    rows = _baseline_rows("rulearena/nba/20260430.023413.workflow")
    assert len(rows) == 46

    disagreements = []
    for row in rows:
        gold = "True" if float(row["expected_output"]) == 1.0 else "False"
        score, _ = benchmark.calculate_score(gold, row["predicted_output"])
        if score != _as_correct(row["correct"]):
            disagreements.append((row["case_name"], row["predicted_output"]))
    assert not disagreements, f"{len(disagreements)} of 46 cases scored differently"
