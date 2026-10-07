"""Checks for the Table 2/3 task registry and its pre-launch guards.

These read the saved baseline CSVs and make no network or model call, so they
can run before any paid task is launched.
"""

import json

import pytest

from scripts import aflow_table_tasks as registry
from scripts.aflow_table_tasks import TASKS, TableTask
from scripts.run_aflow_table_task import build_rows, guard_workspace


def test_registry_covers_the_eight_table_columns():
    labels = {task.label for task in TASKS.values()}
    assert labels == {
        "MuSR Murder", "MuSR Object", "MuSR Team",
        "NaturalPlan Meeting", "NaturalPlan Trip",
        "RuleArena NBA", "MedCalc Formulas", "MedCalc Rules",
    }
    # MuSR Object is wired already; the other seven are this handoff's work.
    assert len(registry.PENDING) == 7
    assert "musr_object" not in registry.PENDING


@pytest.mark.parametrize("key,cases", [
    ("musr_object", 106), ("musr_murder", 100), ("musr_team", 100),
    ("naturalplan_meeting", 100), ("naturalplan_trip", 100),
    ("rulearena_nba", 46), ("medcalc_formulas", 660), ("medcalc_rules", 380),
])
def test_baseline_has_the_case_count_the_table_column_needs(key, cases):
    assert registry.baseline_cell(TASKS[key])["cases"] == cases


def test_baseline_cell_reproduces_accuracy_and_cost_per_hundred():
    """hero_table.py reports mean(correct) and mean(cost) * 100."""
    cell = registry.baseline_cell(TASKS["musr_murder"])
    assert cell["accuracy"] == pytest.approx(0.68)
    assert cell["usd_per_100"] == pytest.approx(0.4717, abs=1e-4)


def test_medcalc_columns_share_one_run_so_one_test_pass_covers_both():
    group = registry.export_group("MedCalcTest")
    assert {task.key for task in group} == {"medcalc_formulas", "medcalc_rules"}
    assert sum(task.expected_cases for task in group) == 1040
    # The two columns must not overlap, or a case would be scored twice.
    formulas = set(TASKS["medcalc_formulas"].category_filter)
    rules = set(TASKS["medcalc_rules"].category_filter)
    assert not formulas & rules


def test_every_table_column_was_produced_with_the_same_model():
    assert {task.baseline_model for task in TASKS.values()} == {
        registry.TABLE_BASELINE_MODEL}


def test_a_different_model_is_reported_as_not_comparable():
    """Running the tables at Flash-Lite would not be comparable to the columns."""
    for task in TASKS.values():
        problems = registry.check_model_match(task, aflow_model="gemini-2.5-flash-lite")
        assert problems and "different models" in problems[0], task.key


def test_same_model_on_another_host_is_reported_as_a_host_difference():
    """DeepSeek-V3.1 via OpenRouter is the column's model through another host."""
    for task in TASKS.values():
        problems = registry.check_model_match(task)
        assert len(problems) == 1
        assert "same model as the column" in problems[0]
        assert "SiliconFlow" in problems[0]
    # Same model and same host raises nothing at all.
    assert registry.check_model_match(
        TASKS["musr_murder"], aflow_model=registry.TABLE_BASELINE_MODEL,
        aflow_host=registry.TABLE_BASELINE_HOST) == []


def test_provider_naming_does_not_hide_the_same_model():
    """The two providers label the same weights differently."""
    assert registry.model_family("together_ai/deepseek-ai/DeepSeek-V3.1") ==         registry.model_family("deepseek/deepseek-chat-v3.1")
    assert registry.model_family("gemini-2.5-flash-lite") !=         registry.model_family("deepseek/deepseek-chat-v3.1")


def _rows_from_baseline(task: TableTask):
    """An export that matches the table exactly, built from the table itself."""
    rows = registry.read_baseline(task)
    gold = registry.baseline_gold(task)
    out = []
    for row, value in zip(rows, gold):
        if task.gold == "bool":
            target = "True" if value else "False"
        elif task.gold == "instance_digest":
            target = registry.canonical_instance(row["expected_output"])
        else:
            target = str(row["expected_output"])
        out.append({"case_name": row["case_name"], "target": target})
    return out


@pytest.mark.parametrize("key", sorted(TASKS))
def test_a_faithful_export_passes_the_alignment_check(key):
    task = TASKS[key]
    blockers, _ = registry.check_case_alignment(task, _rows_from_baseline(task))
    assert blockers == []


def test_a_short_export_is_blocked():
    task = TASKS["musr_murder"]
    blockers, _ = registry.check_case_alignment(task, _rows_from_baseline(task)[:-1])
    assert any("99 cases" in problem for problem in blockers)


def test_a_substituted_case_is_blocked():
    task = TASKS["musr_murder"]
    rows = _rows_from_baseline(task)
    rows[3]["case_name"] = "not_a_real_case"
    blockers, _ = registry.check_case_alignment(task, rows)
    assert any("not in the export" in problem for problem in blockers)
    assert any("not in the table" in problem for problem in blockers)


def test_a_changed_gold_answer_is_blocked():
    task = TASKS["musr_murder"]
    rows = _rows_from_baseline(task)
    rows[5]["target"] = str(int(rows[5]["target"]) + 1)
    blockers, _ = registry.check_case_alignment(task, rows)
    assert any("gold answer differs" in problem for problem in blockers)


def test_reordering_the_same_cases_warns_but_does_not_block():
    """MedCalc's saved order is not reproducible, but its case set is.

    Mean accuracy and mean cost do not depend on order, so a reordering of the
    same cases and golds is reported rather than treated as a mismatch.
    """
    task = TASKS["medcalc_formulas"]
    rows = _rows_from_baseline(task)
    blockers, warnings = registry.check_case_alignment(task, list(reversed(rows)))
    assert blockers == []
    assert any("different order" in warning for warning in warnings)


def test_boolean_gold_is_compared_as_a_boolean():
    """Every RuleArena NBA gold is a boolean saved as 1.0 or 0.0."""
    task = TASKS["rulearena_nba"]
    gold = registry.baseline_gold(task)
    assert set(map(type, gold)) == {bool}
    assert sum(gold) == 34 and len(gold) == 46

    rows = _rows_from_baseline(task)
    blockers, _ = registry.check_case_alignment(task, rows)
    assert blockers == []

    flipped = "False" if rows[0]["target"] == "True" else "True"
    rows[0]["target"] = flipped
    blockers, _ = registry.check_case_alignment(task, rows)
    assert any("gold answer differs" in problem for problem in blockers)


def test_instance_gold_matches_across_python_repr_and_json():
    """The saved column stores a dict repr; the export stores JSON."""
    instance = {"num_people": 2, "golden_plan": ["a", "b"], "n": 1.5}
    assert registry.instance_digest(str(instance)) == \
        registry.instance_digest(json.dumps(instance, sort_keys=True))
    assert registry.instance_digest({"a": 1, "b": 2}) == \
        registry.instance_digest({"b": 2, "a": 1})
    assert registry.instance_digest({"a": 1}) != registry.instance_digest({"a": 2})


def test_musr_rows_carry_a_split_qualified_case_name():
    """Case names repeat across MuSR splits, so the export qualifies them."""
    task = TASKS["musr_murder"]

    class Case:
        name = "ex003"
        input_args = ("a narrative", "who did it?", ["Alice", "Bob"])
        expected_output = 1

    rows = build_rows(task, [Case()])
    assert rows[0]["case_name"] == "murder_mysteries_test/ex003"
    assert rows[0]["target"] == "1"
    assert rows[0]["n_choices"] == 2
    assert "0. Alice" in rows[0]["input"] and "1. Bob" in rows[0]["input"]
    # A qualified name still matches the table's bare name.
    assert rows[0]["case_name"].split("/")[-1] == "ex003"


def test_medcalc_rows_carry_the_limits_its_scorer_needs():
    task = TASKS["medcalc_formulas"]

    class Case:
        name = "test.0001"
        input_args = ("patient note", "what is the dose?")
        expected_output = 12.5
        metadata = {"lower_limit": 11.9, "upper_limit": 13.1,
                    "output_type": "decimal", "category": "dosage"}

    row = build_rows(task, [Case()])[0]
    assert row["lower_limit"] == 11.9 and row["upper_limit"] == 13.1
    assert row["output_type"] == "decimal" and row["category"] == "dosage"


def test_the_live_musr_object_workspace_is_refused():
    for workspace in ["runs/table2_musr3__musr_object__matched__seed1",
                      "runs/table2_musr1__musr_object__matched__seed1"]:
        with pytest.raises(SystemExit):
            guard_workspace(workspace)
    guard_workspace("runs/musr_murder__matched__seed1")
