"""Regression checks for validation-only NSGA-II table selection."""

import csv
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def table(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts" / "learning_table.py"
    spec = importlib.util.spec_from_file_location("learning_table_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OPTIMIZER_DIR = str(tmp_path / "optimize")
    module.RESULTS_DIR = str(tmp_path / "results")
    return module


def summary(table, rows, task="musr_team", filename="nsga2_summary.csv"):
    path = Path(table.OPTIMIZER_DIR) / task / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["method", "model", "correct", "cost", "frontier", "valid"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def row(method, accuracy, cost, frontier=True, valid=True, model="DeepSeek-V3.1"):
    return dict(method=method, model=model, correct=accuracy, cost=cost, frontier=frontier, valid=valid)


def result(table, folder, subtask="team", exists=True):
    path = Path(table.RESULTS_DIR) / "musr" / subtask / folder / "results.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if exists:
        # Selection must work without parsing a test score or even a valid CSV.
        path.write_text("test contents must not be read for selection")
    return str(path)


def test_validation_rank_ignores_test_and_ineligible_rows(table):
    summary(table, [row("wf_orch", .76, .01), row("pot", .80, .02),
                    row("invalid", 1, 0, valid=False), row("not_front", 1, 0, frontier=False)])
    result(table, "20260101.000000.test_pass_wf_orch_DeepSeek-V3-1")
    chosen = result(table, "20260101.000001.test_pass_pot_DeepSeek-V3-1")
    assert table.find_optimizer_csv("musr", "team") == chosen


def test_ties_use_cost_then_file_order_and_normalize_names(table):
    summary(table, [row("expensive", .8, .03), row("POT", .8, .01), row("later", .8, .01)])
    chosen = result(table, "20260101.000000.test_pass_pot_deepseek_v3_1")
    result(table, "20260101.000001.test_pass_later_DeepSeek-V3-1")
    assert table.find_optimizer_csv("musr", "team") == chosen


def test_latest_matching_folder_with_csv(table):
    summary(table, [row("pot", .8, .01)])
    result(table, "20260101.000000.test_pass_pot_DeepSeek-V3-1")
    chosen = result(table, "20260102.000000.test_pass_pot_DeepSeek-V3-1")
    result(table, "20260103.000000.test_pass_pot_DeepSeek-V3-1", exists=False)
    assert table.find_optimizer_csv("musr", "team") == chosen


def test_missing_validation_choice_never_falls_back(table, capsys):
    summary(table, [row("pot", .8, .01), row("wf_orch", .76, .02)])
    result(table, "20260101.000000.test_pass_wf_orch_DeepSeek-V3-1")
    assert table.find_optimizer_csv("musr", "team") is None
    assert "musr/team" in capsys.readouterr().err


def test_missing_summary_returns_none(table):
    assert table.find_optimizer_csv("musr", "team") is None
    assert table.find_optimizer_csv("unknown", "task") is None


def test_murder_requires_corrected_sweep_and_heldout_test(table, capsys):
    rows = [row("structured_baseline", .76, .01, model="gemini-2.5-flash")]
    legacy = summary(table, rows, task="musr_murder")
    legacy_bytes = legacy.read_bytes()
    result(table, "20260101.000000.test_pass_structured_baseline_gemini-2-5-flash", subtask="murder")
    assert table.find_optimizer_csv("musr", "murder") is None
    assert "legacy" in capsys.readouterr().err
    summary(table, rows, task="musr_murder", filename="nsga2_summary.validation.csv")
    assert table.find_optimizer_csv("musr", "murder") is None
    chosen = result(table, "20260102.000000.valsplit_heldout50.test_pass_structured_baseline_gemini-2-5-flash", subtask="murder")
    assert table.find_optimizer_csv("musr", "murder") == chosen
    assert legacy.read_bytes() == legacy_bytes
