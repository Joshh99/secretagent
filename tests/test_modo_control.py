"""Checks for the random-search control and failed-run reporting."""

import json
import csv
import hashlib
import math
import sys
from itertools import product
from types import SimpleNamespace

import pytest

from scripts.run_modo_control import (
    MODEL, audit_outputs, check_fixed_models, initial_population, random_vectors,
    save_result, validation_front,
)
from scripts import run_modo_control as control
from secretagent.optimize.encoder import SearchDimension


def test_random_search_shares_initial_population_and_has_unique_draws():
    sizes = [3, 4]
    initial, _ = initial_population(sizes, seed=17)
    draws = random_vectors(sizes, seed=17, count=10)
    assert draws[:len(initial)] == initial
    assert len(draws) == len(set(draws)) == 10
    assert set(draws) <= set(product(range(3), range(4)))
    assert draws == random_vectors(sizes, seed=17, count=10)


def test_random_search_rejects_unmatched_budget():
    initial, _ = initial_population([3, 4], seed=17)
    with pytest.raises(ValueError, match="smaller"):
        random_vectors([3, 4], seed=17, count=len(initial) - 1)
    with pytest.raises(ValueError, match="exceeds"):
        random_vectors([3, 4], seed=17, count=13)


def test_front_uses_all_evaluations_and_drops_failed_or_duplicate_points():
    evaluated = [
        ([0], 0.8, 0.001),
        ([1], 0.9, 0.002),
        ([2], 0.7, 0.003),
        ([3], 0.9, 0.002),
        ([4], 1.0, math.inf),
    ]
    assert [vec for vec, _, _ in validation_front(evaluated)] == [[0], [1]]


def test_fixed_model_check_catches_hidden_method_override():
    dims = [SearchDimension("toplevel_method", ["safe", "hidden"]),
            SearchDimension("llm.model", [MODEL])]
    compound = {"toplevel_method": {
        "safe": ["ptools.answer.method=simulate"],
        "hidden": ["ptools.answer.model=another-model"],
    }}
    with pytest.raises(ValueError, match="another model"):
        check_fixed_models(dims, compound)


def test_failed_config_is_json_null_not_infinity(tmp_path):
    args = SimpleNamespace(task="sports", mode="fixed", seed=1)
    space = tmp_path / "space.yaml"
    space.write_text("test: true", encoding="utf-8")
    path = tmp_path / "summary.json"
    dims = [SearchDimension("toplevel_method", ["simple"])]
    save_result(path, args, space, dims, [([0], 0.0, math.inf)], [],
                [{"cheapest_cost": math.inf}],
                ["case-1"], "hash")
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["unique_evaluated"] == 1
    assert result["candidates"][0]["failed"] is True
    assert result["candidates"][0]["validation_cost_per_case"] is None
    assert result["generations"][0]["cheapest_cost"] is None


def test_test_dry_run_uses_only_validation_selected_configs(tmp_path, monkeypatch, capsys):
    space = control.ROOT / "benchmarks/bbh/sports_understanding/nsga2.yaml"
    validation = tmp_path / "fixed.json"
    validation.write_text(json.dumps({
        "mode": "fixed", "task": "sports", "seed": 1,
        "space_sha256": hashlib.sha256(space.read_bytes()).hexdigest(),
        "code_revision": control.current_revision(),
        "front_vectors": [[0, 0, 0, 0, 0], [1, 0, 0, 0, 0]],
    }), encoding="utf-8")
    monkeypatch.setattr(control, "check_cases", lambda *args, **kwargs: (["case-1"], "hash"))
    output = tmp_path / "test-output"
    monkeypatch.setattr(sys, "argv", ["run_modo_control.py", "--task", "sports",
                                     "--mode", "test", "--seed", "1", "--out", str(output),
                                     "--validation-summary", str(validation),
                                     "--aflow-dir", str(tmp_path / "aflow"), "--dry-run"])
    control.main()
    assert "2 validation-selected configurations" in capsys.readouterr().out
    assert not output.exists()


def test_audit_rejects_silent_case_exception(tmp_path):
    run = tmp_path / "cases" / "one"
    run.mkdir(parents=True)
    rows = [
        {"case_name": "a", "expt_name": "fixed_s1_001", "predicted_output": "yes", "cost": "0.1"},
        {"case_name": "b", "expt_name": "fixed_s1_001",
         "predicted_output": "**exception raised**: timeout", "cost": "0.0"},
    ]
    with (run / "results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    (run / "results.jsonl").write_text("{}\n{}\n", encoding="utf-8")
    audit = audit_outputs(tmp_path / "cases", "fixed", 1, [([0], 0.5, 0.05)], ["a", "b"])
    assert audit["case_evaluations"] == 2
    assert audit["inference_usd_total"] == pytest.approx(0.1)
    assert any("exception" in issue for issue in audit["issues"])
