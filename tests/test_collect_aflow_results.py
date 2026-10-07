import csv
import importlib.util
import json
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "scripts" / "collect_aflow_results.py"
spec = importlib.util.spec_from_file_location("collect_aflow_results", SOURCE)
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


def put(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_only_complete_cost_corrected_cell_is_exported(tmp_path):
    cell = tmp_path / "cell"
    put(cell / "manifest.json", {
        "aflow_patch_sha256": collector.EXPECTED_PATCH_SHA256,
        "model_call_cache_mode": "record",
        "aflow_dataset": "MuSRObjectPlacements", "task": "musr_object",
        "seed_condition": "matched", "search_seed": 1,
        "executor_model": "gemini-2.5-flash-lite",
    })
    wf = cell / "MuSRObjectPlacements" / "workflows"
    put(wf / "results.json", [{"round": 1, "score": 0.6}])
    put(wf / "search_usage.json", {"optimizer_cost": 1.2,
                                   "failed_proposals": 0})
    validation = wf / "round_1" / "cases.csv"
    validation.parent.mkdir(parents=True)
    with validation.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["prediction", "score", "cost"])
        writer.writeheader()
        writer.writerows({"prediction": "1", "score": 1, "cost": 0.002}
                         for _ in range(50))
    test = cell / "test_logs" / "MuSRObjectPlacements" / "round_1"
    usage = {"round": 1, "n_cases": 50, "score": 0.7,
             "tracked_cost": 0.1, "harness_total_cost": 0.1,
             "cost_per_case": 0.002}
    put(test / "usage.json", usage)
    put(test / "failure_audit.json", {"n_cases": 50, "connection_errors": 0})

    rows = collector.cell_rows(cell)
    assert len(rows) == 1
    assert rows[0]["validation_cases"] == 50
    assert rows[0]["inference_usd_per_100"] == 0.2

    usage["harness_total_cost"] = 1.0  # Shared-counter bug in older output.
    put(test / "usage.json", usage)
    assert collector.cell_rows(cell) == []
