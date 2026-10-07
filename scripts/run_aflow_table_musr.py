"""Score validation-selected AFlow MuSR Object on Table 2's full 106-case test.

The primary AFlow protocol uses 50 held-out cases to match the optimizer
comparison. Table 2's MuSR Object cells use the full 106-case test. This script
exports that exact split and, after the search finishes, scores the selected
workflow on it without changing the AFlow search or its original 50-case test.
"""

import argparse
import asyncio
import csv
import hashlib
import importlib
import json
import math
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmarks" / "musr"
BASELINE = (ROOT / "benchmarks" / "COMMON" / "results" / "musr" / "object"
            / "20260425.122302.workflow" / "results.csv")


def prepare(aflow: Path):
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(BENCHMARK))
    from expt import load_dataset
    from export_aflow_datasets import musr_object_rows, write_jsonl

    dataset = load_dataset("object_placements_test").configure(shuffle_seed=42)
    rows = musr_object_rows(dataset, "object_placements_test")
    with BASELINE.open(encoding="utf-8", newline="") as stream:
        old_rows = list(csv.DictReader(stream))
    if len(rows) != 106 or len(old_rows) != len(rows):
        raise ValueError(f"Table 2 MuSR Object must have 106 cases; got {len(rows)} and {len(old_rows)}")
    for new, old in zip(rows, old_rows):
        if (new["case_name"].split("/")[-1] != old["case_name"]
                or int(new["target"]) != int(float(old["expected_output"]))):
            raise ValueError(f"Table 2 case or gold mismatch: {new['case_name']}")
    path = aflow / "data" / "datasets" / "musrobjectplacements_table_test.jsonl"
    write_jsonl(rows, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".sha256").write_text(digest + "\n", encoding="utf-8")
    print(f"Table 2 MuSR Object: 106 matching case IDs and answers; SHA-256 {digest}")
    return path, digest, [row["case_name"] for row in rows]


def prepared_export(aflow: Path):
    path = aflow / "data" / "datasets" / "musrobjectplacements_table_test.jsonl"
    recorded = path.with_suffix(".sha256").read_text(encoding="utf-8").strip()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != recorded:
        raise ValueError("full MuSR Object test export checksum changed")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != 106:
        raise ValueError("full MuSR Object test export needs 106 cases")
    return path, digest, [row["case_name"] for row in rows]


async def score(aflow: Path, workspace: str, data_path: Path, digest: str,
                case_names: list[str], out_tag: str):
    sys.path.insert(0, str(aflow))
    from benchmarks.musr_object import MuSRObjectBenchmark
    from scripts.async_llm import LLMsConfig
    from selection import round_means, single_best

    ws = aflow / workspace
    manifest = json.loads((ws / "manifest.json").read_text(encoding="utf-8"))
    if manifest["task"] != "musr_object" or manifest["seed_condition"] != "matched":
        raise ValueError("workspace is not the matched MuSR Object run")
    # A run is verified against the exact AFlow source it ran with, so every
    # archived patch is a candidate and the manifest's checksum picks one out.
    # Keeping only the current patch here would have meant that rebuilding it,
    # which adding the Table 2/3 datasets required, retrospectively made an
    # already-finished search unscoreable.
    upstream = ROOT / "benchmarks" / "COMMON" / "aflow-rebuttal" / "upstream"
    archived = {hashlib.sha256(path.read_bytes()).hexdigest(): path
                for path in sorted(upstream.glob("aflow_changes*.patch"))}
    if manifest["aflow_patch_sha256"] not in archived:
        raise ValueError(
            f"no archived AFlow patch matches the one this search ran with "
            f"({manifest['aflow_patch_sha256'][:12]}); archived: "
            f"{sorted(p.name for p in archived.values())}")
    print(f"AFlow source verified against {archived[manifest['aflow_patch_sha256']].name}")
    results = json.loads((ws / "MuSRObjectPlacements" / "workflows" / "results.json")
                         .read_text(encoding="utf-8"))
    round_id = single_best(round_means(results))
    module = workspace.replace("/", ".").replace("\\", ".")
    graph = importlib.import_module(
        f"{module}.MuSRObjectPlacements.workflows.round_{round_id}.graph")
    model = manifest["executor_model"]
    workflow = graph.Workflow(name="MuSRObjectPlacements",
                              llm_config=LLMsConfig.default().get(model),
                              dataset="MuSRObjectPlacements")
    out = ws / out_tag / f"round_{round_id}"
    if out.exists():
        raise FileExistsError(f"table pass already exists: {out}")
    out.mkdir(parents=True)
    benchmark = MuSRObjectBenchmark("MuSRObjectPlacements", str(data_path), str(out))
    score_value, mean_cost, total_cost = await benchmark.run_evaluation(workflow, None)
    csv_files = list(out.glob("*.csv"))
    if len(csv_files) != 1:
        raise ValueError("expected exactly one table-test CSV")
    with csv_files[0].open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(case_names):
        raise ValueError("table-test case count changed")
    bad = [i for i, row in enumerate(rows, 1)
           if (row.get("prediction") or "").strip().casefold() == "connection error."]
    usage = workflow.llm.get_usage_summary()
    row_cost = sum(float(row["cost"]) for row in rows)
    if bad or not math.isclose(row_cost, total_cost, rel_tol=1e-6, abs_tol=1e-8) \
            or not math.isclose(usage["total_cost"], total_cost, rel_tol=1e-6, abs_tol=1e-8):
        raise ValueError(f"invalid table pass: {len(bad)} connection errors, "
                         f"row sum ${row_cost}, harness ${total_cost}, tracker ${usage['total_cost']}")
    summary = {"task": "musr_object", "table": "2 and 3", "round": round_id,
               "selection": "highest validation accuracy, then lowest validation cost",
               "test_cases": len(rows), "case_names": case_names,
               "test_export_sha256": digest, "test_accuracy": score_value,
               "inference_usd_per_100": mean_cost * 100,
               "inference_usd_total": total_cost,
               "model_calls": usage["call_count"],
               "experiment_commit": manifest["secretagent_commit"],
               "table_script_commit": subprocess.run(
                   ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                   capture_output=True, text=True, check=True).stdout.strip(),
               "table_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "aflow_patch_sha256": manifest["aflow_patch_sha256"]}
    (out / "table_result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Table 2/3 MuSR Object: {score_value:.4f} accuracy, ${mean_cost * 100:.4f}/100 cases; "
          f"round {round_id}, {len(rows)} cases, {usage['call_count']} model calls")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--aflow-dir", type=Path, required=True)
    ap.add_argument("--workspace", default="runs/table2_musr3__musr_object__matched__seed1__gemini_2_5_flash_lite")
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--out-tag", default="table2_full_test",
                    help="Unique output folder; change for a retry to preserve old artifacts")
    args = ap.parse_args()
    if args.prepare_only and args.score_only:
        ap.error("choose either --prepare-only or --score-only")
    if not args.out_tag.replace("_", "").replace("-", "").isalnum():
        ap.error("--out-tag must contain only letters, digits, dashes, or underscores")
    aflow = args.aflow_dir.resolve()
    path, digest, names = prepared_export(aflow) if args.score_only else prepare(aflow)
    if args.prepare_only:
        return
    import os
    os.environ["AFLOW_CACHE_MODE"] = "record"
    os.environ["AFLOW_CACHE_DIR"] = str(aflow / args.workspace / "model_calls" / args.out_tag)
    old_cwd = Path.cwd()
    try:
        os.chdir(aflow)
        asyncio.run(score(aflow, args.workspace, path, digest, names, args.out_tag))
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
