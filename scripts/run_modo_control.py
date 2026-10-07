#!/usr/bin/env python
"""Run the fixed-model comparison or a paired MODO search control."""

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import random
import shlex
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

from secretagent.optimize.encoder import (
    SearchDimension, decode_dict, decode_modular, dim_sizes,
    modular_space_from_yaml, space_size,
)
from secretagent.optimize.pareto import EvalCache, run_exhaustive, run_nsga2
from secretagent.dataset import Dataset


ROOT = Path(__file__).resolve().parents[1]
MODEL = "gemini/gemini-2.5-flash-lite"
TASKS = {
    "sports": ("bbh/sports_understanding", "nsga2.yaml", "valid"),
    "finqa": ("finqa", "nsga2.yaml", "valid"),
    "musr_object": ("musr", "nsga2_object.yaml", "object_placements_val"),
}
TEST_SPLITS = {"sports": ("test", 100), "finqa": ("test", 300),
               "musr_object": ("object_placements_test", 50)}
SHUFFLE_SEEDS = {"sports": 137, "finqa": None, "musr_object": 42}
AFLOW_DATASETS = {
    "sports": "sportsunderstanding",
    "finqa": "finqa",
    "musr_object": "musrobjectplacements",
}


def initial_population(sizes, seed, pop_size=12):
    rng = random.Random(seed)
    drawn = [tuple(rng.randint(0, size - 1) for size in sizes)
             for _ in range(pop_size)]
    return list(dict.fromkeys(drawn)), rng


def check_cases(task, benchmark, split, aflow_dir, count=50, stage="validate"):
    if task == "musr_object":
        sys.path.insert(0, str(benchmark))
        ds = importlib.import_module("expt").load_dataset(split)
    else:
        ds = Dataset.model_validate_json(
            (benchmark / "data" / f"{split}.json").read_text(encoding="utf-8"))
    ds.configure(shuffle_seed=SHUFFLE_SEEDS[task], n=count)
    if len(ds.cases) != count or any(case.expected_output is None for case in ds.cases):
        raise ValueError(f"{task}: expected {count} labeled {stage} cases")
    exported = aflow_dir / "data" / "datasets" / f"{AFLOW_DATASETS[task]}_{stage}.jsonl"
    if not exported.is_file():
        raise FileNotFoundError(f"missing AFlow {stage} export: {exported}")
    checksum_file = ROOT / "benchmarks/COMMON/aflow-rebuttal/upstream/dataset_checksums.txt"
    checksums = dict(
        (entry.lstrip("*"), digest)
        for digest, entry in (line.split() for line in checksum_file.read_text(encoding="utf-8").splitlines()
                              if line.strip() and not line.startswith("#"))
    )
    export_hash = hashlib.sha256(exported.read_bytes()).hexdigest()
    if export_hash != checksums[exported.name]:
        raise ValueError(f"{task}: AFlow {stage} export does not match the archived SHA-256")
    rows = [json.loads(line) for line in exported.read_text(encoding="utf-8").splitlines()]
    names = [case.name for case in ds.cases]
    export_names = [row["case_name"].split("/")[-1] for row in rows]
    if names != export_names:
        raise ValueError(f"{task}: MODO and AFlow {stage} case IDs differ")
    for case, row in zip(ds.cases, rows):
        gold = row["target"] if task == "sports" else row["answer"] if task == "finqa" else row["target"]
        expected = ("yes" if case.expected_output else "no") if task == "sports" else (
            str(case.expected_output) if task == "musr_object" else case.expected_output)
        if gold != expected:
            raise ValueError(f"{task}: gold answer differs for {case.name}")
    return names, export_hash


def current_revision():
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def audit_outputs(cases_dir, mode, seed, evaluated, case_names):
    expected_runs = {f"{mode}_s{seed}_{i:03d}" for i in range(1, len(evaluated) + 1)}
    seen = set()
    issues = []
    case_evaluations = 0
    search_usd = 0.0
    for path in cases_dir.rglob("results.csv"):
        with path.open(encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        if not rows:
            issues.append(f"empty results: {path}")
            continue
        run_name = rows[0].get("expt_name")
        if run_name in seen:
            issues.append(f"duplicate result name: {run_name}")
        seen.add(run_name)
        if [row.get("case_name") for row in rows] != case_names:
            issues.append(f"case IDs or order differ: {path}")
        if any(row.get("expt_name") != run_name for row in rows):
            issues.append(f"mixed experiment names: {path}")
        if any((row.get("predicted_output") or "").startswith("**exception raised**")
               or row.get("_error") or row.get("_timeout") for row in rows):
            issues.append(f"case-level exception or timeout: {path}")
        costs = [row.get("cost") for row in rows]
        if any(value in (None, "") for value in costs):
            issues.append(f"missing case cost: {path}")
        else:
            try:
                search_usd += sum(float(value) for value in costs)
            except ValueError:
                issues.append(f"invalid case cost: {path}")
        jsonl = path.with_name("results.jsonl")
        if not jsonl.is_file() or len(jsonl.read_text(encoding="utf-8").splitlines()) != len(rows):
            issues.append(f"missing or mismatched JSONL: {path}")
        case_evaluations += len(rows)
    if seen != expected_runs:
        issues.append(f"result names differ: expected {sorted(expected_runs)}, found {sorted(seen)}")
    return {"case_evaluations": case_evaluations, "inference_usd_total": search_usd,
            "issues": issues}


def check_fixed_models(dims, compound):
    for method_index in range(dims[0].size):
        vector = [method_index] + [0] * (len(dims) - 1)
        overrides = decode_modular(dims, vector, compound)
        for override in overrides:
            key, _, value = override.partition("=")
            if (key == "llm.model" or key.endswith(".model")) and value != MODEL:
                raise ValueError(f"method {dims[0].values[method_index]} uses another model: {override}")


def random_vectors(sizes, seed, count):
    initial, rng = initial_population(sizes, seed)
    if count < len(initial):
        raise ValueError("budget is smaller than the unique starting population")
    all_vectors = list(product(*(range(size) for size in sizes)))
    if count > len(all_vectors):
        raise ValueError("budget exceeds the search space")
    used = set(initial)
    remaining = [vec for vec in all_vectors if vec not in used]
    rng.shuffle(remaining)
    return initial + remaining[:count - len(initial)]


def validation_front(evaluated):
    """Keep nondominated valid configurations from every evaluated candidate."""
    front = []
    seen_fitness = set()
    for vector, accuracy, cost in evaluated:
        if not math.isfinite(cost):
            continue
        fitness = (accuracy, cost)
        if fitness in seen_fitness:
            continue
        dominated = any(
            math.isfinite(other_cost)
            and other_accuracy >= accuracy and other_cost <= cost
            and (other_accuracy > accuracy or other_cost < cost)
            for _, other_accuracy, other_cost in evaluated
        )
        if not dominated:
            front.append((vector, accuracy, cost))
            seen_fitness.add(fitness)
    return front


def save_result(path, args, space_file, dims, evaluated, front, gen_log,
                case_names, export_hash, paired=None, audit=None, elapsed_s=0.0,
                revision=None):
    revision = revision or current_revision()
    rows = [
        {"vector": vec, "configuration": decode_dict(dims, vec),
         "validation_accuracy": acc if math.isfinite(acc) else None,
         "validation_cost_per_case": cost if math.isfinite(cost) else None,
         "failed": not math.isfinite(cost)}
        for vec, acc, cost in evaluated
    ]
    result = {
        "task": args.task, "mode": args.mode, "seed": args.seed,
        "space_file": str(space_file),
        "space_sha256": hashlib.sha256(space_file.read_bytes()).hexdigest(),
        "code_revision": revision,
        "validation_split": TASKS[args.task][2], "validation_cases": 50,
        "validation_case_ids": case_names,
        "aflow_validation_export_sha256": export_hash,
        "cache_enabled": False, "model_call_cache_mode": args.cache_mode,
        "model_call_cache_root": os.environ.get("SECRETAGENT_CALL_CACHE_ROOT"),
        "unique_evaluated": len(rows),
        "wall_time_s": elapsed_s,
        "audit": audit,
        "initial_population": [list(vec) for vec in initial_population(dim_sizes(dims), args.seed)[0]],
        "paired_summary": str(paired) if paired else None,
        "candidates": rows,
        "front_vectors": [vec for vec, _, _ in front],
        "generations": gen_log,
    }
    path.write_text(json.dumps(json_safe(result), indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved {len(rows)} configurations to {path}")


def save_test_result(path, args, space_file, dims, evaluated, case_names,
                     export_hash, validation_summary, audit, elapsed_s, revision=None):
    result = {
        "task": args.task, "mode": "test", "seed": args.seed,
        "code_revision": revision or current_revision(),
        "space_sha256": hashlib.sha256(space_file.read_bytes()).hexdigest(),
        "validation_summary": str(validation_summary),
        "test_split": TEST_SPLITS[args.task][0],
        "test_case_ids": case_names,
        "aflow_test_export_sha256": export_hash,
        "cache_enabled": False, "model_call_cache_mode": args.cache_mode,
        "model_call_cache_root": os.environ.get("SECRETAGENT_CALL_CACHE_ROOT"),
        "wall_time_s": elapsed_s,
        "audit": audit,
        "candidates": [
            {"vector": vec, "configuration": decode_dict(dims, vec),
             "test_accuracy": acc if math.isfinite(acc) else None,
             "test_cost_per_case": cost if math.isfinite(cost) else None,
             "failed": not math.isfinite(cost)}
            for vec, acc, cost in evaluated
        ],
    }
    path.write_text(json.dumps(json_safe(result), indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved test results for {len(evaluated)} validation-selected configurations to {path}")


# Paths whose contents can change what a run measures. Anything else (notes,
# drafts, images) cannot, so a commit touching only those is allowed to sit
# between a validation run and its test pass.
_EVALUATION_SUFFIXES = (".py", ".yaml", ".yml", ".json", ".toml", ".lock", ".txt")
_EVALUATION_ROOTS = ("src/", "scripts/", "benchmarks/", "tests/")


def inert_revision_drift(recorded: str, current: str) -> tuple[bool, list[str]]:
    """True when nothing between two revisions can affect a measurement.

    Committing results or notes moves HEAD, which would otherwise make every
    validation summary unusable for its own test pass.
    """
    if recorded == current:
        return True, []
    proc = subprocess.run(["git", "-C", str(ROOT), "diff", "--name-only",
                           f"{recorded}..{current}"],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        return False, ["<could not diff the two revisions>"]
    changed = [line.strip().replace("\\", "/") for line in proc.stdout.splitlines() if line.strip()]
    relevant = [c for c in changed
                if c.startswith(_EVALUATION_ROOTS) and c.endswith(_EVALUATION_SUFFIXES)]
    return not relevant, relevant


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--mode", required=True, choices=["fixed", "nsga", "random", "test"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True, help="New, empty run directory")
    ap.add_argument("--paired-summary", help="Fresh NSGA summary.json for random mode")
    ap.add_argument("--validation-summary", help="Fixed-model summary.json for test mode")
    ap.add_argument("--aflow-dir", required=True,
                    help="Prepared AFlow checkout holding the archived dataset exports")
    ap.add_argument("--override", action="append", default=[], metavar="KEY=VALUE",
                    help="Extra dotlist override applied to every configuration. "
                         "Recorded in summary.json so the run stays auditable. "
                         "Repeatable.")
    ap.add_argument("--only-method", action="append", default=[], metavar="NAME",
                    help="Evaluate only these top-level methods. Use to rerun a "
                         "single configuration without repeating clean ones.")
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--cache-mode", choices=("record", "replay"), default="record")
    ap.add_argument("--cache-source", help="Existing run directory holding model_calls; required for replay")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.mode == "random" and not args.paired_summary:
        ap.error("random mode needs --paired-summary from a fresh paired NSGA run")
    if args.mode != "random" and args.paired_summary:
        ap.error("--paired-summary is only for random mode")
    if (args.mode == "test") != bool(args.validation_summary):
        ap.error("test mode needs --validation-summary; other modes must omit it")
    if (args.cache_mode == "replay") != bool(args.cache_source):
        ap.error("replay needs --cache-source; recording must omit it")
    if args.cache_source and not (Path(args.cache_source) / "model_calls").is_dir():
        ap.error("--cache-source has no model_calls directory")
    if args.mode in ("nsga", "random") and args.task == "musr_object":
        ap.error("the full-space search control is scoped to Sports and FinQA")

    task_dir, space_name, split = TASKS[args.task]
    benchmark = ROOT / "benchmarks" / task_dir
    space_file = benchmark / space_name
    dims, compound, meta = modular_space_from_yaml(str(space_file))

    if args.only_method:
        method_dim = next((d for d in dims if d.key == "toplevel_method"), None)
        if method_dim is None:
            ap.error("--only-method needs a toplevel_method dimension in the space")
        unknown = [m for m in args.only_method if m not in method_dim.values]
        if unknown:
            ap.error(f"unknown method(s) {unknown}; available: {list(method_dim.values)}")
        # size is a read-only property derived from values
        method_dim.values = [m for m in method_dim.values if m in args.only_method]
        print(f"restricted to methods: {list(method_dim.values)}")
    if args.mode in ("fixed", "test"):
        dims = [SearchDimension(d.key, [MODEL] if d.key == "llm.model" or d.key.endswith(".model")
                                else d.values) for d in dims]
        check_fixed_models(dims, compound)

    paired = None
    if args.mode == "random":
        paired = Path(args.paired_summary).resolve()
        info = json.loads(paired.read_text(encoding="utf-8"))
        expected_hash = hashlib.sha256(space_file.read_bytes()).hexdigest()
        if (info.get("mode"), info.get("task"), info.get("seed"),
                info.get("space_sha256"), info.get("code_revision"),
                info.get("cache_enabled"),
                info.get("initial_population")) != (
                "nsga", args.task, args.seed, expected_hash, current_revision(), False,
                [list(vec) for vec in initial_population(dim_sizes(dims), args.seed)[0]]):
            ap.error("paired summary does not match task, seed, space, or cache policy")
        count = info["unique_evaluated"]
        vectors = random_vectors(dim_sizes(dims), args.seed, count)

    validation_summary = None
    if args.mode == "test":
        validation_summary = Path(args.validation_summary).resolve()
        info = json.loads(validation_summary.read_text(encoding="utf-8"))
        if (info.get("mode"), info.get("task"), info.get("seed"),
                info.get("space_sha256")) != (
                "fixed", args.task, args.seed,
                hashlib.sha256(space_file.read_bytes()).hexdigest()):
            ap.error("validation summary does not match task, seed, or space")
        recorded_revision = info.get("code_revision")
        inert, relevant = inert_revision_drift(recorded_revision, current_revision())
        if not inert:
            ap.error("validation ran at %s but these files changed since: %s"
                     % (recorded_revision[:8], ", ".join(relevant)))
        if recorded_revision != current_revision():
            print(f"validation ran at {recorded_revision[:8]}, now at "
                  f"{current_revision()[:8]}; no evaluation code changed between them")
        vectors = info["front_vectors"]
        if not vectors:
            ap.error("validation front is empty; no test configurations were selected")
        split, count = TEST_SPLITS[args.task]
    else:
        count = 50

    out = Path(args.out).resolve()
    if out.exists():
        ap.error(f"output already exists: {out}")
    print(f"{args.mode}: {args.task}, {space_size(dims)} possible configurations")
    print(f"{split}: {count} cases; model-call cache {args.cache_mode}; "
          f"Cachier off; output: {out}")
    case_names, export_hash = check_cases(
        args.task, benchmark, split, Path(args.aflow_dir), count,
        "test" if args.mode == "test" else "validate")
    print(f"AFlow export matches all {count} case IDs and answers; SHA-256 {export_hash}")
    if args.mode == "random":
        print(f"random: {len(vectors)} unique configurations, paired with {paired}")
    if args.mode == "test":
        print(f"test: {len(vectors)} validation-selected configurations from {validation_summary}")
    if args.dry_run:
        if args.mode in ("fixed", "test"):
            for method in dims[0].values:
                print(f"  method: {method}; all model choices: {MODEL}")
        sample = [0] * len(dims)
        print("First configuration overrides:")
        for override in decode_modular(dims, sample, compound):
            print(f"  {override}")
        print("No files written and no model calls made.")
        return

    status = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if status:
        ap.error("commit or otherwise clean the experiment checkout before a paid run; "
                 "git status --short lists the remaining files")

    run_revision = current_revision()

    out.mkdir(parents=True)
    os.environ["SECRETAGENT_CALL_CACHE_MODE"] = args.cache_mode
    source = Path(args.cache_source).resolve() if args.cache_source else out
    os.environ["SECRETAGENT_CALL_CACHE_ROOT"] = str(source / "model_calls")
    cases_dir = out / "cases"
    if "command" in meta:
        base_command = shlex.split(meta["command"])
    else:
        base_command = [sys.executable, "-m", "secretagent.cli.expt", "run",
                        "--interface", meta["interface"]]
        if "evaluator" in meta:
            base_command += ["--evaluator", meta["evaluator"]]
    base_dotlist = [f"dataset.split={split}", f"dataset.n={count}",
                    "cachier.enable_caching=false", "evaluate.record_details=true",
                    f"evaluate.result_dir={cases_dir}"]
    if SHUFFLE_SEEDS[args.task] is not None:
        base_dotlist.append(f"dataset.shuffle_seed={SHUFFLE_SEEDS[args.task]}")
    for extra in args.override:
        if "=" not in extra:
            ap.error(f"--override expects KEY=VALUE, got {extra!r}")
        base_dotlist.append(extra)
    common = dict(dims=dims, fixed_overrides=[], base_command=base_command,
                  base_dotlist=base_dotlist, cwd=str(benchmark), timeout=args.timeout,
                  metric="correct", compound_overrides=compound,
                  expt_prefix=f"{args.mode}_s{args.seed}")
    start = time.monotonic()
    if args.mode == "fixed":
        _, evaluated, gen_log = run_exhaustive(**common)
    elif args.mode == "nsga":
        _, evaluated, gen_log = run_nsga2(
            **common, pop_size=12, n_gen=5, seed=args.seed)
    else:
        cache = EvalCache(**common)
        evaluated = [(list(vec), *cache(vec)) for vec in vectors]
        gen_log = []
    front = validation_front(evaluated) if args.mode != "test" else []
    elapsed_s = time.monotonic() - start
    audit = audit_outputs(cases_dir, args.mode, args.seed, evaluated, case_names)
    if args.mode == "test":
        save_test_result(out / "summary.json", args, space_file, dims,
                         evaluated, case_names, export_hash, validation_summary,
                         audit, elapsed_s, revision=run_revision)
    else:
        save_result(out / "summary.json", args, space_file, dims, evaluated,
                    front, gen_log, case_names, export_hash, paired, audit, elapsed_s,
                    revision=run_revision)
    failed = sum(not math.isfinite(cost) for _, _, cost in evaluated)
    if failed or audit["issues"]:
        raise SystemExit(f"{failed} configurations failed; {len(audit['issues'])} audit issues. "
                         f"Inspect {out / 'summary.json'} before reporting results")


if __name__ == "__main__":
    main()
