#!/usr/bin/env python
"""Run one AFlow search in its own workspace and record its inputs."""
import argparse
import csv
import hashlib
import os
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

TASKS = {
    "sports": {
        "aflow_dataset": "SportsUnderstanding",
        "benchmark_dir": "benchmarks/bbh/sports_understanding",
        "interface": "are_sports_in_sentence_consistent",
        "module": "ptools",
        "conf": "conf/conf.yaml",
        "template": "SportsUnderstanding",
    },
    "finqa": {
        "aflow_dataset": "FinQA",
        "benchmark_dir": "benchmarks/finqa",
        "interface": "answer_finqa",
        "module": "ptools",
        "conf": "conf/conf.yaml",
        "template": "FinQA",
    },
    "musr_object": {
        "aflow_dataset": "MuSRObjectPlacements",
        "benchmark_dir": "benchmarks/musr",
        "interface": "answer_question",
        "module": "ptools_object",
        "conf": "conf/object_workflow.yaml",
        "template": "SportsUnderstanding",
    },
    # The remaining Table 2/3 groups. The interface is the task itself rather
    # than the workflow wrapper around it, matching musr_object above, because
    # the seed prompt describes what to solve and not how the baseline solved
    # it. The template is the question-type operator set: SportsUnderstanding
    # for the qa tasks, FinQA for the numeric one.
    "musr_murder": {
        "aflow_dataset": "MuSRMurderMysteries",
        "benchmark_dir": "benchmarks/musr",
        "interface": "answer_question",
        "module": "ptools_murder",
        "conf": "conf/murder_workflow.yaml",
        "template": "SportsUnderstanding",
    },
    "musr_team": {
        "aflow_dataset": "MuSRTeamAllocation",
        "benchmark_dir": "benchmarks/musr",
        "interface": "answer_question",
        "module": "ptools_team",
        "conf": "conf/team_workflow.yaml",
        "template": "SportsUnderstanding",
    },
    "naturalplan_meeting": {
        "aflow_dataset": "NaturalPlanMeeting",
        "benchmark_dir": "benchmarks/natural_plan",
        "interface": "meeting_planning",
        "module": "ptools_meeting",
        "conf": "conf/meeting.yaml",
        "template": "SportsUnderstanding",
    },
    "naturalplan_trip": {
        "aflow_dataset": "NaturalPlanTrip",
        "benchmark_dir": "benchmarks/natural_plan",
        "interface": "trip_planning",
        "module": "ptools_trip",
        "conf": "conf/trip.yaml",
        "template": "SportsUnderstanding",
    },
    "rulearena_nba": {
        "aflow_dataset": "RuleArenaNBA",
        "benchmark_dir": "benchmarks/rulearena/nba",
        "interface": "compute_nba_answer",
        "module": "ptools",
        "conf": "conf/conf.yaml",
        "template": "SportsUnderstanding",
    },
    "medcalc": {
        "aflow_dataset": "MedCalcTest",
        "benchmark_dir": "benchmarks/medcalc",
        "interface": "calculate_medical_value",
        "module": "ptools",
        "conf": "conf/workflow.yaml",
        "template": "FinQA",
    },
}


def slug(text):
    """Cell names become Python module paths in test_pass.py, so only
    identifier-safe characters. A hyphen here fails at import time, not at
    search time, which would waste the whole run."""
    out = "".join(c if c.isalnum() else "_" for c in text).strip("_").lower()
    while "__" in out:
        out = out.replace("__", "_")
    return out


def cell_name(task, condition, seed, executor, run_tag):
    return f"{slug(run_tag)}__{slug(task)}__{slug(condition)}__seed{seed}__{slug(executor)}"


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def same_text(left, right):
    return Path(left).is_file() and Path(right).is_file() and (
        Path(left).read_text(encoding="utf-8") == Path(right).read_text(encoding="utf-8"))


def executor_config(aflow, model_name):
    """The executor's resolved settings, without its API key.

    Records the provider pin along with the model, so a finished run says which
    host served it and at what quantization rather than leaving that to be
    inferred from the date.
    """
    import yaml

    path = Path(aflow) / "config" / "config2.yaml"
    if not path.is_file():
        return None
    models = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("models") or {}
    config = models.get(model_name)
    if config is None:
        return None
    return {key: value for key, value in config.items() if key != "api_key"}


def openrouter_credit(aflow, model_name, needed=5.0):
    """Remaining credit on the key this executor uses, or None if not OpenRouter.

    A key that runs out mid-search does not stop the search. Every call is
    rejected with a 403, every case scores 0, and the optimizer keeps evolving
    from those zeros, so hours of wall time produce a poisoned trajectory that
    has to be thrown away. That is what happened on 2026-09-24: a $100 cap was
    reached at about 01:00 and 294 of 447 candidates were lost before anyone
    noticed. Checking before launch costs one request.
    """
    import json as _json
    import urllib.request

    config = executor_config(aflow, model_name) or {}
    if "openrouter.ai" not in str(config.get("base_url", "")):
        return None
    path = Path(aflow) / "config" / "config2.yaml"
    import yaml
    entry = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get(
        "models", {}).get(model_name, {})
    key = entry.get("api_key")
    if not key or key == "<YOUR_KEY>":
        return None
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/key",
        headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(request, timeout=30) as response:
        data = _json.load(response)["data"]
    limit, usage = data.get("limit"), data.get("usage") or 0.0
    remaining = None if limit is None else max(limit - usage, 0.0)
    return {"usage": usage, "limit": limit, "remaining": remaining,
            "enough": remaining is None or remaining >= needed}


def git_commit(repo):
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def audit_validation(workflows_dir, expected_cases=50):
    """Stop if a scored validation case is actually a failed model call.

    expected_cases is the size of this task's validation export, not a fixed
    50. RuleArena NBA has 42 validation cases, which is what its optimizer
    used, so a hardcoded 50 rejected a run that had scored every case.
    """
    files = sorted(workflows_dir.glob("round_*/*.csv"))
    if not files:
        raise RuntimeError(f"no validation case results in {workflows_dir}")
    for path in files:
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if len(rows) != expected_cases:
            raise RuntimeError(f"{path}: expected {expected_cases} cases, found {len(rows)}")
        failed = sum((row.get("prediction") or "").strip().casefold() == "connection error."
                     for row in rows)
        if failed:
            raise RuntimeError(f"{path}: {failed}/{expected_cases} model calls failed; "
                               "validation score is invalid")
        # A key that empties mid-search rejects every call with a 403, and the
        # case is then scored 0 as though the workflow were wrong. Left
        # unchecked the optimizer evolves from those zeros for hours.
        capped = sum("limit exceeded" in (row.get("prediction") or "").lower()
                     for row in rows)
        if capped:
            raise RuntimeError(f"{path}: {capped}/{expected_cases} calls were "
                               "rejected for exceeding an API limit; the score is "
                               "not a measurement. Restore credit and restart.")
    print(f"Validated {len(files)} AFlow validation passes: no connection errors")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--condition", required=True, choices=["matched", "unmatched"])
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--executor", default="gemini-2.5-flash-lite")
    ap.add_argument("--run-tag", default="iclr2027_full",
                    help="Separates full runs from earlier pilots. Use a new tag for a new run set.")
    ap.add_argument("--optimizer", default="gemini-3.1-pro-preview")
    ap.add_argument("--max-candidates", required=True, type=int,
                    help="Total candidates to evaluate, including the round-1 seed. "
                         "Matched to MODO's config count for this task.")
    ap.add_argument("--concurrency", type=int, default=None,
                    help="Per-cell concurrent case evaluations. AFlow defaults to 50, "
                         "so N cells at once means N*50 in flight.")
    ap.add_argument("--validation-repeats", type=int, default=1)
    ap.add_argument("--sample", type=int, default=4)
    ap.add_argument("--early-stop", default="false", choices=["true", "false"])
    ap.add_argument("--aflow-dir", required=True,
                    help="Path to the pinned and prepared AFlow checkout")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--min-credit", type=float, default=5.0,
                    help="Refuse to launch if the executor's OpenRouter key has "
                         "less than this much credit left.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.max_candidates < 2:
        ap.error("--max-candidates must be at least 2")
    if not slug(args.run_tag):
        ap.error("--run-tag must contain at least one letter or digit")

    spec = TASKS[args.task]
    if spec["interface"] is None:
        sys.exit(f"task {args.task!r} has no interface wired up yet in TASKS")

    aflow = Path(args.aflow_dir)
    repo = Path(args.repo)
    patch = repo / "benchmarks" / "COMMON" / "aflow-rebuttal" / "upstream" / "aflow_changes.patch"
    scorer_copy = repo / "benchmarks" / "COMMON" / "aflow-rebuttal" / "upstream" / "benchmarks_musr_object.py"
    name = cell_name(args.task, args.condition, args.seed, args.executor, args.run_tag)
    workspace = f"runs/{name}"
    ws_abs = aflow / workspace
    if ws_abs.exists():
        sys.exit(f"run path already exists: {ws_abs}; choose a new --run-tag")

    interpreter = Path(args.aflow_dir) / ".venv" / (
        "Scripts/python.exe" if os.name == "nt" else "bin/python")
    aflow_py = str(interpreter.resolve())
    if not interpreter.is_file():
        sys.exit(f"AFlow virtual environment is missing: {interpreter}")

    seed_cmd = [
        "uv", "run", "python", "scripts/make_aflow_seed_prompts.py",
        "--benchmark-dir", spec["benchmark_dir"],
        "--interface", spec["interface"],
        "--dataset", spec["aflow_dataset"],
        "--condition", args.condition,
        "--module", spec["module"],
        "--conf", spec["conf"],
        "--workspace-module", workspace.replace("/", "."),
        "--out", str(ws_abs),
    ]
    search_cmd = [
        aflow_py, "run.py",
        "--dataset", spec["aflow_dataset"],
        "--optimized_path", workspace,
        "--sample", str(args.sample),
        # AFlow evaluates round self.round + 1 each iteration, so max_rounds=N
        # yields N+1 candidates counting the seed. Subtract so the flag means
        # what it says.
        "--max_rounds", str(max(args.max_candidates - 1, 1)),
        "--validation_rounds", str(args.validation_repeats),
        "--check_convergence", args.early_stop,
        "--opt_model_name", args.optimizer,
        "--exec_model_name", args.executor,
        "--seed", str(args.seed),
    ]
    test_cmd = [
        aflow_py, "test_pass.py",
        "--dataset", spec["aflow_dataset"],
        "--workspace", workspace,
        "--exec-model", args.executor,
        "--mode", "pareto",
        "--out-root", f"{workspace}/test_logs",
    ]

    # These copies are archived with the experiment and match the pilot templates.
    template_src = repo / "benchmarks" / "COMMON" / "aflow-rebuttal" / "upstream" / "templates" / spec["template"]
    template_dst = ws_abs / spec["aflow_dataset"] / "workflows" / "template"
    if not template_src.is_dir():
        sys.exit(f"no operator template at {template_src}")
    if not patch.is_file():
        sys.exit(f"missing pinned AFlow patch: {patch}")
    patch_check = subprocess.run(
        ["git", "apply", "--check", "--reverse", str(patch)],
        cwd=aflow, capture_output=True, text=True,
    )
    if patch_check.returncode:
        sys.exit(f"AFlow checkout does not match the archived patch: {patch_check.stderr.strip()}")
    for archived, live in (
        ("selection.py", "selection.py"),
        ("test_pass.py", "test_pass.py"),
        ("benchmarks_musr_object.py", "benchmarks/musr_object.py"),
        ("call_cache.py", "scripts/call_cache.py"),
    ):
        if not same_text(repo / "benchmarks" / "COMMON" / "aflow-rebuttal" / "upstream" / archived,
                         aflow / live):
            sys.exit(f"AFlow file {live} differs from archived {archived}; run prepare_aflow.py")

    # The list above predates the Table 2/3 adapters, so verify the whole
    # installed set rather than a hand-kept subset. An edited live adapter
    # would otherwise score a run by a rule that is not the archived one.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from prepare_aflow import COPIES
    for archived, live in COPIES.items():
        if not same_text(repo / "benchmarks" / "COMMON" / "aflow-rebuttal" / "upstream" / archived,
                         aflow / live):
            sys.exit(f"AFlow file {live} differs from archived {archived}; run prepare_aflow.py")

    prompt_path = ws_abs / spec["aflow_dataset"] / "workflows" / "round_1" / "prompt.py"
    data_dir = aflow / "data" / "datasets"
    ds = spec["aflow_dataset"].lower()
    checksum_file = repo / "benchmarks" / "COMMON" / "aflow-rebuttal" / "upstream" / "dataset_checksums.txt"
    checksums = dict(
        (entry.lstrip("*"), digest)
        for digest, entry in (line.split() for line in checksum_file.read_text(encoding="utf-8").splitlines()
                              if line.strip() and not line.startswith("#"))
    )
    for stage in ("validate", "test"):
        filename = f"{ds}_{stage}.jsonl"
        path = data_dir / filename
        if not path.is_file() or sha256_file(path) != checksums[filename]:
            sys.exit(f"AFlow dataset export differs from the archived SHA-256: {path}")

    env = dict(os.environ)
    if args.concurrency:
        env["AFLOW_MAX_CONCURRENT"] = str(args.concurrency)
    env["AFLOW_CACHE_MODE"] = "record"
    cache_root = ws_abs / "model_calls"
    search_env = dict(env, AFLOW_CACHE_DIR=str(cache_root / "search"))
    test_env = dict(env, AFLOW_CACHE_DIR=str(cache_root / "test"))

    if not args.dry_run:
        status = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        if status:
            sys.exit("commit or otherwise clean the experiment checkout before a paid run")
        credit = openrouter_credit(aflow, args.executor, needed=args.min_credit)
        if credit is not None:
            if credit["remaining"] is None:
                print(f"OpenRouter key has no spending cap; used ${credit['usage']:.2f}")
            elif not credit["enough"]:
                sys.exit(f"OpenRouter key has ${credit['remaining']:.2f} left of "
                         f"${credit['limit']:.2f} and this run needs at least "
                         f"${args.min_credit:.2f}. Raise the cap before launching; "
                         f"a key that empties mid-search poisons it rather than "
                         f"stopping it.")
            else:
                print(f"OpenRouter key: ${credit['remaining']:.2f} of "
                      f"${credit['limit']:.2f} remaining")
        subprocess.run(seed_cmd, cwd=repo, check=True)
        shutil.copytree(template_src, template_dst)
        for pkg in (aflow / "runs", ws_abs, ws_abs / spec["aflow_dataset"],
                    ws_abs / spec["aflow_dataset"] / "workflows", template_dst):
            pkg.mkdir(parents=True, exist_ok=True)
            (pkg / "__init__.py").touch()

    manifest = {
        "cell": name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "task": args.task,
        "run_tag": args.run_tag,
        "aflow_dataset": spec["aflow_dataset"],
        "seed_condition": args.condition,
        "search_seed": args.seed,
        "executor_model": args.executor,
        "optimizer_model": args.optimizer,
        "max_candidates": args.max_candidates,
        "concurrency": args.concurrency,
        "model_call_cache_mode": "record",
        "model_call_cache_root": str(cache_root),
        "validation_repeats": args.validation_repeats,
        "early_stop": args.early_stop == "true",
        "workspace": workspace,
        "secretagent_commit": git_commit(repo),
        "aflow_commit": git_commit(aflow),
        "aflow_patch_sha256": sha256_file(patch),
        "selection_sha256": sha256_file(repo / "benchmarks" / "COMMON" / "aflow-rebuttal" / "upstream" / "selection.py"),
        "test_pass_sha256": sha256_file(repo / "benchmarks" / "COMMON" / "aflow-rebuttal" / "upstream" / "test_pass.py"),
        "musr_scorer_sha256": sha256_file(scorer_copy) if args.task == "musr_object" else None,
        # Every installed added file, so the scorer a run used is recoverable
        # whichever task it was. The hand-kept musr entry above is left in
        # place so older manifests stay readable.
        "added_file_sha256": {live: sha256_file(aflow / live) for live in COPIES.values()},
        # Which host served the executor, and at what settings. OpenRouter
        # routes to whichever host is cheapest unless pinned, and its default
        # for DeepSeek-V3.1 serves it at fp4 while the Table 2/3 columns were
        # served at fp8. Without this a reader cannot tell the two apart, so
        # the resolved config is recorded with the key stripped out.
        "executor_config": executor_config(aflow, args.executor),
        "seed_prompt_sha256": sha256_file(prompt_path) if not args.dry_run and prompt_path.exists() else None,
        "seed_prompt_path": str(prompt_path),
        "operator_template_from": str(template_src),
        "operator_template_sha256": {p.name: sha256_file(p) for p in template_src.iterdir() if p.is_file()},
        "operators": sorted(json.loads((template_src / "operator.json").read_text(encoding="utf-8")))
        if (template_src / "operator.json").exists() else None,
        "datasets": {
            f"{ds}_validate.jsonl": checksums[f"{ds}_validate.jsonl"],
            f"{ds}_test.jsonl": checksums[f"{ds}_test.jsonl"],
        },
        "commands": {
            "seed": " ".join(seed_cmd),
            "search": " ".join(search_cmd),
            "test": " ".join(test_cmd),
        },
    }

    if not args.dry_run:
        (ws_abs / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps(manifest, indent=1))

    if args.dry_run:
        print("\n[dry run] nothing launched. To run this cell, from the AFlow directory:")
        print("  " + " ".join(search_cmd))
        print("  " + " ".join(test_cmd))
        return

    subprocess.run(search_cmd, cwd=aflow, check=True, env=search_env)
    validation_export = data_dir / f"{ds}_validate.jsonl"
    validation_cases = sum(1 for line in
                           validation_export.read_text(encoding="utf-8").splitlines()
                           if line.strip())
    audit_validation(ws_abs / spec["aflow_dataset"] / "workflows", validation_cases)
    subprocess.run(test_cmd, cwd=aflow, check=True, env=test_env)


if __name__ == "__main__":
    main()
