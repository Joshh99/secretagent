# Held-out test pass for an optimized AFlow workflow.
#
# Selects the best round by MEAN VALIDATION score from workspace results.json
# (selection on validation only), then scores that round's graph once on the
# held-out *_test.jsonl. Bypasses optimizer.test() (hardcoded rounds=[1] and a
# separate workflows_test dir) — same eval path as validation, full control.
#
# Usage:
#   .venv/Scripts/python.exe test_pass.py --dataset SportsUnderstanding [--round N]
import argparse
import asyncio
import csv
import importlib
import json
import os
from collections import defaultdict

from selection import pareto_front, round_means, single_best

from scripts.async_llm import LLMsConfig
from benchmarks.bbh import BBHBenchmark
from benchmarks.finqa import FinQABenchmark
from benchmarks.musr_object import MuSRObjectBenchmark
from benchmarks.naturalplan_meeting import NaturalPlanMeetingBenchmark
from benchmarks.naturalplan_trip import NaturalPlanTripBenchmark
from benchmarks.rulearena_nba import RuleArenaNBABenchmark
from benchmarks.medcalc import MedCalcBenchmark

BENCH = {"SportsUnderstanding": BBHBenchmark, "FinQA": FinQABenchmark,
         "MuSRObjectPlacements": MuSRObjectBenchmark,
         "MuSRMurderMysteries": MuSRObjectBenchmark,
         "MuSRTeamAllocation": MuSRObjectBenchmark,
         "NaturalPlanMeeting": NaturalPlanMeetingBenchmark,
         "NaturalPlanTrip": NaturalPlanTripBenchmark,
         "RuleArenaNBA": RuleArenaNBABenchmark,
         "MedCalcTest": MedCalcBenchmark}
EXEC_MODEL = "gemini-2.5-flash-lite"


def audit_case_results(log_dir: str, previous_files: set[str], expected_cases: int):
    """Reject incomplete or transport-failed scoring before reporting accuracy."""
    files = set(os.listdir(log_dir)) - previous_files
    csv_files = [name for name in files if name.endswith(".csv")]
    if len(csv_files) != 1:
        raise RuntimeError(f"expected one new case-results CSV in {log_dir}; found {csv_files}")
    path = os.path.join(log_dir, csv_files[0])
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != expected_cases:
        raise RuntimeError(f"{path}: expected {expected_cases} cases, found {len(rows)}")
    # Three ways a case can be scored 0 without that 0 being a measurement.
    # Only the first was checked, which is how a run whose key had expired, and
    # a workflow that never ran because of a NameError, both reported an
    # accuracy as though the model had simply answered badly.
    def _rows_matching(predicate):
        return [i for i, row in enumerate(rows, 1)
                if predicate((row.get("prediction") or "").strip())]

    failed = _rows_matching(lambda p: p.casefold() == "connection error.")
    capped = _rows_matching(lambda p: "limit exceeded" in p.lower())
    broke = _rows_matching(
        lambda p: any(m in p for m in ("Traceback (most recent call last)",
                                       "is not defined", "**exception raised"))
        or p.startswith(("NameError", "TypeError", "AttributeError",
                         "ImportError", "SyntaxError", "IndentationError")))
    with open(os.path.join(log_dir, "failure_audit.json"), "w", encoding="utf-8") as f:
        json.dump({"n_cases": expected_cases, "connection_errors": len(failed),
                   "api_limit_rejections": len(capped),
                   "workflow_exceptions": len(broke),
                   "failed_case_rows": failed, "api_limit_case_rows": capped,
                   "workflow_exception_case_rows": broke,
                   "results_csv": path}, f, indent=1)
    for count, what in ((failed, "model calls failed"),
                        (capped, "calls were rejected for exceeding an API limit"),
                        (broke, "cases raised inside the workflow")):
        if count:
            raise RuntimeError(f"{path}: {len(count)}/{expected_cases} {what}; "
                               "accuracy is invalid. See failure_audit.json")


def _means(dataset: str, workspace: str):
    with open(f"{workspace}/{dataset}/workflows/results.json", encoding="utf-8") as f:
        entries = json.load(f)
    return round_means(entries)


def rounds_to_test(dataset: str, workspace: str, mode: str = "pareto"):
    """Rounds to evaluate on the held-out split, per PREREGISTRATION.md D2.

    mode="pareto" returns the whole validation Pareto front, which is the primary
    result. mode="single" returns one round for a table cell. Upstream's rule of
    breaking ties toward the earliest round is deliberately not used: on Sports it
    reported AFlow's own untouched seed, since rounds 1, 2, 4, 7 and 8 all tied.
    """
    means = _means(dataset, workspace)
    for r in sorted(means):
        m = means[r]
        print(f"  round {r:2d}: val score {m['score']:.4f}  "
              f"val cost/case ${m['cost']:.6f}  ({m['repeats']} repeats)")
    if mode == "single":
        chosen = [single_best(means)]
    else:
        chosen = pareto_front(means)
    print(f"SELECTED ({mode}, validation only): {chosen}")
    return chosen


async def evaluate_round(dataset: str, r: int, exec_model: str, out_root: str, workspace: str):
    """Score one round's graph on the held-out split and record its usage."""
    # workspace is a path; the graph is imported as a module, so it has to be
    # expressed in dotted form. Every run gets its own workspace so that two
    # cells cannot import each other's round_N.graph.
    ws_mod = workspace.strip("/").replace("/", ".").replace("\\", ".")
    graph_mod = importlib.import_module(f"{ws_mod}.{dataset}.workflows.round_{r}.graph")

    exec_cfg = LLMsConfig.default().get(exec_model)
    wf = graph_mod.Workflow(name=dataset, llm_config=exec_cfg, dataset=dataset)

    log_dir = f"{out_root}/{dataset}/round_{r}"
    os.makedirs(log_dir, exist_ok=True)
    bench = BENCH[dataset](
        name=dataset,
        file_path=f"data/datasets/{dataset.lower()}_test.jsonl",
        log_path=log_dir,
    )
    previous_files = set(os.listdir(log_dir))
    score, avg_cost, total_cost = await bench.run_evaluation(wf, va_list=None)
    usage = wf.llm.get_usage_summary()
    # Count only non-blank lines: a trailing newline would otherwise inflate the
    # denominator of every per-case figure.
    with open(bench.file_path, encoding="utf-8") as f:
        n = sum(1 for line in f if line.strip())
    audit_case_results(log_dir, previous_files, n)
    record = {"round": r, "n_cases": n, "score": score,
              "exec_model": exec_model,
              "calls": usage["call_count"],
              "input_tokens": usage["total_input_tokens"],
              "output_tokens": usage["total_output_tokens"],
              "tracked_cost": usage["total_cost"],
              "harness_total_cost": total_cost,
              "calls_per_case": usage["call_count"] / n,
              "in_tok_per_case": usage["total_input_tokens"] / n,
              "out_tok_per_case": usage["total_output_tokens"] / n,
              "cost_per_case": usage["total_cost"] / n}
    with open(f"{log_dir}/usage.json", "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1)
    print(f"TEST RESULT dataset={dataset} round={r} "
          f"score={score:.4f} avg_cost=${avg_cost:.6f} total_cost=${total_cost:.4f} "
          f"calls/case={usage['call_count']/n:.2f} out_tok/case={usage['total_output_tokens']/n:.1f}")
    return record


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(BENCH))
    ap.add_argument("--round", type=int, default=None,
                    help="Evaluate this round only, bypassing selection")
    ap.add_argument("--mode", choices=["pareto", "single"], default="pareto",
                    help="pareto evaluates the whole validation front (primary result); "
                         "single evaluates one round for a table cell")
    ap.add_argument("--exec-model", default=EXEC_MODEL,
                    help="Executor model. Must match the one the search ran with.")
    ap.add_argument("--out-root", default="test_logs")
    ap.add_argument("--workspace", default="workspace",
                    help="Workspace root the search wrote to. Must be the same path "
                         "the seed generator and run.py used for this cell.")
    args = ap.parse_args()

    rounds = ([args.round] if args.round is not None
              else rounds_to_test(args.dataset, args.workspace, args.mode))

    records = []
    for r in rounds:
        records.append(await evaluate_round(args.dataset, r, args.exec_model,
                                            args.out_root, args.workspace))

    if len(records) > 1:
        print("\nHeld-out front:")
        for rec in sorted(records, key=lambda x: x["cost_per_case"]):
            print(f"  round {rec['round']:2d}: test {rec['score']:.4f} "
                  f"at ${rec['cost_per_case']:.6f}/case")


if __name__ == "__main__":
    asyncio.run(main())
