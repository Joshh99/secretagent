#!/usr/bin/env python3
"""Monitor the status of benchmark experiments in this repo.

Usage:
    uv run scripts/benchmark_status.py [--task TASK] [--subtask SUBTASK]
"""

import argparse
import os
import re
import sys

BENCHMARKS_DIR = os.path.join(os.path.dirname(__file__), "..", "benchmarks")

TASKS = [
    "bbh/date_understanding",
    "bbh/geometric_shapes",
    "bbh/penguins_in_a_table",
    "bbh/sports_understanding",
    "designbench/vanilla",
    "designbench/react",
    "designbench/vue",
    "finqa/finqa",
    "medagentbench/medagentbench",
    "medcalc/equation",
    "medcalc/rule",
    "musr/murder",
    "musr/object",
    "musr/team",
    "natural_plan/calendar",
    "natural_plan/meeting",
    "natural_plan/trip",
    "rulearena/airlines",
    "rulearena/tax",
    "rulearena/nba",
    "tabwmp/tabwmp",
    "tau_bench/retail",
]

STRATEGIES = [
    "workflow",
    "pot",
    "react",
    "structured_baseline",
    "unstructured_baseline",
]

# Pattern: 2026mmdd.hhmmss.STRATEGY
RESULT_DIR_RE = re.compile(r"^2026\d{4}\.\d{6}\.(.+)$")


def find_result_dirs(parent, strategy):
    """Find all result directories in parent matching the given strategy name."""
    if not os.path.isdir(parent):
        return []
    matches = []
    for name in os.listdir(parent):
        m = RESULT_DIR_RE.match(name)
        if m and m.group(1) == strategy:
            matches.append(os.path.join(parent, name))
    return matches


def find_all_result_dirs(parent):
    """Find all timestamped result directories in parent, returning (name, strategy) pairs."""
    if not os.path.isdir(parent):
        return []
    results = []
    for name in sorted(os.listdir(parent)):
        m = RESULT_DIR_RE.match(name)
        if m:
            results.append((name, m.group(1)))
    return results


def check_benchmark(task_subtask, full=False):
    """Check a single TASK/SUBTASK and return (score, details)."""
    task, subtask = task_subtask.split("/")
    results_dir = os.path.join(BENCHMARKS_DIR, "results", task, subtask)

    if not os.path.isdir(results_dir):
        return 0, [f"  result directory benchmarks/results/{task}/{subtask} not found"]

    # Candidate locations for local results copy and llm_cache:
    #   benchmarks/TASK/results  or  benchmarks/TASK/SUBTASK/results
    local_results_candidates = [
        os.path.join(BENCHMARKS_DIR, task, "results"),
        os.path.join(BENCHMARKS_DIR, task, subtask, "results"),
    ]
    # llm_cache: benchmarks/TASK/llm_cache or benchmarks/TASK/SUBTASK/llm_cache
    llm_cache_candidates = [
        os.path.join(BENCHMARKS_DIR, task, "llm_cache"),
        os.path.join(BENCHMARKS_DIR, task, subtask, "llm_cache"),
    ]

    score = 0
    details = []

    for strategy in STRATEGIES:
        dirs = find_result_dirs(results_dir, strategy)
        if not dirs:
            details.append(f"  {strategy}: missing")
            continue

        # Use the latest directory (sorted lexicographically = chronologically)
        result_dir = sorted(dirs)[-1]
        dir_name = os.path.basename(result_dir)
        s_score = 0
        s_details = []

        # Check config.yaml
        if os.path.isfile(os.path.join(result_dir, "config.yaml")):
            s_score += 5
        else:
            s_details.append("no config.yaml")

        # Check results.csv
        if os.path.isfile(os.path.join(result_dir, "results.csv")):
            s_score += 5
        else:
            s_details.append("no results.csv")

        # Check local results copy
        has_local_copy = False
        for local_results in local_results_candidates:
            if os.path.isdir(os.path.join(local_results, dir_name)):
                has_local_copy = True
                break
        if has_local_copy:
            s_score += 5
        else:
            s_details.append("no local results copy")

        # Check llm_cache
        has_llm_cache = any(os.path.isdir(d) for d in llm_cache_candidates)
        if has_llm_cache:
            s_score += 5
        else:
            s_details.append("no llm_cache")

        score += s_score
        status = f"{s_score}/20"
        if s_details:
            status += f" ({', '.join(s_details)})"
        details.append(f"  {strategy}: {status}  [{dir_name}]")

    if full:
        scored_dirs = set()
        for strategy in STRATEGIES:
            for d in find_result_dirs(results_dir, strategy):
                scored_dirs.add(os.path.basename(d))
        all_dirs = find_all_result_dirs(results_dir)
        extras = [(name, strat) for name, strat in all_dirs if name not in scored_dirs]
        if extras:
            details.append("  *extra result directories:")
            for name, strat in extras:
                details.append(f"    {name}")

    return score, details


def main():
    parser = argparse.ArgumentParser(description="Monitor benchmark experiment status.")
    parser.add_argument("--task", help="Restrict to a specific task (e.g. musr)")
    parser.add_argument("--subtask", help="Restrict to a specific subtask (e.g. murder)")
    parser.add_argument("--full", action="store_true", help="List extra (non-scored) result directories")
    args = parser.parse_args()

    benchmarks = TASKS
    if args.task:
        benchmarks = [t for t in benchmarks if t.split("/")[0] == args.task]
    if args.subtask:
        benchmarks = [t for t in benchmarks if t.split("/")[1] == args.subtask]

    if not benchmarks:
        print(f"No matching benchmarks found for --task={args.task} --subtask={args.subtask}")
        return 1

    max_possible = len(STRATEGIES) * 20
    total_score = 0
    total_possible = 0

    print(f"Benchmark Status Report")
    print(f"{'=' * 70}")
    print()

    for task_subtask in benchmarks:
        total_possible += max_possible
        score, details = check_benchmark(task_subtask, full=args.full)
        total_score += score
        print(f"{task_subtask}: {score}/{max_possible}")
        for line in details:
            print(line)
        print()

    print(f"{'=' * 70}")
    print(f"Total: {total_score}/{total_possible}")

    return 0 if total_score == total_possible else 1


if __name__ == "__main__":
    sys.exit(main())
