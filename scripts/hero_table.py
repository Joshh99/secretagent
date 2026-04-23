#!/usr/bin/env python3
"""Build hero tables of cost and correctness across tasks and strategies.

For each task in benchmark_status.TASKS, loads the latest result
directory for each strategy and computes mean +/- sem for cost and
correctness.  Prints two DataFrames: one for correctness, one for cost.

Usage:
    uv run scripts/hero_table.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from benchmark_status import TASKS, STRATEGIES, RESULT_DIR_RE

RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"

STRATEGY_ORDER = ["workflow", "react", "pot", "structured_baseline", "unstructured_baseline"]


def find_latest_result_dir(parent: Path, strategy: str) -> Path | None:
    """Find the latest result directory for a strategy under parent."""
    if not parent.is_dir():
        return None
    matches = []
    for d in parent.iterdir():
        if not d.is_dir():
            continue
        m = RESULT_DIR_RE.match(d.name)
        if m and m.group(1) == strategy:
            matches.append(d)
    if not matches:
        return None
    return sorted(matches)[-1]


def build_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build cost and correctness DataFrames across tasks and strategies.

    Returns (cost_df, correct_df) with tasks as rows and strategies as columns.
    Cell values are formatted as "mean +/- sem".
    """
    cost_rows = []
    correct_rows = []

    for task_subtask in TASKS:
        task, subtask = task_subtask.split("/")
        parent = RESULTS_DIR / task / subtask

        cost_row = {"task": task_subtask}
        correct_row = {"task": task_subtask}

        for strategy in STRATEGY_ORDER:
            result_dir = find_latest_result_dir(parent, strategy)
            if result_dir is None:
                cost_row[strategy] = ""
                correct_row[strategy] = ""
                continue

            csv_path = result_dir / "results.csv"
            if not csv_path.exists():
                cost_row[strategy] = ""
                correct_row[strategy] = ""
                continue

            df = pd.read_csv(csv_path)

            for metric, row in [("cost", cost_row), ("correct", correct_row)]:
                if metric not in df.columns:
                    row[strategy] = ""
                    continue
                mean = df[metric].mean()
                sem = df[metric].sem()
                row[strategy] = f"{mean:.4f} +/- {sem:.4f}"

        cost_rows.append(cost_row)
        correct_rows.append(correct_row)

    cost_df = pd.DataFrame(cost_rows).set_index("task")[STRATEGY_ORDER]
    correct_df = pd.DataFrame(correct_rows).set_index("task")[STRATEGY_ORDER]
    return cost_df, correct_df


def _avg_row(df: pd.DataFrame, label: str, exclude_prefix: str | None = None) -> pd.DataFrame:
    """Compute an average row from a table of "mean +/- sem" strings.

    Extracts the mean values, averages across tasks, and returns a
    single-row DataFrame with the same columns.
    """
    subset = df
    if exclude_prefix:
        subset = df[~df.index.str.startswith(exclude_prefix)]

    row = {"task": label}
    for col in df.columns:
        vals = []
        for cell in subset[col]:
            if cell and "+/-" in str(cell):
                vals.append(float(cell.split("+/-")[0].strip()))
        if vals:
            avg = np.mean(vals)
            se = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
            row[col] = f"{avg:.4f} +/- {se:.4f}"
        else:
            row[col] = ""
    return pd.DataFrame([row]).set_index("task")


def main():
    cost_df, correct_df = build_tables()

    correct_df = pd.concat([correct_df, _avg_row(correct_df, "AVERAGE")])
    cost_df = pd.concat([
        cost_df,
        _avg_row(cost_df, "AVERAGE"),
        _avg_row(cost_df, "AVERAGE (excl tau_bench)", exclude_prefix="tau_bench/"),
    ])

    print("=== Correctness ===")
    print(correct_df.to_string())
    print()
    print("=== Cost ===")
    print(cost_df.to_string())


if __name__ == "__main__":
    main()
