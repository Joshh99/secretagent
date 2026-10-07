"""Record which saved result file feeds each entry of the paper's main tables.

Run once, inside the full repository, with its Python environment:
  python make_index.py --repo REPO --out index.json

It uses the repository's own lookup code (scripts/hero_table.py and
scripts/table2_shortcut_recompute.py), so the index lists exactly the runs those
scripts pick. Three learning-table entries the lookup code does not find are added
by hand below, with the reason. For every entry the index stores the file, a
checksum of the original full file, the number of questions, and the accuracy and
cost computed the way the table scripts compute them.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

# The paper's main table (tab:hero) shows these three methods; the appendix tables
# (hero.tex, hero-cost.tex) show all six.
HERO_STRATEGIES = ["workflow", "react", "pot", "structured_baseline", "unstructured_baseline", "unstructured_thinking"]

LEARNING_ROWS = ["human/human", "ReAct/human", "ReAct/learned", "codedist/human", "codedist/learned",
                 "orch-wfseed/human", "orch-toolseed/human", "orch/learned"]
LEARNING_COLS = ["musr/murder", "musr/object", "musr/team", "natural_plan/meeting", "natural_plan/trip",
                 "rulearena/nba", "medcalc/formulas", "medcalc/rules"]

# Entries the lookup code misses. Each one matches the paper's accuracy and cost exactly.
EXTRA = {
    # The tool-seeded NBA run sits one folder deeper, under without_rulebook/.
    ("orch-toolseed/human", "rulearena/nba"):
        "benchmarks/COMMON/orchestrator-results/seed_from_ptools/rulearena_nba/results/without_rulebook/"
        "20260504.005211.test_deepseek_v3_1/results.csv",
    # The MedCalc runs of the orchestrator with induced tools are not in the lookup's MedCalc list.
    ("orch/learned", "medcalc/formulas"): "benchmarks/COMMON/orchestrator-induced-ptools-results/medcalc/formulas/results.csv",
    ("orch/learned", "medcalc/rules"): "benchmarks/COMMON/orchestrator-induced-ptools-results/medcalc/rules/results.csv",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(repo, path):
    df = pd.read_csv(path)
    correct = None
    if "correct" in df:  # some benchmarks have no correct column; the tables leave them blank
        col = df["correct"]
        if col.dtype == object:
            col = col.map({"True": 1.0, "true": 1.0, "False": 0.0, "false": 0.0})
        correct = float(col.astype(float).mean())
    rel = Path(path).resolve().relative_to(repo).as_posix()
    return {"file": rel, "sha256": sha256(path), "n": len(df), "correct": correct,
            "cost100": float(df["cost"].mean() * 100) if "cost" in df else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    repo = a.repo.resolve()
    sys.path.insert(0, str(repo / "scripts"))
    import hero_table as ht  # noqa: E402
    import table2_shortcut_recompute as t2  # noqa: E402

    entries = []
    for task_subtask in ht.TASKS:
        task, subtask = task_subtask.split("/")
        parent = ht.RESULTS_DIR / task / subtask
        for strategy in HERO_STRATEGIES:
            note = ""
            if strategy == "unstructured_thinking":
                d = ht.find_latest_result_dir(parent, "zs_cot_prompt")
                note = "zs_cot_prompt"
                if d is None:
                    d, note = ht.find_latest_result_dir(parent, "unstructured_baseline"), "unstructured_baseline"
            else:
                d = ht.find_latest_result_dir(parent, strategy)
            if d is None or not (d / "results.csv").exists():
                continue
            e = {"table": "hero", "task": task_subtask, "method": strategy, **summarize(repo, d / "results.csv")}
            if note:
                e["source_strategy"] = note
            entries.append(e)

    finders = dict(t2.ROWS)
    for row in LEARNING_ROWS:
        for col in LEARNING_COLS:
            path = EXTRA.get((row, col))
            path = repo / path if path else t2.find_cell_csv(row, finders[row], col)
            if not path:
                raise SystemExit(f"no saved run for {row} / {col}")
            entries.append({"table": "learning", "row": row, "column": col, **summarize(repo, path)})

    a.out.write_text(json.dumps({"task_names": ht.TASK_TO_LATEX, "entries": entries}, indent=1), encoding="utf-8")
    print(f"{sum(e['table'] == 'hero' for e in entries)} main-table entries, "
          f"{sum(e['table'] == 'learning' for e in entries)} learning-table entries -> {a.out}")


if __name__ == "__main__":
    main()
