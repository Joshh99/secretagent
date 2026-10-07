"""Rebuild the paper's main tables from the saved results and compare them with the paper.

Runs offline with plain Python 3.9 or newer. No packages, no API key, no model calls.

  python rebuild_tables.py           # write the rebuilt tables to tables/rebuilt/
  python rebuild_tables.py --check   # also compare every number with the paper; exit 1 on any difference

Tables covered:
  tab:hero (main results, compact), and the appendix versions hero.tex (accuracy) and hero-cost.tex
  tab:learning and tab:learning-cost (learned components, including the AFlow row)

Where the numbers come from:
  tables/index.json lists, for every table entry, the saved result file it reads. Those files sit
  under benchmarks/COMMON/ with their original names. Each holds one row per question: its ID,
  whether the answer was right, and what it cost. tables/expected_tables.json holds the numbers as
  printed in the paper, read straight from the paper's LaTeX.

How each number is computed:
  Accuracy: the share of questions answered correctly.
  Cost in the main results tables: the average cost of the questions that have a recorded cost,
    times 100 (USD per 100 questions).
  Cost in the learning cost table: total recorded cost divided by all questions, times 100.
    A question with no recorded cost adds nothing (pure Python steps make no model call).
  Murder in the learning tables: only the 50 test questions AFlow never saw during its search,
    except CodeDist, which was tested on a different set of 75 questions.
  The AFlow row comes from aflow/build_aflow_numbers.py.
  Averages: the mean of the printed (rounded) values in the row or column, as in the paper.
"""
import argparse
import csv
import json
import subprocess
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "tables"
OUT = TABLES / "rebuilt"

HERO_COMPACT = ["workflow", "react", "structured_baseline"]
HERO_ALL = ["workflow", "react", "pot", "structured_baseline", "unstructured_baseline", "unstructured_thinking"]
LEARNING_ROWS = {  # name printed in the paper -> name in index.json
    "Engineered Workflow Baseline/human": "human/human", "ReAct/human": "ReAct/human",
    "ReAct/learned": "ReAct/learned", "CodeDist/human": "codedist/human", "CodeDist/learned": "codedist/learned",
    "Orch-WfSeed/human": "orch-wfseed/human", "Orch-ToolSeed/human": "orch-toolseed/human",
    "Orch-ToolSeed/learned": "orch/learned", "AFlow/human": "aflow",
}
LEARNING_COLS = ["musr/murder", "musr/object", "musr/team", "natural_plan/meeting", "natural_plan/trip",
                 "rulearena/nba", "medcalc/formulas", "medcalc/rules"]
AFLOW_COLS = {"musr/murder": "Murder", "musr/object": "Object", "musr/team": "Team",
              "natural_plan/meeting": "Meeting", "medcalc/formulas": "Formulas", "medcalc/rules": "Rules"}
NOT_ON_HELD_OUT = {"codedist/human", "codedist/learned"}


def fmt(x):
    return "--" if x is None else f"{x:.2f}"


def mean(xs):
    return sum(xs) / len(xs)


def read_rows(rel):
    """One saved result file: a list of (question_id, correct or None, cost or None)."""
    out = []
    with open(ROOT / rel, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            c = r.get("correct", "")
            correct = None if c == "" else float(c in ("1", "1.0", "True", "true"))
            k = r.get("cost", "")
            out.append((r["case_name"].split("/")[-1], correct, None if k == "" else float(k)))
    return out


def hero_value(rows, metric):
    if metric == "correct":
        vals = [c for _, c, _ in rows if c is not None]
        return mean(vals) if vals else None
    vals = [k for _, _, k in rows if k is not None]
    return mean(vals) * 100 if vals else None


def learning_value(rows, metric, held_out=None):
    if held_out is not None:
        rows = [r for r in rows if r[0] in held_out]
    if metric == "correct":
        vals = [c for _, c, _ in rows if c is not None]
        return mean(vals) if vals else None
    return sum(k for _, _, k in rows if k is not None) / len(rows) * 100


def average(printed):
    """Mean of the printed values, worked out exactly and rounded half up (0.535 becomes 0.54)."""
    vals = [Decimal(v) for v in printed if v != "--"]
    if not vals:
        return "--"
    return str((sum(vals) / len(vals)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def build_hero(index, expected):
    names = {v: k for k, v in index["task_names"].items()}
    files = {(e["task"], e["method"]): e["file"] for e in index["entries"] if e["table"] == "hero"}
    cache = {}

    def cell(task, method, metric):
        rel = files.get((task, method))
        if rel is None:
            return "--"
        if rel not in cache:
            cache[rel] = read_rows(rel)
        return fmt(hero_value(cache[rel], metric))

    tables = {}
    for table, columns in (("hero-compact", [(m, "correct") for m in HERO_COMPACT] + [(m, "cost") for m in HERO_COMPACT]),
                           ("hero", [(m, "correct") for m in HERO_ALL]),
                           ("hero-cost", [(m, "cost") for m in HERO_ALL])):
        rows = {}
        for shown in expected[table]:
            if shown == "Average":
                continue
            task = names[shown.replace("$\\tau$", "$\\tau$")]
            rows[shown] = [cell(task, m, metric) for m, metric in columns]
        rows["Average"] = [average([r[i] for r in rows.values()]) for i in range(len(columns))]
        tables[table] = rows
    return tables


def build_learning(index, aflow):
    held = set(json.loads((ROOT / "aflow" / "data" / "murder_heldout_ids.json").read_text(encoding="utf-8")))
    files = {(e["row"], e["column"]): e["file"] for e in index["entries"] if e["table"] == "learning"}
    tables = {"learning": {}, "learning-cost": {}}
    for shown, row in LEARNING_ROWS.items():
        acc, cost = [], []
        for col in LEARNING_COLS:
            if row == "aflow":
                key = AFLOW_COLS.get(col)
                acc.append(fmt(aflow["per_seed"][key]["accuracy_mean"]) if key else "--")
                cost.append(fmt(aflow["per_seed"][key]["cost_mean"]) if key else "--")
                continue
            rows = read_rows(files[(row, col)])
            ids = held if col == "musr/murder" and row not in NOT_ON_HELD_OUT else None
            acc.append(fmt(learning_value(rows, "correct", ids)))
            cost.append(fmt(learning_value(rows, "cost", ids)))
        avg = (lambda xs: "--") if row == "aflow" else average
        tables["learning"][shown] = acc + [avg(acc)]
        tables["learning-cost"][shown] = cost + [avg(cost)]
    return tables


def write(tables):
    OUT.mkdir(exist_ok=True)
    for name, rows in tables.items():
        lines = [f"# {name} (rebuilt)", ""] + [f"{k} | " + " | ".join(v) for k, v in rows.items()]
        (OUT / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="compare every number with the paper")
    args = ap.parse_args()
    index = json.loads((TABLES / "index.json").read_text(encoding="utf-8"))
    expected = json.loads((TABLES / "expected_tables.json").read_text(encoding="utf-8"))

    done = subprocess.run([sys.executable, str(ROOT / "aflow" / "build_aflow_numbers.py")], capture_output=True, text=True)
    if done.returncode:
        raise SystemExit("the AFlow script failed:\n" + done.stdout + done.stderr)
    aflow = json.loads((ROOT / "aflow" / "outputs" / "aflow_numbers.json").read_text(encoding="utf-8"))

    tables = {**build_hero(index, expected), **build_learning(index, aflow)}
    write(tables)
    print(f"Wrote {len(tables)} tables to {OUT}")
    if not args.check:
        return
    total, wrong = 0, []
    for name, rows in expected.items():
        for row, printed in rows.items():
            got = tables[name].get(row)
            if got is None:
                wrong.append(f"{name}: row '{row}' was not rebuilt")
                continue
            for i, (p, g) in enumerate(zip(printed, got)):
                total += 1
                if p != g:
                    wrong.append(f"{name}: '{row}', column {i + 1}: rebuilt {g}, paper says {p}")
    for w in wrong:
        print("FAIL ", w)
    print(f"{total - len(wrong)} of {total} numbers match the paper.")
    sys.exit(1 if wrong else 0)


if __name__ == "__main__":
    main()
