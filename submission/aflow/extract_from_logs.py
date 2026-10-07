"""Make the small data files in data/ from the full AFlow run folders.

You do not need to run this. It needs the full run folders (about 3 GB of saved
model calls), which are not in the supplementary material. It is here so you can
see exactly how each file in data/ was made.

What it keeps: question IDs, whether each answer was right, cost numbers, token
counts, validation scores per round, and which rounds failed and why. What it
leaves out: question text, model answers and the saved model calls.

Usage:
  python extract_from_logs.py --runs RUNS --datasets DATASETS --analysis ANALYSIS --baselines REPO
    RUNS       folder holding one folder per AFlow search
    DATASETS   folder holding AFlow's *_test.jsonl files (input text -> question ID)
    ANALYSIS   folder holding selection/, b1_test_cost.json, cost_ledger_final.json,
               b4_google_cost.json, b2_murder_rescore.json, murder_heldout_ids.json
    REPO       repository root that the baseline paths in b2_murder_rescore.json start from
    AUDIT      folder holding the billing audit's MedCalc_s<seed>_call_map.csv files

  (the last option is --billing-audit AUDIT)
"""
import argparse
import csv
import glob
import hashlib
import json
import math
import os
import shutil
from pathlib import Path

OUT = Path(__file__).resolve().parent / "data"
SUFFIX = "__deepseek_v31_atlascloud"

TASKS = {  # task: (AFlow dataset folder, dataset file)
    "musr_murder": ("MuSRMurderMysteries", "musrmurdermysteries_test.jsonl"),
    "musr_object": ("MuSRObjectPlacements", "musrobjectplacements_test.jsonl"),
    "musr_team": ("MuSRTeamAllocation", "musrteamallocation_test.jsonl"),
    "naturalplan_meeting": ("NaturalPlanMeeting", "naturalplanmeeting_test.jsonl"),
    "medcalc": ("MedCalcTest", "medcalctest_test.jsonl"),
}

# Every search attempt: (task, seed, try number, run tag, note). The folder name is
# "<tag>__<task>__matched__seed<seed>__deepseek_v31_atlascloud".
ATTEMPTS = [
    ("musr_murder", 1, 1, "tables23_restart2", "reused pilot"),
    ("musr_murder", 2, 1, "aamas_b", ""),
    ("musr_murder", 3, 1, "aamas_b", ""),
    ("musr_object", 1, 1, "aamas_b", ""),
    ("musr_object", 1, 2, "aamas_b_r2", ""),
    ("musr_object", 1, 3, "aamas_b_r3", ""),
    ("musr_object", 2, 1, "aamas_b", ""),
    ("musr_object", 2, 2, "aamas_b_r2", ""),
    ("musr_object", 2, 3, "aamas_b_r3", ""),
    ("musr_object", 3, 1, "aamas_b", ""),
    ("musr_team", 1, 1, "aamas_b", ""),
    ("musr_team", 1, 2, "aamas_b_r2", ""),
    ("musr_team", 2, 1, "aamas_b", ""),
    ("musr_team", 2, 2, "aamas_b_r2", ""),
    ("musr_team", 3, 1, "aamas_b", ""),
    ("naturalplan_meeting", 1, 1, "aamas_b", ""),
    ("naturalplan_meeting", 2, 1, "aamas_b", ""),
    ("naturalplan_meeting", 3, 1, "aamas_b", ""),
    ("medcalc", 1, 1, "aamas_b", ""),
    ("medcalc", 2, 1, "aamas_b", ""),
    ("medcalc", 3, 1, "tables23_restart2", "reused pilot"),
]

# Settings worth keeping from each search's manifest.json. Paths are left out, and so is
# our own code revision, which would point to a named repository during anonymous review.
SETTINGS = ["run_tag", "aflow_dataset", "search_seed", "executor_model", "optimizer_model",
            "max_candidates", "concurrency", "validation_repeats", "early_stop",
            "aflow_commit", "aflow_patch_sha256", "selection_sha256",
            "test_pass_sha256", "musr_scorer_sha256", "seed_prompt_sha256", "operators"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def is_damaged(row):
    """A validation answer lost to the network or to an API limit."""
    pred = (row.get("prediction") or "").strip()
    return pred.casefold() == "connection error." or "limit exceeded" in pred.lower()


def failure_kind(error):
    network = ("onnection" in error) or ("APIConnectionError" in error) or ("timed out" in error)
    return "network" if network else "code"


def attempt_record(runs, task, seed, try_no, tag, note):
    folder = f"{tag}__{task}__matched__seed{seed}{SUFFIX}"
    ws = runs / folder
    dataset = TASKS[task][0]
    wf = ws / dataset / "workflows"
    entries = json.loads((wf / "results.json").read_text(encoding="utf-8"))
    round_dirs = sorted(int(p.name.split("_")[1]) for p in wf.glob("round_*") if p.is_dir())
    damaged = {}
    for d in wf.glob("round_*"):
        n = sum(is_damaged(r) for f in d.glob("*.csv") for r in read_csv(f))
        if n:
            damaged[int(d.name.split("_")[1])] = n
    usage = json.loads((wf / "search_usage.json").read_text(encoding="utf-8"))
    failures = [{"round": f.get("attempted_round"),
                 "kind": failure_kind(f.get("error", "")),
                 "error_type": f.get("error", "").split(":")[0]}
                for f in usage.get("failures") or []]
    manifest = json.loads((ws / "manifest.json").read_text(encoding="utf-8"))
    return {
        "id": f"{task}_seed{seed}_try{try_no}",
        "task": task, "seed": seed, "try": try_no, "tag": tag, "note": note,
        "numbered_rounds": max(round_dirs),
        "validation": [{"round": e["round"], "score": e.get("score"), "avg_cost": e.get("avg_cost")}
                       for e in entries],
        "damaged_validation_rounds": {str(k): v for k, v in sorted(damaged.items())},
        "failed_proposals": failures,
        "settings": {k: manifest.get(k) for k in SETTINGS if k in manifest},
    }, folder


def selected_test(runs, datasets, folder, task, rnd):
    """Per question results of the chosen workflow, with the question text replaced by its ID."""
    dataset, dfile = TASKS[task]
    logs = "test_logs_aamas_b_r53" if folder.startswith("tables23_restart2__musr_murder") else "test_logs"
    d = runs / folder / logs / dataset / f"round_{rnd}"
    rows = read_csv(glob.glob(str(d / "*.csv"))[0])
    usage = json.loads((d / "usage.json").read_text(encoding="utf-8"))
    ds = [json.loads(line) for line in open(datasets / dfile, encoding="utf-8") if line.strip()]
    # MedCalc has one pair of questions that are identical in every field (test.0217 and
    # test.0218). Identical questions get their IDs in order, and every ID must be used once.
    by_input = {}
    for r in ds:
        by_input.setdefault(r["input"], []).append(r)
    out = []
    for r in rows:
        q = by_input[r["inputs"]].pop(0)
        out.append({"question_id": q["case_name"].split("/")[-1],
                    "category": q.get("category", ""),
                    "correct": float(r["score"]),
                    "aflow_cost_estimate": float(r["cost"])})
    if any(by_input.values()) or len(out) != len(ds):
        raise SystemExit(f"{dfile}: test results and question list do not match one to one")
    keep =["round", "n_cases", "score", "calls", "input_tokens", "output_tokens"]
    return out, {k: usage[k] for k in keep}


def main():
    ap = argparse.ArgumentParser()
    for name in ("--runs", "--datasets", "--analysis", "--baselines", "--billing-audit"):
        ap.add_argument(name, required=True, type=Path)
    a = ap.parse_args()
    if OUT.exists():
        raise SystemExit(f"{OUT} already exists; remove it first if you want to rebuild it")
    (OUT / "selected_tests").mkdir(parents=True)

    selections = {}
    for f in sorted((a.analysis / "selection").glob("*.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        selections[rec["search"]] = rec
    shutil.copytree(a.analysis / "selection", OUT / "selection_records")

    ledger = json.loads((a.analysis / "cost_ledger_final.json").read_text(encoding="utf-8"))
    b1 = {s["workspace"]: s for s in json.loads((a.analysis / "b1_test_cost.json").read_text(encoding="utf-8"))}
    b4 = {o["attempt"]: o for o in json.loads((a.analysis / "b4_google_cost.json").read_text(encoding="utf-8"))}

    attempts, usage_out = [], {}
    for task, seed, try_no, tag, note in ATTEMPTS:
        rec, folder = attempt_record(a.runs, task, seed, try_no, tag, note)
        sel = selections.get(folder)
        rec["accepted"] = sel is not None
        rec["selected_round"] = sel["selected_round"] if sel else None
        calls = ledger[folder]
        rec["billed_usd"] = {
            "search": round(calls["search"]["openrouter_billed"], 6),
            "tests": round(sum(v["openrouter_billed"] for k, v in calls.items() if k.startswith("test")), 6),
            "served_by": sorted({p for v in calls.values() for p in v["providers"] if p != "none"}),
        }
        opt = b4[folder.replace(SUFFIX, "")]
        rec["optimizer_tokens"] = {"calls": opt["optimizer_calls"], "input": opt["prompt_tokens"],
                                   "output_incl_thinking": opt["output_tokens_incl_thinking"]}
        if sel:
            rows, usage = selected_test(a.runs, a.datasets, folder, task, sel["selected_round"])
            name = f"{task}_seed{seed}.csv"
            with open(OUT / "selected_tests" / name, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            chosen = next(r for r in b1[folder]["rounds"] if r["selected"])
            usage.update({"billed_test_usd": chosen["billed"], "billed_calls_matched": chosen["assigned_calls"],
                          "aflow_calls": chosen["aflow_calls"], "file": f"selected_tests/{name}"})
            usage_out[f"{task}_seed{seed}"] = usage
        attempts.append(rec)
        print(rec["id"], "accepted" if rec["accepted"] else "rejected", rec["selected_round"])

    (OUT / "attempts.json").write_text(json.dumps(attempts, indent=1), encoding="utf-8")
    (OUT / "selected_test_usage.json").write_text(json.dumps(usage_out, indent=1), encoding="utf-8")

    ids = json.loads((a.analysis / "murder_heldout_ids.json").read_text(encoding="utf-8"))
    held = sorted(c.split("/")[-1] for c in ids["held_out_case_names"])
    (OUT / "murder_heldout_ids.json").write_text(json.dumps(held, indent=1), encoding="utf-8")

    # Murder baselines: question ID, correct, cost, from each run's saved results.csv.
    sources, rows_out = [], []
    for row in json.loads((a.analysis / "b2_murder_rescore.json").read_text(encoding="utf-8")):
        if not row.get("source") or not row.get("n_heldout"):
            continue
        rel = os.path.normpath(row["source"]).replace("\\", "/")
        path = a.baselines / rel
        if sha256(path) != row["sha256"]:
            raise SystemExit(f"checksum changed: {path}")
        for r in read_csv(path):
            correct = str(r["correct"]).strip().lower() in ("true", "1", "1.0")
            cost = r.get("cost", "")
            cost = "" if cost in ("", "nan", "NaN") or not math.isfinite(float(cost)) else float(cost)
            rows_out.append({"method": row["row"], "question_id": str(r["case_name"]).split("/")[-1],
                             "correct": int(correct), "cost": cost})
        sources.append({"method": row["row"], "source": rel, "sha256": row["sha256"]})
    with open(OUT / "murder_baselines.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["method", "question_id", "correct", "cost"])
        w.writeheader()
        w.writerows(rows_out)
    (OUT / "murder_baselines_sources.json").write_text(json.dumps(sources, indent=1), encoding="utf-8")

    # MedCalc bills per question. The billing audit matched every billed call of the three
    # chosen MedCalc tests to the question it answered. Here the calls are added up per
    # question, so the Formulas and Rules bills can be rebuilt without the call logs.
    rows_out = []
    for seed in (1, 2, 3):
        per_q = {}
        for r in read_csv(a.billing_audit / f"MedCalc_s{seed}_call_map.csv"):
            key = (r["case_names"], r["partition"])
            billed, calls = per_q.get(key, (0.0, 0))
            per_q[key] = (billed + float(r["billed"] or 0), calls + 1)
        for (names, part), (billed, calls) in sorted(per_q.items()):
            rows_out.append({"seed": seed, "question_ids": names, "column": part,
                             "billed_usd": round(billed, 10), "calls": calls})
    with open(OUT / "medcalc_billed_by_question.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["seed", "question_ids", "column", "billed_usd", "calls"])
        w.writeheader()
        w.writerows(rows_out)

    # Murder bills per question, made by murder_bills_by_question.py from the same call logs.
    murder = Path(__file__).with_name("murder_bills_by_question.csv")
    if not murder.exists():
        raise SystemExit("run murder_bills_by_question.py first")
    shutil.copy(murder, OUT / "murder_bills_by_question.csv")
    print("done:", OUT)


if __name__ == "__main__":
    main()
