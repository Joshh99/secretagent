#!/usr/bin/env python
"""Collect every AFlow cell into one CSV and print a short table.

Reads each runs/<cell>/ directory, so it works on a partial matrix and can be
rerun as cells finish. Writes the columns the result package specifies.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

COLUMNS = [
    "task", "arm", "condition", "seed", "executor", "candidate_id", "generation",
    "validation_cases", "test_cases", "validation_accuracy", "test_accuracy",
    "inference_usd_per_case", "inference_usd_per_100", "search_usd_total",
    "failures", "artifact_path",
]

PATCH = (Path(__file__).resolve().parents[1] / "benchmarks" / "COMMON"
         / "aflow-rebuttal" / "upstream" / "aflow_changes.patch")
EXPECTED_PATCH_SHA256 = hashlib.sha256(PATCH.read_bytes()).hexdigest()


def load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def cell_rows(cell_dir: Path):
    manifest = load_json(cell_dir / "manifest.json")
    if (not manifest or manifest.get("aflow_patch_sha256") != EXPECTED_PATCH_SHA256
            or manifest.get("model_call_cache_mode") != "record"):
        return []

    dataset = manifest["aflow_dataset"]
    wf = cell_dir / dataset / "workflows"
    results = load_json(wf / "results.json") or []
    usage = load_json(wf / "search_usage.json") or {}

    val = {}
    for e in results:
        if e.get("score") is None:
            continue
        val.setdefault(e["round"], []).append(e["score"])
    val_mean = {r: sum(v) / len(v) for r, v in val.items()}
    validation_csvs = sorted(wf.glob("round_*/*.csv"))
    if not validation_csvs:
        return []
    validation_sizes = []
    for path in validation_csvs:
        with path.open(encoding="utf-8", newline="") as stream:
            cases = list(csv.DictReader(stream))
        if any((case.get("prediction") or "").strip().casefold() == "connection error."
               for case in cases):
            return []
        validation_sizes.append(len(cases))
    if len(set(validation_sizes)) != 1 or validation_sizes[0] != 50:
        return []
    val_n = validation_sizes[0]

    rows = []
    for usage_path in sorted((cell_dir / "test_logs").rglob("usage.json")):
        rec = load_json(usage_path)
        if not rec:
            continue
        audit = load_json(usage_path.with_name("failure_audit.json"))
        if (not audit or audit.get("connection_errors") != 0
                or audit.get("n_cases") != rec.get("n_cases")):
            continue
        if not math.isclose(rec.get("tracked_cost", -1),
                            rec.get("harness_total_cost", -2), rel_tol=1e-6,
                            abs_tol=1e-8):
            # Old AFlow code used a shared running cost in every case row.
            continue
        r = rec["round"]
        if r not in val_mean:
            continue
        per_case = rec["harness_total_cost"] / rec["n_cases"]
        rows.append({
            "task": manifest["task"],
            "arm": "aflow",
            "condition": manifest["seed_condition"],
            "seed": manifest["search_seed"],
            "executor": rec.get("exec_model") or manifest["executor_model"],
            "candidate_id": r,
            # Candidate N is produced by generation N-1; candidate 1 is the seed.
            "generation": max(r - 1, 0),
            "validation_cases": val_n,
            "test_cases": rec["n_cases"],
            "validation_accuracy": round(val_mean[r], 6) if r in val_mean else None,
            "test_accuracy": round(rec["score"], 6),
            "inference_usd_per_case": round(per_case, 9),
            "inference_usd_per_100": round(per_case * 100, 6),
            "search_usd_total": usage.get("optimizer_cost"),
            "failures": usage.get("failed_proposals"),
            "artifact_path": str(usage_path.parent),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="C:/Users/STUDENT/aflow/runs")
    ap.add_argument("--out", default="results.csv")
    args = ap.parse_args()

    root = Path(args.runs_root)
    if not root.is_dir():
        raise SystemExit(f"no runs directory at {root}")

    rows = []
    for cell in sorted(p for p in root.iterdir() if p.is_dir()):
        rows.extend(cell_rows(cell))

    if not rows:
        print(f"no completed cells under {root}")
        return

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {args.out}\n")

    hdr = f"{'task':<13}{'cond':<11}{'seed':>5}{'cand':>6}{'val':>8}{'test':>8}{'usd/case':>12}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        va = f"{r['validation_accuracy']:.3f}" if r["validation_accuracy"] is not None else "-"
        print(f"{r['task']:<13}{r['condition']:<11}{r['seed']:>5}{r['candidate_id']:>6}"
              f"{va:>8}{r['test_accuracy']:>8.3f}{r['inference_usd_per_case']:>12.2e}")


if __name__ == "__main__":
    main()
