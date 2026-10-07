"""Run the Table 2/3 AFlow searches as a queue with a fixed number in flight.

Each job is one task group at one seed. Jobs are independent: they write to
their own workspace and share nothing, so the only reason to cap them is the
executor host's throughput. The cap is a command-line option because that
ceiling is a property of the provider, not of this code.

Longest jobs start first, which shortens the total when the groups differ in
length as much as these do: a NaturalPlan search takes hours and a RuleArena
NBA search takes about one.

Nothing is retried automatically. A failed job is left failed and reported at
the end, because a silent retry would hide the kind of fault the smoke runs
were meant to surface.
"""

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Minutes per candidate measured on the smoke runs, used only to order the
# queue. A wrong estimate costs a little scheduling efficiency, nothing else.
ESTIMATED_MINUTES_PER_CANDIDATE = {
    "naturalplan_meeting": 8.1,
    "naturalplan_trip": 8.1,
    "medcalc": 8.1,
    "rulearena_nba": 1.4,
    "musr_object": 3.6,
    "musr_murder": 3.6,
    "musr_team": 3.6,
}


def run_one(job, args, log_dir):
    task, seed = job
    name = f"{task}__seed{seed}"
    log_path = log_dir / f"{name}.log"
    command = [
        "uv", "run", "python", "scripts/run_aflow_cell.py",
        "--task", task, "--condition", "matched", "--seed", str(seed),
        "--executor", args.executor, "--optimizer", args.optimizer,
        "--run-tag", args.run_tag,
        "--max-candidates", str(args.max_candidates),
        "--concurrency", str(args.concurrency),
        "--aflow-dir", args.aflow_dir,
    ]
    started = time.time()
    print(f"[{datetime.now():%H:%M:%S}] start  {name}", flush=True)
    with log_path.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream,
                                stderr=subprocess.STDOUT)
    minutes = (time.time() - started) / 60
    status = "ok" if result.returncode == 0 else f"FAILED ({result.returncode})"
    print(f"[{datetime.now():%H:%M:%S}] {status:14s} {name}  {minutes:.1f} min"
          f"  log {log_path}", flush=True)
    return {"task": task, "seed": seed, "minutes": minutes,
            "returncode": result.returncode, "log": str(log_path)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", required=True,
                    help="Comma-separated task keys from run_aflow_cell.py")
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--run-tag", required=True)
    ap.add_argument("--aflow-dir", required=True)
    ap.add_argument("--executor", default="deepseek/deepseek-chat-v3.1")
    ap.add_argument("--optimizer", default="gemini-3.1-pro-preview")
    ap.add_argument("--max-candidates", type=int, default=54)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--in-flight", type=int, default=4,
                    help="Searches running at once, capped by the host's throughput")
    ap.add_argument("--log-dir", default=None)
    args = ap.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    jobs = [(task, seed) for task in tasks for seed in seeds]
    # Longest first, so a long job never starts last.
    jobs.sort(key=lambda j: -ESTIMATED_MINUTES_PER_CANDIDATE.get(j[0], 5.0))

    log_dir = Path(args.log_dir) if args.log_dir else (
        Path(args.aflow_dir) / "runs" / f"{args.run_tag}__logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    estimate = sum(ESTIMATED_MINUTES_PER_CANDIDATE.get(t, 5.0) * args.max_candidates
                   for t, _ in jobs) / 60
    print(f"{len(jobs)} searches, {args.in_flight} at a time, "
          f"{args.max_candidates} candidates each")
    print(f"executor {args.executor}, optimizer {args.optimizer}, "
          f"concurrency {args.concurrency}")
    print(f"serial estimate {estimate:.1f}h, so roughly "
          f"{estimate / args.in_flight:.1f}h if the host keeps up")
    print(f"logs in {log_dir}\n", flush=True)

    started = datetime.now(timezone.utc)
    with ThreadPoolExecutor(max_workers=args.in_flight) as pool:
        results = list(pool.map(lambda job: run_one(job, args, log_dir), jobs))

    elapsed = (datetime.now(timezone.utc) - started).total_seconds() / 3600
    failed = [r for r in results if r["returncode"] != 0]
    print(f"\n{len(results) - len(failed)}/{len(results)} searches finished, "
          f"{elapsed:.1f}h wall")
    for r in failed:
        print(f"  FAILED {r['task']} seed {r['seed']}: see {r['log']}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
