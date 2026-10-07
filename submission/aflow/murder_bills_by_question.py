"""Work out what each Murder test question cost, for the three chosen Murder workflows.

Why this exists: the paper scores Murder on the 50 test questions AFlow never saw during
its search, so the cost should cover the same 50 questions. The bill is recorded per
model call, not per question, so each call has to be tied to its question.

How it works:
  1. Most calls contain the full text of their question, so they are matched by text,
     the same way the MedCalc billing audit matched its calls.
  2. About one call per question cannot be matched that way. It is AFlow's voting step,
     which only sees the candidate answers ("Several answers have been generated ...").
  3. AFlow saved its own cost estimate for every question, made from that question's
     tokens. Subtracting the matched calls leaves exactly what each question's voting
     call(s) cost in AFlow's estimate. The script checks that these leftovers add up to
     the voting calls' total, then splits the voting calls' bill and tokens over the
     questions in proportion to the leftovers.
  4. It checks that the per question bills add up to the full test bill and that the
     tokens add up to AFlow's own token count.

Needs the full run folders, so it is run once, before the small data files are made.
Writes murder_bills_by_question.csv next to this file.

Usage: python murder_bills_by_question.py --runs RUNS --datasets DATASETS --analysis ANALYSIS
"""
import argparse
import bisect
import csv
import json
from decimal import Decimal
from pathlib import Path

SEEDS = {  # label in b1_test_cost.json: (chosen round, model_calls subfolder, test logs subfolder)
    "Murder s1": (53, "test_aamas_b_r53", "test_logs_aamas_b_r53"),
    "Murder s2": (43, "test", "test_logs"),
    "Murder s3": (33, "test", "test_logs"),
}
# AFlow's own per question cost estimate uses these prices (USD per million tokens).
TRACKER_IN, TRACKER_OUT = Decimal("0.27"), Decimal("1")
MILLION = Decimal(10**6)


def tracked(tokens_in, tokens_out):
    return (tokens_in * TRACKER_IN + tokens_out * TRACKER_OUT) / MILLION


def one_seed(base, chosen, calls_sub, logs_sub, inputs):
    rounds = []
    for rd in (base / logs_sub / "MuSRMurderMysteries").iterdir():
        if (rd / "usage.json").exists():
            u = json.loads((rd / "usage.json").read_text(encoding="utf-8"))
            rounds.append(((rd / "usage.json").stat().st_mtime_ns, u))
    rounds.sort(key=lambda x: x[0])
    ends = [r[0] for r in rounds]
    idx = next(i for i, (_, u) in enumerate(rounds) if u["round"] == chosen)
    usage = rounds[idx][1]

    matched = {q: {"billed": Decimal(0), "in": 0, "out": 0, "calls": 0} for q in inputs.values()}
    voting = {"billed": Decimal(0), "in": 0, "out": 0, "calls": 0}
    for f in (base / "model_calls" / calls_sub).rglob("*.json"):
        # A front test runs its rounds one after another; a call belongs to the first round
        # whose usage.json was written after it.
        if bisect.bisect_left(ends, f.stat().st_mtime_ns) != idx:
            continue
        e = json.loads(f.read_bytes())
        us = (e.get("response") or {}).get("usage") or {}
        cost = Decimal(str(us["cost"])) if us.get("cost") is not None else Decimal(0)
        pi, co = us.get("prompt_tokens") or 0, us.get("completion_tokens") or 0
        text = "\n".join(str(m.get("content") or "") for m in (e.get("request") or {}).get("messages", []))
        hits = [q for inp, q in inputs.items() if inp[:200] in text and inp in text]
        if len(hits) > 1:
            raise SystemExit(f"{f}: matches more than one question")
        v = matched[hits[0]] if hits else voting
        if not hits and not text.startswith("\nSeveral answers have been generated"):
            raise SystemExit(f"{f}: unmatched call that is not a voting call")
        v["billed"] += cost; v["in"] += pi; v["out"] += co; v["calls"] += 1

    csv_path = next((base / logs_sub / "MuSRMurderMysteries" / f"round_{chosen}").glob("*.csv"))
    aflow_estimate = {inputs[r["inputs"]]: Decimal(r["cost"])
                      for r in csv.DictReader(open(csv_path, newline="", encoding="utf-8"))}
    leftover = {q: aflow_estimate[q] - tracked(v["in"], v["out"]) for q, v in matched.items()}
    if min(leftover.values()) < Decimal("-1e-12"):
        raise SystemExit("a question's matched calls cost more than AFlow's own estimate for it")
    if abs(sum(leftover.values()) - tracked(voting["in"], voting["out"])) > Decimal("1e-9"):
        raise SystemExit("the leftovers do not add up to the voting calls")

    total_left = sum(leftover.values())
    rows = []
    for q, v in sorted(matched.items()):
        share = leftover[q] / total_left
        rows.append({"question_id": q,
                     "billed_usd": v["billed"] + voting["billed"] * share,
                     "input_tokens": Decimal(v["in"]) + voting["in"] * share,
                     "output_tokens": Decimal(v["out"]) + voting["out"] * share,
                     "matched_calls": v["calls"]})
    checks = {
        "bill": sum(r["billed_usd"] for r in rows),
        "input_tokens": sum(r["input_tokens"] for r in rows) - usage["input_tokens"],
        "output_tokens": sum(r["output_tokens"] for r in rows) - usage["output_tokens"],
        "voting_calls": voting["calls"],
        "voting_share_of_bill": voting["billed"] / sum(r["billed_usd"] for r in rows),
    }
    return rows, checks


def main():
    ap = argparse.ArgumentParser()
    for name in ("--runs", "--datasets", "--analysis"):
        ap.add_argument(name, required=True, type=Path)
    a = ap.parse_args()
    ds = [json.loads(l) for l in open(a.datasets / "musrmurdermysteries_test.jsonl", encoding="utf-8") if l.strip()]
    inputs = {r["input"]: r["case_name"].split("/")[-1] for r in ds}
    if len(inputs) != 100:
        raise SystemExit("expected 100 Murder questions with distinct texts")
    b1 = {e["seed"]: e for e in json.loads((a.analysis / "b1_test_cost.json").read_text(encoding="utf-8"))}
    out = []
    for label, (chosen, calls_sub, logs_sub) in SEEDS.items():
        rows, checks = one_seed(a.runs / b1[label]["workspace"], chosen, calls_sub, logs_sub, inputs)
        billed = Decimal(str(next(r["billed"] for r in b1[label]["rounds"] if r["selected"])))
        if abs(checks["bill"] - billed) > Decimal("1e-5"):
            raise SystemExit(f"{label}: question bills add up to {checks['bill']}, test bill is {billed}")
        print(label, {k: (round(float(v), 6) if isinstance(v, Decimal) else v) for k, v in checks.items()}, flush=True)
        for r in rows:
            out.append({"seed": int(label[-1]), "question_id": r["question_id"],
                        "billed_usd": f"{r['billed_usd']:.10f}", "input_tokens": f"{r['input_tokens']:.3f}",
                        "output_tokens": f"{r['output_tokens']:.3f}", "matched_calls": r["matched_calls"]})
    with open(Path(__file__).with_name("murder_bills_by_question.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)


if __name__ == "__main__":
    main()
