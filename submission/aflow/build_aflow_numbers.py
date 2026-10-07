"""Rebuild every AFlow number in the paper from the small files in data/.

Runs offline with plain Python 3.9 or newer. No packages, no API key, no model calls.

  python build_aflow_numbers.py           # write outputs/aflow_numbers.json and outputs/aflow_report.md
  python build_aflow_numbers.py --check   # also compare with the numbers printed in the paper

What it computes:
  1. Which workflow each search picks (best validation score, after dropping rounds
     hurt by network failures), and checks it against the saved choice.
  2. Accuracy and cost of each chosen workflow (the AFlow row of the main tables and
     the per-seed table in the supplementary appendix).
  3. The Murder results of the other methods on the 50 test questions AFlow did not
     use for validation.
  4. The question-by-question Murder comparison with bootstrap intervals.
  5. The table of all 21 search attempts and the cost totals.
"""
import argparse
import csv
import json
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "outputs"

# DeepSeek-V3.1 prices at Together AI, used to price every AFlow row in the main tables.
TOGETHER_IN, TOGETHER_OUT = 0.60, 1.70      # USD per million tokens
# Gemini 3.1 Pro list prices, used to estimate the optimizer's cost.
GEMINI_IN, GEMINI_OUT = 2.0, 12.0           # USD per million tokens
# MedCalc question categories that make up the "Formulas" column. The rest are "Rules".
FORMULAS = {"lab test", "physical", "dosage"}
TASK_NAMES = {"musr_murder": "Murder", "musr_object": "Object", "musr_team": "Team",
              "naturalplan_meeting": "Meeting"}
COLUMNS = ["Murder", "Object", "Team", "Meeting", "Formulas", "Rules"]


def load_json(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean(xs):
    return sum(xs) / len(xs)


# ---------------------------------------------------------------- 1. selection

def validation_means(attempt):
    """Mean validation score and cost per round, leaving out failed and damaged rounds."""
    damaged = {int(r) for r in attempt["damaged_validation_rounds"]}
    by_round = {}
    for e in attempt["validation"]:
        if e["score"] is None or e["round"] in damaged:
            continue
        by_round.setdefault(e["round"], []).append((e["score"], e["avg_cost"]))
    return {r: {"score": mean([s for s, _ in v]), "cost": mean([c for _, c in v])}
            for r, v in by_round.items()}


def pick_round(means):
    """Highest validation score; ties go to the lower validation cost, then the earlier round."""
    return min(means, key=lambda r: (-means[r]["score"], means[r]["cost"], r))


def check_selections(attempts):
    problems = []
    for a in attempts:
        if not a["accepted"]:
            continue
        chosen = pick_round(validation_means(a))
        record = json.loads((DATA / "selection_records" /
                             f"{a['tag']}__{a['task']}__matched__seed{a['seed']}__deepseek_v31_atlascloud.json")
                            .read_text(encoding="utf-8"))
        if not (chosen == a["selected_round"] == record["selected_round"]):
            problems.append(f"{a['id']}: recomputed {chosen}, saved {record['selected_round']}")
    return problems


# ------------------------------------------------- 2. accuracy and cost per seed

def medcalc_bills(seed, total_billed):
    """Billed cost of the Formulas and Rules questions of one chosen MedCalc workflow.

    Every billed call was matched to the question it answered (see medcalc_billed_by_question.csv),
    so each column's bill is the sum over its own questions.
    """
    rows = [r for r in load_csv(DATA / "medcalc_billed_by_question.csv") if int(r["seed"]) == seed]
    bills = {"Formulas": 0.0, "Rules": 0.0}
    for r in rows:
        bills[r["column"]] += float(r["billed_usd"])
    if abs(sum(bills.values()) - total_billed) > 1e-6:
        raise SystemExit(f"MedCalc seed {seed}: question bills add up to {sum(bills.values())}, "
                         f"but the test bill is {total_billed}")
    return bills


def seed_results(usage, held_out):
    """Accuracy and cost of one chosen workflow.

    Two costs, both in USD per 100 test questions:
      billed:   what AtlasCloud actually charged for the test run (the number in the paper)
      together: the same token counts priced at Together AI's DeepSeek-V3.1 rates, which is
                how most other rows of the cost table were priced
    """
    rows = load_csv(DATA / usage["file"])
    together = (usage["input_tokens"] * TOGETHER_IN + usage["output_tokens"] * TOGETHER_OUT) / 1e6
    billed = usage["billed_test_usd"]
    n = usage["n_cases"]
    if rows[0]["category"]:  # MedCalc: split into the Formulas and Rules columns
        seed = int(Path(usage["file"]).stem.rsplit("_seed", 1)[1])
        bills = medcalc_bills(seed, billed)
        out = {}
        total_estimate = sum(float(r["aflow_cost_estimate"]) for r in rows)
        for col, in_col in (("Formulas", lambda c: c in FORMULAS), ("Rules", lambda c: c not in FORMULAS)):
            part = [r for r in rows if in_col(r["category"])]
            share = sum(float(r["aflow_cost_estimate"]) for r in part) / total_estimate
            out[col] = {"accuracy": mean([float(r["correct"]) for r in part]), "n": len(part),
                        "billed": bills[col] / len(part) * 100,
                        "together": together * share / len(part) * 100}
        return out, billed, together
    costs = {"billed": billed / n * 100, "together": together / n * 100}
    if "murder" in usage["file"]:  # accuracy and cost both on the 50 held out questions
        seed = int(Path(usage["file"]).stem.rsplit("_seed", 1)[1])
        part = [r for r in rows if r["question_id"] in held_out]
        per_q = [r for r in load_csv(DATA / "murder_bills_by_question.csv") if int(r["seed"]) == seed]
        if abs(sum(float(r["billed_usd"]) for r in per_q) - billed) > 1e-5:
            raise SystemExit(f"Murder seed {seed}: question bills do not add up to the test bill")
        if abs(sum(float(r["input_tokens"]) for r in per_q) - usage["input_tokens"]) > 0.01:
            raise SystemExit(f"Murder seed {seed}: question tokens do not add up to the test tokens")
        held = [r for r in per_q if r["question_id"] in held_out]
        costs = {"billed": sum(float(r["billed_usd"]) for r in held) / len(held) * 100,
                 "together": sum(float(r["input_tokens"]) * TOGETHER_IN + float(r["output_tokens"]) * TOGETHER_OUT
                                 for r in held) / 1e6 / len(held) * 100}
        return {"Murder": {"accuracy": mean([float(r["correct"]) for r in part]),
                           "n": len(part), **costs}}, billed, together
    acc = mean([float(r["correct"]) for r in rows])
    if abs(acc - usage["score"]) > 1e-9:
        raise SystemExit(f"{usage['file']}: accuracy {acc} differs from AFlow's own {usage['score']}")
    task = Path(usage["file"]).stem.rsplit("_seed", 1)[0]
    return {TASK_NAMES[task]: {"accuracy": acc, "n": len(rows), **costs}}, billed, together


def per_seed_table(usages, held_out):
    table = {c: {"accuracy": [], "billed": [], "together": []} for c in COLUMNS}
    billed_vs_together = []
    for key in sorted(usages, key=lambda k: (k.rsplit("_seed", 1)[0], int(k.rsplit("_seed", 1)[1]))):
        results, billed, together = seed_results(usages[key], held_out)
        billed_vs_together.append(billed / together)
        for col, v in results.items():
            for k in ("accuracy", "billed", "together"):
                table[col][k].append(v[k])
    summary = {}
    for col, v in table.items():
        summary[col] = {
            "accuracy_per_seed": v["accuracy"], "cost_per_seed": v["billed"],
            "accuracy_mean": mean(v["accuracy"]), "accuracy_sd": statistics.stdev(v["accuracy"]),
            "cost_mean": mean(v["billed"]), "cost_sd": statistics.stdev(v["billed"]),
            "together_per_seed": v["together"], "together_mean": mean(v["together"]),
        }
    return summary, billed_vs_together


# --------------------------------------------- 3. Murder baselines on the held out 50

def murder_baselines(held_out):
    rows = load_csv(DATA / "murder_baselines.csv")
    order = [s["method"] for s in load_json("murder_baselines_sources.json")]
    out = {}
    for m in order:
        mine = [r for r in rows if r["method"] == m]
        held = [r for r in mine if r["question_id"] in held_out]
        def summary(rs):
            # Cost per 100 questions = recorded spend on these questions / number of questions.
            # A question with no recorded cost (Orch-WfSeed timed out on two) adds nothing,
            # the same way it counts as a wrong answer in the accuracy.
            costs = [float(r["cost"]) for r in rs if r["cost"] != ""]
            return {"n": len(rs), "accuracy": mean([int(r["correct"]) for r in rs]),
                    "cost": sum(costs) / len(rs) * 100, "n_with_cost": len(costs)}
        out[m] = {"held_out": summary(held), "all": summary(mine)}
    return out, order


# --------------------------------------------- 4. question-by-question Murder comparison

def paired_murder(usages, held_out, order):
    """AFlow (mean of 3 seeds per question) minus each baseline, 95% bootstrap over questions.

    The random generator is seeded once with 0 and shared across baselines in a fixed
    order, so the intervals come out the same every time.
    """
    aflow = {q: [] for q in held_out}
    for key in ("musr_murder_seed1", "musr_murder_seed2", "musr_murder_seed3"):
        for r in load_csv(DATA / usages[key]["file"]):
            if r["question_id"] in aflow:
                aflow[r["question_id"]].append(float(r["correct"]))
    assert all(len(v) == 3 for v in aflow.values()), "every held out question needs 3 seeds"
    a = {q: mean(v) for q, v in aflow.items()}
    rows = load_csv(DATA / "murder_baselines.csv")
    rng = random.Random(0)
    out = {}
    for m in order:
        b = {r["question_id"]: float(r["correct"]) for r in rows if r["method"] == m and r["question_id"] in a}
        qs = sorted(b)
        d = [a[q] - b[q] for q in qs]
        boots = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(10000))
        out[m] = {"n": len(qs), "baseline": mean([b[q] for q in qs]), "aflow": mean([a[q] for q in qs]),
                  "difference": mean(d), "low": boots[249], "high": boots[9749]}
    return out


# --------------------------------------------- 5. attempts table and cost totals

def attempts_table(attempts):
    rows = []
    for a in attempts:
        means = validation_means(a)
        scored = {e["round"] for e in a["validation"] if e["score"] is not None}
        damaged_val = {int(r) for r in a["damaged_validation_rounds"]}
        code = sum(f["kind"] == "code" for f in a["failed_proposals"])
        network = sum(f["kind"] == "network" for f in a["failed_proposals"])
        damaged = len(damaged_val) + network
        clean = len(scored - damaged_val)
        original = clean >= 50 and damaged <= 1
        amended = a["numbered_rounds"] >= 54 and damaged <= 1
        if amended != a["accepted"]:
            raise SystemExit(f"{a['id']}: amended rule says {amended}, records say {a['accepted']}")
        t = a["optimizer_tokens"]
        rows.append({
            "id": a["id"], "clean": clean, "parse": code, "damaged": damaged,
            "original": "accept" if original else "reject", "amended": "accept" if amended else "reject",
            "selected_round": a["selected_round"],
            "selected_validation": means[a["selected_round"]]["score"] if a["selected_round"] else None,
            "executor_usd": a["billed_usd"]["search"],
            "optimizer_usd": t["input"] / 1e6 * GEMINI_IN + t["output_incl_thinking"] / 1e6 * GEMINI_OUT,
            "tests_usd": a["billed_usd"]["tests"], "served_by": a["billed_usd"]["served_by"],
            "accepted": a["accepted"],
        })
    return rows


def cost_totals(rows, usages):
    acc = [r for r in rows if r["accepted"]]
    rej = [r for r in rows if not r["accepted"]]
    per_search = [r["executor_usd"] + r["optimizer_usd"] for r in acc]
    return {
        "accepted_executor": sum(r["executor_usd"] for r in acc),
        "accepted_optimizer": sum(r["optimizer_usd"] for r in acc),
        "per_search_min": min(per_search), "per_search_max": max(per_search),
        "selected_tests": sum(u["billed_test_usd"] for u in usages.values()),
        "all_accepted_tests": sum(r["tests_usd"] for r in acc),
        "rejected_executor": sum(r["executor_usd"] for r in rej),
        "rejected_optimizer": sum(r["optimizer_usd"] for r in rej),
        "only_atlascloud": all(r["served_by"] == ["AtlasCloud"] for r in rows),
    }


# ---------------------------------------------------------------- check and report

def fmt(x, like):
    """Format x with as many decimals as the printed paper value `like`."""
    d = len(like.split(".")[1]) if "." in like else 0
    return f"{x:.{d}f}"


def compare(numbers, expected):
    results = []

    def check(label, value, printed):
        got = fmt(value, printed) if isinstance(value, float) else str(value)
        if got == "-0.000" or got == "-0.00":
            got = got[1:]
        printed = printed.lstrip("+")
        got = got.lstrip("+")
        results.append((label, got, printed, got == printed))

    for col, v in expected["aflow_row"]["accuracy"].items():
        check(f"AFlow row accuracy, {col}", numbers["per_seed"][col]["accuracy_mean"], v)
    for col, v in expected["aflow_row"]["cost"].items():
        check(f"AFlow row cost (billed), {col}", numbers["per_seed"][col]["cost_mean"], v)
    for col, v in expected["aflow_row_at_together_prices"].items():
        check(f"AFlow cost at Together AI prices, {col}", numbers["per_seed"][col]["together_mean"], v)
    ratios = numbers["billed_vs_together"]
    check("lowest bill as a share of the Together AI price", min(ratios), expected["bill_share_of_together"]["low"])
    check("highest bill as a share of the Together AI price", max(ratios), expected["bill_share_of_together"]["high"])
    for col, e in expected["per_seed"].items():
        got = numbers["per_seed"][col]
        for i, v in enumerate(e["accuracy"]):
            check(f"seed {i + 1} accuracy, {col}", got["accuracy_per_seed"][i], v)
        for i, v in enumerate(e["cost"]):
            check(f"seed {i + 1} cost, {col}", got["cost_per_seed"][i], v)
        check(f"accuracy SD, {col}", got["accuracy_sd"], e["accuracy_sd"])
        check(f"cost SD, {col}", got["cost_sd"], e["cost_sd"])
    for m, e in expected["murder_baselines"].items():
        got = numbers["murder_baselines"][m][e["questions"]]
        check(f"Murder {m}, accuracy ({e['questions']})", got["accuracy"], e["accuracy"])
        check(f"Murder {m}, cost ({e['questions']})", got["cost"], e["cost"])
    for m, e in expected["paired"].items():
        got = numbers["paired_murder"][m]
        check(f"paired {m}, n", got["n"], e["n"])
        for k in ("baseline", "aflow", "difference", "low", "high"):
            check(f"paired {m}, {k}", got[k], e[k])
    by_id = {r["id"]: r for r in numbers["attempts"]}
    for e in expected["attempts"]:
        got = by_id[e["id"]]
        for k in ("clean", "parse", "damaged", "original", "amended", "selected_round"):
            check(f"attempt {e['id']}, {k}", got[k], str(e[k]))
        if e["selected_validation"] != "--":
            check(f"attempt {e['id']}, validation", got["selected_validation"], e["selected_validation"])
        check(f"attempt {e['id']}, executor", got["executor_usd"], e["executor"])
        check(f"attempt {e['id']}, optimizer", got["optimizer_usd"], e["optimizer"])
    for k, v in expected["totals"].items():
        check(f"total {k}", numbers["cost_totals"][k], v)
    return results


def report(numbers):
    lines = ["# AFlow numbers rebuilt from data/", "",
             "## AFlow row (mean of three seeds)", "",
             "| | " + " | ".join(COLUMNS) + " |", "|---" * (len(COLUMNS) + 1) + "|",
             "| accuracy | " + " | ".join(f"{numbers['per_seed'][c]['accuracy_mean']:.2f}" for c in COLUMNS) + " |",
             "| actual bill per 100 (USD) | " + " | ".join(f"{numbers['per_seed'][c]['cost_mean']:.2f}" for c in COLUMNS) + " |",
             "| at Together AI prices per 100 (USD) | " + " | ".join(f"{numbers['per_seed'][c]['together_mean']:.2f}" for c in COLUMNS) + " |",
             "", "## Murder, other methods, on the 50 held out questions", "",
             "| method | n | accuracy | cost per 100 |", "|---|---:|---:|---:|"]
    for m, v in numbers["murder_baselines"].items():
        h = v["held_out"]
        lines.append(f"| {m} | {h['n']} | {h['accuracy']:.3f} | {h['cost']:.3f} |")
    lines += ["", "## Murder, question by question (AFlow minus baseline)", "",
              "| method | n | difference | 95% interval |", "|---|---:|---:|---|"]
    for m, v in numbers["paired_murder"].items():
        lines.append(f"| {m} | {v['n']} | {v['difference']:+.3f} | [{v['low']:+.3f}, {v['high']:+.3f}] |")
    lines += ["", "## Search attempts", "",
              "| attempt | clean | parse | damaged | original | amended | selected (val.) | executor | optimizer |",
              "|---|---:|---:|---:|---|---|---|---:|---:|"]
    for r in numbers["attempts"]:
        sel = f"{r['selected_round']} ({r['selected_validation']:.2f})" if r["selected_round"] else "--"
        lines.append(f"| {r['id']} | {r['clean']} | {r['parse']} | {r['damaged']} | {r['original']} | "
                     f"{r['amended']} | {sel} | {r['executor_usd']:.2f} | {r['optimizer_usd']:.2f} |")
    t = numbers["cost_totals"]
    lines += ["", "## Cost totals (USD)", "",
              f"- Accepted searches, executor (billed): {t['accepted_executor']:.2f}",
              f"- Accepted searches, optimizer (estimate): {t['accepted_optimizer']:.2f}",
              f"- One accepted search, both together: {t['per_search_min']:.2f} to {t['per_search_max']:.2f}",
              f"- Testing the 15 chosen workflows (billed): {t['selected_tests']:.2f}",
              f"- All tests of accepted searches (billed): {t['all_accepted_tests']:.2f}",
              f"- Rejected attempts: executor {t['rejected_executor']:.2f}, optimizer {t['rejected_optimizer']:.2f}",
              f"- Every priced call served by AtlasCloud: {t['only_atlascloud']}",
              f"- Billed test cost as a share of the Together AI price: "
              f"{min(numbers['billed_vs_together']):.2f} to {max(numbers['billed_vs_together']):.2f}"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="compare with the numbers printed in the paper")
    args = ap.parse_args()

    attempts = load_json("attempts.json")
    usages = load_json("selected_test_usage.json")
    held_out = set(load_json("murder_heldout_ids.json"))

    problems = check_selections(attempts)
    if problems:
        raise SystemExit("selection does not match the saved records:\n  " + "\n  ".join(problems))
    per_seed, ratio = per_seed_table(usages, held_out)
    baselines, order = murder_baselines(held_out)
    rows = attempts_table(attempts)
    numbers = {"per_seed": per_seed, "murder_baselines": baselines,
               "paired_murder": paired_murder(usages, held_out, order),
               "attempts": rows, "cost_totals": cost_totals(rows, usages), "billed_vs_together": ratio}

    OUT.mkdir(exist_ok=True)
    (OUT / "aflow_numbers.json").write_text(json.dumps(numbers, indent=1), encoding="utf-8")
    (OUT / "aflow_report.md").write_text(report(numbers), encoding="utf-8")
    print("All 15 chosen workflows match their saved selection records.")
    print(f"Wrote {OUT / 'aflow_numbers.json'} and {OUT / 'aflow_report.md'}")

    if args.check:
        results = compare(numbers, json.loads((HERE / "expected.json").read_text(encoding="utf-8")))
        failed = [r for r in results if not r[3]]
        for label, got, printed, ok in failed:
            print(f"FAIL  {label}: rebuilt {got}, paper says {printed}")
        print(f"{len(results) - len(failed)} of {len(results)} numbers match the paper.")
        sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
