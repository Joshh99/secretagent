"""Collect Table 2/3 AFlow runs: cost split, contamination watch, seed aggregation.

Three jobs, all read-only over finished or in-flight run directories.

`costs` separates what the search spent from what running the chosen workflow
costs. Those answer different questions and the paper reports them separately:
the search is a one-time price paid to find a workflow, while the running cost
is what a user pays per case afterwards. The split comes from the recorded call
archive, where the search phase and each test pass write to their own directory.

`watch` scans per-case CSVs for the failure that silently corrupted six earlier
scoring passes, where litellm returned the string "Connection error." and it was
graded as a wrong answer. Run it while a search is in flight, not afterwards.

`table` aggregates the seeds of one task into the mean and standard error that
go into Tables 2 and 3, matching how hero_table.py reports every other column.

Prices come from the AFlow checkout's own ModelPricing, so there is one source
of truth for cost and this cannot drift from what the runs recorded.
"""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


CONNECTION_ERROR = "connection error."
# A rejected call is scored as a wrong answer, so it has to be found the same
# way. 11,708 of these went unnoticed for two hours on 2026-09-24.
LIMIT_ERROR = "limit exceeded"


class Pricing:
    """AFlow's own price table, so costs here match the costs it recorded.

    The table is read out of the source rather than imported, because importing
    scripts.async_llm drags in AFlow's whole dependency chain (tree_sitter and
    the rest) which this repo's environment does not carry. Reading the literal
    keeps one source of truth without needing AFlow installed.
    """

    def __init__(self, prices: dict):
        self.prices = prices

    def get_price(self, model_name: str, token_type: str) -> float:
        if model_name in self.prices:
            return self.prices[model_name][token_type]
        hits = [key for key in self.prices if key in model_name]
        if len(hits) == 1:
            return self.prices[hits[0]][token_type]
        if len(hits) > 1:
            raise KeyError(f"model {model_name!r} matches several price keys "
                           f"{sorted(hits)}")
        raise KeyError(f"no price for model {model_name!r} in the AFlow checkout's "
                       "ModelPricing. Add it there before reporting cost.")


def load_pricing(aflow_dir: Path) -> Pricing:
    import ast

    source = (Path(aflow_dir) / "scripts" / "async_llm.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ModelPricing":
            for item in node.body:
                if isinstance(item, ast.Assign) and any(
                        getattr(t, "id", None) == "PRICES" for t in item.targets):
                    return Pricing(ast.literal_eval(item.value))
    raise ValueError(f"no ModelPricing.PRICES found in {aflow_dir}")


# OpenRouter's fp8 hosts for deepseek-chat-v3.1 charge different rates, from
# its public listing on 2026-09-23, in USD per 1K tokens. AFlow's ModelPricing
# carries one rate per model because it must price a call as it makes it; this
# is what makes the reported total exact once the host is known.
PROVIDER_PRICES = {
    "SiliconFlow": {"input": 0.00027, "output": 0.001},
    "Novita": {"input": 0.00027, "output": 0.001},
    "AtlasCloud": {"input": 0.00030, "output": 0.00095},
}


def walk_calls(run_dir: Path):
    """Yield (phase, model, input_tokens, output_tokens) per recorded call.

    The phase is the first directory under model_calls, which is `search` for
    the search itself and the test pass's own tag for a scoring pass.
    """
    root = run_dir / "model_calls"
    if not root.is_dir():
        return
    for path in root.rglob("*.json"):
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        usage = (entry.get("response") or {}).get("usage") or {}
        if usage.get("prompt_tokens") is None:
            continue
        phase = path.relative_to(root).parts[0]
        model = (entry.get("request") or {}).get("model", "unknown")
        provider = (entry.get("response") or {}).get("provider")
        yield (phase, model, provider,
               usage["prompt_tokens"], usage.get("completion_tokens") or 0)


def costs(run_dir: Path, pricing) -> dict:
    """Cost by phase and model, with the search and running totals separated."""
    per = defaultdict(lambda: [0, 0, 0])
    for phase, model, provider, tin, tout in walk_calls(run_dir):
        row = per[(phase, model, provider)]
        row[0] += 1
        row[1] += tin
        row[2] += tout

    report = {"run": run_dir.name, "phases": {}, "search_usd": 0.0, "running_usd": 0.0}
    for (phase, model, provider), (n, tin, tout) in sorted(
            per.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2] or "")):
        # The fp8 hosts charge different rates, so price each call by the host
        # that actually served it. AFlow's own table is the fallback for a
        # model whose response carries no provider, such as the Gemini
        # optimizer, which is not routed through OpenRouter.
        rates = PROVIDER_PRICES.get(provider) or {
            "input": pricing.get_price(model, "input"),
            "output": pricing.get_price(model, "output")}
        usd = (tin / 1000) * rates["input"] + (tout / 1000) * rates["output"]
        label = f"{model} @ {provider}" if provider else model
        report["phases"].setdefault(phase, {})[label] = {
            "calls": n, "input_tokens": tin, "output_tokens": tout,
            "provider": provider, "usd": usd}
        if phase == "search":
            report["search_usd"] += usd
        else:
            report["running_usd"] += usd
    return report


def scan_cases(run_dir: Path) -> dict:
    """Per-case CSV health. Non-zero `connection_errors` means stop the run."""
    rows = errors = files = limited = 0
    bad_files = []
    for path in run_dir.rglob("*.csv"):
        try:
            with path.open(encoding="utf-8", newline="") as stream:
                data = list(csv.DictReader(stream))
        except (OSError, ValueError):
            continue
        if not data:
            continue
        files += 1
        rows += len(data)
        bad = sum(1 for row in data
                  if (row.get("prediction") or "").strip().casefold() == CONNECTION_ERROR)
        capped = sum(1 for row in data
                     if LIMIT_ERROR in (row.get("prediction") or "").lower())
        if bad or capped:
            errors += bad
            limited += capped
            bad_files.append((str(path.relative_to(run_dir)), bad + capped, len(data)))
    return {"csv_files": files, "case_rows": rows,
            "connection_errors": errors, "limit_errors": limited,
            "bad_files": bad_files}


def read_results(run_dirs: list[Path]) -> list[dict]:
    """The table_result.json each scored seed leaves behind."""
    out = []
    for run_dir in run_dirs:
        for path in sorted(run_dir.rglob("table_result.json")):
            out.append(json.loads(path.read_text(encoding="utf-8")))
    return out


def aggregate(results: list[dict]) -> dict:
    """Mean and standard error across seeds, as hero_table.py reports a column.

    A single seed has no spread, so its error is reported as None rather than
    zero. Reporting zero would claim a precision the run cannot support.
    """
    def stats(values):
        n = len(values)
        mean = sum(values) / n
        if n < 2:
            return mean, None
        var = sum((v - mean) ** 2 for v in values) / (n - 1)
        return mean, math.sqrt(var / n)

    accuracy, error = stats([r["test_accuracy"] for r in results])
    cost, cost_error = stats([r["inference_usd_per_100"] for r in results])
    cases = {r["test_cases"] for r in results}
    return {"seeds": len(results), "test_cases": sorted(cases),
            "accuracy": accuracy, "accuracy_stderr": error,
            "usd_per_100": cost, "usd_per_100_stderr": cost_error,
            "rounds": [r.get("round") for r in results]}


def _fmt(value, error, places=4):
    if value is None:
        return "n/a"
    if error is None:
        return f"{value:.{places}f} (1 seed)"
    return f"{value:.{places}f} +/- {error:.{places}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    c = sub.add_parser("costs", help="Search cost against running cost")
    c.add_argument("run_dirs", nargs="+", type=Path)
    c.add_argument("--aflow-dir", type=Path, required=True)
    c.add_argument("--json", type=Path)

    w = sub.add_parser("watch", help="Scan per-case CSVs for contamination")
    w.add_argument("run_dirs", nargs="+", type=Path)

    t = sub.add_parser("table", help="Aggregate seeds into a table row")
    t.add_argument("run_dirs", nargs="+", type=Path)
    t.add_argument("--label", default="task")
    t.add_argument("--json", type=Path)

    args = ap.parse_args()

    if args.command == "watch":
        worst = 0
        for run_dir in args.run_dirs:
            scan = scan_cases(run_dir)
            worst = max(worst, scan["connection_errors"] + scan["limit_errors"])
            flag = "CONTAMINATED" if (scan["connection_errors"] or scan["limit_errors"]) else "clean"
            print(f"{run_dir.name}: {scan['case_rows']} rows in "
                  f"{scan['csv_files']} CSVs, {scan['connection_errors']} "
                  f"connection errors, {scan['limit_errors']} api-limit "
                  f"rejections [{flag}]")
            for path, bad, total in scan["bad_files"]:
                print(f"    {path}: {bad}/{total}")
        # Non-zero exit so a watcher loop can stop a run on contamination.
        return 1 if worst else 0

    if args.command == "costs":
        pricing = load_pricing(args.aflow_dir.resolve())
        reports = []
        for run_dir in args.run_dirs:
            report = costs(run_dir, pricing)
            reports.append(report)
            print(f"\n=== {report['run']} ===")
            for phase, models in report["phases"].items():
                kind = "search" if phase == "search" else "running"
                print(f"  {phase} ({kind})")
                for model, row in models.items():
                    print(f"    {model:30s} {row['calls']:6d} calls  "
                          f"{row['input_tokens']:>10,} in  "
                          f"{row['output_tokens']:>10,} out  ${row['usd']:.4f}")
            print(f"  search cost  ${report['search_usd']:.4f}  (one time, to find the workflow)")
            print(f"  running cost ${report['running_usd']:.4f}  (to score the test split)")
        if args.json:
            args.json.write_text(json.dumps(reports, indent=2), encoding="utf-8")
            print(f"\nsaved to {args.json}")
        return 0

    results = read_results(args.run_dirs)
    if not results:
        print("no table_result.json found under those directories")
        return 1
    summary = aggregate(results)
    summary["label"] = args.label
    print(f"{args.label}: {summary['seeds']} seed(s), "
          f"{summary['test_cases']} test cases")
    print(f"  accuracy    {_fmt(summary['accuracy'], summary['accuracy_stderr'])}")
    print(f"  USD/100     {_fmt(summary['usd_per_100'], summary['usd_per_100_stderr'])}")
    print(f"  rounds      {summary['rounds']}")
    if summary["seeds"] < 3:
        print("  NOTE: fewer than 3 seeds, so the spread is not yet meaningful")
    if args.json:
        args.json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
