"""Export and verify one Table 2/3 task for AFlow, without launching a paid run.

This is the shared launcher behind the seven remaining Table 2/3 columns. It
follows the MuSR Object check in `run_aflow_table_musr.py`: rebuild the exact
case set the saved table column was scored on, compare case IDs and gold answers
against that column, and refuse to go further when they differ.

What it will not do:

  * touch the running `table2_musr3` workspace, or any MuSR Object workspace
  * start a paid run for a task whose AFlow benchmark adapter is not written
  * add a second cache; a live run records through the existing
    AFLOW_CACHE_MODE=record recorder, into that run's own model_calls directory

Usage:

    uv run python scripts/run_aflow_table_task.py list
    uv run python scripts/run_aflow_table_task.py check --task musr_murder \
        --aflow-dir C:/Users/STUDENT/aflow
    uv run python scripts/run_aflow_table_task.py check --all --aflow-dir C:/Users/STUDENT/aflow
"""

import argparse
import contextlib
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataclasses import replace

import aflow_table_tasks as registry
from aflow_table_tasks import ROOT, TASKS, TableTask


# Workspaces this script must never write into. The MuSR Object search is live.
PROTECTED_WORKSPACE_PREFIXES = ("table2_musr",)


def _benchmark_dir(name: str) -> Path:
    return ROOT / "benchmarks" / name


@contextlib.contextmanager
def _in_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@contextlib.contextmanager
def _unchanged(*directories: Path):
    """Put back any file these directories held before the block ran.

    Importing a benchmark's ptools can rewrite its own prompt templates, and on
    Windows that rewrite changes the line endings. A checking run that leaves
    the checkout dirty would then be refused by the paid runner, so anything
    touched here is restored byte for byte.
    """
    before = {}
    for directory in directories:
        if directory.is_dir():
            for path in directory.rglob("*"):
                if path.is_file():
                    before[path] = path.read_bytes()
    try:
        yield
    finally:
        for path, content in before.items():
            if path.is_file() and path.read_bytes() != content:
                path.write_bytes(content)


def _load_expt(name: str):
    """Import one benchmark's expt.py under its own module name.

    Every benchmark calls its module `expt`, so importing them by that name
    would hand back whichever one was imported first. Loading each from its
    file under a distinct name keeps the loaders separate when one process
    checks several tasks.
    """
    import importlib.util

    bench = _benchmark_dir(name)
    if str(bench) not in sys.path:
        # expt.py imports its siblings (ptools, eval_utils, accuracy).
        sys.path.insert(0, str(bench))
    module_name = f"_table_expt_{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, bench / "expt.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_cases(task: TableTask):
    """Rebuild the saved column's case list with the same code path it used.

    Each loader mirrors the benchmark's own `load_dataset`, then applies the
    shuffle seed and size the saved config.yaml recorded, so the order matches.
    """
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))

    if task.loader == "musr":
        expt = _load_expt("musr")
        return expt.load_dataset(task.split).configure(**task.configure).cases

    if task.loader == "natural_plan":
        expt = _load_expt("natural_plan")
        dataset = expt.load_dataset(
            task.split,
            prompt_mode=task.extra.get("prompt_mode", "0shot"),
            partition=task.extra.get("partition"))
        return dataset.configure(**task.configure).cases

    if task.loader == "rulearena":
        # rulearena binds its prompt templates by relative path while ptools is
        # imported, so both the import and the load need its own directory.
        bench = _benchmark_dir("rulearena")
        with _unchanged(bench / "prompt_templates"), _in_directory(bench):
            expt = _load_expt("rulearena")
            dataset = expt.load_dataset(task.extra["domain"], task.split)
        return dataset.configure(**task.configure).cases

    if task.loader == "medcalc":
        # MedCalc loads from HuggingFace, so this step needs the network even
        # though it makes no model call.
        bench = _benchmark_dir("medcalc")
        with _unchanged(bench / "prompt_templates"):
            expt = _load_expt("medcalc")
            dataset = expt.load_dataset(task.split)
        if task.category_filter:
            # The saved run filtered by category inside load_dataset and only
            # then shuffled, so filtering after the shuffle would give a
            # different case order than the table column was scored on.
            wanted = {name.lower() for name in task.category_filter}
            kept = [case for case in dataset.cases
                    if str((case.metadata or {}).get("category", "")).lower() in wanted]
            dataset = type(dataset)(name=dataset.name, split=dataset.split, cases=kept)
        return dataset.configure(**task.configure).cases

    raise ValueError(f"{task.key}: unknown loader {task.loader!r}")


def build_rows(task: TableTask, cases) -> list[dict]:
    """Turn the loaded cases into AFlow JSONL rows.

    AFlow reads `input` and `target`. Where the secretagent scorer needs more
    than a scalar gold, the extra fields ride alongside and the task's adapter
    reads them back.
    """
    rows = []
    for case in cases:
        if task.loader == "musr":
            narrative, question, choices = case.input_args
            numbered = "\n".join(f"{i}. {choice}" for i, choice in enumerate(choices))
            rows.append({
                "input": f"{narrative}\n\n## Question\n{question}\n\n## Choices\n{numbered}",
                "target": str(case.expected_output),
                # Case.name is a per-split index, so qualify it to stay unique.
                "case_name": f"{task.split}/{case.name}",
                "n_choices": len(choices),
            })

        elif task.loader == "natural_plan":
            # The scorer replays the plan against the instance, so the whole
            # instance is the gold and travels as JSON.
            rows.append({
                "input": case.input_args[0],
                "target": json.dumps(case.expected_output, sort_keys=True, default=str),
                "case_name": case.name,
            })

        elif task.loader == "rulearena":
            problem_text, domain, rules_text, metadata_json, forms_text = case.input_args
            parts = [f"## Rules\n{rules_text}" if rules_text else "",
                     f"## Forms\n{forms_text}" if forms_text else "",
                     f"## Problem\n{problem_text}"]
            rows.append({
                "input": "\n\n".join(part for part in parts if part),
                "target": str(case.expected_output),
                "case_name": case.name,
                "domain": domain,
                "metadata": metadata_json,
            })

        elif task.loader == "medcalc":
            note, question = case.input_args
            meta = case.metadata or {}
            rows.append({
                "input": f"{note}\n\n## Question\n{question}",
                "target": str(case.expected_output),
                "case_name": case.name,
                # calculate_accuracy needs all four to score a case.
                "lower_limit": meta.get("lower_limit"),
                "upper_limit": meta.get("upper_limit"),
                "output_type": meta.get("output_type"),
                "category": meta.get("category"),
            })

        else:
            raise ValueError(f"{task.key}: unknown loader {task.loader!r}")
    return rows


def load_validation_cases(task: TableTask):
    """The cases AFlow selects its winning workflow on.

    Taken from the split the saved MODO optimizer config recorded for this
    task, so both sides pick a workflow by looking at the same cases.
    """
    spec = task.validation
    if spec is None:
        raise ValueError(f"{task.key}: no validation split recorded yet")
    if spec.get("stratified"):
        # MedCalc has no recorded validation split. Use the benchmark's own
        # stratified_sample, by calculator_name, over the train split with the
        # date category dropped so validation covers what the table reports.
        # Importing medcalc's ptools is what rewrites its prompt templates, so
        # the import has to sit inside the guard, not before it.
        bench = _benchmark_dir("medcalc")
        with _unchanged(bench / "prompt_templates"):
            expt = _load_expt("medcalc")
            cases = expt.load_dataset(spec["split"]).cases
        # Reproduce the official validation set first, then take our 50 from
        # it, so ours is a subset of it rather than an independent draw. No
        # category filter: the official set is drawn from the whole train split
        # and filtered by category only afterwards.
        official = expt.stratified_sample(cases, spec["official_n"], seed=42)
        return expt.stratified_sample(official, spec["stratified"], seed=42)

    stand_in = replace(task, split=spec["split"],
                       configure=spec.get("configure", {}),
                       extra={**task.extra, **spec.get("extra", {})},
                       # A validation export is not filtered to one table
                       # column; the whole split is what the search saw.
                       category_filter=spec.get("category_filter", ()))
    return load_cases(stand_in)


def export_paths(task: TableTask, aflow_dir: Path) -> dict:
    """Where AFlow looks for this task's data, by the names run.py expects."""
    data = Path(aflow_dir) / "data" / "datasets"
    stem = task.export.lower()
    return {"validate": data / f"{stem}_validate.jsonl",
            "test": data / f"{stem}_test.jsonl"}


def update_checksums(entries: dict) -> Path:
    """Record each export's SHA-256 where run_aflow_cell.py verifies it.

    Edits in place rather than rewriting the file. It carries comments and a
    separate section for the raw sources the exports are built from, and
    rewriting it sorted would drop both.
    """
    path = (ROOT / "benchmarks" / "COMMON" / "aflow-rebuttal" / "upstream"
            / "dataset_checksums.txt")
    lines = path.read_text(encoding="utf-8").splitlines()

    def entry_name(line: str) -> str | None:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            return None
        parts = stripped.split(None, 1)
        return parts[1].lstrip("*").strip() if len(parts) == 2 else None

    remaining = dict(entries)
    last_entry = -1
    for i, line in enumerate(lines):
        name = entry_name(line)
        if name is None:
            continue
        last_entry = i
        if name in remaining:
            lines[i] = f"{remaining.pop(name)} *{name}"
    # New exports go with the others, above any trailing comment section.
    insert_at = last_entry + 1 if last_entry >= 0 else len(lines)
    for name, digest in sorted(remaining.items()):
        lines.insert(insert_at, f"{digest} *{name}")
        insert_at += 1

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def adapter_present(task: TableTask, aflow_dir: Path) -> bool:
    """Whether AFlow already has the benchmark class this task needs."""
    module = task.aflow_benchmark.rsplit(".", 1)[0]
    return (Path(aflow_dir) / Path(*module.split("."))).with_suffix(".py").is_file()


# A dataset name has to appear in all of these before AFlow can run it. They
# are tracked AFlow files, so editing them changes the archived patch, which
# must not happen while the MuSR Object search is still running.
REGISTRATION_FILES = ("run.py", "scripts/evaluator.py", "test_pass.py")


def registration_gaps(task: TableTask, aflow_dir: Path) -> list[str]:
    """AFlow files that do not yet know this task's dataset name."""
    gaps = []
    for name in REGISTRATION_FILES:
        path = Path(aflow_dir) / name
        if not path.is_file():
            gaps.append(f"{name} (missing)")
        elif task.export not in path.read_text(encoding="utf-8", errors="replace"):
            gaps.append(name)
    return gaps


def guard_workspace(workspace: str) -> None:
    name = Path(workspace).name
    if name.startswith(PROTECTED_WORKSPACE_PREFIXES):
        raise SystemExit(f"refusing to touch the live MuSR Object workspace: {workspace}")


def current_revision() -> str:
    proc = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                          capture_output=True, text=True)
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def check(task: TableTask, aflow_dir: Path, write: bool) -> dict:
    """Dry-run check for one task. Returns a report; never calls a model."""
    report = {"task": task.key, "label": task.label, "blockers": [], "warnings": []}

    cell = registry.baseline_cell(task)
    report["baseline"] = {
        "path": task.baseline,
        "cases": cell["cases"],
        "accuracy": round(cell["accuracy"], 4),
        "usd_per_100": round(cell["usd_per_100"], 4),
        "model": task.baseline_model,
    }

    try:
        cases = load_cases(task)
        rows = build_rows(task, cases)
    except Exception as error:                     # noqa: BLE001 - reported, not raised
        report["blockers"].append(f"could not rebuild the split: "
                                  f"{type(error).__name__}: {error}")
        report["warnings"].extend(registry.check_model_match(task))
        report["ready"] = False
        return report

    report["exported_cases"] = len(rows)
    blockers, warnings = registry.check_case_alignment(task, rows)
    report["blockers"].extend(blockers)
    report["warnings"].extend(warnings)
    report["warnings"].extend(registry.check_model_match(task))

    if not adapter_present(task, aflow_dir):
        report["blockers"].append(
            f"AFlow has no {task.aflow_benchmark}; the adapter must be written "
            f"and archived before a paid run")

    gaps = registration_gaps(task, aflow_dir)
    report["registration_gaps"] = gaps
    if gaps:
        report["blockers"].append(
            f"AFlow does not know the dataset {task.export}; register it in "
            f"{', '.join(gaps)} and rebuild the archived patch")

    if task.validation is None:
        report["blockers"].append(
            "no validation split recorded for this task yet, so AFlow has "
            "nothing to select a workflow on")
    else:
        report["validation_split"] = task.validation["split"]
        if task.validation.get("overlaps_test"):
            report["warnings"].append(
                f"validation is drawn from {task.validation['split']}, the same "
                f"split the table reports, matching what MODO did for this "
                f"column; the overlap applies to both sides and belongs in the paper")

    if write and not report["blockers"]:
        paths = export_paths(task, aflow_dir)
        validation_rows = build_rows(task, load_validation_cases(task))
        report["validation_cases"] = len(validation_rows)

        # One AFlow dataset can fill more than one table column. MedCalc is one
        # 1040-case run partitioned into Formulas and Rules afterwards, so the
        # test export is every column sharing this dataset, not just this one.
        # Exporting a single column would have scored the run on 660 cases and
        # left the Rules column empty.
        group = registry.export_group(task.export)
        if len(group) > 1:
            test_rows = []
            for member in group:
                test_rows.extend(build_rows(member, load_cases(member)))
            report["test_cases_exported"] = len(test_rows)
            report["test_columns"] = [member.label for member in group]
        else:
            test_rows = rows

        digests = {
            paths["validate"].name: registry.write_jsonl(validation_rows, paths["validate"]),
            paths["test"].name: registry.write_jsonl(test_rows, paths["test"]),
        }
        report["exports"] = {name: str(paths[key]) for key, name in
                             (("validate", paths["validate"].name),
                              ("test", paths["test"].name))}
        report["export_sha256"] = digests
        report["checksums_file"] = str(update_checksums(digests))

    report["ready"] = not report["blockers"]
    return report


def print_report(report: dict) -> None:
    base = report["baseline"]
    print(f"\n=== {report['label']} ===")
    print(f"  baseline      {base['path']}")
    print(f"  column        {base['cases']} cases, accuracy {base['accuracy']}, "
          f"${base['usd_per_100']}/100 at {base['model']}")
    if "exported_cases" in report:
        print(f"  exported      {report['exported_cases']} cases")
    if "validation_split" in report:
        print(f"  validation    {report['validation_split']}"
              + (f", {report['validation_cases']} cases" if "validation_cases" in report else ""))
    for name, digest in (report.get("export_sha256") or {}).items():
        print(f"  export        {name}  {digest[:16]}")
    for warning in report["warnings"]:
        print(f"  WARNING       {warning}")
    for blocker in report["blockers"]:
        print(f"  BLOCKER       {blocker}")
    print(f"  status        {'ready to launch' if report['ready'] else 'not ready'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="Show every Table 2/3 column and its status")

    checker = sub.add_parser("check", help="Dry-run check; makes no model call")
    checker.add_argument("--task", choices=sorted(TASKS))
    checker.add_argument("--all", action="store_true",
                         help="Check the seven pending columns")
    checker.add_argument("--aflow-dir", type=Path, required=True)
    checker.add_argument("--write-export", action="store_true",
                         help="Write the verified JSONL export into the AFlow checkout")
    checker.add_argument("--json", type=Path, help="Also save the report as JSON")

    args = ap.parse_args()

    if args.command == "list":
        print(f"{'task':22s} {'cases':>6s}  {'adapter':<52s} baseline")
        for task in TASKS.values():
            print(f"{task.label:22s} {task.expected_cases:6d}  "
                  f"{task.adapter_status:<52s} {task.baseline}")
        print(f"\nAll eight columns were produced at {registry.TABLE_BASELINE_MODEL}.")
        print(f"The AFlow runs are at {registry.AFLOW_EXECUTOR_MODEL}.")
        return

    if not args.task and not args.all:
        ap.error("pass --task or --all")

    keys = registry.PENDING if args.all else [args.task]
    reports = []
    for key in keys:
        report = check(TASKS[key], args.aflow_dir.resolve(), args.write_export)
        print_report(report)
        reports.append(report)

    ready = [r["task"] for r in reports if r["ready"]]
    blocked = [r["task"] for r in reports if not r["ready"]]
    print(f"\nready: {len(ready)}  blocked: {len(blocked)}")
    if blocked:
        print("blocked: " + ", ".join(blocked))

    if args.json:
        args.json.write_text(json.dumps(
            {"revision": current_revision(), "reports": reports}, indent=2),
            encoding="utf-8")
        print(f"report saved to {args.json}")


if __name__ == "__main__":
    main()
