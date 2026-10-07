#!/usr/bin/env python
"""Check or prepare an AFlow clone for this comparison without touching credentials."""

import argparse
import shutil
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmarks" / "COMMON" / "aflow-rebuttal" / "upstream"
PIN = "3f457218fc716093fe53f6df8a5d5e6379d66346"
COPIES = {
    "selection.py": "selection.py",
    "test_pass.py": "test_pass.py",
    "benchmarks_musr_object.py": "benchmarks/musr_object.py",
    # Archived since the first pilot but never listed here, so a checkout built
    # by this script alone was missing it and scripts/evaluator.py would not
    # import. Found while preparing a fresh checkout for the Table 2/3 runs.
    "benchmarks_finqa.py": "benchmarks/finqa.py",
    "call_cache.py": "scripts/call_cache.py",
    # The optimizer LLM sometimes emits the two characters \n between Python
    # statements instead of a line break, so prompt.py does not parse and the
    # round is abandoned before it is ever evaluated. graph_utils imports this
    # to repair that one fault at write time.
    "aflow_prompt_source.py": "scripts/aflow_prompt_source.py",
    # Table 2/3 adapters. The two scorer files are verbatim copies of the
    # secretagent originals so both sides grade a case by the same rule;
    # tests/test_aflow_table_adapters.py fails if either drifts.
    "scorers_natural_plan.py": "benchmarks/scorers_natural_plan.py",
    "scorers_medcalc.py": "benchmarks/scorers_medcalc.py",
    "benchmarks_naturalplan_meeting.py": "benchmarks/naturalplan_meeting.py",
    "benchmarks_naturalplan_trip.py": "benchmarks/naturalplan_trip.py",
    "benchmarks_rulearena_nba.py": "benchmarks/rulearena_nba.py",
    "benchmarks_medcalc.py": "benchmarks/medcalc.py",
}


def git(aflow, *args):
    return subprocess.run(["git", "-C", str(aflow), *args],
                          capture_output=True, text=True)


def same_text(left, right):
    return right.is_file() and left.read_text(encoding="utf-8") == right.read_text(encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--aflow-dir", type=Path, required=True)
    ap.add_argument("--apply", action="store_true",
                    help="Apply the archived patch and copy the three added source files")
    args = ap.parse_args()
    aflow = args.aflow_dir.resolve()
    revision = git(aflow, "rev-parse", "HEAD")
    if revision.returncode or revision.stdout.strip() != PIN:
        ap.error(f"AFlow must be checked out at {PIN}")

    patch = SOURCE / "aflow_changes.patch"
    reverse = git(aflow, "apply", "--check", "--reverse", str(patch))
    if reverse.returncode:
        forward = git(aflow, "apply", "--check", str(patch))
        if forward.returncode:
            ap.error("AFlow files match neither the pinned source nor the archived patch")
        if not args.apply:
            ap.error("patch is ready but not applied; rerun with --apply")
        applied = git(aflow, "apply", str(patch))
        if applied.returncode:
            ap.error(f"could not apply patch: {applied.stderr.strip()}")

    mismatches = []
    for source_name, destination_name in COPIES.items():
        source = SOURCE / source_name
        destination = aflow / destination_name
        if not same_text(source, destination):
            if args.apply:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            else:
                mismatches.append(destination_name)
    if mismatches:
        ap.error(f"added files need copying: {', '.join(mismatches)}; rerun with --apply")

    # Every operator template imports its own operator_an and op_prompt by the
    # absolute module path workspace.<name>.workflows.template, so that
    # directory has to exist in the checkout even though each run also gets its
    # own copy. The pilot checkout had it left over from earlier runs, which is
    # why this was never noticed; a checkout built only from this script failed
    # at round 1 with "No module named 'workspace.SportsUnderstanding'".
    # The files are the archived templates unchanged, so no recorded
    # operator_template_sha256 moves.
    missing_templates = []
    for template in sorted(p for p in (SOURCE / "templates").iterdir() if p.is_dir()):
        destination = aflow / "workspace" / template.name / "workflows" / "template"
        for source in sorted(p for p in template.iterdir() if p.is_file()):
            if not same_text(source, destination / source.name):
                if args.apply:
                    destination.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination / source.name)
                else:
                    missing_templates.append(f"workspace/{template.name}/{source.name}")
        if args.apply:
            # The template is imported as a package, so every level needs one.
            for level in (aflow / "workspace", destination.parent.parent,
                          destination.parent, destination):
                level.mkdir(parents=True, exist_ok=True)
                (level / "__init__.py").touch()
    if missing_templates:
        ap.error(f"operator templates need installing: "
                 f"{', '.join(missing_templates)}; rerun with --apply")

    config = aflow / "config" / "config2.yaml"
    if not config.is_file():
        ap.error("create config/config2.yaml from upstream/config2.yaml.redacted with your own keys")
    expected = yaml.safe_load((SOURCE / "config2.yaml.redacted").read_text(encoding="utf-8"))["models"]
    actual = yaml.safe_load(config.read_text(encoding="utf-8"))["models"]
    for name, fields in expected.items():
        current = actual.get(name) or {}
        if any(current.get(key) != value for key, value in fields.items() if key != "api_key"):
            ap.error(f"local config settings differ from the archived template for {name}")
        if not current.get("api_key") or current["api_key"] == "<YOUR_KEY>":
            ap.error(f"provide your own API key for {name}")
    print(f"AFlow source and added files verified at {PIN}")
    print("Local model settings match the archived redacted template")
    print("Operator templates are archived in the experiment repo and copied by run_aflow_cell.py")
    print("API keys remain in the local AFlow config and are never copied by this script")


if __name__ == "__main__":
    main()
