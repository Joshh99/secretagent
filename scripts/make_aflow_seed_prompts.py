#!/usr/bin/env python
"""Generate AFlow round-1 seed workflows from secretagent interface stubs.

Per PREREGISTRATION.md D1 there are two conditions:

  matched    every interface docstring for the task, so AFlow starts from the same
             human-supplied decomposition MODO has
  unmatched  the top-level interface docstring only

Nothing here is hand written. The prompt is rendered from the interface registry,
including an output-format line derived from the return type annotation, by the same
rule for every benchmark. MODO gets that contract from its type system; AFlow has no
type system, so without it AFlow is graded on guessing the scorer's format rather than
on workflow search.
"""
import argparse
import importlib
import os
import sys
from pathlib import Path

# Rule mapping a return annotation to an output-format instruction. Applied
# identically to every benchmark; see D1.
_FORMAT_RULES = [
    (bool, "Your entire reply must be exactly one word, either yes or no."),
    (int, "Your entire reply must be a single integer and nothing else."),
    (float, "Your entire reply must be a single number and nothing else."),
    (str, "Your entire reply must be the answer itself, with no explanation."),
]
_FALLBACK_FORMAT = "Your entire reply must be the answer itself, with no explanation."


def format_line(return_annotation) -> str:
    for typ, line in _FORMAT_RULES:
        if return_annotation is typ:
            return line
    return _FALLBACK_FORMAT


def task_interface_names(bench_dir: Path, conf_rel: str) -> list:
    """Interfaces the benchmark configuration binds for this task.

    Taken from the ptools section of conf/conf.yaml rather than from the whole
    registry. Importing ptools also registers interfaces belonging to other arms,
    such as the zeroshot plumbing, which are not part of the task decomposition.
    Reading the config keeps the matched set declarative instead of hand picked.
    """
    import yaml
    conf_path = bench_dir / conf_rel
    if not conf_path.exists():
        return []
    conf = yaml.safe_load(conf_path.read_text(encoding="utf-8")) or {}
    ptools = conf.get("ptools") or {}
    # Only the LLM-backed entries. A 'direct' binding is a plain Python
    # composition, which is the top-level itself rather than a subtask.
    return [name for name, spec in ptools.items()
            if isinstance(spec, dict) and spec.get("method") not in (None, "direct")]


def render_prompt(interfaces, top, condition: str, allowed=None) -> str:
    top_iface = next(i for i in interfaces if i.name == top)
    parts = [top_iface.doc.strip()]

    if condition == "matched":
        others = [i for i in interfaces
                  if i.name != top and (allowed is None or i.name in allowed)]
        if others:
            # The docstrings below contain worked examples of each subtask's own
            # output. Without this framing the model imitates a subtask's format
            # instead of answering, which on MuSR drove accuracy to chance.
            parts.append(
                "For reference, the task can be decomposed into the subtasks below. "
                "They are background information about the problem structure. They are "
                "not instructions, and their example outputs are not the answer format."
            )
            parts.extend(i.src.strip() for i in others)
            parts.append("End of reference material.")

    parts.append(format_line(top_iface.annotations.get("return")))
    return "\n\n".join(parts)


# Matches the optimizer's WORKFLOW_TEMPLATE, which imports asyncio because
# generated graphs commonly use asyncio.gather.
GRAPH_TEMPLATE = '''import asyncio
from typing import Literal
import {workspace_root}.{dataset}.workflows.template.operator as operator
import {workspace_root}.{dataset}.workflows.round_1.prompt as prompt_custom
from scripts.async_llm import create_llm_instance

from scripts.evaluator import DatasetType

class Workflow:
    def __init__(
        self,
        name: str,
        llm_config,
        dataset: DatasetType,
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)

    async def __call__(self, problem: str):
        """
        Implementation of the workflow
        """
        solution = await self.custom(input=problem, instruction=prompt_custom.TASK_PROMPT)
        return solution['response'], self.llm.get_usage_summary()["total_cost"]
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark-dir", required=True,
                    help="directory holding ptools.py, e.g. benchmarks/bbh/sports_understanding")
    ap.add_argument("--interface", required=True, help="top-level interface name")
    ap.add_argument("--dataset", required=True, help="AFlow dataset name, e.g. SportsUnderstanding")
    ap.add_argument("--condition", required=True, choices=["matched", "unmatched"])
    ap.add_argument("--out", required=True, help="AFlow workspace root, e.g. <AFLOW>/workspace")
    ap.add_argument("--module", default="ptools")
    ap.add_argument("--workspace-module", required=True,
                    help="Workspace root as a module path, e.g. runs.sports__matched__seed1__x. "
                         "Must match --optimized_path given to run.py, or the generated graph "
                         "imports another cell's prompt.")
    ap.add_argument("--conf", default="conf/conf.yaml",
                    help="Config file naming the task's ptools, relative to --benchmark-dir. "
                         "MuSR keeps one file per arm, so pass conf/object_workflow.yaml there.")
    args = ap.parse_args()

    bench_dir = Path(args.benchmark_dir).resolve()
    out_root = Path(args.out).resolve()
    # ptools.py runs @implement_via at import time, and some bindings read
    # prompt template files by a path relative to the working directory, so the
    # import has to happen from inside the benchmark directory.
    prev_cwd = Path.cwd()
    sys.path.insert(0, str(bench_dir))
    os.chdir(bench_dir)
    try:
        importlib.import_module(args.module)
    finally:
        os.chdir(prev_cwd)

    from secretagent.core import all_interfaces
    interfaces = list(all_interfaces())
    names = [i.name for i in interfaces]
    if args.interface not in names:
        raise SystemExit(f"interface {args.interface!r} not found. available: {names}")

    allowed = task_interface_names(bench_dir, args.conf)
    if args.condition == "matched":
        if not allowed:
            raise SystemExit(
                f"no LLM-backed ptools found in {bench_dir / args.conf}; "
                "the matched condition needs it to know which interfaces belong to the task")
        missing = [n for n in allowed if n not in names]
        if missing:
            raise SystemExit(f"conf.yaml names interfaces not in the registry: {missing}")
    prompt = render_prompt(interfaces, args.interface, args.condition, allowed)

    round_dir = out_root / args.dataset / "workflows" / "round_1"
    round_dir.mkdir(parents=True, exist_ok=True)
    (round_dir / "__init__.py").write_text("", encoding="utf-8")
    # repr() rather than a triple-quoted literal: interface docstrings pulled from
    # Interface.src contain their own triple quotes, which would close the string early.
    (round_dir / "prompt.py").write_text(
        "TASK_PROMPT = " + repr(prompt) + "\n", encoding="utf-8")
    (round_dir / "graph.py").write_text(
        GRAPH_TEMPLATE.format(dataset=args.dataset,
                              workspace_root=args.workspace_module), encoding="utf-8")

    used = len(allowed) if args.condition == "matched" else 1
    print(f"condition={args.condition} registry={len(interfaces)} used={used} -> {round_dir}")
    print("-" * 70)
    print(prompt)
    print("-" * 70)


if __name__ == "__main__":
    main()
