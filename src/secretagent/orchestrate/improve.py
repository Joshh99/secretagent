"""Pipeline improvement loop: chain transforms to iteratively improve a pipeline."""

from __future__ import annotations

import inspect
import json
import logging
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence

from pydantic import BaseModel

from secretagent import config
from secretagent.orchestrate.catalog import PtoolCatalog
from secretagent.orchestrate.pipeline import (
    Pipeline, _entry_signature_from_interface,
)
from secretagent.orchestrate.profiler import PipelineProfile, profile_from_results

if TYPE_CHECKING:
    from secretagent.core import Interface
    from secretagent.dataset import Dataset
    from secretagent.evaluate import Evaluator
    from secretagent.orchestrate.transforms.base import PipelineTransform

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Transform registry (mirrors _FACTORIES in core.py)
# ---------------------------------------------------------------------------

_TRANSFORMS: dict[str, PipelineTransform] = {}


def register_transform(name: str, transform: PipelineTransform) -> None:
    _TRANSFORMS[name] = transform


def get_transform(name: str) -> PipelineTransform:
    if name not in _TRANSFORMS:
        raise KeyError(f'unknown transform: {name!r} (registered: {list(_TRANSFORMS)})')
    return _TRANSFORMS[name]


# ---------------------------------------------------------------------------
# Improvement report
# ---------------------------------------------------------------------------

class ImprovementReport(BaseModel):
    before_profile: PipelineProfile
    after_profile: PipelineProfile | None = None
    iterations: list[dict] = []
    improved: bool = False
    best_accuracy: float = 0.0


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def improve_pipeline(
    pipeline: Pipeline,
    result_dirs: Sequence[str | Path],
    catalog: PtoolCatalog,
    transforms: list[PipelineTransform] | None = None,
    max_iterations: int | None = None,
    run_eval_fn: Callable[[], Sequence[str | Path]] | None = None,
    target_accuracy: float | None = None,
) -> ImprovementReport:
    """Run improvement transforms on a pipeline using profiling data.

    Args:
        pipeline: the current pipeline to improve
        result_dirs: directories with results.jsonl for profiling
        catalog: ptool catalog available to the pipeline
        transforms: explicit list of transforms (default: from config or registry)
        max_iterations: how many improvement rounds (default: from config or 1)
        run_eval_fn: callback that re-runs the experiment and returns new
            result directories.  If provided, the loop re-profiles after
            each iteration and keeps improvements (rolls back regressions).
        target_accuracy: stop early when this accuracy is reached.
    """
    if transforms is None:
        transform_names = config.get('orchestrate.improve.transforms', [])
        if transform_names:
            transforms = [get_transform(n) for n in transform_names]
        else:
            transforms = list(_TRANSFORMS.values())

    if max_iterations is None:
        max_iterations = config.get('orchestrate.improve.max_iterations', 1)

    before = profile_from_results(result_dirs, pipeline_source=pipeline.source)
    iterations: list[dict] = []

    log.info('improvement loop starting: accuracy=%.1f%%, %d cases',
             before.accuracy * 100, before.n_cases)
    from secretagent.orchestrate.transforms.base import format_profiling_summary
    print(f'\n[improve] === Initial Profile ===')
    print(format_profiling_summary(before))

    profile = before
    best_accuracy = profile.accuracy
    current_result_dirs = list(result_dirs)

    for i in range(max_iterations):
        print(f'\n[improve] === Iteration {i + 1}/{max_iterations} ===')
        proposals = []
        results = []

        for t in transforms:
            if not t.should_apply(profile):
                log.debug('transform %s: skipped (should_apply=False)', t.name)
                continue
            try:
                proposal = t.propose(profile, catalog)
                proposals.append(proposal.model_dump())
            except NotImplementedError:
                log.debug('transform %s: propose not implemented', t.name)
                continue
            try:
                result = t.apply(proposal, pipeline, catalog)
                results.append(result.model_dump())
                if result.success:
                    print(f'[improve] {t.name}: {result.message}')
                    # Apply config overrides (e.g. model downgrades)
                    if result.new_config:
                        dotlist = [f'{k}={v}' for k, v in result.new_config.items()]
                        config.configure(dotlist=dotlist)
                        log.info('applied config overrides: %s', dotlist)
            except NotImplementedError:
                log.debug('transform %s: apply not implemented', t.name)
                continue

        iterations.append({'proposals': proposals, 'results': results})

        # Re-evaluate if we have a callback
        if run_eval_fn and any(r.get('success') for r in results):
            print('[improve] re-evaluating after transforms...')
            try:
                new_dirs = run_eval_fn()
                new_profile = profile_from_results(
                    new_dirs, pipeline_source=pipeline.source,
                )
                print(f'[improve] accuracy: {profile.accuracy:.1%} -> {new_profile.accuracy:.1%}')
                print(format_profiling_summary(new_profile))

                if new_profile.accuracy >= profile.accuracy:
                    current_result_dirs = list(new_dirs)
                    profile = new_profile
                    if new_profile.accuracy > best_accuracy:
                        best_accuracy = new_profile.accuracy
                    print(f'[improve] kept improvements (accuracy={new_profile.accuracy:.1%})')
                else:
                    print(f'[improve] regression detected, keeping previous state')
                    # Note: ptool state was already modified by transforms.
                    # The caller is responsible for rollback if needed.
            except Exception as e:
                log.warning('re-evaluation failed: %s', e)
                print(f'[improve] re-evaluation failed: {e}')

        # Early exit if target reached
        if target_accuracy is not None and best_accuracy >= target_accuracy:
            print(f'[improve] target accuracy {target_accuracy:.1%} reached!')
            break

    after = profile_from_results(
        current_result_dirs, pipeline_source=pipeline.source,
    ) if run_eval_fn else None

    return ImprovementReport(
        before_profile=before,
        after_profile=after,
        iterations=iterations,
        improved=best_accuracy > before.accuracy,
        best_accuracy=best_accuracy,
    )


# ---------------------------------------------------------------------------
# Supervisor-driven improvement
# ---------------------------------------------------------------------------

class IterationRecord(BaseModel):
    iteration: int
    train_accuracy: float
    train_cost: float
    eval_accuracy: float | None = None
    eval_cost: float | None = None
    supervisor_cost: float = 0.0
    reasoning: str = ''
    kept: bool = False
    code_snapshot: str = ''
    config_overrides: list[str] = []


class SupervisorReport(BaseModel):
    iterations: list[IterationRecord] = []
    best_iteration: int = 0
    best_train_accuracy: float = 0.0
    final_eval_accuracy: float | None = None
    total_supervisor_cost: float = 0.0
    best_code: str = ''
    best_config_overrides: list[str] = []


def _format_failure_traces(
    result_dir: Path,
    max_cases: int = 5,
    max_output_chars: int = 150,
) -> str:
    """Read results.jsonl and format failure cases as compact call chains."""
    jsonl_path = Path(result_dir) / 'results.jsonl'
    if not jsonl_path.exists():
        return 'No results.jsonl found.'

    failures = []
    with open(jsonl_path) as f:
        for line in f:
            record = json.loads(line)
            if not record.get('correct'):
                failures.append(record)

    if not failures:
        return 'All cases passed!'

    # Select diverse failures: group by error type, pick one per group
    error_groups: dict[str, list[dict]] = {}
    for rec in failures:
        rollout = rec.get('rollout', [])
        # Key: first exception or "wrong_answer"
        key = 'wrong_answer'
        for step in rollout:
            out = str(step.get('output', ''))
            if out.startswith('**exception'):
                key = out[:60]
                break
        error_groups.setdefault(key, []).append(rec)

    selected: list[dict] = []
    # One per error group (sorted by frequency)
    for _key, recs in sorted(error_groups.items(), key=lambda kv: -len(kv[1])):
        if len(selected) >= max_cases:
            break
        selected.append(recs[0])

    # Fill remaining with highest-cost failures
    if len(selected) < max_cases:
        used = {r.get('case_name') for r in selected}
        by_cost = sorted(failures, key=lambda r: r.get('cost', 0), reverse=True)
        for rec in by_cost:
            if rec.get('case_name') not in used:
                selected.append(rec)
                used.add(rec.get('case_name'))
                if len(selected) >= max_cases:
                    break

    lines = []
    for rec in selected:
        meta_parts = []
        for mk in ('calculator_name', 'category', 'ques_type'):
            if mk in rec:
                meta_parts.append(f'{mk}={rec[mk]}')
        meta_str = f' ({", ".join(meta_parts)})' if meta_parts else ''
        lines.append(f'Case {rec.get("case_name", "?")}:{meta_str}')

        # Show input (truncated)
        inp = rec.get('input_args', rec.get('predicted_output', ''))
        lines.append(f'  Input: {str(inp)[:200]}')

        # Show rollout steps
        for step in rec.get('rollout', []):
            func = step.get('func', '?')
            output = str(step.get('output', ''))[:max_output_chars]
            cost = step.get('stats', {}).get('cost', 0)
            lines.append(f'  -> {func}() = {output} [${cost:.4f}]')

        pred = rec.get('predicted_output', '?')
        exp = rec.get('expected_output', '?')
        lines.append(f'  Predicted: {pred}, Expected: {exp}')
        lines.append('')

    return '\n'.join(lines)


def _format_iteration_history(iterations: list[IterationRecord]) -> str:
    """Condensed history: last 3 full detail, rest one-line."""
    if not iterations:
        return ''
    lines = []
    for rec in iterations[:-3]:
        status = 'KEPT' if rec.kept else 'ROLLED BACK'
        lines.append(
            f'Iter {rec.iteration}: accuracy={rec.train_accuracy:.1%}, '
            f'cost=${rec.train_cost:.4f} — {status}'
        )
    for rec in iterations[-3:]:
        status = 'KEPT' if rec.kept else 'ROLLED BACK'
        lines.append(
            f'Iter {rec.iteration}: accuracy={rec.train_accuracy:.1%}, '
            f'cost=${rec.train_cost:.4f} — {status}'
        )
        if rec.reasoning:
            # Show first 200 chars of reasoning
            lines.append(f'  Reasoning: {rec.reasoning[:200]}')
    return '\n'.join(lines)


def _extract_initial_code(entry_interface: Interface) -> str:
    """Get the function body of the entry interface's implementation."""
    impl = entry_interface.implementation
    if impl is None:
        return '    pass'
    fn = impl.implementing_fn
    # If it's an OrchestrateFactory with a pipeline
    if hasattr(fn, 'pipeline') and fn.pipeline is not None:
        return fn.pipeline.code
    # If it's a DirectFactory, get the actual function
    actual_fn = getattr(fn, 'direct_fn', None) or getattr(fn, '_fn', None)
    if actual_fn is None:
        actual_fn = fn
    # Try to get source
    try:
        src = inspect.getsource(actual_fn)
        # Strip decorators and def line to get just the body
        src_lines = src.split('\n')
        body_start = 0
        for i, line in enumerate(src_lines):
            stripped = line.strip()
            if stripped.startswith('def '):
                body_start = i + 1
                break
        body = '\n'.join(src_lines[body_start:])
        return textwrap.dedent(body)
    except (TypeError, OSError):
        return '    pass'


def _build_namespace(
    tool_interfaces: list,
    ptools_module: Any = None,
) -> dict[str, Any]:
    """Build exec namespace for pipeline compilation."""
    import builtins
    import json as json_mod
    import re as re_mod
    from secretagent.core import Interface

    namespace: dict[str, Any] = {
        '__builtins__': builtins,
        'json': json_mod,
        're': re_mod,
    }
    for iface in tool_interfaces:
        namespace[iface.name] = iface

    # Add non-Interface objects from ptools module (functions, modules, constants)
    if ptools_module is not None:
        import types
        for name in dir(ptools_module):
            if name.startswith('__'):
                continue
            obj = getattr(ptools_module, name)
            if isinstance(obj, Interface):
                continue
            # Include callables, modules, and data objects
            if (callable(obj) or isinstance(obj, types.ModuleType)
                    or isinstance(obj, (str, list, dict, set, tuple, int, float))):
                namespace[name] = obj

    return namespace


def _describe_namespace_utilities(
    namespace: dict[str, Any],
    tool_interfaces: list,
) -> str:
    """Build a description of non-tool items in the namespace."""
    tool_names = {iface.name for iface in tool_interfaces}
    skip = {'__builtins__', 'json', 're'}
    lines = []
    for name, obj in sorted(namespace.items()):
        if name in tool_names or name in skip:
            continue
        if callable(obj):
            # Try to get a one-line description
            sig = ''
            try:
                sig = f'({", ".join(inspect.signature(obj).parameters.keys())})'
            except (ValueError, TypeError):
                pass
            doc = (getattr(obj, '__doc__', '') or '').split('\n')[0][:80]
            lines.append(f'- `{name}{sig}`: {doc}' if doc else f'- `{name}{sig}`')
        elif isinstance(obj, str) and len(obj) > 20:
            lines.append(f'- `{name}` (str, {len(obj)} chars)')
        elif isinstance(obj, (list, dict)):
            lines.append(f'- `{name}` ({type(obj).__name__}, {len(obj)} items)')
    return '\n'.join(lines) if lines else ''


def _compile_pipeline(code: str, entry_sig: str, namespace: dict):
    """Compile pipeline code without Pipeline's indentation normalization.

    Pipeline._compile has a first-line normalization that breaks try/except
    blocks. This function does a clean compile: dedent, re-indent under
    the entry signature, exec.
    """
    func_name = entry_sig.split('(')[0].replace('def ', '').strip()
    indented_body = textwrap.indent(textwrap.dedent(code), '    ')
    func_src = f'{entry_sig}\n{indented_body}'
    exec_ns = dict(namespace)
    exec(func_src, exec_ns)
    fn = exec_ns[func_name]

    # Wrap in a Pipeline-like callable for compatibility
    p = Pipeline.__new__(Pipeline)
    p.code = code
    p.entry_signature = entry_sig
    p._fn = fn
    return p


def _compile_with_retry(
    code: str, entry_sig: str, namespace: dict,
    recompose_fn, recompose_kwargs: dict,
    max_retries: int = 2,
) -> Pipeline | None:
    """Try to compile pipeline code, retrying with error feedback."""
    for attempt in range(1 + max_retries):
        try:
            return _compile_pipeline(code, entry_sig, namespace)
        except Exception as e:
            if attempt >= max_retries:
                print(f'[supervisor] compile failed after {attempt + 1} attempts: {e}')
                return None
            print(f'[supervisor] compile error (retry {attempt + 1}): {e}')
            # Append error context to custom instructions for retry
            error_ctx = (
                f'\n\nYour previous code had a compile error:\n```\n{e}\n```\n'
                f'Previous code:\n```python\n{code}\n```\n'
                f'Fix the syntax error. Return corrected code.'
            )
            retry_kwargs = dict(recompose_kwargs)
            retry_kwargs['custom_instructions'] = (
                retry_kwargs.get('custom_instructions', '') + error_ctx
            )
            code, _reasoning, _cfg, _stats = recompose_fn(**retry_kwargs)
    return None


def improve_with_supervisor(
    entry_interface: Interface,
    tool_interfaces: list[Interface],
    catalog: PtoolCatalog,
    evaluator: Evaluator,
    train_dataset: Dataset,
    eval_dataset: Dataset | None = None,
    supervisor_model: str = 'gemini/gemini-3.1-pro-preview',
    max_iterations: int = 10,
    target_accuracy: float | None = None,
    custom_instructions: str = '',
    model_choices: str = '',
    output_dir: Path | None = None,
    ptools_module: Any = None,
) -> SupervisorReport:
    """Iteratively improve a pipeline using a supervisor LLM.

    The supervisor sees profiling data and failure traces, and outputs
    improved pipeline code + optional config overrides. Hill climbing:
    keep improvements, rollback regressions.

    Args:
        entry_interface: the workflow Interface to improve
        tool_interfaces: ptools available to the pipeline
        catalog: ptool catalog for the supervisor prompt
        evaluator: evaluator to measure performance
        train_dataset: training dataset for evaluation
        eval_dataset: optional held-out dataset for periodic eval
        supervisor_model: LLM model for the supervisor
        max_iterations: max improvement iterations
        target_accuracy: stop when reached
        custom_instructions: extra text for the supervisor prompt
        model_choices: formatted model table for the supervisor
        output_dir: directory to save iteration artifacts
        ptools_module: the ptools module for namespace construction
    """
    from secretagent.orchestrate.composer import recompose
    from secretagent.orchestrate.transforms.base import format_profiling_summary

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / 'iterations').mkdir(exist_ok=True)

    entry_sig = _entry_signature_from_interface(entry_interface)
    namespace = _build_namespace(tool_interfaces, ptools_module)
    current_code = _extract_initial_code(entry_interface)

    # Build a description of utility functions available in the namespace
    # so the supervisor knows it can use them
    utility_desc = _describe_namespace_utilities(namespace, tool_interfaces)
    if utility_desc:
        custom_instructions = (
            f'{custom_instructions}\n\n'
            f'## Utility functions available in scope\n'
            f'These helper functions and variables are also available '
            f'(in addition to the tools listed above):\n{utility_desc}'
        ).strip()

    iterations: list[IterationRecord] = []
    best_code = current_code
    best_config_overrides: list[str] = []
    total_supervisor_cost = 0.0

    # Save original state for rollback
    original_impl = entry_interface.implementation

    # --- Initial evaluation ---
    print('\n[supervisor] === Initial Evaluation ===')
    with config.configuration(evaluate=dict(
        expt_name='rc_iter0', record_details=True,
    )):
        csv_path = evaluator.evaluate(train_dataset, entry_interface)
    result_dir = csv_path.parent
    profile = profile_from_results([result_dir])
    best_accuracy = profile.accuracy
    best_result_dir = result_dir

    rec0 = IterationRecord(
        iteration=0, train_accuracy=profile.accuracy,
        train_cost=profile.avg_cost, kept=True, code_snapshot=current_code,
    )
    iterations.append(rec0)

    print(f'[supervisor] baseline: accuracy={profile.accuracy:.1%}, '
          f'avg_cost=${profile.avg_cost:.4f}')
    prof_summary = format_profiling_summary(profile)
    print(prof_summary)

    if output_dir:
        iter_dir = output_dir / 'iterations' / 'iter_000_baseline'
        iter_dir.mkdir(exist_ok=True)
        (iter_dir / 'code.py').write_text(
            f'{entry_sig}\n{textwrap.indent(textwrap.dedent(current_code), "    ")}'
        )

    # --- Improvement loop ---
    no_improve_count = 0

    for i in range(1, max_iterations + 1):
        print(f'\n[supervisor] === Iteration {i}/{max_iterations} ===')

        # 1. Build context
        prof_summary = format_profiling_summary(profile)
        failure_traces = _format_failure_traces(best_result_dir)
        history_text = _format_iteration_history(iterations)

        # 2. Call supervisor
        print('[supervisor] calling supervisor LLM...')
        new_code, reasoning, cfg_overrides, sup_stats = recompose(
            current_code=current_code,
            catalog=catalog,
            entry_signature=entry_sig,
            profiling_summary=prof_summary,
            failure_traces=failure_traces,
            iteration_history=history_text,
            custom_instructions=custom_instructions,
            model_choices=model_choices,
            model=supervisor_model,
        )
        sup_cost = sup_stats.get('cost', 0.0)
        total_supervisor_cost += sup_cost
        print(f'[supervisor] supervisor cost: ${sup_cost:.4f}')
        if reasoning:
            print(f'[supervisor] reasoning: {reasoning[:200]}')

        # Save iteration artifacts
        if output_dir:
            iter_dir = output_dir / 'iterations' / f'iter_{i:03d}'
            iter_dir.mkdir(exist_ok=True)
            (iter_dir / 'code_before.py').write_text(
                f'{entry_sig}\n{textwrap.indent(textwrap.dedent(current_code), "    ")}'
            )
            (iter_dir / 'code_after.py').write_text(
                f'{entry_sig}\n{textwrap.indent(textwrap.dedent(new_code), "    ")}'
            )
            (iter_dir / 'reasoning.txt').write_text(reasoning)
            if cfg_overrides:
                (iter_dir / 'config_overrides.txt').write_text(
                    '\n'.join(cfg_overrides)
                )

        # 3. Check if code actually changed
        if new_code.strip() == current_code.strip() and not cfg_overrides:
            print('[supervisor] no changes proposed, skipping')
            iterations.append(IterationRecord(
                iteration=i, train_accuracy=profile.accuracy,
                train_cost=profile.avg_cost, supervisor_cost=sup_cost,
                reasoning=reasoning, kept=False, code_snapshot=new_code,
            ))
            no_improve_count += 1
            if no_improve_count >= 3:
                print('[supervisor] 3 consecutive no-change iterations, stopping')
                break
            continue

        # 4. Compile new pipeline (with retry on syntax errors)
        recompose_kwargs = dict(
            current_code=current_code, catalog=catalog,
            entry_signature=entry_sig, profiling_summary=prof_summary,
            failure_traces=failure_traces, iteration_history=history_text,
            custom_instructions=custom_instructions,
            model_choices=model_choices, model=supervisor_model,
        )
        pipeline = _compile_with_retry(
            new_code, entry_sig, namespace,
            recompose, recompose_kwargs,
        )
        if pipeline is None:
            iterations.append(IterationRecord(
                iteration=i, train_accuracy=profile.accuracy,
                train_cost=profile.avg_cost, supervisor_cost=sup_cost,
                reasoning=reasoning, kept=False, code_snapshot=new_code,
            ))
            continue

        # 5. Apply config overrides
        saved_config = dict(config.GLOBAL_CONFIG) if cfg_overrides else None
        if cfg_overrides:
            print(f'[supervisor] applying config: {cfg_overrides}')
            try:
                config.configure(dotlist=cfg_overrides)
            except Exception as e:
                print(f'[supervisor] config override failed: {e}')

        # 6. Rebind entry interface to new pipeline
        old_impl = entry_interface.implementation
        entry_interface.implement_via('direct', fn=pipeline)

        # 7. Re-evaluate
        print('[supervisor] re-evaluating on train set...')
        with config.configuration(evaluate=dict(
            expt_name=f'rc_iter{i}', record_details=True,
        )):
            try:
                csv_path = evaluator.evaluate(train_dataset, entry_interface)
            except Exception as e:
                print(f'[supervisor] evaluation failed: {e}')
                entry_interface.implementation = old_impl
                if saved_config is not None:
                    config.GLOBAL_CONFIG = saved_config
                iterations.append(IterationRecord(
                    iteration=i, train_accuracy=profile.accuracy,
                    train_cost=profile.avg_cost, supervisor_cost=sup_cost,
                    reasoning=reasoning, kept=False, code_snapshot=new_code,
                ))
                continue

        new_result_dir = csv_path.parent
        new_profile = profile_from_results([new_result_dir])

        print(f'[supervisor] accuracy: {profile.accuracy:.1%} -> {new_profile.accuracy:.1%}')
        print(f'[supervisor] cost: ${profile.avg_cost:.4f} -> ${new_profile.avg_cost:.4f}')

        # 8. Keep or rollback
        kept = new_profile.accuracy >= best_accuracy
        if kept:
            no_improve_count = 0
            best_accuracy = new_profile.accuracy
            best_code = new_code
            best_result_dir = new_result_dir
            current_code = new_code
            profile = new_profile
            if cfg_overrides:
                best_config_overrides = cfg_overrides
            print(f'[supervisor] KEPT (best accuracy: {best_accuracy:.1%})')
        else:
            no_improve_count += 1
            entry_interface.implementation = old_impl
            if saved_config is not None:
                config.GLOBAL_CONFIG = saved_config
            print(f'[supervisor] ROLLED BACK (accuracy dropped)')

        iterations.append(IterationRecord(
            iteration=i, train_accuracy=new_profile.accuracy,
            train_cost=new_profile.avg_cost, supervisor_cost=sup_cost,
            reasoning=reasoning, kept=kept, code_snapshot=new_code,
            config_overrides=cfg_overrides,
        ))

        # 9. Check stopping criteria
        if target_accuracy is not None and best_accuracy >= target_accuracy:
            print(f'[supervisor] target accuracy {target_accuracy:.1%} reached!')
            break
        if no_improve_count >= 3:
            print('[supervisor] 3 consecutive non-improvements, stopping')
            break

    # --- Final report ---
    best_iter = max(
        (r for r in iterations if r.kept),
        key=lambda r: r.train_accuracy,
        default=iterations[0],
    )

    report = SupervisorReport(
        iterations=iterations,
        best_iteration=best_iter.iteration,
        best_train_accuracy=best_accuracy,
        total_supervisor_cost=total_supervisor_cost,
        best_code=best_code,
        best_config_overrides=best_config_overrides,
    )

    # Save report
    if output_dir:
        (output_dir / 'best_code.py').write_text(
            f'{entry_sig}\n{textwrap.indent(textwrap.dedent(best_code), "    ")}'
        )
        (output_dir / 'report.json').write_text(
            report.model_dump_json(indent=2)
        )

    print(f'\n[supervisor] === Summary ===')
    print(f'Iterations: {len(iterations) - 1}')
    print(f'Best accuracy: {best_accuracy:.1%} (iteration {best_iter.iteration})')
    print(f'Total supervisor cost: ${total_supervisor_cost:.4f}')

    return report
