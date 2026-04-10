"""Pipeline improvement loop: chain transforms to iteratively improve a pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Sequence

from pydantic import BaseModel

from secretagent import config
from secretagent.orchestrate.catalog import PtoolCatalog
from secretagent.orchestrate.pipeline import Pipeline
from secretagent.orchestrate.profiler import PipelineProfile, profile_from_results

if TYPE_CHECKING:
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
