"""Pipeline improvement loop: chain transforms to iteratively improve a pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel

from secretagent import config
from secretagent.orchestrate.catalog import PtoolCatalog
from secretagent.orchestrate.pipeline import Pipeline
from secretagent.orchestrate.profiler import PipelineProfile, profile_from_results
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


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def improve_pipeline(
    pipeline: Pipeline,
    result_dirs: Sequence[str | Path],
    catalog: PtoolCatalog,
    transforms: list[PipelineTransform] | None = None,
    max_iterations: int | None = None,
) -> ImprovementReport:
    """Run improvement transforms on a pipeline using profiling data.

    Args:
        pipeline: the current pipeline to improve
        result_dirs: directories with results.jsonl for profiling
        catalog: ptool catalog available to the pipeline
        transforms: explicit list of transforms (default: from config or registry)
        max_iterations: how many improvement rounds (default: from config or 1)
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

    profile = before
    for i in range(max_iterations):
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
            except NotImplementedError:
                log.debug('transform %s: apply not implemented', t.name)
                continue

        iterations.append({'proposals': proposals, 'results': results})

    return ImprovementReport(
        before_profile=before,
        iterations=iterations,
    )
