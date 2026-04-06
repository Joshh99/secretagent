"""Prune transform: remove ptools that contribute negligible accuracy lift."""

from __future__ import annotations

from secretagent.orchestrate.catalog import PtoolCatalog
from secretagent.orchestrate.profiler import PipelineProfile
from secretagent.orchestrate.transforms.base import (
    PipelineTransform, TransformProposal, TransformResult,
)
from secretagent.orchestrate.pipeline import Pipeline


class PruneTransform(PipelineTransform):
    """Remove ptools whose accuracy lift is negligible.

    Implementation guide:
    Use the profiler's lift data to identify ptools that add cost but
    don't meaningfully improve accuracy. Generate a new pipeline via
    the LLM that omits the pruned ptools.
    """

    name = 'prune'
    requires_llm = True

    def should_apply(self, profile: PipelineProfile) -> bool:
        return any(
            pp.lift is not None and pp.lift < 0.02
            for pp in profile.ptool_profiles.values()
        )

    def propose(
        self, profile: PipelineProfile, catalog: PtoolCatalog,
    ) -> TransformProposal:
        raise NotImplementedError('TODO: implement prune transform')

    def apply(
        self,
        proposal: TransformProposal,
        pipeline: Pipeline,
        catalog: PtoolCatalog,
    ) -> TransformResult:
        raise NotImplementedError('TODO: implement prune transform')
