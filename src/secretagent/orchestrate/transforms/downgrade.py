"""Downgrade transform: switch expensive ptools to cheaper implementations."""

from __future__ import annotations

from secretagent.orchestrate.catalog import PtoolCatalog
from secretagent.orchestrate.profiler import PipelineProfile
from secretagent.orchestrate.transforms.base import (
    PipelineTransform, TransformProposal, TransformResult,
)
from secretagent.orchestrate.pipeline import Pipeline


class DowngradeTransform(PipelineTransform):
    """Switch high-cost ptools to cheaper model implementations.

    Implementation guide:
    Identify ptools consuming a disproportionate share of cost and
    reconfigure them to use a cheaper LLM model. This is a config-only
    change — no LLM generation needed.
    """

    name = 'downgrade'
    requires_llm = False

    def should_apply(self, profile: PipelineProfile) -> bool:
        return any(
            pp.cost_fraction > 0.4
            for pp in profile.ptool_profiles.values()
        )

    def propose(
        self, profile: PipelineProfile, catalog: PtoolCatalog,
    ) -> TransformProposal:
        raise NotImplementedError('TODO: implement downgrade transform')

    def apply(
        self,
        proposal: TransformProposal,
        pipeline: Pipeline,
        catalog: PtoolCatalog,
    ) -> TransformResult:
        raise NotImplementedError('TODO: implement downgrade transform')
