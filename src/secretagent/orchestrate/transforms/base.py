"""Base class for pipeline transforms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from secretagent.orchestrate.catalog import PtoolCatalog
from secretagent.orchestrate.pipeline import Pipeline
from secretagent.orchestrate.profiler import PipelineProfile


class TransformProposal(BaseModel):
    transform_name: str
    rationale: str
    changes: list[dict[str, Any]] = []


class TransformResult(BaseModel):
    success: bool
    new_pipeline_code: str | None = None
    new_config: dict[str, Any] | None = None
    new_ptools: list[dict[str, Any]] | None = None  # list of {name, src, doc}
    message: str = ''


class PipelineTransform(ABC):
    """Abstract base class for pipeline improvement transforms.

    Subclasses implement should_apply/propose/apply to suggest and execute
    improvements to a generated pipeline based on profiling data.
    """

    name: str = ''
    requires_llm: bool = False

    @abstractmethod
    def should_apply(self, profile: PipelineProfile) -> bool:
        """Return True if this transform is applicable given the profile."""
        ...

    @abstractmethod
    def propose(
        self, profile: PipelineProfile, catalog: PtoolCatalog,
    ) -> TransformProposal:
        """Analyze the profile and propose changes."""
        ...

    @abstractmethod
    def apply(
        self,
        proposal: TransformProposal,
        pipeline: Pipeline,
        catalog: PtoolCatalog,
    ) -> TransformResult:
        """Apply the proposed changes to produce a new pipeline."""
        ...

    def _generate_code(
        self, prompt: str, entry_signature: str, model: str | None = None,
    ) -> str:
        """Call LLM, extract code block, ruff-fix it."""
        from secretagent import config
        from secretagent.llm_util import llm
        from secretagent.orchestrate.composer import _extract_code, _ruff_fix

        model = model or config.get('llm.model', 'claude-haiku-4-5-20251001')
        text, _stats = llm(prompt, model)
        code = _extract_code(text)
        return _ruff_fix(code, entry_signature)

    def _validate_code(
        self, code: str, entry_signature: str, namespace: dict[str, Any],
    ) -> Pipeline:
        """Compile code into a Pipeline; raises on failure."""
        return Pipeline(code, entry_signature, namespace)
