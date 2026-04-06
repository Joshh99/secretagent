"""Tests for the pipeline transforms system."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import yaml

from secretagent.orchestrate.improve import (
    register_transform, get_transform, _TRANSFORMS,
    improve_pipeline, ImprovementReport,
)
from secretagent.orchestrate.transforms.base import (
    PipelineTransform, TransformProposal, TransformResult,
)
from secretagent.orchestrate.profiler import PipelineProfile, PtoolProfile
from secretagent.orchestrate.pipeline import Pipeline
from secretagent.orchestrate.catalog import PtoolCatalog


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def simple_profile():
    return PipelineProfile(
        accuracy=0.7,
        total_cost=0.1,
        avg_cost=0.01,
        n_cases=10,
        n_cases_with_rollout=10,
        ptool_profiles={
            'a': PtoolProfile(
                name='a', n_calls=10, cost_fraction=0.6,
                accuracy_when_correct=0.9, accuracy_when_incorrect=0.5,
                lift=0.01,
            ),
            'b': PtoolProfile(
                name='b', n_calls=10, cost_fraction=0.4,
                accuracy_when_correct=0.8, accuracy_when_incorrect=0.3,
            ),
        },
    )


@pytest.fixture
def empty_catalog():
    return PtoolCatalog([])


# ── Registry tests ───────────────────────────────────────────────────

class TestTransformRegistry:
    def test_register_and_get(self):
        # Transforms are registered at import time via transforms/__init__.py
        assert get_transform('prune') is not None
        assert get_transform('downgrade') is not None

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError, match='unknown transform'):
            get_transform('nonexistent_transform_xyz')

    def test_all_six_registered(self):
        expected = {'prune', 'downgrade', 'induce', 'expand', 'repair', 'restructure'}
        assert expected <= set(_TRANSFORMS.keys())


# ── Base class helper tests ──────────────────────────────────────────

class TestGenerateCodeHelper:
    @patch('secretagent.orchestrate.transforms.base.llm')
    @patch('secretagent.orchestrate.transforms.base._ruff_fix')
    @patch('secretagent.orchestrate.transforms.base._extract_code')
    def test_generate_code(self, mock_extract, mock_ruff, mock_llm):
        mock_llm.return_value = ('```python\nreturn 42\n```', {})
        mock_extract.return_value = 'return 42'
        mock_ruff.return_value = 'return 42'

        t = get_transform('induce')
        result = t._generate_code('test prompt', 'def f(x: str) -> int:')
        assert result == 'return 42'
        mock_llm.assert_called_once()


class TestValidateCode:
    def test_valid_code_returns_pipeline(self):
        t = get_transform('prune')
        p = t._validate_code('return 42', 'def f() -> int:', {})
        assert isinstance(p, Pipeline)
        assert p() == 42

    def test_invalid_code_raises(self):
        t = get_transform('prune')
        with pytest.raises(Exception):
            t._validate_code('def !!!invalid', 'def f():', {})


# ── Stub tests ───────────────────────────────────────────────────────

class TestStubsRaiseNotImplemented:
    @pytest.mark.parametrize('name', [
        'prune', 'downgrade', 'induce', 'expand', 'repair', 'restructure',
    ])
    def test_propose_raises(self, name, simple_profile, empty_catalog):
        t = get_transform(name)
        with pytest.raises(NotImplementedError):
            t.propose(simple_profile, empty_catalog)

    @pytest.mark.parametrize('name', [
        'prune', 'downgrade', 'induce', 'expand', 'repair', 'restructure',
    ])
    def test_apply_raises(self, name, empty_catalog):
        t = get_transform(name)
        proposal = TransformProposal(transform_name=name, rationale='test')
        pipeline = Pipeline('return 1', 'def f() -> int:', {})
        with pytest.raises(NotImplementedError):
            t.apply(proposal, pipeline, empty_catalog)


# ── should_apply logic tests ────────────────────────────────────────

class TestShouldApplyLogic:
    def test_prune_triggers_on_low_lift(self):
        t = get_transform('prune')
        profile = PipelineProfile(ptool_profiles={
            'a': PtoolProfile(name='a', lift=0.01),
        })
        assert t.should_apply(profile) is True

    def test_prune_skips_when_no_lift_data(self):
        t = get_transform('prune')
        profile = PipelineProfile(ptool_profiles={
            'a': PtoolProfile(name='a', lift=None),
        })
        assert t.should_apply(profile) is False

    def test_prune_skips_when_lift_is_high(self):
        t = get_transform('prune')
        profile = PipelineProfile(ptool_profiles={
            'a': PtoolProfile(name='a', lift=0.15),
        })
        assert t.should_apply(profile) is False

    def test_downgrade_triggers_on_high_cost(self):
        t = get_transform('downgrade')
        profile = PipelineProfile(ptool_profiles={
            'a': PtoolProfile(name='a', cost_fraction=0.5),
        })
        assert t.should_apply(profile) is True

    def test_downgrade_skips_on_low_cost(self):
        t = get_transform('downgrade')
        profile = PipelineProfile(ptool_profiles={
            'a': PtoolProfile(name='a', cost_fraction=0.2),
        })
        assert t.should_apply(profile) is False

    def test_induce_always_applies(self):
        t = get_transform('induce')
        assert t.should_apply(PipelineProfile()) is True

    def test_expand_triggers_on_expensive_inaccurate(self):
        t = get_transform('expand')
        profile = PipelineProfile(
            accuracy=0.8,
            ptool_profiles={
                'a': PtoolProfile(name='a', cost_fraction=0.5, accuracy_when_correct=0.5),
            },
        )
        assert t.should_apply(profile) is True

    def test_expand_skips_accurate_ptool(self):
        t = get_transform('expand')
        profile = PipelineProfile(
            accuracy=0.8,
            ptool_profiles={
                'a': PtoolProfile(name='a', cost_fraction=0.5, accuracy_when_correct=0.9),
            },
        )
        assert t.should_apply(profile) is False

    def test_repair_triggers_on_errors(self):
        from secretagent.orchestrate.profiler import ErrorPattern
        t = get_transform('repair')
        profile = PipelineProfile(ptool_profiles={
            'a': PtoolProfile(
                name='a',
                error_patterns=[ErrorPattern(pattern='err', frequency=3)],
            ),
        })
        assert t.should_apply(profile) is True

    def test_repair_skips_no_errors(self):
        t = get_transform('repair')
        profile = PipelineProfile(ptool_profiles={
            'a': PtoolProfile(name='a'),
        })
        assert t.should_apply(profile) is False

    def test_restructure_always_applies(self):
        t = get_transform('restructure')
        assert t.should_apply(PipelineProfile()) is True


# ── Improvement loop test ────────────────────────────────────────────

class TestImprovePipeline:
    def test_with_stubs_produces_report(self, tmp_path):
        # Create a result dir
        d = tmp_path / '20260101.120000.test'
        d.mkdir()
        (d / 'config.yaml').write_text(yaml.dump({'evaluate': {'expt_name': 'test'}}))
        records = [
            {
                'correct': True, 'cost': 0.01, 'latency': 1.0,
                'rollout': [
                    {'func': 'a', 'args': ['x'], 'kw': {}, 'output': 'ok',
                     'stats': {'cost': 0.01, 'latency': 1.0, 'input_tokens': 100, 'output_tokens': 50}},
                ],
            },
        ]
        with open(d / 'results.jsonl', 'w') as f:
            for rec in records:
                f.write(json.dumps(rec) + '\n')

        pipeline = Pipeline('return "ok"', 'def f(x: str) -> str:', {})
        catalog = PtoolCatalog([])

        report = improve_pipeline(
            pipeline=pipeline,
            result_dirs=[d],
            catalog=catalog,
        )

        assert isinstance(report, ImprovementReport)
        assert report.before_profile.n_cases == 1
        assert len(report.iterations) == 1  # max_iterations defaults to 1
