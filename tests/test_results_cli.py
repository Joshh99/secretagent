import json
import pytest
import pandas as pd
from pathlib import Path

from omegaconf import OmegaConf
from typer.testing import CliRunner

from secretagent import config
from secretagent.cli.results import app


runner = CliRunner()


@pytest.fixture(autouse=True)
def clean_config():
    saved = config.GLOBAL_CONFIG.copy()
    yield
    config.GLOBAL_CONFIG = saved


def _make_expt(base, dirname, expt_name, cfg_dict, rows):
    """Create a fake experiment directory with config.yaml and results.csv."""
    d = base / dirname
    d.mkdir(parents=True, exist_ok=True)
    # write config
    full_cfg = {**cfg_dict, 'evaluate': {'expt_name': expt_name, 'result_dir': str(base)}}
    with open(d / 'config.yaml', 'w') as f:
        f.write(OmegaConf.to_yaml(OmegaConf.create(full_cfg)))
    # write results.csv
    df = pd.DataFrame(rows)
    df['expt_name'] = expt_name
    df.to_csv(d / 'results.csv', index=False)
    return d


@pytest.fixture
def two_expts(tmp_path):
    """Create two experiment directories with different configs and results."""
    d1 = _make_expt(tmp_path, '20260101.120000.baseline', 'baseline',
               {'llm': {'model': 'model-a'}},
               [{'correct': 1, 'latency': 1.0, 'cost': 0.01},
                {'correct': 0, 'latency': 2.0, 'cost': 0.02},
                {'correct': 1, 'latency': 1.5, 'cost': 0.015},
                {'correct': 1, 'latency': 1.2, 'cost': 0.012}])
    d2 = _make_expt(tmp_path, '20260102.120000.improved', 'improved',
               {'llm': {'model': 'model-b'}},
               [{'correct': 1, 'latency': 0.8, 'cost': 0.008},
                {'correct': 1, 'latency': 1.5, 'cost': 0.015},
                {'correct': 1, 'latency': 1.0, 'cost': 0.01},
                {'correct': 0, 'latency': 1.1, 'cost': 0.011}])
    return d1, d2


def _dirs_as_args(dirs):
    """Convert directory paths to CLI args."""
    return [str(d) for d in dirs]


# --- list tests ---

def test_list_shows_all(two_expts):
    result = runner.invoke(app, ['list', '--latest', '0'] + _dirs_as_args(two_expts))
    assert result.exit_code == 0
    assert 'baseline' in result.output
    assert 'improved' in result.output
    assert '4' in result.output  # each has 4 rows


def test_list_filter_by_check(two_expts):
    result = runner.invoke(app, ['list', '--latest', '0',
                                  '--check', 'evaluate.expt_name=baseline']
                           + _dirs_as_args(two_expts))
    assert result.exit_code == 0
    assert 'baseline' in result.output
    assert 'improved' not in result.output


def test_list_latest_default(two_expts):
    result = runner.invoke(app, ['list'] + _dirs_as_args(two_expts))
    assert result.exit_code == 0
    # latest=1 means one per tag; these have different tags so both appear
    assert 'baseline' in result.output
    assert 'improved' in result.output


def test_list_no_results(tmp_path):
    # A nonexistent path raises ValueError (no config.yaml)
    result = runner.invoke(app, ['list', str(tmp_path / 'nonexistent')])
    assert result.exit_code != 0


def test_list_no_args():
    result = runner.invoke(app, ['list'])
    assert result.exit_code == 1


# --- average tests ---

def test_average_shows_metrics(two_expts):
    result = runner.invoke(app, ['average', '--latest', '0',
                                  '--metric', 'correct', '--metric', 'latency']
                           + _dirs_as_args(two_expts))
    assert result.exit_code == 0
    assert 'mean' in result.output
    assert 'sem' in result.output


def test_average_single_expt(two_expts):
    result = runner.invoke(app, ['average', '--metric', 'correct',
                                  '--check', 'evaluate.expt_name=baseline']
                           + _dirs_as_args(two_expts))
    assert result.exit_code == 0
    # only one experiment should appear (grouped by path)
    assert 'mean' in result.output


def test_average_shows_cost(two_expts):
    result = runner.invoke(app, ['average', '--latest', '0', '--metric', 'cost']
                           + _dirs_as_args(two_expts))
    assert result.exit_code == 0
    assert 'cost' in result.output


# --- pair tests ---

def test_pair_runs(two_expts):
    result = runner.invoke(app, ['pair', '--latest', '0',
                                  '--metric', 'correct', '--metric', 'latency']
                           + _dirs_as_args(two_expts))
    assert result.exit_code == 0
    assert 'latency_t' in result.output


def test_pair_needs_two_expts(two_expts):
    result = runner.invoke(app, ['pair', '--metric', 'correct',
                                  '--check', 'evaluate.expt_name=baseline']
                           + _dirs_as_args(two_expts))
    assert result.exit_code != 0


def test_pair_custom_metric(two_expts):
    result = runner.invoke(app, ['pair', '--latest', '0', '--metric', 'cost']
                           + _dirs_as_args(two_expts))
    assert result.exit_code == 0
    assert 'cost' in result.output


# --- compare tests ---

def test_compare_shows_diffs(two_expts):
    result = runner.invoke(app, ['compare', '--latest', '0'] + _dirs_as_args(two_expts))
    assert result.exit_code == 0
    assert 'llm.model' in result.output
    assert 'model-a' in result.output
    assert 'model-b' in result.output


def test_compare_needs_two(two_expts):
    result = runner.invoke(app, ['compare', '--check', 'evaluate.expt_name=baseline']
                           + _dirs_as_args(two_expts))
    assert result.exit_code == 1
    assert 'Need at least 2' in result.output


# --- config file option ---

def test_config_file_option(two_expts, tmp_path):
    cfg_file = tmp_path / 'test_cfg.yaml'
    cfg_file.write_text(OmegaConf.to_yaml(OmegaConf.create(
        {'evaluate': {'result_dir': str(tmp_path)}})))
    # reset config so result_dir is not set
    config.GLOBAL_CONFIG = OmegaConf.create()
    result = runner.invoke(app, ['--config-file', str(cfg_file), 'list', '--latest', '0']
                           + _dirs_as_args(two_expts))
    assert result.exit_code == 0
    assert 'baseline' in result.output
