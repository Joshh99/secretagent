import importlib.util

import pytest

from secretagent.dataset import Dataset, Case
from secretagent.learn.baselines import RoteLearner


def _make_dataset(cases_data):
    """Create a Dataset from a list of (input_args, input_kw, expected_output) tuples."""
    cases = [
        Case(name=f'c{i}', input_args=args, input_kw=kw, expected_output=out)
        for i, (args, kw, out) in enumerate(cases_data)
    ]
    return Dataset(name='test', cases=cases)


def _load_learned(path):
    """Import the generated learned.py module."""
    spec = importlib.util.spec_from_file_location('learned', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- learn() tests ---


def test_learn_creates_file(tmp_path):
    ds = _make_dataset([(['a'], None, 'x')])
    outpath = RoteLearner().load(ds).learn(tmp_path / 'out', 'my_func')
    assert outpath.exists()
    assert outpath.name == 'learned.py'


def test_learn_returns_most_common(tmp_path):
    ds = _make_dataset([
        (['hello'], None, 'world'),
        (['hello'], None, 'world'),
        (['hello'], None, 'earth'),
    ])
    outpath = RoteLearner().load(ds).learn(tmp_path / 'out', 'my_func')
    mod = _load_learned(outpath)
    assert mod.my_func('hello') == 'world'


def test_learn_returns_none_for_unseen(tmp_path):
    ds = _make_dataset([(['a'], None, 'x')])
    outpath = RoteLearner().load(ds).learn(tmp_path / 'out', 'my_func')
    mod = _load_learned(outpath)
    assert mod.my_func('unseen') is None


def test_learn_multiple_inputs(tmp_path):
    ds = _make_dataset([
        (['a'], None, '1'),
        (['b'], None, '2'),
    ])
    outpath = RoteLearner().load(ds).learn(tmp_path / 'out', 'f')
    mod = _load_learned(outpath)
    assert mod.f('a') == '1'
    assert mod.f('b') == '2'


def test_learn_with_kwargs(tmp_path):
    ds = _make_dataset([
        ([], {'x': 1, 'y': 2}, 'ok'),
    ])
    outpath = RoteLearner().load(ds).learn(tmp_path / 'out', 'f')
    mod = _load_learned(outpath)
    assert mod.f(x=1, y=2) == 'ok'
    assert mod.f(x=1, y=99) is None


def test_learn_with_mixed_args_and_kwargs(tmp_path):
    ds = _make_dataset([
        (['pos'], {'key': 'val'}, 'result'),
    ])
    outpath = RoteLearner().load(ds).learn(tmp_path / 'out', 'f')
    mod = _load_learned(outpath)
    assert mod.f('pos', key='val') == 'result'


def test_learn_creates_outdir(tmp_path):
    ds = _make_dataset([(['a'], None, 'x')])
    outdir = tmp_path / 'nested' / 'dir'
    outpath = RoteLearner().load(ds).learn(outdir, 'f')
    assert outdir.exists()
    assert outpath.exists()


def test_learn_function_named_after_interface(tmp_path):
    ds = _make_dataset([(['a'], None, 'x')])
    outpath = RoteLearner().load(ds).learn(tmp_path / 'out', 'classify')
    mod = _load_learned(outpath)
    assert hasattr(mod, 'classify')
    assert mod.classify('a') == 'x'
