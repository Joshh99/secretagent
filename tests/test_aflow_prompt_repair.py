"""Checks the one repair we make to prompts the optimizer writes.

The optimizer LLM sometimes emits the two characters ``\\n`` between Python
statements instead of a real line break. ``prompt.py`` then fails to parse,
``graph_utils.load_graph`` raises before the round is ever evaluated, and
because ``max_retries`` is 1 the round is abandoned with the round counter
incremented anyway. A bad prompt spends one of the candidate slots and yields
nothing, so a search finishes well short of the candidate count the paper
reports.

The repair has to be narrow to be defensible: it may only change formatting
outside strings and comments, and only when the original does not parse and
the result does. These tests pin that contract. No network or model call is
made.
"""

import ast
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "benchmarks" / "COMMON" / "aflow-rebuttal" / "upstream"


def _load_repair():
    path = UPSTREAM / "aflow_prompt_source.py"
    spec = importlib.util.spec_from_file_location("aflow_prompt_source", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.repair_escaped_linebreaks


repair = _load_repair()


def test_valid_source_is_returned_untouched():
    source = "A = 'one'\nB = 'two'\n"
    repaired, count = repair(source)
    assert repaired == source
    assert count == 0


def test_escaped_break_between_statements_is_repaired():
    # What the optimizer actually writes: the closing quote, then a literal
    # backslash-n, then the next assignment on the same physical line.
    source = "A = 'one'\\nB = 'two'\n"
    with pytest.raises(SyntaxError):
        ast.parse(source)
    repaired, count = repair(source)
    assert count == 1
    ast.parse(repaired)
    namespace = {}
    exec(repaired, namespace)
    assert namespace["A"] == "one"
    assert namespace["B"] == "two"


def test_escapes_inside_strings_are_left_alone():
    """The prompts are full of \\n inside their text; those must survive."""
    source = "A = 'first line\\nsecond line'\\nB = 'tail'\n"
    repaired, count = repair(source)
    assert count == 1
    namespace = {}
    exec(repaired, namespace)
    # The break between the two statements became real; the one inside the
    # string stayed an escape and still renders as a newline in the prompt.
    assert namespace["A"] == "first line\nsecond line"
    assert namespace["B"] == "tail"


def test_unrelated_syntax_error_is_left_broken():
    """We repair one fault. Anything else is AFlow's own failure and stays."""
    source = "A = 'unterminated\nB = 2\n"
    repaired, count = repair(source)
    assert repaired == source
    assert count == 0


def test_repair_is_refused_when_the_result_still_does_not_parse():
    source = "def f(:\\n    return 1\n"
    repaired, count = repair(source)
    assert repaired == source
    assert count == 0


def test_saved_prompts_from_the_first_restart_are_repaired():
    """Regression against the real run that exposed this.

    Skipped when the run directories are absent, since they are not committed.
    """
    runs = Path("C:/Users/STUDENT/aflow-b/runs")
    prompts = sorted(runs.glob("tables23_restart1__*matched*/**/prompt.py"))
    if not prompts:
        pytest.skip("tables23_restart1 run directories are not present")
    broken = repaired = 0
    for path in prompts:
        source = path.read_text(encoding="utf-8")
        try:
            ast.parse(source)
            continue
        except SyntaxError:
            broken += 1
        fixed, count = repair(source)
        if count:
            ast.parse(fixed)  # every repair we accept must parse
            repaired += 1
    assert broken > 0
    # 66 of 67 when this was written; the remaining one is an unrelated
    # syntax error. Pinning the rate rather than the count keeps this useful
    # if more rounds are added to the directory.
    assert repaired / broken > 0.9
