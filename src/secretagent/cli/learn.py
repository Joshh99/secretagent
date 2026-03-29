from pathlib import Path
from typing import Optional

import typer

from secretagent.learn.baselines import RoteLearner
from secretagent.learn.examples import extract_examples
from secretagent.learn.traces import extract_ptp_traces

app = typer.Typer()
_EXTRA_ARGS = {"allow_extra_args": True, "allow_interspersed_args": False}

@app.callback()
def main():
    """Learn implementations from recorded interface calls."""

@app.command(context_settings=_EXTRA_ARGS)
def rote(
    ctx: typer.Context,
    interface: str = typer.Option(..., help="Interface name to extract, e.g. 'consistent_sports'"),
    latest: int = typer.Option(1, help='Keep latest k dirs per tag; 0 for all'),
    check: Optional[list[str]] = typer.Option(None, help='Config constraint like key=value'),
    train_dir: str = typer.Option('/tmp/rote_train', help='Directory to store collected data'),
):
    """Learn a rote (lookup-based) implementation from recorded calls."""
    learner = RoteLearner(interface_name=interface, train_dir=train_dir)
    learner.learn([Path(a) for a in ctx.args], latest=latest, check=check)


@app.command(context_settings={"allow_extra_args": True, "allow_interspersed_args": True})
def examples(
    ctx: typer.Context,
    output: str = typer.Option('examples.json', help='Output JSON file path'),
    interface: Optional[list[str]] = typer.Option(None, help='Interface names to extract (repeatable)'),
    only_correct: bool = typer.Option(True, help='Only include examples from correct predictions'),
    max_per_interface: Optional[int] = typer.Option(None, help='Max examples per interface'),
    latest: int = typer.Option(1, help='Keep latest k dirs per tag; 0 for all'),
    check: Optional[list[str]] = typer.Option(None, help='Config constraint like key=value'),
):
    """Extract in-context examples from recorded rollouts.

    Collects successful input/output traces and saves them in the JSON
    format expected by SimulateFactory's example_file parameter.

    Example::

        uv run -m secretagent.cli.learn examples results/* --output examples.json
    """
    extract_examples(
        dirs=[Path(a) for a in ctx.args],
        output_file=output,
        interfaces=interface,
        only_correct=only_correct,
        max_per_interface=max_per_interface,
        latest=latest,
        check=check,
    )


@app.command(context_settings={"allow_extra_args": True, "allow_interspersed_args": True})
def traces(
    ctx: typer.Context,
    output: str = typer.Option('traces.txt', help='Output trace file path'),
    only_correct: bool = typer.Option(True, help='Only include traces from correct predictions'),
    max_traces: int = typer.Option(3, help='Max number of traces to include'),
    max_output_chars: int = typer.Option(200, help='Max chars per step output'),
    latest: int = typer.Option(1, help='Keep latest k dirs per tag; 0 for all'),
    check: Optional[list[str]] = typer.Option(None, help='Config constraint like key=value'),
):
    """Extract PTP (Program Trace Prompting) traces from recorded rollouts.

    Formats execution traces as doctest-style chains with abbreviated
    inputs. Use with method=ptp and trace_file=<output>.

    Example::

        uv run -m secretagent.cli.learn traces results/* --output traces.txt
        # Then use: ptools.answer_question.method=ptp ptools.answer_question.trace_file=traces.txt
    """
    extract_ptp_traces(
        dirs=[Path(a) for a in ctx.args],
        output_file=output,
        only_correct=only_correct,
        max_traces=max_traces,
        max_output_chars=max_output_chars,
        latest=latest,
        check=check,
    )


if __name__ == '__main__':
    app()
