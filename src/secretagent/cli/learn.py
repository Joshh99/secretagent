from pathlib import Path
from typing import Optional

import typer

from secretagent.learn.baselines import RoteLearner
from secretagent.learn.codedistill import CodeDistillLearner, EndToEndDistillLearner, distill_all
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
    learned_dir: str = typer.Option('/tmp/rote_train', help='Directory to store collected data'),
):
    """Learn a rote (lookup-based) implementation from recorded calls."""
    learner = RoteLearner(interface_name=interface, train_dir=learned_dir)
    learner.learn([Path(a) for a in ctx.args], latest=latest, check=check)


@app.command(context_settings=_EXTRA_ARGS)
def codedistill(
    ctx: typer.Context,
    interface: str = typer.Option(..., help="Interface name to distill, e.g. 'sport_for'"),
    latest: int = typer.Option(1, help='Keep latest k dirs per tag; 0 for all'),
    check: Optional[list[str]] = typer.Option(None, help='Config constraint like key=value'),
    learned_dir: str = typer.Option('/tmp/codedistill_train', help='Directory to store learned code'),
    model: str = typer.Option('claude-opus-4-6', help='LLM model for code generation'),
    n_candidates: int = typer.Option(3, help='Number of candidate versions per round'),
    max_rounds: int = typer.Option(3, help='Maximum refinement rounds'),
    only_correct: bool = typer.Option(True, help='Only use rollouts that produced correct final answers'),
):
    """Learn a code-distilled implementation from recorded calls.

    Prompts an LLM to generate Python code that implements the interface,
    using multi-round refinement and ensemble selection.

    Example::

        uv run -m secretagent.cli.learn codedistill --interface sport_for recordings/*
    """
    learner = CodeDistillLearner(
        interface_name=interface,
        train_dir=learned_dir,
        model=model,
        n_candidates=n_candidates,
        max_rounds=max_rounds,
        only_correct=only_correct,
    )
    learner.learn([Path(a) for a in ctx.args], latest=latest, check=check)


@app.command(context_settings=_EXTRA_ARGS)
def e2e_codedistill(
    ctx: typer.Context,
    interface: str = typer.Option(..., help="Top-level interface name, e.g. 'calendar_scheduling'"),
    dataset_file: str = typer.Option(..., help="Path to dataset JSON file, e.g. 'data/train.json'"),
    output_field: Optional[str] = typer.Option(None, help="Field to extract from expected_output dict, e.g. 'golden_plan'"),
    learned_dir: str = typer.Option('/tmp/e2e_codedistill', help='Directory to store learned code'),
    model: str = typer.Option('claude-opus-4-6', help='LLM model for code generation'),
    n_candidates: int = typer.Option(3, help='Number of candidate versions per round'),
    max_rounds: int = typer.Option(3, help='Maximum refinement rounds'),
):
    """Learn an end-to-end implementation directly from a dataset.

    Instead of learning intermediate interfaces from recorded rollouts,
    this generates a complete solution function from (input, output) pairs
    in a dataset JSON. The LLM is prompted to structure the code as
    parse -> solve -> format.

    Example::

        uv run -m secretagent.cli.learn e2e-codedistill \\
          --interface calendar_scheduling \\
          --dataset-file data/calendar_train.json \\
          --output-field golden_plan \\
          --learned-dir learned
    """
    learner = EndToEndDistillLearner(
        interface_name=interface,
        train_dir=learned_dir,
        dataset_file=dataset_file,
        output_field=output_field,
        model=model,
        n_candidates=n_candidates,
        max_rounds=max_rounds,
    )
    learner.learn_from_dataset()


@app.command(context_settings=_EXTRA_ARGS)
def codedistill_all(
    ctx: typer.Context,
    latest: int = typer.Option(1, help='Keep latest k dirs per tag; 0 for all'),
    check: Optional[list[str]] = typer.Option(None, help='Config constraint like key=value'),
    learned_dir: str = typer.Option('/tmp/codedistill_train', help='Directory to store learned code'),
    model: str = typer.Option('claude-opus-4-6', help='LLM model for code generation'),
    n_candidates: int = typer.Option(3, help='Number of candidate versions per round'),
    max_rounds: int = typer.Option(3, help='Maximum refinement rounds'),
    max_wrong_rate: float = typer.Option(0.05, help='Max fraction of wrong (non-None) answers to enable'),
):
    """Auto-distill all interfaces found in recordings.

    Discovers all interfaces called in recorded rollouts, runs codedistill
    on each. Enables interfaces where the generated code rarely returns
    wrong answers (wrong_rate <= max_wrong_rate). Abstentions (returning
    None) are safe because they backoff to the LLM.

    Example::

        uv run -m secretagent.cli.learn codedistill-all --learned-dir learned recordings/*
    """
    distill_all(
        dirs=[Path(a) for a in ctx.args],
        train_dir=learned_dir,
        max_wrong_rate=max_wrong_rate,
        model=model,
        n_candidates=n_candidates,
        max_rounds=max_rounds,
        latest=latest,
        check=check,
    )


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
