"""Baseline learners that extract patterns from recorded interface calls.

Example usage:

    # Learn from a single recording directory
    uv run -m secretagent.learn.baselines rote --interface consistent_sports recordings/20260319.103600.workflow

    # Learn from multiple directories (glob)
    uv run -m secretagent.learn.baselines rote --interface consistent_sports recordings/*

    # Only latest 2 per tag, with config constraint
    uv run -m secretagent.learn.baselines rote --interface consistent_sports recordings/* --latest 2 --check llm.model=claude-haiku-4-5-20251001
"""

import textwrap
from collections import Counter
from pathlib import Path
from typing import Optional

import typer

from secretagent import config, savefile
from secretagent.dataset import Dataset
from secretagent.learn.utils import collect_interface_data


def _make_hashable(obj):
    """Convert a JSON-decoded object to a hashable form."""
    if isinstance(obj, list):
        return tuple(_make_hashable(x) for x in obj)
    if isinstance(obj, dict):
        return tuple(sorted((k, _make_hashable(v)) for k, v in obj.items()))
    return obj


class RoteLearner:
    """Extract input/output statistics from a Dataset.

    Accumulates a dict mapping each unique input to a Counter of outputs.
    """

    def __init__(self):
        self.counts: dict[tuple, Counter] = {}

    def load(self, dataset: Dataset) -> "RoteLearner":
        """Load cases from a Dataset."""
        for case in dataset.cases:
            self._observe(case)
        return self

    def _observe(self, case):
        args_key = _make_hashable(case.input_args or [])
        kw_key = _make_hashable(case.input_kw or {})
        input_key = (args_key, kw_key)
        output_key = _make_hashable(case.expected_output)
        if input_key not in self.counts:
            self.counts[input_key] = Counter()
        self.counts[input_key][output_key] += 1

    @property
    def total_observations(self) -> int:
        return sum(c.total() for c in self.counts.values())

    def learn(self, outdir: str | Path, interface_name: str) -> Path:
        """Write a learned.py file with a function that returns the most common output.

        The generated function accepts *args, **kw and looks up the input
        in a precomputed dict, returning the most common output or None.
        """
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        # Build lookup: input_key -> most_common_output (unhashable original form)
        lookup = {}
        for input_key, counter in self.counts.items():
            best_output, _ = counter.most_common(1)[0]
            lookup[input_key] = best_output
        outpath = outdir / 'learned.py'
        outpath.write_text(textwrap.dedent(f"""\
            \"\"\"Auto-generated rote-learned implementation for {interface_name}.\"\"\"


            def _make_hashable(obj):
                if isinstance(obj, list):
                    return tuple(_make_hashable(x) for x in obj)
                if isinstance(obj, dict):
                    return tuple(sorted((k, _make_hashable(v)) for k, v in obj.items()))
                return obj


            _LOOKUP = {repr(lookup)}


            def {interface_name}(*args, **kw):
                args_key = _make_hashable(list(args))
                kw_key = _make_hashable(kw)
                return _LOOKUP.get((args_key, kw_key))
        """))
        return outpath

    def report(self) -> list[dict]:
        """Return per-input statistics.

        Each dict contains:
            input: the input (args, kw) as a hashable tuple
            count: number of times this input was seen
            p_input: probability of seeing this input
            most_common_output: the most frequent output for this input
            p_most_common: probability of the most common output given this input
        """
        total = self.total_observations
        results = []
        for input_key, counter in sorted(self.counts.items(), key=lambda x: -x[1].total()):
            n = counter.total()
            best_output, best_count = counter.most_common(1)[0]
            results.append({
                'input': input_key,
                'count': n,
                'p_input': n / total if total else 0.0,
                'most_common_output': best_output,
                'p_most_common': best_count / n,
            })
        return results


app = typer.Typer()

_EXTRA_ARGS = {"allow_extra_args": True, "allow_interspersed_args": False}


def _get_dirs(ctx: typer.Context, latest: int = 1, check: Optional[list[str]] = None) -> list[Path]:
    """Resolve recording directories from extra CLI args via savefile.filter_paths.

    Directories without results.jsonl are silently skipped.
    """
    if not ctx.args:
        raise ValueError('No paths provided.')
    dirs = savefile.filter_paths(ctx.args, latest=latest, dotlist=check or [])
    return [d for d in dirs if (Path(d) / 'results.jsonl').exists()]


@app.command(context_settings=_EXTRA_ARGS)
def rote(
    ctx: typer.Context,
    interface: str = typer.Option(..., help="Interface name to extract, e.g. 'consistent_sports'"),
    latest: int = typer.Option(1, help='Keep latest k dirs per tag; 0 for all'),
    check: Optional[list[str]] = typer.Option(None, help='Config constraint like key=value'),
    train_dir: str = typer.Option('/tmp/rote_train', help='Directory to store collected data'),
):
    """Learn input/output statistics from recorded interface calls.

    Recording directories are passed as extra positional arguments.
    They are filtered through savefile.filter_paths().
    """
    config.configure(cfg={'train_dir': train_dir})
    dirs = _get_dirs(ctx, latest=latest, check=check)
    out_dir, dataset = collect_interface_data(dirs, interface, file_under='rote')
    learner = RoteLearner().load(dataset)
    report = learner.report()
    total = learner.total_observations
    typer.echo(f"Interface: {interface}")
    typer.echo(f"Directories: {len(dirs)}")
    typer.echo(f"Data saved to: {out_dir}")
    typer.echo(f"Total observations: {total}")
    typer.echo(f"Unique inputs: {len(report)}")
    typer.echo()
    for entry in report:
        typer.echo(f"  input: {entry['input']}")
        typer.echo(f"    p(input)={entry['p_input']:.3f}  "
                    f"most_common={entry['most_common_output']}  "
                    f"p(most_common)={entry['p_most_common']:.3f}  "
                    f"count={entry['count']}")


if __name__ == '__main__':
    app()
