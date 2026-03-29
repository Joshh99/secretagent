"""CLI for grid search optimization over config spaces.

Usage::

    # Run a sweep from a YAML space definition
    uv run -m secretagent.cli.optimize sweep \\
        --command "uv run python benchmarks/musr/expt.py run --config-file conf/murder.yaml" \\
        --space-file sweep_space.yaml \\
        dataset.n=10 cachier.enable_caching=false

    # Load and display sweep results
    uv run -m secretagent.cli.optimize summary sweep_results.csv
"""

import typer
import yaml

from secretagent.optimize import SearchSpace, GridSearchRunner

app = typer.Typer()


@app.callback()
def callback():
    """Grid search optimizer for secretagent configurations."""


@app.command(context_settings={
    "allow_extra_args": True, "allow_interspersed_args": False,
})
def sweep(
    ctx: typer.Context,
    command: str = typer.Option(..., help="Base command to run (quoted)"),
    space_file: str = typer.Option(..., help="YAML file defining search space"),
    prefix: str = typer.Option("sweep", help="Experiment name prefix"),
    cwd: str = typer.Option(None, help="Working directory for subprocess"),
    timeout: int = typer.Option(1800, help="Timeout per config in seconds"),
    metric: str = typer.Option("correct", help="Metric column to optimize"),
    output: str = typer.Option("sweep_summary.csv", help="Output summary CSV path"),
):
    """Run grid search over a config space."""
    # Load search space from YAML
    with open(space_file) as f:
        space_dict = yaml.safe_load(f)

    # Support both top-level dict and nested under 'search_space' key
    if 'search_space' in space_dict:
        space_dict = space_dict['search_space']

    space = SearchSpace(space_dict)

    runner = GridSearchRunner(
        command=command,
        space=space,
        base_dotlist=ctx.args,
        expt_prefix=prefix,
        cwd=cwd,
        timeout=timeout,
        metric=metric,
    )

    df = runner.run_all()

    # Save and display
    if output:
        runner.save_summary(output)

    print('\n' + '=' * 60)
    print('SWEEP RESULTS (sorted by accuracy)')
    print('=' * 60)
    display_cols = [c for c in df.columns
                    if c not in ('csv_path', 'config_idx')]
    print(df[display_cols].to_string(index=False))

    # Print best config
    best = df.iloc[0] if not df.empty else None
    if best is not None and best.get('accuracy') is not None:
        print(f'\nBest: {best["expt_name"]} — {best["accuracy"]:.1%}')
        for k in space.keys:
            print(f'  {k} = {best.get(k, "?")}')


@app.command()
def summary(
    csv_path: str = typer.Argument(..., help="Path to sweep_summary.csv"),
    top_n: int = typer.Option(10, help="Show top N results"),
):
    """Display results from a saved sweep summary."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    df = df.sort_values('accuracy', ascending=False).head(top_n)
    print(df.to_string(index=False))


if __name__ == '__main__':
    app()
