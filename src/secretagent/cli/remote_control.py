"""Remote Control: supervisor-driven pipeline hill climbing.

Usage (from benchmark directory, e.g. benchmarks/medcalc/):

    uv run python -m secretagent.cli.remote_control run \
        --config-file conf/workflow.yaml \
        --n-train 110 --n-eval 110 \
        --max-iterations 10

    # With custom instructions and model switching:
    uv run python -m secretagent.cli.remote_control run \
        --config-file conf/workflow.yaml \
        --custom-instructions "Focus on scoring system calculators" \
        --model-change models.json \
        --supervisor-model gemini/gemini-3.1-pro-preview
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

os.environ['PYTHONUNBUFFERED'] = '1'

import typer

_EXTRA_ARGS = {"allow_extra_args": True, "allow_interspersed_args": False}
app = typer.Typer(pretty_exceptions_enable=False)


def _load_custom_instructions(value: str) -> str:
    """Load custom instructions from text or @filepath."""
    if not value:
        return ''
    if value.startswith('@'):
        path = Path(value[1:])
        if path.exists():
            return path.read_text()
        print(f'Warning: instruction file {path} not found, using as text')
    return value


def _load_model_choices(path_str: str) -> str:
    """Load model choices from JSON and format as table."""
    if not path_str:
        return ''
    path = Path(path_str)
    if not path.exists():
        print(f'Warning: model file {path} not found')
        return ''
    models = json.loads(path.read_text())
    lines = ['| Model | Input $/1M | Output $/1M |',
             '|-------|-----------|------------|']
    for m in models:
        name = m.get('name', m.get('model', '?'))
        cin = m.get('input_cost', m.get('cost_in', '?'))
        cout = m.get('output_cost', m.get('cost_out', '?'))
        lines.append(f'| {name} | ${cin} | ${cout} |')
    return '\n'.join(lines)


def _generate_plots(report, output_dir: Path):
    """Generate accuracy/cost plots from iteration data."""
    try:
        # /// script
        # requires-python = ">=3.11"
        # dependencies = ["matplotlib"]
        # ///
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('[plots] matplotlib not available, skipping plots')
        return

    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)

    iters = report.iterations
    xs = [r.iteration for r in iters]
    train_acc = [r.train_accuracy for r in iters]
    train_cost = [r.train_cost for r in iters]
    kept = [r.kept for r in iters]

    # --- Accuracy over iterations ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(xs, train_acc, 'b-o', label='Train accuracy', markersize=4)

    # Mark kept vs rolled back
    for x, acc, k in zip(xs, train_acc, kept):
        if k:
            ax.plot(x, acc, 'go', markersize=8, zorder=5)
        elif x > 0:
            ax.plot(x, acc, 'rx', markersize=8, zorder=5)

    # Best accuracy line
    best_acc = max(train_acc)
    ax.axhline(y=best_acc, color='green', linestyle='--', alpha=0.5,
               label=f'Best: {best_acc:.1%}')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Accuracy')
    ax.set_title('Remote Control: Accuracy over Iterations')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / 'accuracy_over_iterations.png', dpi=150)
    plt.close(fig)
    print(f'[plots] saved {plots_dir / "accuracy_over_iterations.png"}')

    # --- Cost over iterations ---
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(xs, train_cost, 'r-o', label='Avg cost/case', markersize=4)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Avg cost per case ($)')
    ax1.set_title('Remote Control: Cost over Iterations')

    # Cumulative supervisor cost
    sup_costs = [r.supervisor_cost for r in iters]
    cum_sup = []
    total = 0.0
    for sc in sup_costs:
        total += sc
        cum_sup.append(total)
    ax2 = ax1.twinx()
    ax2.plot(xs, cum_sup, 'b--', label='Cumulative supervisor cost', alpha=0.7)
    ax2.set_ylabel('Cumulative supervisor cost ($)')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / 'cost_over_iterations.png', dpi=150)
    plt.close(fig)
    print(f'[plots] saved {plots_dir / "cost_over_iterations.png"}')

    # --- Accuracy vs Cost scatter ---
    fig, ax = plt.subplots(figsize=(8, 6))
    for r in iters:
        color = 'green' if r.kept else 'red'
        marker = 'o' if r.kept else 'x'
        ax.plot(r.train_cost, r.train_accuracy, marker=marker, color=color,
                markersize=8)
        ax.annotate(str(r.iteration), (r.train_cost, r.train_accuracy),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel('Avg cost per case ($)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Remote Control: Accuracy vs Cost')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / 'accuracy_vs_cost.png', dpi=150)
    plt.close(fig)
    print(f'[plots] saved {plots_dir / "accuracy_vs_cost.png"}')


@app.command(context_settings=_EXTRA_ARGS)
def run(
    ctx: typer.Context,
    config_file: str = typer.Option(..., help='Starting config YAML'),
    n_train: int = typer.Option(110, help='Training set size'),
    n_eval: int = typer.Option(110, help='Eval set size'),
    max_iterations: int = typer.Option(10, help='Max improvement iterations'),
    target_accuracy: float = typer.Option(None, help='Stop when reached'),
    supervisor_model: str = typer.Option(
        'gemini/gemini-3.1-pro-preview', help='Supervisor LLM model',
    ),
    custom_instructions: str = typer.Option(
        '', help='Extra instructions (text or @filepath)',
    ),
    model_change: str = typer.Option(
        '', help='JSON file with model choices for supervisor',
    ),
    train_split: str = typer.Option('train', help='HF split for training'),
    eval_split: str = typer.Option('test', help='HF split for evaluation'),
):
    """Run supervisor-driven pipeline improvement on a benchmark.

    Iteratively improves a pipeline by calling a supervisor LLM that sees
    profiling data and failure traces. Hill climbs on accuracy, rolls back
    regressions. Reports best config on a held-out eval set.

    Run from a benchmark directory (e.g. benchmarks/medcalc/).
    Extra args are dotlist config overrides.
    """
    benchmark_dir = Path.cwd()

    # Add benchmark dir to path so we can import ptools, expt, etc.
    sys.path.insert(0, str(benchmark_dir))
    project_root = benchmark_dir
    while project_root != project_root.parent:
        if (project_root / 'src').exists():
            sys.path.insert(0, str(project_root / 'src'))
            break
        project_root = project_root.parent

    from secretagent import config
    from secretagent.core import Interface, all_interfaces, implement_via_config
    from secretagent.orchestrate.catalog import PtoolCatalog
    from secretagent.orchestrate.improve import improve_with_supervisor

    # --- Load config ---
    cfg_path = Path(config_file)
    if not cfg_path.is_absolute():
        cfg_path = benchmark_dir / cfg_path
    config.configure(yaml_file=str(cfg_path), dotlist=ctx.args)
    config.set_root(benchmark_dir)

    # Enable caching by default
    if config.get('cachier.enable_caching') is None:
        config.configure(cachier=dict(enable_caching=True))

    # --- Create/load ptools_evolved.py ---
    evolved_path = benchmark_dir / 'ptools_evolved.py'
    if not evolved_path.exists():
        import shutil
        base_path = benchmark_dir / 'ptools.py'
        shutil.copy2(base_path, evolved_path)
        print(f'Created ptools_evolved.py from ptools.py')
    else:
        print(f'Using existing ptools_evolved.py')

    # Import ptools_evolved as the ptools module
    import importlib.util
    spec = importlib.util.spec_from_file_location('ptools', str(evolved_path))
    ptools_module = importlib.util.module_from_spec(spec)
    sys.modules['ptools'] = ptools_module
    spec.loader.exec_module(ptools_module)

    # Try to import benchmark-specific evaluator and dataset loader
    try:
        from expt import load_dataset, stratified_sample
        evaluator_module = __import__('expt')
    except ImportError as e:
        print(f'Error: could not import expt module from {benchmark_dir}: {e}')
        raise typer.Exit(1)

    # Get evaluator class
    evaluator_cls = None
    for name in dir(evaluator_module):
        obj = getattr(evaluator_module, name)
        if (isinstance(obj, type) and issubclass(obj, __import__(
                'secretagent.evaluate', fromlist=['Evaluator']).Evaluator)
                and name != 'Evaluator'):
            evaluator_cls = obj
            break
    if evaluator_cls is None:
        from secretagent.evaluate import ExactMatchEvaluator
        evaluator_cls = ExactMatchEvaluator

    evaluator = evaluator_cls()

    # --- Bind interfaces ---
    implement_via_config(ptools_module, config.require('ptools'))

    entry_point_name = config.get('evaluate.entry_point', 'calculate_medical_value')
    entry_interface = getattr(ptools_module, entry_point_name)

    # --- Load datasets ---
    print(f'\n=== Loading datasets ===')
    train_dataset = load_dataset(train_split)
    if hasattr(evaluator_module, 'stratified_sample'):
        train_dataset.cases = stratified_sample(
            train_dataset.cases, n_train,
            seed=config.get('dataset.shuffle_seed', 42),
        )
    else:
        train_dataset = train_dataset.configure(n=n_train)
    print(f'Train: {len(train_dataset.cases)} cases from {train_split}')

    eval_dataset = None
    if n_eval > 0:
        eval_dataset = load_dataset(eval_split)
        if hasattr(evaluator_module, 'stratified_sample'):
            eval_dataset.cases = stratified_sample(
                eval_dataset.cases, n_eval,
                seed=config.get('dataset.shuffle_seed', 42),
            )
        else:
            eval_dataset = eval_dataset.configure(n=n_eval)
        print(f'Eval: {len(eval_dataset.cases)} cases from {eval_split}')

    # --- Build catalog ---
    tool_interfaces = [
        iface for iface in all_interfaces()
        if iface.name != entry_point_name and iface.implementation is not None
    ]
    catalog = PtoolCatalog.from_interfaces(
        all_interfaces(), exclude=[entry_point_name],
    )

    # --- Load optional args ---
    instructions = _load_custom_instructions(custom_instructions)
    model_choices_text = _load_model_choices(model_change)

    # --- Output directory ---
    timestamp = datetime.now().strftime('%Y%m%d.%H%M%S')
    output_dir = benchmark_dir / 'results' / f'{timestamp}.remote_control'

    # --- Print setup summary ---
    print(f'\n=== Remote Control ===')
    print(f'Benchmark: {benchmark_dir.name}')
    print(f'Config: {cfg_path.name}')
    print(f'Entry point: {entry_point_name}')
    print(f'Supervisor: {supervisor_model}')
    print(f'Train: {len(train_dataset.cases)} cases, Eval: {len(eval_dataset.cases) if eval_dataset else 0} cases')
    print(f'Max iterations: {max_iterations}')
    if target_accuracy:
        print(f'Target accuracy: {target_accuracy:.1%}')
    if instructions:
        print(f'Custom instructions: {instructions[:100]}...')
    if model_choices_text:
        print(f'Model choices loaded')
    print(f'Output: {output_dir}')

    # --- Run improvement loop ---
    report = improve_with_supervisor(
        entry_interface=entry_interface,
        tool_interfaces=tool_interfaces,
        catalog=catalog,
        evaluator=evaluator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        supervisor_model=supervisor_model,
        max_iterations=max_iterations,
        target_accuracy=target_accuracy,
        custom_instructions=instructions,
        model_choices=model_choices_text,
        output_dir=output_dir,
        ptools_module=ptools_module,
    )

    # --- Final eval on held-out set ---
    # Re-get entry_interface from the (possibly reloaded) ptools module
    entry_interface = getattr(ptools_module, entry_point_name)
    if eval_dataset:
        print(f'\n=== Final Evaluation on Held-Out Set ({len(eval_dataset.cases)} cases) ===')
        final_dir = output_dir / 'final_eval'
        final_dir.mkdir(exist_ok=True)
        with config.configuration(evaluate=dict(
            expt_name='rc_final_eval',
            result_dir=str(final_dir),
            record_details=True,
        )):
            csv_path = evaluator.evaluate(eval_dataset, entry_interface)
        import pandas as pd
        df = pd.read_csv(csv_path)
        eval_acc = df['correct'].mean()
        eval_cost = df.get('cost', pd.Series([0])).mean()
        report.final_eval_accuracy = eval_acc
        print(f'Final eval accuracy: {eval_acc:.1%}')
        print(f'Final eval avg cost: ${eval_cost:.4f}')

        # Save updated report
        (output_dir / 'report.json').write_text(
            report.model_dump_json(indent=2)
        )

    # --- Generate plots ---
    print(f'\n=== Generating Plots ===')
    _generate_plots(report, output_dir)

    # --- Final summary ---
    print(f'\n{"=" * 60}')
    print(f'=== Remote Control Complete ===')
    print(f'{"=" * 60}')
    print(f'Best train accuracy: {report.best_train_accuracy:.1%} '
          f'(iteration {report.best_iteration})')
    if report.final_eval_accuracy is not None:
        print(f'Final eval accuracy: {report.final_eval_accuracy:.1%}')
    print(f'Total supervisor cost: ${report.total_supervisor_cost:.4f}')
    print(f'Output saved to: {output_dir}')

    # Print iteration summary table
    print(f'\nIteration log:')
    print(f'  {"Iter":>4}  {"Train Acc":>9}  {"Cost/case":>9}  {"Sup $":>7}  {"Status":>10}')
    for r in report.iterations:
        status = 'KEPT' if r.kept else ('BASELINE' if r.iteration == 0 else 'ROLLBACK')
        print(f'  {r.iteration:>4}  {r.train_accuracy:>8.1%}  ${r.train_cost:>8.4f}  '
              f'${r.supervisor_cost:>6.4f}  {status:>10}')


if __name__ == '__main__':
    app()
