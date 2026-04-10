"""MUSR self-improvement runner.

Iteratively profiles and evolves ptools to beat baselines.

Usage:
    uv run python benchmarks/musr/self_improve.py \
        --config-file conf/murder_self_improve.yaml

    # override target and iterations
    uv run python benchmarks/musr/self_improve.py \
        --config-file conf/murder_self_improve.yaml \
        --target-accuracy 0.72 \
        --max-iterations 5 \
        --train-n 25

    # quick test on 4 examples
    uv run python benchmarks/musr/self_improve.py \
        --config-file conf/murder_self_improve.yaml \
        --max-iterations 1 --train-n 4 dataset.n=4
"""

import importlib
import sys
from pathlib import Path

import pandas as pd
import typer

_BENCHMARK_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _BENCHMARK_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / 'src'))
sys.path.insert(0, str(_BENCHMARK_DIR))

from secretagent import config
from secretagent.core import implement_via_config
from secretagent.dataset import Dataset, Case
from secretagent.experimental.improve import (
    improve_ptool_within_workflow, _apply_variant, _get_ptool_info,
)
from secretagent.orchestrate.profiler import profile_from_results
from secretagent.orchestrate.transforms.base import format_profiling_summary

# Reuse from expt.py
from expt import MUSREvaluator, load_dataset, _resolve_module

app = typer.Typer()


def _pick_weakest_ptool(profile, exclude=None):
    """Pick the ptool most worth evolving from profiling data.

    Prefers high-cost ptools (where improvement has impact) and skips
    trivial utility ptools like extract_index.
    """
    exclude = exclude or set()
    # Skip utility ptools that just parse/extract — evolving their
    # docstrings rarely helps, the real reasoning happens elsewhere
    skip_utilities = {'extract_index', 'raw_answer', 'format_answer'}
    best_name = None
    best_score = -float('inf')

    for name, pp in profile.ptool_profiles.items():
        if name in exclude or name in skip_utilities or pp.n_calls < 3:
            continue
        # Prioritize ptools that consume significant cost (reasoning ptools)
        # and have room to improve
        error_count = sum(e.frequency for e in pp.error_patterns)
        error_rate = error_count / pp.n_calls if pp.n_calls else 0.0
        # Score = cost_fraction (big is important) + error_rate (buggy)
        score = pp.cost_fraction + error_rate * 0.5
        if score > best_score:
            best_score = score
            best_name = name
    return best_name


@app.command(context_settings={
    'allow_extra_args': True,
    'allow_interspersed_args': False,
})
def run(
    ctx: typer.Context,
    config_file: str = typer.Option(..., help='Config YAML file'),
    target_accuracy: float = typer.Option(0.72, help='Accuracy target to beat'),
    max_iterations: int = typer.Option(5, help='Max improvement iterations'),
    train_n: int = typer.Option(25, help='Cases for evolution fitness eval'),
    population_size: int = typer.Option(3, help='Variants per generation'),
    n_generations: int = typer.Option(2, help='Evolutionary generations per ptool'),
):
    """Run self-improvement loop on MUSR."""

    # --- Setup ---
    cfg_path = Path(config_file)
    if not cfg_path.is_absolute():
        cfg_path = _BENCHMARK_DIR / cfg_path
    config.configure(yaml_file=str(cfg_path), dotlist=ctx.args)
    config.set_root(_BENCHMARK_DIR)

    split = config.require('dataset.split')
    ptools_module = importlib.import_module(_resolve_module(split))
    implement_via_config(ptools_module, config.require('ptools'))

    # Load datasets: train and eval must be DISJOINT to prevent overfitting
    full_dataset = load_dataset(split)
    eval_n = config.get('dataset.n') or len(full_dataset.cases)
    eval_dataset = full_dataset.configure(
        shuffle_seed=config.get('dataset.shuffle_seed'),
        n=eval_n,
    )
    # Train cases: take from the END of the shuffled dataset (after eval set)
    all_cases = full_dataset.configure(shuffle_seed=42).cases
    train_cases = all_cases[eval_n:eval_n + train_n]
    if len(train_cases) < train_n:
        # Not enough remaining — use a different shuffle seed for train
        train_cases = full_dataset.configure(shuffle_seed=99).cases[:train_n]
    print(f'Train/eval disjoint: {len(set(c.name for c in train_cases) & set(c.name for c in eval_dataset.cases))} overlap')

    entry_point = config.require('evaluate.entry_point')
    workflow_interface = getattr(ptools_module, entry_point)
    evaluator = MUSREvaluator()

    print(f'=== MUSR Self-Improvement ===')
    print(f'split: {split}')
    print(f'actor model: {config.get("llm.model")}')
    print(f'big model: {config.get("improve.model")}')
    print(f'eval set: {len(eval_dataset.cases)} cases, train set: {train_n} cases')
    print(f'target accuracy: {target_accuracy:.0%}')
    print(f'max iterations: {max_iterations}')
    print(f'evolution: pop={population_size}, gen={n_generations}')

    # --- Initial evaluation ---
    print(f'\n=== Initial Evaluation ===')
    csv_path = evaluator.evaluate(eval_dataset, workflow_interface)
    df = pd.read_csv(csv_path)
    initial_accuracy = df['correct'].mean()
    result_dir = csv_path.parent
    print(f'Initial accuracy: {initial_accuracy:.1%} ({df["correct"].sum()}/{len(df)})')

    # --- Profile (FREE) ---
    profile = profile_from_results([result_dir])
    print(f'\n=== Initial Profile (FREE) ===')
    print(format_profiling_summary(profile))

    if initial_accuracy >= target_accuracy:
        print(f'\nAlready at target! ({initial_accuracy:.1%} >= {target_accuracy:.0%})')
        return

    # --- Self-improvement loop ---
    best_accuracy = initial_accuracy
    evolved_ptools = []
    already_evolved = set()

    for iteration in range(1, max_iterations + 1):
        print(f'\n{"=" * 50}')
        print(f'=== Iteration {iteration}/{max_iterations} ===')
        print(f'{"=" * 50}')

        # Pick weakest ptool (skip ones we already evolved this round)
        target_ptool = _pick_weakest_ptool(profile, exclude=already_evolved)
        if target_ptool is None:
            print('No more ptools to evolve. Stopping.')
            break

        print(f'\nTarget ptool: {target_ptool}')
        pp = profile.ptool_profiles.get(target_ptool)
        if pp:
            error_count = sum(e.frequency for e in pp.error_patterns)
            print(f'  cost_fraction: {pp.cost_fraction:.1%}')
            print(f'  errors: {error_count}')
            print(f'  correct_rate: {pp.accuracy_when_correct:.2f}')
            print(f'  incorrect_rate: {pp.accuracy_when_incorrect:.2f}')

        # Evolve it
        print(f'\nEvolving {target_ptool}...')
        prof_summary = format_profiling_summary(profile)

        try:
            result = improve_ptool_within_workflow(
                ptool_name=target_ptool,
                workflow_interface=workflow_interface,
                train_cases=train_cases,
                population_size=population_size,
                n_generations=n_generations,
                profiling_summary=prof_summary,
            )
        except Exception as e:
            print(f'Evolution failed: {e}')
            already_evolved.add(target_ptool)
            continue

        if not result['improved']:
            print(f'No improvement found for {target_ptool}.')
            already_evolved.add(target_ptool)
            continue

        # Apply the improvement (save original for rollback)
        from secretagent.core import all_interfaces
        ptool = None
        for iface in all_interfaces():
            if iface.name == target_ptool:
                ptool = iface
                break

        if ptool is None:
            print(f'Could not find interface {target_ptool}')
            continue

        # Save original state for rollback
        original_impl = ptool.implementation
        original_doc = ptool.doc
        original_src = ptool.src

        _apply_variant(ptool, result['code'], _get_ptool_info(ptool))
        print(f'Applied improved {target_ptool}')
        print(f'  evolution fitness: accuracy={result["fitness"]["accuracy"]:.2f}, '
              f'cost={result["fitness"]["cost"]:.4f}')

        # Re-evaluate on full eval set
        print(f'\nRe-evaluating on full eval set...')
        csv_path = evaluator.evaluate(eval_dataset, workflow_interface)
        df = pd.read_csv(csv_path)
        new_accuracy = df['correct'].mean()
        result_dir = csv_path.parent
        print(f'Accuracy: {best_accuracy:.1%} -> {new_accuracy:.1%}')

        # Re-profile (FREE)
        profile = profile_from_results([result_dir])
        print(f'\n--- Profile after iteration {iteration} (FREE) ---')
        print(format_profiling_summary(profile))

        if new_accuracy > best_accuracy:
            best_accuracy = new_accuracy
            evolved_ptools.append({
                'ptool': target_ptool,
                'accuracy_after': new_accuracy,
                'code_len': len(result['code']),
            })
            print(f'Improvement kept! Best accuracy: {best_accuracy:.1%}')
        else:
            # Rollback to original implementation
            ptool.implementation = original_impl
            ptool.doc = original_doc
            ptool.src = original_src
            print(f'Regression — rolled back {target_ptool}.')

        already_evolved.add(target_ptool)

        if best_accuracy >= target_accuracy:
            print(f'\nTarget accuracy {target_accuracy:.0%} reached!')
            break

    # --- Summary ---
    print(f'\n{"=" * 50}')
    print(f'=== Summary ===')
    print(f'{"=" * 50}')
    print(f'Initial accuracy: {initial_accuracy:.1%}')
    print(f'Best accuracy:    {best_accuracy:.1%}')
    print(f'Target:           {target_accuracy:.0%}')
    print(f'Evolved ptools:   {len(evolved_ptools)}')
    for e in evolved_ptools:
        print(f'  {e["ptool"]}: accuracy_after={e["accuracy_after"]:.1%}')
    if best_accuracy >= target_accuracy:
        print(f'\nSUCCESS: Beat baseline!')
    else:
        print(f'\nDid not reach target. Consider more iterations or different approach.')


if __name__ == '__main__':
    app()
