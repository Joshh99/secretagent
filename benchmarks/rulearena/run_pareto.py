"""Run NSGA-II Pareto optimization on rulearena benchmark domains."""

import typer

from search_spaces import DOMAIN_SPACES
from secretagent.optimize.encoder import decode, decode_dict, space_size
from secretagent.optimize.pareto import run_nsga2
from secretagent.optimize.viz import plot_pareto_frontier

app = typer.Typer()

# Short labels for human-readable plot annotations and terminal output
_WORKFLOW_LABELS = {
    "ptools.l1_extract_workflow": "L1",
    "ptools.l0f_cot_workflow": "L0F",
    "ptools.l0_oracle_workflow": "L0",
    "ptools.l3_react_workflow": "L3",
}

_MODEL_LABELS = {
    "together_ai/deepseek-ai/DeepSeek-V3": "DSv3",
    "together_ai/deepseek-ai/DeepSeek-V3.1": "DSv3.1",
    "claude-haiku-4-5-20251001": "Haiku",
    "together_ai/Qwen/Qwen3.5-9B": "Qwen9B",
    "together_ai/google/gemma-3n-E4B-it": "Gemma4B",
    "together_ai/openai/gpt-oss-20b": "GPToss20B",
}


def _short_label(dims, chrom):
    """Build a short human-readable label from a decoded chromosome."""
    config = decode_dict(dims, chrom)
    parts = []
    for key, val in config.items():
        if "fn" in key or "method" in key:
            parts.append(_WORKFLOW_LABELS.get(val, val.split(".")[-1]))
        elif "model" in key:
            parts.append(_MODEL_LABELS.get(val, val.split("/")[-1]))
        else:
            parts.append(str(val))
    return " + ".join(parts)


@app.command()
def main(
    domain: str = typer.Option("airline", help="Domain: airline, nba, tax"),
    pop_size: int = typer.Option(10, help="Population size"),
    n_gen: int = typer.Option(5, help="Number of generations"),
    dataset_n: int = typer.Option(5, help="Instances per evaluation"),
    seed: int = typer.Option(42, help="Random seed"),
    timeout: int = typer.Option(600, help="Timeout per config (seconds)"),
    no_plot: bool = typer.Option(False, help="Skip plot generation"),
):
    """Run NSGA-II multi-objective optimization over rulearena configs."""
    if domain not in DOMAIN_SPACES:
        raise typer.BadParameter(f"Unknown domain: {domain}. Choose from {list(DOMAIN_SPACES)}")

    dims, fixed = DOMAIN_SPACES[domain]()

    base_command = "uv run python expt.py run"
    base_dotlist = [
        f"dataset.n={dataset_n}",
        f"dataset.domain={domain}",
    ]

    frontier, all_evaluated = run_nsga2(
        dims=dims,
        fixed_overrides=fixed,
        base_command=base_command,
        base_dotlist=base_dotlist,
        cwd=None,
        timeout=timeout,
        metric="correct",
        expt_prefix=f"nsga_{domain}",
        pop_size=pop_size,
        n_gen=n_gen,
        seed=seed,
        label_fn=_short_label,
    )

    total = space_size(dims)
    print()
    print("=" * 70)
    print(f"PARETO FRONTIER ({domain}, {total} total configs in space)")
    print("=" * 70)
    print(f"  {'Config':<25} {'Accuracy':>10} {'Cost/q':>10}")
    for chrom, acc, cost in frontier:
        label = _short_label(dims, chrom)
        print(f"  {label:<25} {acc:>9.1%} ${cost:>9.4f}")

    print()
    if len(all_evaluated) > len(frontier):
        print(f"Dominated configs:")
        frontier_keys = {tuple(c) for c, _, _ in frontier}
        for chrom, acc, cost in all_evaluated:
            if tuple(chrom) not in frontier_keys:
                label = _short_label(dims, chrom)
                print(f"  {label:<25} {acc:>9.1%} ${cost:>9.4f}")
        print()

    print(f"{len(frontier)} Pareto-optimal / {len(all_evaluated)} total evaluated")

    if not no_plot:
        metric_name = "F1" if domain == "nba" else "Accuracy"
        plot_results = [
            (_short_label(dims, chrom), acc, cost)
            for chrom, acc, cost in all_evaluated
        ]

        output_path = f"results/pareto_{domain}.png"
        plot_pareto_frontier(
            results=plot_results,
            title=f"Pareto Frontier: rulearena {domain}",
            output_path=output_path,
            metric_name=metric_name,
            show=False,
        )


if __name__ == "__main__":
    app()
