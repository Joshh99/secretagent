"""Pareto frontier visualization."""

from pathlib import Path

import matplotlib.pyplot as plt


def plot_pareto_frontier(
    results: list[tuple[str, float, float]],
    title: str = "Pareto Frontier",
    output_path: str | Path = "pareto.png",
    metric_name: str = "Accuracy",
    show: bool = True,
):
    """Plot accuracy vs cost with frontier points highlighted.

    Args:
        results: list of (label, accuracy_or_metric, cost_per_q) tuples.
            All points are plotted; dominated points are drawn faded/hollow.
        title: plot title.
        output_path: where to save the PNG.
        metric_name: y-axis label (e.g. "Accuracy", "F1").
        show: if True, call plt.show() after saving.
    """
    if not results:
        print("No results to plot.")
        return

    # Identify Pareto-optimal points (non-dominated)
    frontier_mask = []
    for i, (_, acc_i, cost_i) in enumerate(results):
        dominated = False
        for j, (_, acc_j, cost_j) in enumerate(results):
            if i != j and acc_j >= acc_i and cost_j <= cost_i and (acc_j > acc_i or cost_j < cost_i):
                dominated = True
                break
        frontier_mask.append(not dominated)

    frontier = [(lb, a, c) for (lb, a, c), on in zip(results, frontier_mask) if on]
    dominated = [(lb, a, c) for (lb, a, c), on in zip(results, frontier_mask) if not on]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Dominated points: hollow, faded
    if dominated:
        ax.scatter(
            [c for _, _, c in dominated],
            [a for _, a, _ in dominated],
            marker="o", facecolors="none", edgecolors="#999999",
            linewidths=1.2, s=80, zorder=2, label="Dominated",
        )
        for label, acc, cost in dominated:
            ax.annotate(
                label, (cost, acc),
                textcoords="offset points", xytext=(6, -8),
                fontsize=8, color="#999999",
            )

    # Frontier points: filled, connected by line
    frontier_sorted = sorted(frontier, key=lambda x: x[2])  # sort by cost
    if frontier_sorted:
        costs = [c for _, _, c in frontier_sorted]
        accs = [a for _, a, _ in frontier_sorted]

        if len(frontier_sorted) > 1:
            ax.plot(costs, accs, "-", color="#2196F3", linewidth=1.5, alpha=0.6, zorder=3)

        ax.scatter(
            costs, accs,
            marker="*", c="#2196F3", s=150, zorder=4, label="Pareto-optimal",
        )
        for label, acc, cost in frontier_sorted:
            ax.annotate(
                label, (cost, acc),
                textcoords="offset points", xytext=(6, 6),
                fontsize=9, fontweight="bold", color="#1565C0",
            )

    ax.set_xlabel("Cost per query ($)", fontsize=11)
    ax.set_ylabel(metric_name, fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")

    if show:
        plt.show()
    plt.close(fig)
