#!/usr/bin/env python3
# /// script
# dependencies = ["pandas", "matplotlib"]
# ///
"""Plot optimization results: baselines vs NSGA-II summary.

Usage:
    uv run scripts/optimize_plot.py rulearena_nba
    uv run scripts/optimize_plot.py --output my_plot.png finqa
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OPTIMIZE_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "COMMON" / "optimize-results"


def plot_validation_summary(benchmark, summary_path, selection_path, output):
    """Plot saved final-front flags, without recomputing or reading old runs."""
    df = pd.read_csv(summary_path, float_precision="round_trip")
    required = {"cost", "correct", "valid", "frontier", "method", "model",
                "eval_index", "chromosome"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing summary columns: {required - set(df.columns)}")
    for column in ("valid", "frontier"):
        values = df[column].astype(str).str.lower()
        if not values.isin(["true", "false"]).all():
            raise ValueError(f"Invalid {column} flags")
        df[column] = values.eq("true")
    valid = df[df["valid"]].copy()
    if valid.empty or not valid["eval_index"].is_unique:
        raise ValueError("Expected complete evaluations with unique indices")
    if not all(math.isfinite(x) and x > 0 for x in valid["cost"]):
        raise ValueError("Complete evaluations need finite positive costs")
    if not all(math.isfinite(y) and 0 <= y <= 1 for y in valid["correct"]):
        raise ValueError("Invalid validation accuracy")
    front = valid[valid["frontier"]].sort_values("cost")
    if front.empty or (df["frontier"] & ~df["valid"]).any():
        raise ValueError("Expected a saved frontier of complete evaluations")

    selection = None
    if selection_path:
        selection = json.loads(Path(selection_path).read_text(encoding="utf-8"))
        selected_rows = front[front["eval_index"] == selection["eval_index"]]
        if len(selected_rows) != 1:
            raise ValueError("Frozen selection is absent from the saved frontier")
        selected = selected_rows.iloc[0]
        if (json.loads(selected["chromosome"]) != json.loads(selection["chromosome"])
                or not math.isclose(selected["correct"], selection["correct"])
                or not math.isclose(selected["cost"], selection["cost"])):
            raise ValueError("Summary disagrees with the frozen selection")

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    other = valid[~valid["frontier"]]
    ax.scatter(other["cost"] * 100, other["correct"] * 100,
               s=22, color="0.65", alpha=0.65, edgecolors="none",
               label="Other complete evaluations", zorder=2)
    ax.plot(front["cost"] * 100, front["correct"] * 100,
            "o-", color="#2463a6", markersize=7, linewidth=1.3,
            label="Final validation frontier", zorder=3)
    if selection:
        ax.scatter([selected["cost"] * 100], [selected["correct"] * 100],
                   marker="*", s=180, color="#142c49", zorder=4,
                   label="Selected on validation")

    names = {"react": "ReAct", "structured_baseline": "Structured baseline",
             "zs_cot": "Zero-shot CoT", "workflow": "Workflow"}
    # Offsets keep the four Murder labels legible at paper-panel size.
    offsets = {"react": (-12, -52), "structured_baseline": (-32, 38),
               "zs_cot": (-35, -57), "workflow": (23, 24)}
    for row in front.itertuples():
        model_label = {"gpt-oss-20b": "GPT-OSS-20B", "gpt-oss-120b": "GPT-OSS-120B"}.get(row.model, row.model)
        label = f"{names.get(row.method, row.method)} [{row.eval_index}]\n{model_label}"
        ax.annotate(label, (row.cost * 100, row.correct * 100),
                    xytext=offsets.get(row.method, (12, 15)),
                    textcoords="offset points", fontsize=10, linespacing=1.2,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=2),
                    arrowprops=dict(arrowstyle="-", color="0.35", lw=0.8),
                    zorder=5)
    ax.set_xscale("log")
    ax.set_ylim(-3, 100)
    ax.set_xlim(valid["cost"].min() * 65, valid["cost"].max() * 180)
    ax.set_xlabel("Execution cost (USD per 100 validation cases; log scale)", fontsize=11)
    ax.set_ylabel("Validation accuracy (%)", fontsize=11)
    title = {"musr_murder": "MuSR Murder"}.get(benchmark, benchmark.replace("_", " "))
    ax.set_title(f"{title}: configuration search", fontsize=13)
    ax.tick_params(labelsize=10)
    ax.grid(axis="y", color="0.9", linewidth=0.6)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)

    evidence = {"summary": str(Path(summary_path).resolve()),
                "summary_sha256": hashlib.sha256(Path(summary_path).read_bytes()).hexdigest(),
                "plot_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "complete_evaluations": len(valid), "excluded_incomplete": len(df) - len(valid),
                "frontier_source": "Saved optimizer final-front flags; no recomputation",
                "cost_unit": "USD per 100 validation cases, using recorded execution cost",
                "frontier": front.to_dict(orient="records"), "selection": selection}
    output.with_suffix(".data.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {output} and PDF: {len(valid)} complete evaluations, "
          f"{len(front)} frontier points, {len(df) - len(valid)} incomplete excluded")


def load_points_from_dirs(parent_dir):
    """Load (cost, correct, dir_name) tuples from results.csv files in subdirectories."""
    points = []
    if not parent_dir.is_dir():
        return points
    for result_dir in sorted(parent_dir.iterdir()):
        csv_path = result_dir / "results.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if "cost" not in df.columns or "correct" not in df.columns:
            continue
        points.append((df["cost"].mean(), df["correct"].mean(), result_dir.name))
    return points


def strip_timestamp(name):
    """Strip the YYYYMMDD.HHMMSS. prefix from a directory name."""
    parts = name.split(".", 2)
    if len(parts) >= 3:
        return parts[2]
    return name


def compute_frontier(points):
    """Return list of bools: True if point is on the Pareto frontier.

    Points are (cost, correct, name) tuples.
    """
    frontier = []
    for i, (x, y, _) in enumerate(points):
        dominated = any(
            ox < x and oy > y
            for j, (ox, oy, _) in enumerate(points) if j != i
        )
        frontier.append(not dominated)
    return frontier


def plot_points(ax, points, frontier, color, frontier_marker, dominated_marker, label=""):
    """Plot points with different markers for frontier vs dominated."""
    for (x, y, _), on_frontier in zip(points, frontier):
        marker = frontier_marker if on_frontier else dominated_marker
        size = 12 if on_frontier else 8
        ax.plot(x, y, marker=marker, color=color, markersize=size, alpha=0.6)

    # Legend entries
    if any(frontier):
        ax.plot([], [], marker=frontier_marker, color=color, markersize=12, alpha=0.6,
                linestyle="None", label=f"{label} (frontier)")
    if not all(frontier):
        ax.plot([], [], marker=dominated_marker, color=color, markersize=8, alpha=0.6,
                linestyle="None", label=f"{label} (dominated)")

    # Draw frontier hull
    frontier_pts = sorted((x, y) for (x, y, _), f in zip(points, frontier) if f)
    if len(frontier_pts) >= 2:
        fx, fy = zip(*frontier_pts)
        ax.plot(fx, fy, color=color, linewidth=1, alpha=0.4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", help="Directory name under optimize-results/")
    parser.add_argument("--output", default=None, help="Output PNG file path (default: <benchmark>_optimize.png)")
    parser.add_argument("--summary", type=Path,
                        help="Plot a validation summary using its saved final-front flags")
    parser.add_argument("--selection", type=Path,
                        help="Frozen validation selection JSON, for --summary mode")
    parser.add_argument("--points-of-interest", type=int, default=0, metavar="K",
                        help="Annotate K points of interest on the NSGA-II frontier")
    parser.add_argument("--downshift", type=float, default=0.75,
                        help="Top label y position as fraction of max correctness (default: 0.75)")
    parser.add_argument("--label", nargs="+", default=None,
                        help="Override labels for points of interest (in order)")
    parser.add_argument("--spacer", type=float, default=0.05,
                        help="Vertical spacing between labels (default: 0.05)")
    args = parser.parse_args()

    if args.summary:
        plot_validation_summary(args.benchmark, args.summary, args.selection,
                                args.output or f"{args.benchmark}_optimize.png")
        return
    if args.selection:
        parser.error("--selection requires --summary")

    bench_dir = OPTIMIZE_DIR / args.benchmark
    if not bench_dir.is_dir():
        raise ValueError(f"Directory not found: {bench_dir}")

    output = args.output or f"{args.benchmark}_optimize.png"

    fig, ax = plt.subplots(figsize=(10, 7))

    # Load and plot baselines (green)
    baseline_points = load_points_from_dirs(bench_dir / "baselines")
    if baseline_points:
        baseline_frontier = compute_frontier(baseline_points)
        plot_points(ax, baseline_points, baseline_frontier, "green", "s", "D", "baseline")

    # Load and plot NSGA-II runs (blue)
    nsga_points = load_points_from_dirs(bench_dir / "nsga_runs")
    if nsga_points:
        nsga_frontier = compute_frontier(nsga_points)
        plot_points(ax, nsga_points, nsga_frontier, "blue", "o", ".", "NSGA-II")

    # Points of interest
    if args.points_of_interest >= 2 and nsga_points:
        import math
        frontier_pts = [(x, y, name) for (x, y, name), f in zip(nsga_points, nsga_frontier) if f]
        if len(frontier_pts) >= 2:
            # Point 1: highest correctness (listed first)
            pt1 = max(frontier_pts, key=lambda p: p[1])
            # Point 2: lowest cost (listed last)
            pt2 = min(frontier_pts, key=lambda p: p[0])

            # Midpoint between pt1 and pt2
            mid_x = (pt1[0] + pt2[0]) / 2
            mid_y = (pt1[1] + pt2[1]) / 2

            # Select K-2 closest to midpoint from remaining frontier points
            remaining = [p for p in frontier_pts if p not in (pt1, pt2)]
            remaining.sort(key=lambda p: math.hypot(p[0] - mid_x, p[1] - mid_y))
            selected = [pt1] + remaining[:max(0, args.points_of_interest - 2)] + [pt2]

            # Place labels above the legend area (upper-right region)
            # Use figure transform to position text
            if args.label and len(args.label) >= len(selected):
                labels = args.label[:len(selected)]
            elif args.label:
                labels = args.label + [strip_timestamp(name) for _, _, name in selected[len(args.label):]]
            else:
                labels = [strip_timestamp(name) for _, _, name in selected]
            label_x = 0.70  # right region in axes coords, left-aligned text extends rightward
            y_max = max(y for _, y, _ in frontier_pts)
            label_y_start = args.downshift * y_max
            label_spacing = args.spacer * y_max

            for idx, ((px, py, _), label_text) in enumerate(zip(selected, labels)):
                text_y = label_y_start - idx * label_spacing
                ax.annotate(
                    label_text,
                    xy=(px, py),
                    xytext=(label_x, text_y),
                    textcoords=("axes fraction", "data"),
                    fontsize=13,
                    color="black",
                    ha="left",
                    arrowprops=dict(
                        arrowstyle="->",
                        color="black",
                        linestyle="dotted",
                        lw=1.5,
                    ),
                )

    ax.legend(fontsize=12, loc="lower right")
    ax.set_xlabel("cost (minimize)", fontsize=14)
    ax.set_ylabel("correct (maximize)", fontsize=14)
    ax.set_title(f"{args.benchmark}: baselines vs NSGA-II", fontsize=16)
    ax.tick_params(labelsize=12)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {output}")


if __name__ == "__main__":
    main()
