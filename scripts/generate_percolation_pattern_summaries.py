#!/usr/bin/env python3
"""Export one balanced percolation summary figure per IQP pattern.

Each output contains:
- top row: three graph-fragmentation panels for the chosen noise levels
- bottom row: largest-component and fragmentation transition plots
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.gridspec import GridSpec

from src.iqp_swap_benchmark.colors import PAPER_COLORS

from scripts.generate_percolation_figure import (
    _compute_transition_statistics,
    apply_noise_percolation,
    create_iqp_interaction_graph,
    draw_edge_from_boundary,
)


COMPONENT_COLORS = [
    PAPER_COLORS["quantum_teal"],
    PAPER_COLORS["purple"],
    PAPER_COLORS["coral"],
    PAPER_COLORS["orange"],
    PAPER_COLORS["teal_dark"],
    PAPER_COLORS["teal_light"],
    "#8B7355",
    "#6B8E23",
    "#DB7093",
    "#20B2AA",
    "#9370DB",
    "#3CB371",
]

PATTERN_COLORS = {
    "dense": PAPER_COLORS["quantum_teal"],
    "local": PAPER_COLORS["purple"],
    "sparse": PAPER_COLORS["coral"],
}


def _bold_tick_labels(ax: plt.Axes) -> None:
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")


def _draw_graph_panel(
    ax: plt.Axes,
    graph: nx.Graph,
    pos: dict[int, np.ndarray],
    noise_level: float,
    node_radius: float,
    node_font_size: float,
    node_edge_width: float,
    removed_marker_size: float,
    removed_marker_width: float,
) -> None:
    if noise_level == 0:
        fragmented = graph.copy()
        removed: set[int] = set()
    else:
        fragmented, removed = apply_noise_percolation(
            graph,
            noise_level,
            seed=int(noise_level * 1000) + 123,
        )

    components = list(nx.connected_components(fragmented))
    components.sort(key=len, reverse=True)

    node_colors: dict[int, str] = {}
    for comp_idx, component in enumerate(components):
        color = COMPONENT_COLORS[comp_idx % len(COMPONENT_COLORS)]
        for node in component:
            node_colors[node] = color

    for u, v in graph.edges():
        if u in removed or v in removed:
            draw_edge_from_boundary(
                ax,
                pos[u],
                pos[v],
                node_radius,
                color="lightgray",
                linewidth=1,
                alpha=0.3,
                linestyle="--",
            )
        else:
            draw_edge_from_boundary(
                ax,
                pos[u],
                pos[v],
                node_radius,
                color=PAPER_COLORS["teal_darkest"],
                linewidth=2,
                alpha=0.8,
            )

    for node in removed:
        x, y = pos[node]
        circle = plt.Circle(
            (x, y),
            node_radius * 0.82,
            facecolor="lightgray",
            edgecolor="gray",
            linewidth=node_edge_width,
            alpha=0.4,
            zorder=2,
        )
        ax.add_patch(circle)
        ax.scatter(
            x,
            y,
            marker="x",
            s=removed_marker_size,
            c=PAPER_COLORS["coral"],
            alpha=0.9,
            zorder=7,
            linewidths=removed_marker_width,
        )

    for node in fragmented.nodes():
        x, y = pos[node]
        circle = plt.Circle(
            (x, y),
            node_radius,
            facecolor=node_colors[node],
            edgecolor=PAPER_COLORS["black"],
            linewidth=node_edge_width,
            zorder=4,
        )
        ax.add_patch(circle)

    comp_sizes = [len(component) for component in components]
    max_comp = max(comp_sizes) if comp_sizes else 0
    n_comps = len(components)
    if noise_level == 0:
        panel_title = f"Original ({graph.number_of_edges()} edges)"
    else:
        panel_title = (
            f"$r = {noise_level}$: {len(removed)} removed\n"
            f"{n_comps} components, largest = {max_comp}"
        )

    ax.set_title(
        panel_title,
        fontsize=30,
        fontweight="black",
        color=PAPER_COLORS["teal_darkest"],
        pad=12,
    )
    xs = [coords[0] for coords in pos.values()]
    ys = [coords[1] for coords in pos.values()]
    x_pad = 0.1 * (max(xs) - min(xs))
    y_pad = 0.1 * (max(ys) - min(ys))
    ax.set_xlim(min(xs) - x_pad, max(xs) + x_pad)
    ax.set_ylim(min(ys) - y_pad, max(ys) + y_pad)
    ax.set_aspect("equal")
    ax.axis("off")


def _draw_largest_component_panel(
    ax: plt.Axes,
    p_values: np.ndarray,
    mean_largest: list[float],
    std_largest: list[float],
    n_qubits: int,
    color: str,
) -> None:
    log_threshold = np.log2(n_qubits)
    ax.errorbar(
        p_values,
        mean_largest,
        yerr=std_largest,
        marker="o",
        markersize=7,
        capsize=4,
        linewidth=2.5,
        color=color,
        ecolor=color,
        alpha=0.92,
    )
    ax.axhline(
        y=log_threshold,
        color=PAPER_COLORS["coral"],
        linestyle="--",
        linewidth=2.3,
        label=r"$O(\log n)$ threshold",
    )
    ax.fill_between(
        p_values,
        0,
        log_threshold,
        alpha=0.14,
        color=PAPER_COLORS["quantum_teal"],
        label="Efficiently simulatable",
    )
    ax.set_title("Largest component", fontsize=33, fontweight="black")
    ax.set_xlabel(
        r"Effective vertex-removal probability $r$",
        fontsize=32,
        fontweight="black",
    )
    ax.set_ylabel("Largest component size", fontsize=32, fontweight="black")
    ax.set_ylim(0, n_qubits + 1)
    ax.tick_params(axis="both", labelsize=25, width=1.8)
    _bold_tick_labels(ax)
    ax.grid(True, alpha=0.3, color=PAPER_COLORS["teal_light"])
    ax.legend(loc="upper right", framealpha=0.95, prop={"size": 23, "weight": "bold"})


def _draw_fragmentation_panel(
    ax: plt.Axes,
    p_values: np.ndarray,
    mean_n_components: list[float],
    std_n_components: list[float],
    color: str,
) -> None:
    ax.errorbar(
        p_values,
        mean_n_components,
        yerr=std_n_components,
        marker="s",
        markersize=7,
        capsize=4,
        linewidth=2.5,
        color=color,
        ecolor=color,
        alpha=0.92,
    )
    ax.set_title("Fragmentation", fontsize=33, fontweight="black")
    ax.set_xlabel(
        r"Effective vertex-removal probability $r$",
        fontsize=32,
        fontweight="black",
    )
    ax.set_ylabel("Number of components", fontsize=32, fontweight="black")
    ax.tick_params(axis="both", labelsize=25, width=1.8)
    _bold_tick_labels(ax)
    ax.grid(True, alpha=0.3, color=PAPER_COLORS["teal_light"])


def plot_pattern_summary(
    output_path: str | Path,
    pattern: str,
    n_qubits: int = 16,
    k: int = 2,
    noise_levels: list[float] = [0.0, 0.25, 0.5],
    n_trials: int = 50,
    show_pattern_title: bool = True,
) -> None:
    """Save one balanced percolation summary figure for a single pattern."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    graph = create_iqp_interaction_graph(n_qubits, k, pattern=pattern)
    np.random.seed(42)
    pos = nx.spring_layout(graph, k=3.1, iterations=120)
    pos = {node: coords * 1.12 for node, coords in pos.items()}
    color = PATTERN_COLORS[pattern]

    fig = plt.figure(figsize=(27.2, 14.2))
    gs = GridSpec(2, 6, figure=fig, height_ratios=[1.22, 0.84], hspace=0.34, wspace=0.82)

    top_axes = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]), fig.add_subplot(gs[0, 4:6])]
    bottom_left = fig.add_subplot(gs[1, 0:3])
    bottom_right = fig.add_subplot(gs[1, 3:6])

    large_graph = n_qubits > 20
    node_radius = 0.075 if large_graph else 0.175
    node_font_size = 18 if large_graph else 35
    node_edge_width = 2.2 if large_graph else 4
    removed_marker_size = 105 if large_graph else 190
    removed_marker_width = 3.2 if large_graph else 5.2
    for ax, noise_level in zip(top_axes, noise_levels):
        _draw_graph_panel(
            ax,
            graph,
            pos,
            noise_level,
            node_radius,
            node_font_size,
            node_edge_width,
            removed_marker_size,
            removed_marker_width,
        )

    p_values, mean_largest, std_largest, mean_n_components, std_n_components = _compute_transition_statistics(
        n_qubits,
        k,
        pattern,
        n_trials,
    )
    _draw_largest_component_panel(bottom_left, p_values, mean_largest, std_largest, n_qubits, color)
    _draw_fragmentation_panel(bottom_right, p_values, mean_n_components, std_n_components, color)

    _pattern_labels = {"rhg": "RHG", "cubic_3d": "Cubic 3D"}
    if show_pattern_title:
        fig.suptitle(
            f"{_pattern_labels.get(pattern, pattern.title())} Pattern",
            fontsize=42,
            fontweight="black",
            color=color,
            y=0.992,
        )
    fig.subplots_adjust(bottom=0.055)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved pattern summary to {output_path}")


def export_pattern_summaries(
    base_dir: str | Path = "paper_images/percolation/pattern_summaries",
    n_qubits: int = 16,
    k: int = 2,
    noise_levels: list[float] = [0.0, 0.25, 0.5],
    n_trials: int = 50,
    show_pattern_title: bool = True,
) -> None:
    """Export one summary figure per pattern."""
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    for pattern in ["dense", "local", "sparse"]:
        plot_pattern_summary(
            output_path=base_dir / f"{pattern}.pdf",
            pattern=pattern,
            n_qubits=n_qubits,
            k=k,
            noise_levels=noise_levels,
            n_trials=n_trials,
            show_pattern_title=show_pattern_title,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Export one percolation summary figure per IQP pattern"
    )
    parser.add_argument("--n-qubits", type=int, default=16)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument(
        "--no-pattern-title",
        action="store_true",
        help="Remove only the top pattern title while keeping panel titles",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("paper_images/percolation/pattern_summaries"),
    )
    args = parser.parse_args()

    export_pattern_summaries(
        base_dir=args.base_dir,
        n_qubits=args.n_qubits,
        k=args.k,
        n_trials=args.n_trials,
        show_pattern_title=not args.no_pattern_title,
    )
