#!/usr/bin/env python3
"""Export one percolation graph figure per IQP pattern.

Each output contains the three graph-fragmentation cases used in the combined
percolation figure for a single pattern, arranged side by side with no titles.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from src.iqp_swap_benchmark.colors import PAPER_COLORS

from scripts.generate_percolation_figure import (
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


def plot_pattern_percolation_triptych(
    output_path: str | Path,
    pattern: str,
    n_qubits: int = 16,
    k: int = 2,
    noise_levels: list[float] = [0.0, 0.25, 0.5],
) -> None:
    """Save a single pattern's three percolation cases as one PDF with no titles."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    graph = create_iqp_interaction_graph(n_qubits, k, pattern=pattern)
    np.random.seed(42)
    pos = nx.spring_layout(graph, k=2.5, iterations=100)

    fig, axes = plt.subplots(1, len(noise_levels), figsize=(9.2 * len(noise_levels), 8.8))
    if len(noise_levels) == 1:
        axes = [axes]

    large_graph = n_qubits > 20
    node_radius = 0.075 if large_graph else 0.13
    node_font_size = 18 if large_graph else 32
    node_edge_width = 2.2 if large_graph else 4
    removed_marker_size = 105 if large_graph else 190
    removed_marker_width = 3.2 if large_graph else 5.2

    for ax, noise_level in zip(axes, noise_levels):
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
            fontsize=24,
            fontweight="bold",
            color=PAPER_COLORS["teal_darkest"],
            pad=14,
        )
        ax.set_aspect("equal")
        ax.axis("off")

    plt.tight_layout(pad=0.4, w_pad=1.2)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved pattern percolation panel to {output_path}")


def export_pattern_percolation_panels(
    base_dir: str | Path = "paper_images/percolation/pattern_panels",
    n_qubits: int = 16,
    k: int = 2,
    noise_levels: list[float] = [0.0, 0.25, 0.5],
) -> None:
    """Export one three-panel percolation graph PDF per pattern."""
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    for pattern in ["dense", "local", "sparse"]:
        plot_pattern_percolation_triptych(
            output_path=base_dir / f"{pattern}.pdf",
            pattern=pattern,
            n_qubits=n_qubits,
            k=k,
            noise_levels=noise_levels,
        )


if __name__ == "__main__":
    export_pattern_percolation_panels()
