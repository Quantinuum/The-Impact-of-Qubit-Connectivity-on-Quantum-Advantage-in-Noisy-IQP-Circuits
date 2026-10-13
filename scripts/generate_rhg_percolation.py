#!/usr/bin/env python3
"""Generate percolation summary figure for an RHG-lattice IQP interaction graph.

Produces a single combined figure in the same layout as the pattern_summaries
(dense.pdf, local.pdf, sparse.pdf):
  - Top row: 3 graph-fragmentation panels (original, r=0.25, r=0.5)
  - Bottom row: largest-component transition + fragmentation transition

The RHG lattice is the topology used for topological MBQC (Fujii & Tamate,
arXiv:1406.6932).  Its noise threshold is connected to error *detection*
rather than error correction, making the hardness argument more robust.

OUTPUT:
-------
  paper_images/percolation/pattern_summaries/rhg.pdf

USAGE:
------
  python scripts/generate_rhg_percolation.py
  python scripts/generate_rhg_percolation.py --n-qubits 90
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.gridspec import GridSpec

from iqp_swap_benchmark.circuit_patterns import (
    rhg_lattice_terms,
    rhg_lattice_dims,
    _rhg_lattice_qubit_counts,
    pattern_summary,
)
from iqp_swap_benchmark.colors import PAPER_COLORS


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

PATTERN_COLOR = PAPER_COLORS["purple"]  # Distinguishes RHG from the other patterns


def _build_rhg_graph(n_qubits: int) -> tuple[nx.Graph, int]:
    """Build interaction graph from rhg_lattice_terms. Returns (graph, actual_n)."""
    lx, ly, lz, actual_n = rhg_lattice_dims(n_qubits)
    terms = rhg_lattice_terms(n_qubits)
    G = nx.Graph()
    G.add_nodes_from(range(actual_n))
    for term in terms:
        if len(term) == 2:
            G.add_edge(term[0], term[1])
    return G, actual_n


def _apply_noise(G: nx.Graph, p: float, seed: int = 42):
    rng = np.random.default_rng(seed)
    removed = {node for node in G.nodes() if rng.random() < p}
    G_frag = G.copy()
    G_frag.remove_nodes_from(removed)
    return G_frag, removed


def _rhg_layout(n_qubits: int) -> dict[int, tuple[float, float]]:
    """2D projection of RHG lattice for visualisation.
    
    Edge qubits are at midpoints of cubic edges; face qubits at face centres.
    We use isometric-like projection of 3D coordinates.
    """
    lx, ly, lz, actual_n = rhg_lattice_dims(n_qubits)
    ne, nf, _ = _rhg_lattice_qubit_counts(lx, ly, lz)

    pos: dict[int, tuple[float, float]] = {}

    # Isometric projection factors
    iso_x = 0.55
    iso_y = 0.35

    # Edge qubits
    idx = 0
    # x-edges: midpoint between (vx,vy,vz) and (vx+1,vy,vz)
    for vx in range(lx):
        for vy in range(ly + 1):
            for vz in range(lz + 1):
                mx = vx + 0.5
                my = vy
                mz = vz
                px = my * 1.0 + mx * iso_x
                py = mz * 1.0 + mx * iso_y
                pos[idx] = (px, py)
                idx += 1
    # y-edges
    for vx in range(lx + 1):
        for vy in range(ly):
            for vz in range(lz + 1):
                mx = vx
                my = vy + 0.5
                mz = vz
                px = my * 1.0 + mx * iso_x
                py = mz * 1.0 + mx * iso_y
                pos[idx] = (px, py)
                idx += 1
    # z-edges
    for vx in range(lx + 1):
        for vy in range(ly + 1):
            for vz in range(lz):
                mx = vx
                my = vy
                mz = vz + 0.5
                px = my * 1.0 + mx * iso_x
                py = mz * 1.0 + mx * iso_y
                pos[idx] = (px, py)
                idx += 1
    assert idx == ne

    # Face qubits
    # xy-faces
    for cx in range(lx):
        for cy in range(ly):
            for vz in range(lz + 1):
                mx = cx + 0.5
                my = cy + 0.5
                mz = vz
                px = my * 1.0 + mx * iso_x
                py = mz * 1.0 + mx * iso_y
                pos[idx] = (px, py)
                idx += 1
    # xz-faces
    for cx in range(lx):
        for vy in range(ly + 1):
            for cz in range(lz):
                mx = cx + 0.5
                my = vy
                mz = cz + 0.5
                px = my * 1.0 + mx * iso_x
                py = mz * 1.0 + mx * iso_y
                pos[idx] = (px, py)
                idx += 1
    # yz-faces
    for vx in range(lx + 1):
        for cy in range(ly):
            for cz in range(lz):
                mx = vx
                my = cy + 0.5
                mz = cz + 0.5
                px = my * 1.0 + mx * iso_x
                py = mz * 1.0 + mx * iso_y
                pos[idx] = (px, py)
                idx += 1
    assert idx == actual_n

    return pos


def _bold_tick_labels(ax: plt.Axes) -> None:
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")


def _draw_graph_panel(
    ax: plt.Axes,
    graph: nx.Graph,
    pos: dict,
    noise_level: float,
    n_qubits: int,
) -> None:
    if noise_level == 0:
        fragmented = graph.copy()
        removed: set[int] = set()
    else:
        fragmented, removed = _apply_noise(
            graph, noise_level, seed=int(noise_level * 1000) + 123,
        )

    components = sorted(nx.connected_components(fragmented), key=len, reverse=True)
    node_colors: dict[int, str] = {}
    for ci, comp in enumerate(components):
        color = COMPONENT_COLORS[ci % len(COMPONENT_COLORS)]
        for node in comp:
            node_colors[node] = color

    scale = min(1.0, 16.0 / n_qubits)
    edge_lw = max(0.6, 2.0 * scale)
    edge_lw_broken = max(0.3, 1.0 * scale)
    node_radius = max(0.04, 0.175 * (scale ** 0.5))
    node_lw = max(0.8, 4.0 * scale)
    x_size = max(30, 190 * scale)
    x_lw = max(1.2, 5.2 * scale)
    show_labels = n_qubits <= 25

    for u, v in graph.edges():
        xu, yu = pos[u]
        xv, yv = pos[v]
        if u in removed or v in removed:
            ax.plot([xu, xv], [yu, yv], color="lightgray", linewidth=edge_lw_broken,
                    alpha=0.3, linestyle="--", zorder=0)
        else:
            ax.plot([xu, xv], [yu, yv], color=PAPER_COLORS["teal_darkest"],
                    linewidth=edge_lw, alpha=0.7, zorder=1)

    for node in removed:
        x, y = pos[node]
        circle = plt.Circle(
            (x, y), node_radius * 0.82,
            facecolor="lightgray", edgecolor="gray",
            linewidth=node_lw * 0.55, alpha=0.4, zorder=2,
        )
        ax.add_patch(circle)
        ax.scatter(x, y, marker="x", s=x_size, c=PAPER_COLORS["coral"],
                   alpha=0.9, zorder=7, linewidths=x_lw)

    for node in fragmented.nodes():
        x, y = pos[node]
        circle = plt.Circle(
            (x, y), node_radius,
            facecolor=node_colors[node], edgecolor=PAPER_COLORS["black"],
            linewidth=node_lw, zorder=4,
        )
        ax.add_patch(circle)
        if show_labels:
            ax.text(x, y, str(node), ha="center", va="center",
                    fontsize=28, fontweight="black",
                    color=PAPER_COLORS["black"], zorder=5)

    xs = [c[0] for c in pos.values()]
    ys = [c[1] for c in pos.values()]
    x_pad = 0.12 * (max(xs) - min(xs) + 0.01)
    y_pad = 0.12 * (max(ys) - min(ys) + 0.01)
    ax.set_xlim(min(xs) - x_pad, max(xs) + x_pad)
    ax.set_ylim(min(ys) - y_pad, max(ys) + y_pad)

    comp_sizes = [len(c) for c in components]
    max_comp = max(comp_sizes) if comp_sizes else 0
    n_comps = len(components)
    if noise_level == 0:
        title = f"Original ({graph.number_of_edges()} edges)"
    else:
        title = (
            f"$r = {noise_level}$: {len(removed)} removed\n"
            f"{n_comps} components, largest = {max_comp}"
        )
    ax.set_title(title, fontsize=30, fontweight="black",
                 color=PAPER_COLORS["teal_darkest"], pad=12)
    ax.set_aspect("equal")
    ax.axis("off")


def _compute_transition_statistics(
    graph: nx.Graph,
    n_trials: int,
) -> tuple[np.ndarray, list[float], list[float], list[float], list[float]]:
    p_values = np.linspace(0, 0.6, 25)
    mean_largest: list[float] = []
    std_largest: list[float] = []
    mean_n_comp: list[float] = []
    std_n_comp: list[float] = []

    for p in p_values:
        largest_sizes = []
        n_components = []
        for trial in range(n_trials):
            G_frag, _ = _apply_noise(graph, p, seed=trial * 1000 + int(p * 10000))
            components = list(nx.connected_components(G_frag))
            if components:
                largest_sizes.append(max(len(c) for c in components))
                n_components.append(len(components))
            else:
                largest_sizes.append(0)
                n_components.append(0)
        mean_largest.append(float(np.mean(largest_sizes)))
        std_largest.append(float(np.std(largest_sizes)))
        mean_n_comp.append(float(np.mean(n_components)))
        std_n_comp.append(float(np.std(n_components)))

    return p_values, mean_largest, std_largest, mean_n_comp, std_n_comp


def _draw_largest_component_panel(
    ax, p_values, mean_largest, std_largest, n_qubits, color,
) -> None:
    log_threshold = np.log2(n_qubits)
    ax.errorbar(
        p_values, mean_largest, yerr=std_largest,
        marker="o", markersize=7, capsize=4, linewidth=2.5,
        color=color, ecolor=color, alpha=0.92,
    )
    ax.axhline(y=log_threshold, color=PAPER_COLORS["coral"],
               linestyle="--", linewidth=2.3,
               label=rf"$\log_2 n = {log_threshold:.1f}$")
    ax.fill_between(p_values, 0, log_threshold,
                    alpha=0.14, color=PAPER_COLORS["quantum_teal"],
                    label="Efficiently simulatable")
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
    ax.legend(loc="upper right", framealpha=0.95,
              prop={"size": 23, "weight": "bold"})


def _draw_fragmentation_panel(
    ax, p_values, mean_n_comp, std_n_comp, color,
) -> None:
    ax.errorbar(
        p_values, mean_n_comp, yerr=std_n_comp,
        marker="s", markersize=7, capsize=4, linewidth=2.5,
        color=color, ecolor=color, alpha=0.92,
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


def plot_rhg_summary(
    output_path: str | Path,
    n_qubits: int = 90,
    noise_levels: list[float] = [0.0, 0.25, 0.5],
    n_trials: int = 100,
    show_pattern_title: bool = True,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    graph, actual_n = _build_rhg_graph(n_qubits)
    pos = _rhg_layout(n_qubits)
    lx, ly, lz, _ = rhg_lattice_dims(n_qubits)

    fig = plt.figure(figsize=(27.2, 14.2))
    gs = GridSpec(2, 6, figure=fig, height_ratios=[1.22, 0.84],
                  hspace=0.34, wspace=0.82)

    top_axes = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[0, 4:6]),
    ]
    bottom_left = fig.add_subplot(gs[1, 0:3])
    bottom_right = fig.add_subplot(gs[1, 3:6])

    for ax, noise_level in zip(top_axes, noise_levels):
        _draw_graph_panel(ax, graph, pos, noise_level, actual_n)

    p_values, mean_lg, std_lg, mean_nc, std_nc = _compute_transition_statistics(
        graph, n_trials,
    )
    _draw_largest_component_panel(
        bottom_left, p_values, mean_lg, std_lg, actual_n, PATTERN_COLOR,
    )
    _draw_fragmentation_panel(
        bottom_right, p_values, mean_nc, std_nc, PATTERN_COLOR,
    )

    n_edges = graph.number_of_edges()
    if show_pattern_title:
        fig.suptitle(
            f"RHG Lattice ({actual_n} qubits, {lx}×{ly}×{lz}, {n_edges} interactions)",
            fontsize=42, fontweight="black", color=PATTERN_COLOR, y=0.992,
        )
    fig.subplots_adjust(bottom=0.055)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved RHG percolation summary to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate percolation summary for RHG-lattice IQP interaction graph"
    )
    parser.add_argument("--n-qubits", type=int, default=90,
                        help="Target qubit count (default: 90 → RHG 2×2×2)")
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument(
        "--no-pattern-title",
        action="store_true",
        help="Remove only the top pattern title while keeping panel titles",
    )
    args = parser.parse_args()

    plot_rhg_summary(
        output_path="paper_images/percolation/pattern_summaries/rhg.pdf",
        n_qubits=args.n_qubits,
        n_trials=args.n_trials,
        show_pattern_title=not args.no_pattern_title,
    )


if __name__ == "__main__":
    main()
