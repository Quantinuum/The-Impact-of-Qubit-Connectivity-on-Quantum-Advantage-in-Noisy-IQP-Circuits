#!/usr/bin/env python3
"""
Generate percolation diagram showing how noise fragments the IQP interaction graph.

This illustrates the key intuition behind the simulatability theorem:
1. Start with an interaction graph where vertices = qubits, edges = entangling gates
2. Under interleaved Pauli noise, qubits are "removed" with effective
   vertex-removal probability r
3. This fragments the graph into small connected components
4. When components are O(log n), classical simulation becomes efficient

NOTE: This uses ACTUAL IQP gate patterns from circuit_patterns.py, not random graphs.
The interaction graph is derived from the k-local diagonal gates in the IQP circuit.
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import networkx as nx
import numpy as np
from itertools import combinations

# Import actual IQP pattern generator and paper colors
from src.iqp_swap_benchmark.circuit_patterns import PatternConfig, build_gate_pattern
from src.iqp_swap_benchmark.colors import PAPER_COLORS


SPARSE_DENSITY = 0.4


def create_iqp_interaction_graph(n_qubits: int, k: int = 2, pattern: str = "dense") -> nx.Graph:
    """
    Create an interaction graph from an ACTUAL IQP circuit pattern.
    
    Uses the same gate pattern generator as your benchmark code.
    Vertices = qubits, Edges = pairs of qubits that share a k-local interaction.
    
    For k=2 (ZZ gates): edge (i,j) means qubits i,j have a ZZ interaction.
    For k>2 (e.g., ZZZ): we add edges for all pairs in each k-local term.
    """
    G = nx.Graph()
    G.add_nodes_from(range(n_qubits))
    
    # Build the actual IQP gate pattern
    config = PatternConfig(
        kind=pattern,
        n_qubits=n_qubits,
        interaction_order=k,
        density=SPARSE_DENSITY if pattern == "sparse" else 1.0,
        seed=42,
    )
    gate_pattern = build_gate_pattern(config)
    
    # Convert gate pattern to interaction graph
    # Each k-local term (q1, q2, ..., qk) induces edges between all pairs
    for term in gate_pattern:
        if len(term) >= 2:
            for i, j in combinations(term, 2):
                G.add_edge(i, j)
    
    return G


def apply_noise_percolation(G: nx.Graph, p: float, seed: int | None = None) -> tuple[nx.Graph, set]:
    """
    Apply percolation: remove each qubit (and its edges) with probability ``p``.

    Here ``p`` is the direct vertex-removal probability retained for API
    compatibility. Figures denote it by ``r`` to distinguish it from the
    single-layer Pauli-channel parameter
    p_channel = p_Z + min(p_X, p_Y). At circuit depth d,
    r = 1 - (1 - 2 p_channel)^d.
    
    This models the effect of interleaved Pauli noise: qubits are effectively
    "removed" (classicalized) with vertex-removal probability r, represented
    by the legacy parameter name ``p`` here, breaking their entangling edges.
    
    Returns the fragmented graph and the set of removed qubits.
    """
    if seed is not None:
        np.random.seed(seed)
    
    removed = set()
    for node in G.nodes():
        if np.random.random() < p:
            removed.add(node)
    
    # Create new graph without removed nodes
    G_frag = G.copy()
    G_frag.remove_nodes_from(removed)
    
    return G_frag, removed


def draw_edge_from_boundary(ax, pos_u, pos_v, node_radius, color='black', 
                            linewidth=1.5, alpha=0.7, linestyle='-'):
    """Draw an edge that connects from node boundaries, not centers."""
    x1, y1 = pos_u
    x2, y2 = pos_v
    
    # Calculate direction vector
    dx = x2 - x1
    dy = y2 - y1
    length = np.sqrt(dx**2 + dy**2)
    
    if length < 2 * node_radius:
        return  # Nodes too close
    
    # Normalize direction
    dx_norm = dx / length
    dy_norm = dy / length
    
    # Offset start and end points by node radius
    x1_edge = x1 + dx_norm * node_radius
    y1_edge = y1 + dy_norm * node_radius
    x2_edge = x2 - dx_norm * node_radius
    y2_edge = y2 - dy_norm * node_radius
    
    ax.plot([x1_edge, x2_edge], [y1_edge, y2_edge], 
            color=color, linewidth=linewidth, alpha=alpha, 
            linestyle=linestyle, zorder=1)


def _compute_transition_statistics(
    n_qubits: int,
    k: int,
    pattern: str,
    n_trials: int,
) -> tuple[np.ndarray, list[float], list[float], list[float], list[float]]:
    """Compute transition statistics for a fixed IQP interaction graph."""
    G_original = create_iqp_interaction_graph(n_qubits, k, pattern=pattern)
    p_values = np.linspace(0, 0.6, 25)

    mean_largest: list[float] = []
    std_largest: list[float] = []
    mean_n_components: list[float] = []
    std_n_components: list[float] = []

    for p in p_values:
        largest_sizes = []
        n_components = []

        for trial in range(n_trials):
            seed = trial * 1000 + int(p * 10000)
            G_frag, _ = apply_noise_percolation(G_original, p, seed=seed)

            components = list(nx.connected_components(G_frag))
            if components:
                largest_sizes.append(max(len(c) for c in components))
                n_components.append(len(components))
            else:
                largest_sizes.append(0)
                n_components.append(0)

        mean_largest.append(float(np.mean(largest_sizes)))
        std_largest.append(float(np.std(largest_sizes)))
        mean_n_components.append(float(np.mean(n_components)))
        std_n_components.append(float(np.std(n_components)))

    return p_values, mean_largest, std_largest, mean_n_components, std_n_components


def plot_single_percolation_panel(
    output_path: str | Path,
    n_qubits: int,
    k: int,
    pattern: str,
    noise_level: float,
):
    """Export a single graph-fragmentation panel used in the combined percolation figure."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pattern_color = {
        "dense": PAPER_COLORS["quantum_teal"],
        "local": PAPER_COLORS["purple"],
        "sparse": PAPER_COLORS["coral"],
    }[pattern]

    component_colors = [
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

    G_original = create_iqp_interaction_graph(n_qubits, k, pattern=pattern)
    np.random.seed(42)
    pos = nx.spring_layout(G_original, k=2.5, iterations=100)

    if noise_level == 0:
        G_frag = G_original.copy()
        removed: set[int] = set()
    else:
        G_frag, removed = apply_noise_percolation(
            G_original,
            noise_level,
            seed=int(noise_level * 1000) + 123,
        )

    components = list(nx.connected_components(G_frag))
    components.sort(key=len, reverse=True)

    node_colors: dict[int, str] = {}
    for comp_idx, component in enumerate(components):
        color = component_colors[comp_idx % len(component_colors)]
        for node in component:
            node_colors[node] = color

    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    large_graph = n_qubits > 20
    node_radius = 0.075 if large_graph else 0.13
    node_font_size = 18 if large_graph else 32
    node_edge_width = 2.2 if large_graph else 4
    removed_marker_size = 105 if large_graph else 190
    removed_marker_width = 3.2 if large_graph else 5.2

    for u, v in G_original.edges():
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

    for node in G_frag.nodes():
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

    comp_sizes = [len(c) for c in components]
    max_comp = max(comp_sizes) if comp_sizes else 0
    n_comps = len(components)
    if noise_level == 0:
        title = f"{pattern.title()} pattern: original graph"
        subtitle = f"{G_original.number_of_edges()} edges"
    else:
        title = f"{pattern.title()} pattern: r = {noise_level}"
        subtitle = f"{len(removed)} removed, {n_comps} components, largest = {max_comp}"

    ax.set_title(
        f"{title}\n{subtitle}",
        fontsize=28,
        fontweight="bold",
        color=pattern_color,
        pad=16,
    )
    ax.set_aspect("equal")
    ax.axis("off")

    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_single_transition_panel(
    output_path: str | Path,
    n_qubits: int,
    k: int,
    pattern: str,
    n_trials: int,
    metric: str,
):
    """Export one transition subplot used in the combined transition figure."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    p_values, mean_largest, std_largest, mean_n_components, std_n_components = (
        _compute_transition_statistics(n_qubits, k, pattern, n_trials)
    )
    color = {
        "dense": PAPER_COLORS["quantum_teal"],
        "local": PAPER_COLORS["purple"],
        "sparse": PAPER_COLORS["coral"],
    }[pattern]

    fig, ax = plt.subplots(figsize=(9, 6.5))
    log_threshold = np.log2(n_qubits)

    if metric == "largest_component":
        ax.errorbar(
            p_values,
            mean_largest,
            yerr=std_largest,
            marker="o",
            markersize=8,
            capsize=4,
            linewidth=2.5,
            color=color,
            ecolor=color,
            alpha=0.9,
        )
        ax.axhline(
            y=log_threshold,
            color=PAPER_COLORS["coral"],
            linestyle="--",
            linewidth=2.5,
            label=r"$O(\log n)$ threshold",
        )
        ax.fill_between(
            p_values,
            0,
            log_threshold,
            alpha=0.15,
            color=PAPER_COLORS["quantum_teal"],
            label="Efficiently simulatable",
        )
        ax.set_ylabel("Largest component size", fontsize=24, fontweight="bold")
        ax.set_title(f"{pattern.title()} pattern: largest component", fontsize=28, fontweight="bold")
        ax.set_ylim(0, n_qubits + 1)
        ax.legend(fontsize=18, loc="upper right", framealpha=0.95)
    elif metric == "component_count":
        ax.errorbar(
            p_values,
            mean_n_components,
            yerr=std_n_components,
            marker="s",
            markersize=8,
            capsize=4,
            linewidth=2.5,
            color=color,
            ecolor=color,
            alpha=0.9,
        )
        ax.set_ylabel("Number of components", fontsize=24, fontweight="bold")
        ax.set_title(f"{pattern.title()} pattern: fragmentation", fontsize=28, fontweight="bold")
    else:
        raise ValueError(f"Unknown transition metric: {metric}")

    ax.set_xlabel(
        r"Effective vertex-removal probability $r$",
        fontsize=24,
        fontweight="bold",
    )
    ax.tick_params(axis="both", labelsize=18)
    ax.grid(True, alpha=0.3, color=PAPER_COLORS["teal_light"])

    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def export_percolation_singles(
    base_dir: str | Path = "paper_images/percolation/singles",
    n_qubits: int = 16,
    k: int = 2,
    noise_levels: list[float] = [0.0, 0.25, 0.5],
    n_trials: int = 50,
) -> None:
    """Export every panel from the combined percolation figures as individual files."""
    base_dir = Path(base_dir)
    graph_dir = base_dir / "graphs"
    transition_dir = base_dir / "transitions"
    graph_dir.mkdir(parents=True, exist_ok=True)
    transition_dir.mkdir(parents=True, exist_ok=True)

    for pattern in ["dense", "local", "sparse"]:
        for noise_level in noise_levels:
            noise_tag = f"p{noise_level:.2f}".replace(".", "_")
            plot_single_percolation_panel(
                output_path=graph_dir / f"{pattern}_{noise_tag}.pdf",
                n_qubits=n_qubits,
                k=k,
                pattern=pattern,
                noise_level=noise_level,
            )

        plot_single_transition_panel(
            output_path=transition_dir / f"{pattern}_largest_component.pdf",
            n_qubits=n_qubits,
            k=k,
            pattern=pattern,
            n_trials=n_trials,
            metric="largest_component",
        )
        plot_single_transition_panel(
            output_path=transition_dir / f"{pattern}_component_count.pdf",
            n_qubits=n_qubits,
            k=k,
            pattern=pattern,
            n_trials=n_trials,
            metric="component_count",
        )


def plot_percolation_combined(
    output_path: str = "paper_images/percolation/percolation_combined.pdf",
    n_qubits: int = 16,
    k: int = 2,
    noise_levels: list[float] = [0.0, 0.25, 0.5],
):
    """
    Generate a combined figure with 3 rows (Dense, Local, Sparse) showing 
    graph fragmentation under increasing noise. Uses paper color palette.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    patterns = ["dense", "local", "sparse"]
    pattern_labels = ["Dense", "Local", "Sparse"]
    pattern_colors = {
        "dense": PAPER_COLORS["quantum_teal"],
        "local": PAPER_COLORS["purple"],
        "sparse": PAPER_COLORS["coral"],
    }
    
    # Paper component colors
    component_colors = [
        PAPER_COLORS["quantum_teal"],
        PAPER_COLORS["purple"],
        PAPER_COLORS["coral"],
        PAPER_COLORS["orange"],
        PAPER_COLORS["teal_dark"],
        PAPER_COLORS["teal_light"],
        "#8B7355",  # Brown
        "#6B8E23",  # Olive
        "#DB7093",  # Pale violet red
        "#20B2AA",  # Light sea green
        "#9370DB",  # Medium purple
        "#3CB371",  # Medium sea green
    ]
    
    n_rows = len(patterns)
    n_cols = len(noise_levels)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10.5 * n_cols, 9.8 * n_rows))
    
    # Precompute graphs and layouts for each pattern
    graphs = {}
    positions = {}
    for pattern in patterns:
        G = create_iqp_interaction_graph(n_qubits, k, pattern=pattern)
        graphs[pattern] = G
        np.random.seed(42)
        positions[pattern] = nx.spring_layout(G, k=2.5, iterations=100)
    
    # Preserve the original 16-qubit styling, but scale graph elements down
    # for larger alternatives so labels and nodes remain distinguishable.
    large_graph = n_qubits > 20
    node_radius = 0.075 if large_graph else 0.13
    node_font_size = 18 if large_graph else 32
    node_edge_width = 2.2 if large_graph else 4
    removed_marker_size = 105 if large_graph else 190
    removed_marker_width = 3.2 if large_graph else 5.2
    
    for row_idx, (pattern, pattern_label) in enumerate(zip(patterns, pattern_labels)):
        G_original = graphs[pattern]
        pos = positions[pattern]
        
        for col_idx, p in enumerate(noise_levels):
            ax = axes[row_idx, col_idx]
            
            if p == 0:
                G_frag = G_original.copy()
                removed = set()
            else:
                G_frag, removed = apply_noise_percolation(G_original, p, seed=int(p * 1000) + 123)
            
            # Find connected components
            components = list(nx.connected_components(G_frag))
            components.sort(key=len, reverse=True)
            
            # Assign colors to components
            node_colors = {}
            for comp_idx, component in enumerate(components):
                color = component_colors[comp_idx % len(component_colors)]
                for node in component:
                    node_colors[node] = color
            
            # Draw edges first (from node boundaries)
            for u, v in G_original.edges():
                if u in removed or v in removed:
                    draw_edge_from_boundary(ax, pos[u], pos[v], node_radius,
                                          color='lightgray', linewidth=1, 
                                          alpha=0.3, linestyle='--')
                else:
                    draw_edge_from_boundary(ax, pos[u], pos[v], node_radius,
                                          color=PAPER_COLORS["teal_darkest"],
                                          linewidth=2, alpha=0.8)
            
            # Draw removed nodes (gray, smaller)
            for node in removed:
                x, y = pos[node]
                circle = plt.Circle((x, y), node_radius * 0.82, 
                                   facecolor='lightgray', edgecolor='gray',
                                   linewidth=node_edge_width, alpha=0.4, zorder=2)
                ax.add_patch(circle)
                ax.scatter(x, y, marker='x', s=removed_marker_size, c=PAPER_COLORS["coral"],
                          alpha=0.9, zorder=7, linewidths=removed_marker_width)
            
            # Draw surviving nodes
            for node in G_frag.nodes():
                x, y = pos[node]
                color = node_colors[node]
                circle = plt.Circle((x, y), node_radius, 
                                   facecolor=color, edgecolor=PAPER_COLORS["black"],
                                   linewidth=node_edge_width, zorder=4)
                ax.add_patch(circle)
            
            # Component size statistics
            comp_sizes = [len(c) for c in components]
            max_comp = max(comp_sizes) if comp_sizes else 0
            n_comps = len(components)
            
            # Title with statistics
            if p == 0:
                title = f"Original ({G_original.number_of_edges()} edges)"
            else:
                title = f"$r = {p}$: {len(removed)} removed\n{n_comps} components, largest = {max_comp}"
            
            ax.set_title(title, fontsize=40, fontweight='bold', 
                        color=PAPER_COLORS["teal_darkest"])
            ax.set_aspect('equal')
            ax.axis('off')
    
    # Add legend at bottom
    legend_elements = [
        mpatches.Patch(facecolor=PAPER_COLORS["quantum_teal"], edgecolor='black',
                      linewidth=5, label='Active qubit'),
        plt.Line2D([0], [0], marker='x', color=PAPER_COLORS["coral"], linestyle='None',
                   markersize=30, markeredgewidth=6, label='Removed by noise'),
        plt.Line2D([0], [0], color=PAPER_COLORS["teal_darkest"], linewidth=6,
                   label='Entangling interaction'),
        plt.Line2D([0], [0], color='lightgray', linewidth=6, linestyle='--', 
                   label='Broken interaction'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=44,
               bbox_to_anchor=(0.5, -0.02), frameon=True, 
               facecolor='white', edgecolor=PAPER_COLORS["teal_light"])
    
    plt.suptitle(
        f"IQP Interaction Graphs Under Percolation Noise ({n_qubits} qubits)",
        fontsize=70, fontweight='bold', y=0.99, color=PAPER_COLORS["teal_darkest"]
    )
    
    plt.tight_layout(rect=(0.06, 0.05, 1, 0.975))

    left_label_x = axes[0, 0].get_position().x0 - 0.05
    for row_idx, pattern_label in enumerate(pattern_labels):
        left_bbox = axes[row_idx, 0].get_position()
        y_center = 0.5 * (left_bbox.y0 + left_bbox.y1)
        fig.text(
            left_label_x,
            y_center,
            pattern_label,
            ha='right',
            va='center',
            fontsize=75,
            fontweight='bold',
            color=pattern_colors[patterns[row_idx]],
            rotation=90,
        )

    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved combined percolation diagram to {output_path}")
    
    return fig


def plot_percolation_figure(
    output_path: str = "paper_images/percolation_diagram.pdf",
    n_qubits: int = 16,
    k: int = 2,
    pattern: str = "dense",
    noise_levels: list[float] = [0.0, 0.15, 0.35],
):
    """
    Generate a multi-panel figure showing graph fragmentation under increasing noise.
    
    Uses actual IQP gate patterns from your codebase.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create initial graph FROM ACTUAL IQP PATTERN
    G_original = create_iqp_interaction_graph(n_qubits, k, pattern=pattern)
    
    # Use spring layout for consistent positioning
    np.random.seed(42)
    pos = nx.spring_layout(G_original, k=2, iterations=50)
    
    # Create figure with subplots
    n_panels = len(noise_levels)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]
    
    # Color palette for connected components
    component_colors = plt.cm.Set3(np.linspace(0, 1, 12))
    
    for idx, (ax, p) in enumerate(zip(axes, noise_levels)):
        if p == 0:
            G_frag = G_original.copy()
            removed = set()
        else:
            # Use consistent random state based on noise level for reproducibility
            G_frag, removed = apply_noise_percolation(G_original, p, seed=int(p * 1000) + 123)
        
        # Find connected components
        components = list(nx.connected_components(G_frag))
        components.sort(key=len, reverse=True)
        
        # Assign colors to components
        node_colors = {}
        for comp_idx, component in enumerate(components):
            color = component_colors[comp_idx % len(component_colors)]
            for node in component:
                node_colors[node] = color
        
        # Draw removed nodes (gray, smaller)
        removed_pos = {n: pos[n] for n in removed}
        if removed_pos:
            nx.draw_networkx_nodes(
                G_original, removed_pos, nodelist=list(removed),
                node_color='lightgray', node_size=200, alpha=0.4,
                ax=ax
            )
            # Draw X through removed nodes
            for node in removed:
                x, y = pos[node]
                ax.scatter(x, y, marker='x', s=100, c='red', alpha=0.7, zorder=5)
        
        # Draw surviving nodes colored by component
        surviving_nodes = list(G_frag.nodes())
        if surviving_nodes:
            colors = [node_colors[n] for n in surviving_nodes]
            surviving_pos = {n: pos[n] for n in surviving_nodes}
            nx.draw_networkx_nodes(
                G_frag, surviving_pos, nodelist=surviving_nodes,
                node_color=colors, node_size=400, edgecolors='black',
                linewidths=1.5, ax=ax
            )
        
        # Draw edges of original graph (gray, dashed for removed)
        for u, v in G_original.edges():
            x = [pos[u][0], pos[v][0]]
            y = [pos[u][1], pos[v][1]]
            if u in removed or v in removed:
                ax.plot(x, y, 'gray', linestyle='--', linewidth=0.5, alpha=0.3)
            else:
                ax.plot(x, y, 'black', linewidth=1.5, alpha=0.7)
        
        # Component size statistics
        comp_sizes = [len(c) for c in components]
        max_comp = max(comp_sizes) if comp_sizes else 0
        n_comps = len(components)
        
        # Title with statistics
        if p == 0:
            title = f"Original interaction graph\n({n_qubits} qubits, {G_original.number_of_edges()} edges)"
        else:
            title = f"After noise ($p={p}$)\n{len(removed)} qubits removed, {n_comps} components"
            if max_comp > 0:
                title += f"\nLargest: {max_comp} qubits"
        
        ax.set_title(title, fontsize=12)
        ax.axis('off')
        
        # Add component boundary highlighting
        for comp_idx, component in enumerate(components):
            if len(component) >= 2:
                comp_pos = np.array([pos[n] for n in component])
                # Draw convex hull or bounding box
                from scipy.spatial import ConvexHull
                if len(comp_pos) >= 3:
                    try:
                        hull = ConvexHull(comp_pos)
                        hull_points = comp_pos[hull.vertices]
                        hull_points = np.vstack([hull_points, hull_points[0]])
                        ax.fill(hull_points[:, 0], hull_points[:, 1], 
                               alpha=0.15, color=component_colors[comp_idx % len(component_colors)])
                    except:
                        pass
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='lightgreen', edgecolor='black', label='Active qubit'),
        plt.Line2D([0], [0], marker='x', color='red', linestyle='None', 
                   markersize=10, label='Removed by noise'),
        plt.Line2D([0], [0], color='black', linewidth=2, label='Entangling interaction'),
        plt.Line2D([0], [0], color='gray', linewidth=1, linestyle='--', label='Broken interaction'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))
    
    plt.suptitle(
        r"Percolation-based fragmentation of IQP interaction graph under noise",
        fontsize=14, fontweight='bold', y=1.02
    )
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved percolation diagram to {output_path}")
    
    return fig


def plot_percolation_transition(
    output_path: str = "paper_images/percolation_transition.pdf",
    n_qubits: int = 20,
    k: int = 2,
    pattern: str = "dense",
    n_trials: int = 50,
):
    """
    Plot how the largest connected component size decreases with noise p.
    Shows the percolation transition.
    
    Uses a FIXED IQP circuit (from actual gate patterns) and varies only the noise.
    Multiple trials represent different random noise realizations on the same circuit.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create the FIXED IQP interaction graph (same circuit for all trials)
    G_original = create_iqp_interaction_graph(n_qubits, k, pattern=pattern)
    
    p_values = np.linspace(0, 0.6, 25)
    
    # For each p, run multiple trials (different noise realizations)
    mean_largest = []
    std_largest = []
    mean_n_components = []
    
    for p in p_values:
        largest_sizes = []
        n_components = []
        
        for trial in range(n_trials):
            # Different noise realization for each trial
            seed = trial * 1000 + int(p * 10000)
            G_frag, removed = apply_noise_percolation(G_original, p, seed=seed)
            
            components = list(nx.connected_components(G_frag))
            if components:
                largest_sizes.append(max(len(c) for c in components))
                n_components.append(len(components))
            else:
                largest_sizes.append(0)
                n_components.append(0)
        
        mean_largest.append(np.mean(largest_sizes))
        std_largest.append(np.std(largest_sizes))
        mean_n_components.append(np.mean(n_components))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Largest component size vs p
    ax1.errorbar(p_values, mean_largest, yerr=std_largest, 
                 marker='o', capsize=3, color='#1f77b4')
    ax1.axhline(y=np.log2(n_qubits), color='red', linestyle='--', 
                label=r'$O(\log n)$ threshold')
    ax1.fill_between(p_values, 0, np.log2(n_qubits), alpha=0.2, color='lightgreen',
                     label='Efficiently simulatable')
    ax1.set_xlabel(r'Effective vertex-removal probability $r$', fontsize=12)
    ax1.set_ylabel('Largest component size', fontsize=12)
    ax1.set_title(f'Fragmentation of {pattern} IQP graph', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Number of components vs p
    ax2.plot(p_values, mean_n_components, marker='s', color='#ff7f0e')
    ax2.set_xlabel(r'Effective vertex-removal probability $r$', fontsize=12)
    ax2.set_ylabel('Number of connected components', fontsize=12)
    ax2.set_title('Graph fragmentation', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'Percolation transition for {n_qubits}-qubit IQP circuit (k={k}, {pattern} pattern)',
                 fontsize=5, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved percolation transition plot to {output_path}")
    
    return fig


def plot_transition_combined(
    output_path: str = "paper_images/percolation/transition_combined.pdf",
    n_qubits: int = 16,
    k: int = 2,
    n_trials: int = 50,
):
    """
    Generate a combined transition plot with 3 rows (Dense, Local, Sparse).
    Each row shows: (1) Largest component size vs p, (2) Number of components vs p.
    Uses paper color palette.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    patterns = ["dense", "local", "sparse"]
    pattern_labels = ["Dense", "Local", "Sparse"]
    
    # Pattern-specific colors from paper palette
    pattern_colors = {
        "dense": PAPER_COLORS["quantum_teal"],
        "local": PAPER_COLORS["purple"],
        "sparse": PAPER_COLORS["coral"],
    }
    
    n_rows = len(patterns)
    fig, axes = plt.subplots(n_rows, 2, figsize=(23, 8.5 * n_rows))
    
    p_values = np.linspace(0, 0.6, 25)
    
    for row_idx, (pattern, pattern_label) in enumerate(zip(patterns, pattern_labels)):
        # Create the FIXED IQP interaction graph
        G_original = create_iqp_interaction_graph(n_qubits, k, pattern=pattern)
        
        # For each p, run multiple trials
        mean_largest = []
        std_largest = []
        mean_n_components = []
        std_n_components = []
        
        for p in p_values:
            largest_sizes = []
            n_components = []
            
            for trial in range(n_trials):
                seed = trial * 1000 + int(p * 10000)
                G_frag, removed = apply_noise_percolation(G_original, p, seed=seed)
                
                components = list(nx.connected_components(G_frag))
                if components:
                    largest_sizes.append(max(len(c) for c in components))
                    n_components.append(len(components))
                else:
                    largest_sizes.append(0)
                    n_components.append(0)
            
            mean_largest.append(np.mean(largest_sizes))
            std_largest.append(np.std(largest_sizes))
            mean_n_components.append(np.mean(n_components))
            std_n_components.append(np.std(n_components))
        
        color = pattern_colors[pattern]
        
        # Left plot: Largest component size vs p
        ax1 = axes[row_idx, 0]
        ax1.errorbar(p_values, mean_largest, yerr=std_largest, 
                     marker='o', markersize=8, capsize=4, linewidth=2.5,
                     color=color, ecolor=color, alpha=0.9,
                     label=f'{pattern_label} pattern')
        
        # O(log n) threshold line
        log_threshold = np.log2(n_qubits)
        ax1.axhline(y=log_threshold, color=PAPER_COLORS["coral"], linestyle='--',
                    linewidth=2.5, label=r'$O(\log n)$ threshold')
        
        # Fill simulatable region
        ax1.fill_between(p_values, 0, log_threshold, alpha=0.15, 
                        color=PAPER_COLORS["quantum_teal"],
                        label='Efficiently simulatable')
        
        ax1.set_xlabel(r'Removal probability $r$', fontsize=40, fontweight='bold',
                      color=PAPER_COLORS["teal_darkest"])
        ax1.set_ylabel('Largest component size', fontsize=40, fontweight='bold',
                      color=PAPER_COLORS["teal_darkest"])
        ax1.set_title(f'{pattern_label} pattern', fontsize=50, fontweight='bold',
                     color=PAPER_COLORS["teal_darkest"])
        ax1.tick_params(axis='both', labelsize=32)
        ax1.grid(True, alpha=0.3, color=PAPER_COLORS["teal_light"])
        ax1.set_ylim(0, n_qubits + 1)
        
        # Legend only on first row
        if row_idx == 0:
            ax1.legend(fontsize=40, loc='upper right', framealpha=0.95,
                      edgecolor=PAPER_COLORS["teal_light"])
        
        # Right plot: Number of components vs p
        ax2 = axes[row_idx, 1]
        ax2.errorbar(p_values, mean_n_components, yerr=std_n_components,
                     marker='s', markersize=8, capsize=4, linewidth=2.5,
                     color=color, ecolor=color, alpha=0.9)
        
        ax2.set_xlabel(r'Removal probability $r$', fontsize=40, fontweight='bold',
                      color=PAPER_COLORS["teal_darkest"])
        ax2.set_ylabel('Number of components', fontsize=40, fontweight='bold',
                      color=PAPER_COLORS["teal_darkest"])
        ax2.set_title(f'{pattern_label} fragmentation', fontsize=50, 
                     fontweight='bold', color=PAPER_COLORS["teal_darkest"])
        ax2.tick_params(axis='both', labelsize=32)
        ax2.grid(True, alpha=0.3, color=PAPER_COLORS["teal_light"])
    
    # Main title
    plt.suptitle(
        f"Percolation Transition for {n_qubits}-qubit IQP Circuits",
        fontsize=60, fontweight='bold', y=0.99, color=PAPER_COLORS["teal_darkest"]
    )
    
    plt.tight_layout(rect=(0.14, 0, 0.985, 0.985))

    left_label_x = 0.06
    for row_idx, pattern_label in enumerate(pattern_labels):
        left_bbox = axes[row_idx, 0].get_position()
        y_center = 0.5 * (left_bbox.y0 + left_bbox.y1) - 0.02
        fig.text(
            left_label_x,
            y_center,
            pattern_label,
            ha='right',
            va='center',
            fontsize=75,
            fontweight='bold',
            color=pattern_colors[patterns[row_idx]],
            rotation=90,
        )

    fig.savefig(output_path, dpi=150, facecolor='white')
    print(f"Saved combined transition plot to {output_path}")
    
    return fig


if __name__ == '__main__':
    # Generate combined percolation diagram (3 rows: Dense, Local, Sparse)
    plot_percolation_combined(
        output_path='paper_images/percolation/percolation_combined.pdf',
        n_qubits=16,
        k=2,
        noise_levels=[0.0, 0.25, 0.5],
    )
    
    # Generate combined transition plot (3 rows: Dense, Local, Sparse)
    plot_transition_combined(
        output_path='paper_images/percolation/transition_combined.pdf',
        n_qubits=16,
        k=2,
        n_trials=50,
    )

    export_percolation_singles(
        base_dir='paper_images/percolation/singles',
        n_qubits=16,
        k=2,
        noise_levels=[0.0, 0.25, 0.5],
        n_trials=50,
    )
    
    print("\nDone!")
