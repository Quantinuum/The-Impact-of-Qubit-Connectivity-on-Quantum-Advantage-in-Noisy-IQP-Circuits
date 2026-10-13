#!/usr/bin/env python3
"""
Generate hardware topology comparison figure.

Shows the public connectivity models used in the paper, as defined in
topologies.py.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
from typing import Optional

# Import actual topology implementations
from src.iqp_swap_benchmark.topologies import (
    ionq_forte,
    quantinuum_helios,
    ibm_washington,
    ibm_washington_positions,
    google_sycamore_topology,
    google_sycamore_positions,
    google_willow,
    google_willow_positions,
    rigetti_ankaa,
    rigetti_ankaa_positions,
    oqc_toshiko,
    oqc_toshiko_positions,
    line_topology,
    ring_topology,
    ladder_topology,
)
from src.iqp_swap_benchmark.colors import PAPER_COLORS, get_hardware_colors


def hardware_target_to_graph(target, max_qubits: Optional[int] = None) -> nx.Graph:
    """Convert a HardwareTarget to a NetworkX graph.
    
    If max_qubits is specified and the topology has more qubits,
    extract a connected subgraph using BFS from node 0.
    """
    G = nx.Graph()
    if target.coupling_map is None:
        # Fully connected - infer n_qubits from name
        n = int(target.name.split('_')[-1])
        G.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i + 1, n):
                G.add_edge(i, j)
    else:
        # Extract unique edges (coupling_map is bidirectional)
        edges = set()
        for a, b in target.coupling_map:
            edges.add((min(a, b), max(a, b)))
        max_node = max(max(e) for e in edges) if edges else 0
        G.add_nodes_from(range(max_node + 1))
        G.add_edges_from(edges)
        
        # If too many qubits, extract a connected subgraph
        if max_qubits and G.number_of_nodes() > max_qubits:
            from collections import deque
            visited: set[int] = set()
            queue = deque([0])
            while queue and len(visited) < max_qubits:
                node = queue.popleft()
                if node in visited:
                    continue
                visited.add(node)
                for neighbor in G.neighbors(node):
                    if neighbor not in visited:
                        queue.append(neighbor)
            
            # Create subgraph and relabel nodes to 0..n-1
            subgraph = G.subgraph(visited).copy()
            old_to_new = {old: new for new, old in enumerate(sorted(visited))}
            G = nx.relabel_nodes(subgraph, old_to_new)
    
    return G


def plot_topology_comparison(
    output_path: str | Path = "paper_images/topologies/topology_comparison.pdf",
    n_qubits: int = 12,
    show_full_topologies: bool = False,
):
    """Generate a multi-panel figure comparing hardware topologies.
    
    Includes all hardware platforms from the compilation benchmark:
    - Trapped-ion: IonQ Forte, Quantinuum Helios (all-to-all)
    - Superconducting: IBM Boston (heavy-hex), Google Sycamore/Willow, Rigetti Ankaa-3, OQC Toshiko
    
    Args:
        output_path: Path to save the figure
        n_qubits: Number of qubits for representative display (used when show_full_topologies=False)
        show_full_topologies: If True, show full hardware topologies; if False, show n_qubits subset
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get color palette
    hw_colors = get_hardware_colors()
    
    # Define topology sizes for full display
    # Use large n_qubits to get full topologies, actual size depends on hardware
    full_n = 200  # Request large, functions will return actual max
    
    if show_full_topologies:
        # Get FULL topologies from your codebase
        # Row 1: Trapped-ion (all-to-all) - show representative sizes
        row1_topologies = [
            ("Quantinuum Helios\n(All-to-All, 98Q)", quantinuum_helios(98), hw_colors["quantinuum_helios"]),
            ("IonQ Forte\n(All-to-All)", ionq_forte(36), hw_colors["ionq_forte"]),  # Forte has 36 qubits
        ]
        # Row 2: Heavy + grid topologies - full hardware sizes
        # Note: ibm_washington returns None if > 127Q, so use 127
        row2_topologies = [
            ("IBM Washington\n(Heavy-Hex)", ibm_washington(127), hw_colors["ibm_washington"]),  # 127Q
            ("OQC Toshiko\n(Sparse Near-Planar, 32Q)", oqc_toshiko(32), hw_colors["oqc_toshiko"]),
            ("Google Sycamore\n(Staggered 9x6, 53Q)", google_sycamore_topology(53), hw_colors["google_sycamore"]),
            ("Google Willow\n(Tapered Triangular, 105Q)", google_willow(105), hw_colors["google_willow"]),
        ]
        # Row 3: Rigetti + simulated topologies
        row3_topologies = [
            ("Rigetti Ankaa-3\n(Defected 12x7 Grid)", rigetti_ankaa(82), hw_colors["rigetti_ankaa"]),  # 82Q
            ("Line\n(Linear Chain)", line_topology(20), hw_colors["line"]),  # Representative size
            ("Ring\n(Circular)", ring_topology(20), hw_colors["ring"]),  # Representative size
            ("Ladder\n(2-Rail)", ladder_topology(20), hw_colors["ladder"]),  # Representative size
        ]
    else:
        # Get ACTUAL topologies from your codebase
        # Row 1: Trapped-ion (all-to-all) - 2 items centered
        row1_topologies = [
            ("Quantinuum Helios\n(All-to-All, 98Q)", quantinuum_helios(n_qubits), hw_colors["quantinuum_helios"]),
            ("IonQ Forte\n(All-to-All, 36Q)", ionq_forte(n_qubits), hw_colors["ionq_forte"]),
        ]
        # Row 2: Heavy + grid topologies - 4 items
        row2_topologies = [
            ("IBM Washington\n(Heavy-Hex, 127Q)", ibm_washington(n_qubits), hw_colors["ibm_washington"]),
            ("OQC Toshiko\n(Sparse Near-Planar, 32Q)", oqc_toshiko(n_qubits), hw_colors["oqc_toshiko"]),
            ("Google Sycamore\n(Staggered 9x6, 53Q)", google_sycamore_topology(n_qubits), hw_colors["google_sycamore"]),
            ("Google Willow\n(Tapered Triangular, 105Q)", google_willow(n_qubits), hw_colors["google_willow"]),
        ]
        # Row 3: Rigetti + simulated topologies - 4 items
        row3_topologies = [
            ("Rigetti Ankaa-3\n(Defected 12x7 Grid, 82Q)", rigetti_ankaa(n_qubits), hw_colors["rigetti_ankaa"]),
            ("Line\n(Linear Chain)", line_topology(n_qubits), hw_colors["line"]),
            ("Ring\n(Circular)", ring_topology(n_qubits), hw_colors["ring"]),
            ("Ladder\n(2-Rail)", ladder_topology(n_qubits), hw_colors["ladder"]),
        ]
    
    # Create custom layout: 2 centered on top, 4+4 below
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(30, 25))
    gs = GridSpec(3, 4, figure=fig, hspace=0.58, wspace=0.36)
    
    # Row 1: 2 items centered (use columns 1 and 2 of 4)
    axes_row1 = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])]
    # Row 2: 4 items
    axes_row2 = [fig.add_subplot(gs[1, i]) for i in range(4)]
    # Row 3: 4 items
    axes_row3 = [fig.add_subplot(gs[2, i]) for i in range(4)]
    
    # Combine all topologies with their axes
    all_items = list(zip(row1_topologies, axes_row1)) + \
                list(zip(row2_topologies, axes_row2)) + \
                list(zip(row3_topologies, axes_row3))
    
    for (name, target, color), ax in all_items:
        
        if target is None:
            ax.text(0.5, 0.5, f"{name}\nNot available for n={n_qubits}",
                   ha='center', va='center', fontsize=18, fontweight='bold')
            ax.axis('off')
            continue
        
        # For full topologies, don't limit max_qubits
        if show_full_topologies:
            G = hardware_target_to_graph(target, max_qubits=None)
        else:
            G = hardware_target_to_graph(target, max_qubits=n_qubits)
        
        # Calculate graph properties
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        max_possible = n_nodes * (n_nodes - 1) // 2
        density = n_edges / max_possible if max_possible > 0 else 0
        avg_degree = 2 * n_edges / n_nodes if n_nodes > 0 else 0
        
        # Adjust visualization parameters based on graph size
        if n_nodes <= 15:
            node_size = 500
            show_labels = True
            edge_width = 2.5
        elif n_nodes <= 30:
            node_size = 200
            show_labels = False
            edge_width = 1.5
        elif n_nodes <= 60:
            node_size = 80
            show_labels = False
            edge_width = 1.0
        else:
            node_size = 40
            show_labels = False
            edge_width = 0.8
        
        # Choose layout based on topology
        if "All-to-All" in name:
            pos = nx.circular_layout(G)
        elif "Sycamore" in name:
            # Sycamore uses the explicit 53Q staggered inter-row device layout
            pos = google_sycamore_positions(n_nodes)
        elif "Willow" in name:
            # Willow uses the explicit 105Q tapered triangular-lattice patch
            pos = google_willow_positions(n_nodes)
        elif "Heavy-Hex" in name:
            # IBM heavy-hex: use Qiskit-style physical coordinates instead of a generic spectral embedding
            pos = ibm_washington_positions(n_nodes)
        elif "OQC" in name or "Toshiko" in name:
            pos = oqc_toshiko_positions(n_nodes)
        elif "Rigetti" in name:
            # Rigetti Ankaa - actual 12x7 defected grid layout
            pos = rigetti_ankaa_positions(n_nodes)
        elif "Line" in name:
            # Linear chain layout
            pos = {i: (i, 0) for i in range(n_nodes)}
        elif "Ring" in name:
            # Circular layout for ring
            pos = nx.circular_layout(G)
        elif "Ladder" in name:
            # Two-rail ladder layout
            pos = {}
            for i in range(n_nodes):
                row = i % 2  # Alternate between top (0) and bottom (1) rail
                col = i // 2
                pos[i] = (col, -row)
        else:
            # Default spring layout
            pos = nx.spring_layout(G, k=2, iterations=100, seed=42)
        
        # Adjust line width for large graphs
        lw = 2 if n_nodes <= 30 else 1
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=PAPER_COLORS["teal_darkest"],
                               width=edge_width, alpha=0.6)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=color, node_size=node_size,
                               edgecolors=PAPER_COLORS["black"], linewidths=lw)
        
        # Draw labels only for small graphs
        if show_labels:
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=12, font_weight='bold',
                                    font_color=PAPER_COLORS["white"])
        
        # Title: hardware name at top, details below graph in smaller font
        ax.set_title(name, fontsize=28, fontweight='bold', color=PAPER_COLORS["teal_darkest"], pad=14)
        # Add subtitle with qubit count and density below the graph
        ax.text(0.5, -0.05, f"{n_nodes}Q, density={density:.2f}",
                transform=ax.transAxes, ha='center', va='top',
                fontsize=18, color=PAPER_COLORS["teal_darkest"])
        ax.axis('off')
        
        # Add colored border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(color)
            spine.set_linewidth(4)
    
    # Add legend explaining the issue - larger text
    fig.text(0.5, 0.02, 
             "Sparse topologies require SWAP gates to implement all-to-all IQP interactions,\n"
             "increasing circuit depth and noise accumulation.",
             ha='center', fontsize=28, style='italic', fontweight='bold',
             color=PAPER_COLORS["teal_darkest"],
             bbox=dict(boxstyle='round', facecolor=PAPER_COLORS["teal_lightest"],
                       alpha=0.9, edgecolor=PAPER_COLORS["quantum_teal"], linewidth=2))
    
    if show_full_topologies:
        title = "Quantum Hardware Topologies"
    else:
        title = f"Quantum Hardware Topologies ({n_qubits} qubits)"
    plt.suptitle(title, 
                 fontsize=42, fontweight='bold', y=0.985, color=PAPER_COLORS["teal_darkest"])
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved topology comparison to {output_path}")
    
    return fig


def export_device_topology_panels(base_dir: str | Path = "paper_images/topologies/devices"):
    """Export one PDF per topology/device without panel titles."""
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    hw_colors = get_hardware_colors()
    items = [
        ("quantinuum_helios", quantinuum_helios(98), hw_colors["quantinuum_helios"]),
        ("ionq_forte", ionq_forte(36), hw_colors["ionq_forte"]),
        ("ibm_washington", ibm_washington(127), hw_colors["ibm_washington"]),
        ("oqc_toshiko", oqc_toshiko(32), hw_colors["oqc_toshiko"]),
        ("google_sycamore", google_sycamore_topology(53), hw_colors["google_sycamore"]),
        ("google_willow", google_willow(105), hw_colors["google_willow"]),
        ("rigetti_ankaa", rigetti_ankaa(82), hw_colors["rigetti_ankaa"]),
        ("line", line_topology(20), hw_colors["line"]),
        ("ring", ring_topology(20), hw_colors["ring"]),
        ("ladder", ladder_topology(20), hw_colors["ladder"]),
    ]

    for name, target, color in items:
        fig, ax = plt.subplots(figsize=(8, 6.5))
        graph = hardware_target_to_graph(target, max_qubits=None)
        n_nodes = graph.number_of_nodes()

        if name == "quantinuum_helios" or name == "ionq_forte":
            pos = nx.circular_layout(graph)
        elif name == "ibm_washington":
            pos = ibm_washington_positions(n_nodes)
        elif name == "google_sycamore":
            pos = google_sycamore_positions(n_nodes)
        elif name == "google_willow":
            pos = google_willow_positions(n_nodes)
        elif name == "rigetti_ankaa":
            pos = rigetti_ankaa_positions(n_nodes)
        elif name == "oqc_toshiko":
            pos = oqc_toshiko_positions(n_nodes)
        elif name == "line":
            pos = {i: (i, 0) for i in range(n_nodes)}
        elif name == "ring":
            pos = nx.circular_layout(graph)
        else:
            pos = {i: (i // 2, -(i % 2)) for i in range(n_nodes)}

        node_size = 220 if n_nodes <= 36 else 110 if n_nodes <= 82 else 70
        edge_width = 1.6 if n_nodes <= 36 else 1.1 if n_nodes <= 82 else 0.9

        nx.draw_networkx_edges(graph, pos, ax=ax, edge_color=PAPER_COLORS["teal_darkest"], width=edge_width, alpha=0.7)
        nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=color, node_size=node_size, edgecolors=PAPER_COLORS["black"], linewidths=1.2)
        ax.axis('off')
        fig.savefig(base_dir / f"{name}.pdf", dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"Saved device topology to {base_dir / f'{name}.pdf'}")


if __name__ == '__main__':
    # Generate full topology version
    plot_topology_comparison(
        output_path='paper_images/topologies/topology_comparison.pdf',
        show_full_topologies=True,
    )
    export_device_topology_panels()
    print("\nDone!")
