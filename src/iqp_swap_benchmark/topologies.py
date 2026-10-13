"""Hardware topology definitions for compiling IQP circuits.

Native Gatesets by Hardware Vendor:
===================================

IonQ (Trapped-Ion, Forte):
    - GPI: General single-qubit rotation - U(θ,φ)
    - GPI2: π/2 rotation gate - GPI2(φ)
    - ZZPhase: Mølmer-Sørensen entangling gate - exp(-i * θ/2 * ZZ)
    - All-to-all connectivity (no routing needed)
    Qiskit basis approximation: rx, ry, rz, rxx

Quantinuum (Trapped-Ion, H1/H2 Series):
    - Rz: Z rotation - exp(-i * θ/2 * Z)
    - PhasedX: Phased X rotation - Rz(-φ) @ Rx(θ) @ Rz(φ)
    - ZZPhase: Parametric ZZ gate - exp(-i * θ/2 * ZZ)
    - ZZMax: Maximum entangling ZZ - exp(-i * π/4 * ZZ)
    - All-to-all connectivity via ion shuttling
    Qiskit basis approximation: rz, rx, ry, rzz

Google (Sycamore/Willow, Superconducting):
    - PhasedXPow: Phased X power gate
    - Rz: Z rotation (virtual, zero-duration)
    - SYC: Sycamore gate - iSWAP-like + CZ component (fsim)
    - CZ: Controlled-Z (Willow native)
    - Tilted 2D grid connectivity
    Qiskit basis approximation: rz, rx, ry, cz

IBM (Washington/Eagle r3, Superconducting):
    - Rz: Z rotation (virtual)
    - SX: √X gate
    - X: Pauli-X
    - CX: CNOT (Eagle native 2Q gate)
    - Heavy-hex connectivity pattern, 127 qubits
    Qiskit basis: cx, id, rz, sx, x

Rigetti (Ankaa-3, Superconducting):
    - Rz: Z rotation (virtual, zero-duration)
    - RX: X rotation (θ = ±π/2 or π)
    - ISWAP: Native 2Q entangling gate
    - Defected 12x7 grid connectivity with tunable couplers
    Qiskit basis approximation: rz, rx, iswap

OQC (Lucy/Toshiko, Superconducting):
    - Rz: Z rotation
    - SX: √X gate
    - ECR: Echoed cross-resonance gate
    Qiskit basis approximation: rz, sx, ecr
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple

Coupling = List[Tuple[int, int]]
GridCoord = Tuple[int, int]


@dataclass(frozen=True)
class HardwareTarget:
    """Static metadata describing one compilation target."""

    name: str
    description: str
    basis_gates: Tuple[str, ...]  # Qiskit-compatible basis for transpilation
    coupling_map: Optional[Coupling]
    allow_swap_marker: bool = True
    vendor: str = "generic"  # Hardware vendor identifier for native compilation

    @property
    def basis_with_swap(self) -> Tuple[str, ...]:
        """Return the counting basis, optionally augmented with `swap`."""
        if not self.allow_swap_marker:
            return self.basis_gates
        if "swap" in self.basis_gates:
            return self.basis_gates
        return self.basis_gates + ("swap",)


def fully_connected(n_qubits: int) -> HardwareTarget:
    """Idealized fully-connected topology (no routing overhead).
    
    Uses generic Qiskit basis gates for comparison baseline.
    Native gateset: Rz, SX, X (1Q) + CX (2Q)
    """
    return HardwareTarget(
        name=f"fully_connected_{n_qubits}",
        description="Idealized all-to-all connectivity",
        basis_gates=("rz", "sx", "x", "cx"),  # Generic basis
        coupling_map=None,
        vendor="generic",
    )


def line_topology(n_qubits: int) -> HardwareTarget:
    """Linear nearest-neighbour topology (simulated)."""
    edges = [(i, i + 1) for i in range(n_qubits - 1)]
    bidirectional = edges + [(b, a) for a, b in edges]
    return HardwareTarget(
        name=f"line_{n_qubits}",
        description="Nearest-neighbour linear array",
        basis_gates=("rz", "sx", "x", "cx"),
        coupling_map=bidirectional,
        vendor="generic",
    )


def ring_topology(n_qubits: int) -> HardwareTarget:
    """Circular ring topology (simulated)."""
    edges = [(i, (i + 1) % n_qubits) for i in range(n_qubits)]
    bidirectional = edges + [(b, a) for a, b in edges]
    return HardwareTarget(
        name=f"ring_{n_qubits}",
        description="Circular nearest-neighbour connectivity",
        basis_gates=("rz", "sx", "x", "cx"),
        coupling_map=bidirectional,
        vendor="generic",
    )


def rectangular_grid_topology(n_qubits: int) -> HardwareTarget:
    """Construct a simple rectangular nearest-neighbour grid."""
    if n_qubits <= 2:
        return line_topology(n_qubits)
    n_cols = int((n_qubits) ** 0.5)
    n_cols = max(2, n_cols)
    n_rows = (n_qubits + n_cols - 1) // n_cols
    edges: Coupling = []
    for r in range(n_rows):
        for c in range(n_cols):
            idx = r * n_cols + c
            if idx >= n_qubits:
                continue
            right = idx + 1
            if c + 1 < n_cols and right < n_qubits:
                edges.append((idx, right))
            down = idx + n_cols
            if r + 1 < n_rows and down < n_qubits:
                edges.append((idx, down))
    bidirectional = edges + [(b, a) for a, b in edges]
    return HardwareTarget(
        name=f"rect_grid_{n_rows}x{n_cols}_{n_qubits}",
        description="Rectangular planar grid",
        basis_gates=("rz", "sx", "x", "cz", "cx"),
        coupling_map=bidirectional,
        vendor="generic",
    )


def ladder_topology(n_qubits: int) -> HardwareTarget:
    """Construct a two-leg ladder used as a simulated sparse topology."""
    if n_qubits < 4:
        return line_topology(n_qubits)
    top = list(range(0, min(n_qubits, (n_qubits + 1) // 2)))
    bottom = list(range(len(top), min(n_qubits, len(top) * 2)))
    mapping = top + bottom
    edges: Coupling = []
    for row in (top, bottom):
        for i in range(len(row) - 1):
            edges.append((row[i], row[i + 1]))
    for i in range(min(len(top), len(bottom))):
        edges.append((top[i], bottom[i]))
    bidirectional = edges + [(b, a) for a, b in edges]
    return HardwareTarget(
        name=f"ladder_{n_qubits}",
        description="Two-row ladder connectivity",
        basis_gates=("rz", "sx", "x", "cx"),
        coupling_map=bidirectional,
        vendor="generic",
    )


def ionq_forte(n_qubits: int) -> HardwareTarget:
    """IonQ Forte trapped-ion all-to-all connectivity.
    
    IonQ Forte: 36-qubit system with all-to-all connectivity.
    True native gateset: GPI, GPI2 (1Q) + ZZPhase/MS (2Q)
    Qiskit approximation: rx, ry, rz (1Q) + rxx (2Q)
    
    Performance:
        - 2Q gate error: ~0.4% (MS gate)
    
    All-to-all connectivity means no SWAP overhead for any circuit.
    """
    edges = [(i, j) for i in range(n_qubits) for j in range(n_qubits) if i != j]
    return HardwareTarget(
        name=f"ionq_forte_{n_qubits}",
        description="IonQ Forte trapped-ion all-to-all (36Q)",
        basis_gates=("rz", "rx", "ry", "rxx"),  # Qiskit approx of GPI/GPI2/MS
        coupling_map=edges,
        vendor="ionq",
    )


def quantinuum_helios(n_qubits: int) -> HardwareTarget:
    """Quantinuum Helios trapped-ion all-to-all connectivity.
    
    Quantinuum Helios: 98-qubit system with all-to-all via ion shuttling.
    This is Quantinuum's newest and largest system (not H2 which was 56Q).
    True native gateset: Rz, PhasedX (1Q) + ZZPhase, ZZMax (2Q)
    Qiskit approximation: rz, rx, ry (1Q) + rzz (2Q)
    
    All-to-all via ion shuttling - no SWAP overhead needed.
    """
    edges = [(i, j) for i in range(n_qubits) for j in range(n_qubits) if i != j]
    return HardwareTarget(
        name=f"quantinuum_helios_{n_qubits}",
        description="Quantinuum Helios trapped-ion all-to-all (98Q)",
        basis_gates=("rz", "rx", "ry", "rzz"),  # Qiskit approx of Rz/PhasedX/ZZPhase
        coupling_map=edges,
        vendor="quantinuum",
    )


# IBM Washington (127Q Eagle r3) coupling map - extracted from qiskit_ibm_runtime.fake_provider.FakeWashingtonV2
# Heavy-hex topology: 127 qubits, 142 unique edges, max degree 3
# Degree distribution: {1: 4, 2: 89, 3: 34}
# Median CX gate error: 1.146% (from FakeWashingtonV2)
_IBM_WASHINGTON_EDGES: Coupling = [
    (0, 1), (0, 14), (1, 2), (2, 3), (3, 4), (4, 5), (4, 15), (5, 6), (6, 7), (7, 8),
    (8, 16), (9, 10), (10, 11), (11, 12), (12, 13), (12, 17), (14, 18), (15, 22),
    (16, 26), (17, 30), (18, 19), (19, 20), (20, 21), (20, 33), (21, 22), (22, 23),
    (23, 24), (24, 25), (24, 34), (25, 26), (26, 27), (27, 28), (28, 29), (28, 35),
    (29, 30), (30, 31), (31, 32), (32, 36), (33, 39), (34, 43), (35, 47), (36, 51),
    (37, 38), (37, 52), (38, 39), (39, 40), (40, 41), (41, 42), (41, 53), (42, 43),
    (43, 44), (44, 45), (45, 46), (45, 54), (46, 47), (47, 48), (48, 49), (49, 50),
    (49, 55), (50, 51), (52, 56), (53, 60), (54, 64), (55, 68), (56, 57), (57, 58),
    (58, 59), (58, 71), (59, 60), (60, 61), (61, 62), (62, 63), (62, 72), (63, 64),
    (64, 65), (65, 66), (66, 67), (66, 73), (67, 68), (68, 69), (69, 70), (70, 74),
    (71, 77), (72, 81), (73, 85), (74, 89), (75, 76), (75, 90), (76, 77), (77, 78),
    (78, 79), (79, 80), (79, 91), (80, 81), (81, 82), (82, 83), (83, 84), (83, 92),
    (84, 85), (85, 86), (86, 87), (87, 88), (87, 93), (88, 89), (90, 94), (91, 98),
    (92, 102), (93, 106), (94, 95), (95, 96), (96, 97), (96, 109), (97, 98), (98, 99),
    (99, 100), (100, 101), (100, 110), (101, 102), (102, 103), (103, 104), (104, 105),
    (104, 111), (105, 106), (106, 107), (107, 108), (108, 112), (110, 118), (111, 122),
    (112, 126), (113, 114), (114, 115), (115, 116), (116, 117), (117, 118), (118, 119),
    (119, 120), (120, 121), (121, 122), (122, 123), (123, 124), (124, 125), (125, 126),
]

# Create bidirectional edges for IBM Washington
_IBM_WASHINGTON_COUPLING: Coupling = (
    _IBM_WASHINGTON_EDGES + [(b, a) for a, b in _IBM_WASHINGTON_EDGES]
)

_IBM_WASHINGTON_COORDS: List[Tuple[int, int]] = [
    (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (0, 12), (0, 13),
    (1, 0), (1, 4), (1, 8), (1, 12),
    (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), (2, 11), (2, 12), (2, 13), (2, 14),
    (3, 2), (3, 6), (3, 10), (3, 14),
    (4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (4, 10), (4, 11), (4, 12), (4, 13), (4, 14),
    (5, 0), (5, 4), (5, 8), (5, 12),
    (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (6, 9), (6, 10), (6, 11), (6, 12), (6, 13), (6, 14),
    (7, 2), (7, 6), (7, 10), (7, 14),
    (8, 0), (8, 1), (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7), (8, 8), (8, 9), (8, 10), (8, 11), (8, 12), (8, 13), (8, 14),
    (9, 0), (9, 4), (9, 8), (9, 12),
    (10, 0), (10, 1), (10, 2), (10, 3), (10, 4), (10, 5), (10, 6), (10, 7), (10, 8), (10, 9), (10, 10), (10, 11), (10, 12), (10, 13), (10, 14),
    (11, 2), (11, 6), (11, 10), (11, 14),
    (12, 1), (12, 2), (12, 3), (12, 4), (12, 5), (12, 6), (12, 7), (12, 8), (12, 9), (12, 10), (12, 11), (12, 12), (12, 13), (12, 14),
]


@lru_cache(maxsize=None)
def ibm_washington(n_qubits: int) -> HardwareTarget | None:
    """IBM Washington (127-qubit Eagle r3 processor) Heavy-Hex topology.
    
    IBM Washington is an Eagle r3 processor with:
    - 127 qubits in heavy-hex layout
    - CX as native 2-qubit gate (median error: 1.146%)
    - Heavy-hex pattern with max degree 3
    
    True native gateset (Eagle r3):
    - CX: Native 2-qubit gate (CNOT), median error 1.146%
    - RZ: Z rotation (virtual, zero-duration)
    - SX: √X gate
    - X: Pauli-X
    
    Returns None if n_qubits > 127.
    
    Note: Coupling map and error rates from qiskit_ibm_runtime.fake_provider.FakeWashingtonV2.
    """
    total_device = 127
    if n_qubits > total_device:
        return None
    
    # For full device, return all edges
    if n_qubits == total_device:
        coupling = _IBM_WASHINGTON_COUPLING
    else:
        path = _ibm_washington_path(n_qubits)
        visited = set(path)
        
        # Relabel nodes to 0..n-1 preserving path order
        old_to_new = {old: new for new, old in enumerate(path)}
        
        # Filter and relabel edges
        edges = []
        for a, b in _IBM_WASHINGTON_EDGES:
            if a in visited and b in visited:
                edges.append((old_to_new[a], old_to_new[b]))
        
        coupling = edges + [(b, a) for a, b in edges]
    
    return HardwareTarget(
        name=f"ibm_washington_{n_qubits}",
        description="IBM Washington (127Q Eagle r3) heavy-hex topology",
        basis_gates=("cx", "id", "rz", "sx", "x"),  # Eagle native: CX
        coupling_map=coupling,
        vendor="ibm",
    )


# Keep old alias for backward compatibility
@lru_cache(maxsize=None)
def ibm_boston(n_qubits: int) -> HardwareTarget | None:
    """Deprecated: Use ibm_washington instead."""
    return ibm_washington(n_qubits)


def ibm_washington_positions(n_qubits: int) -> dict[int, tuple[float, float]]:
    """Return plotting coordinates aligned with IBM's heavy-hex gate map."""
    if n_qubits < 0 or n_qubits > 127:
        raise ValueError("IBM Washington positions are defined for 0 <= n_qubits <= 127")

    if n_qubits == 127:
        coords = _IBM_WASHINGTON_COORDS
    else:
        path = _ibm_washington_path(n_qubits)
        coords = [_IBM_WASHINGTON_COORDS[old_index] for old_index in path]

    return {
        index: (float(col), float(-row))
        for index, (row, col) in enumerate(coords)
    }


def _ibm_washington_path(n_qubits: int) -> List[int]:
    """Return the DFS-selected connected node order used for reduced IBM subgraphs."""
    import networkx as nx

    graph = nx.Graph()
    graph.add_edges_from(_IBM_WASHINGTON_EDGES)

    path: List[int] = []
    visited: set[int] = set()
    stack = [0]

    while stack and len(path) < n_qubits:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        path.append(node)

        neighbors = [(neighbor, graph.degree(neighbor)) for neighbor in graph.neighbors(node) if neighbor not in visited]
        neighbors.sort(key=lambda item: item[1], reverse=True)
        for neighbor, _ in neighbors:
            stack.append(neighbor)

    return path


def google_sycamore_topology(n_qubits: int) -> HardwareTarget:
    """Google Sycamore 53-qubit staggered inter-row topology.

    The device is modeled as a 9x6 staggered layout with odd and even rows
    offset from each other, no same-row couplers, and a single missing site
    at r1c4. Each present qubit couples to its nearest neighbors on the rows
    immediately above and below.
    """
    if n_qubits <= 2:
        return line_topology(n_qubits)

    total_device = 53
    if n_qubits > total_device:
        raise ValueError("Google Sycamore supports at most 53 qubits")

    coords = _google_sycamore_coordinates()[:n_qubits]
    coord_to_index = {coord: index for index, coord in enumerate(coords)}
    present_coords = set(coords)

    edges: Coupling = []

    for row, col in coords:
        if row == 8:
            continue

        if row % 2 == 0:
            neighbor_coords = ((row + 1, col - 1), (row + 1, col))
        else:
            neighbor_coords = ((row + 1, col), (row + 1, col + 1))

        for neighbor in neighbor_coords:
            if neighbor in present_coords:
                edges.append((coord_to_index[(row, col)], coord_to_index[neighbor]))

    bidirectional = edges + [(b, a) for a, b in edges]
    # True native: PhasedXPow, Rz (1Q) + SYC/CZ (2Q)
    # Qiskit approximation uses CZ as closest match
    return HardwareTarget(
        name=f"google_sycamore_{n_qubits}",
        description="Google Sycamore 53Q staggered inter-row lattice",
        basis_gates=("rz", "rx", "ry", "cz"),  # Sycamore native: SYC/CZ
        coupling_map=bidirectional,
        vendor="google",
    )


def google_sycamore_positions(n_qubits: int) -> dict[int, tuple[float, float]]:
    """Return plotting coordinates matching the Sycamore staggered layout."""
    if n_qubits < 0 or n_qubits > 53:
        raise ValueError("Google Sycamore positions are defined for 0 <= n_qubits <= 53")

    coords = _google_sycamore_coordinates()[:n_qubits]
    return {
        index: (float(col) + 0.5 * (row % 2), float(-row))
        for index, (row, col) in enumerate(coords)
    }


def _google_sycamore_coordinates() -> List[GridCoord]:
    missing_sites = {(0, 3)}
    return [
        (row, col)
        for row in range(9)
        for col in range(6)
        if (row, col) not in missing_sites
    ]


def rigetti_ankaa(n_qubits: int) -> HardwareTarget:
    """Rigetti Ankaa-3 defected 12x7 grid topology with ISWAP native gate.

    Ankaa-3 is modeled as an 82-qubit rectangular 12x7 nearest-neighbour grid
    with two missing sites and three missing couplers:

        - missing qubits at r1c6 and r7c6
        - missing vertical couplers at (r2c6, r3c6) and (r4c3, r5c3)
        - missing horizontal coupler at (r7c8, r7c9)
    
    True native gateset:
        - RZ: Z rotation (virtual, zero-duration)
        - RX: X rotation (θ = ±π/2 or π)
        - ISWAP: Native 2Q entangling gate (98.31% fidelity)
    
    Performance (Ankaa-3):
        - Median T1: 37 µs
        - Median T2: 23 µs
        - Median ISWAP fidelity: 98.31% (error: 1.69%)
        - Median readout fidelity: 97.20%
        - Rep rate: 50 kHz
    """
    if n_qubits <= 2:
        return line_topology(n_qubits)

    total_device = 82

    if n_qubits > total_device:
        raise ValueError("Rigetti Ankaa-3 supports at most 82 qubits")

    coords = _rigetti_ankaa_coordinates()[:n_qubits]
    coord_to_index = {coord: index for index, coord in enumerate(coords)}
    present_coords = set(coords)

    edges: Coupling = []

    for coord in coords:
        row, col = coord
        for neighbor in ((row, col + 1), (row + 1, col)):
            if neighbor not in present_coords:
                continue
            if _normalize_grid_edge(coord, neighbor) in _RIGETTI_ANKAA_DISABLED_EDGES:
                continue
            edges.append((coord_to_index[coord], coord_to_index[neighbor]))

    bidirectional = edges + [(b, a) for a, b in edges]
    # True native: RZ (virtual), RX(±π/2, π), ISWAP
    return HardwareTarget(
        name=f"rigetti_ankaa_{n_qubits}",
        description="Rigetti Ankaa-3 defected 12x7 grid (82Q, ISWAP native)",
        basis_gates=("rz", "rx", "iswap"),  # Ankaa native: ISWAP
        coupling_map=bidirectional,
        vendor="rigetti",
    )


def rigetti_ankaa_positions(n_qubits: int) -> dict[int, tuple[float, float]]:
    """Return plotting coordinates matching the Ankaa-3 physical grid."""
    if n_qubits < 0 or n_qubits > 82:
        raise ValueError("Rigetti Ankaa-3 positions are defined for 0 <= n_qubits <= 82")

    coords = _rigetti_ankaa_coordinates()[:n_qubits]
    return {
        index: (float(col), float(-row))
        for index, (row, col) in enumerate(coords)
    }


def _normalize_grid_edge(a: GridCoord, b: GridCoord) -> Tuple[GridCoord, GridCoord]:
    """Return a grid edge in a canonical order for set membership checks."""
    return (a, b) if a <= b else (b, a)


def _rigetti_ankaa_coordinates() -> List[GridCoord]:
    missing_sites = {(0, 5), (6, 5)}
    coords = [
        (row, col)
        for row in range(7)
        for col in range(12)
        if (row, col) not in missing_sites
    ]
    return coords


_RIGETTI_ANKAA_DISABLED_EDGES = {
    _normalize_grid_edge((1, 5), (2, 5)),
    _normalize_grid_edge((3, 2), (4, 2)),
    _normalize_grid_edge((6, 7), (6, 8)),
}


# Keep old Aspen alias for backward compatibility
def rigetti_aspen_topology(n_qubits: int) -> HardwareTarget:
    """Deprecated: Use rigetti_ankaa instead."""
    return rigetti_ankaa(n_qubits)


def google_willow(n_qubits: int) -> HardwareTarget:
    """Google Willow 105-qubit tapered triangular-lattice patch.

    Willow is modeled as the explicit 105-site patch used in this repo,
    defined by a tapered set of row coordinates on a triangular lattice.
    Qubits couple to nearest neighbors along three lattice directions:
    vertical, horizontal, and one diagonal family in the coordinate system
    used below.
    """
    if n_qubits <= 2:
        return line_topology(n_qubits)

    total_device = 105
    if n_qubits > total_device:
        raise ValueError("Google Willow supports at most 105 qubits")

    coords = _google_willow_coordinates()[:n_qubits]
    coord_to_index = {coord: index for index, coord in enumerate(coords)}
    present_coords = set(coords)

    edges: Coupling = []

    for coord in coords:
        row, col = coord
        for neighbor in ((row + 1, col), (row, col + 1), (row + 1, col + 1)):
            if neighbor in present_coords:
                edges.append((coord_to_index[coord], coord_to_index[neighbor]))

    bidirectional = edges + [(b, a) for a, b in edges]
    return HardwareTarget(
        name=f"google_willow_{n_qubits}",
        description="Google Willow tapered triangular-lattice patch (105Q)",
        basis_gates=("rz", "rx", "ry", "cz"),  # Willow native: CZ
        coupling_map=bidirectional,
        vendor="google",
    )


def google_willow_positions(n_qubits: int) -> dict[int, tuple[float, float]]:
    """Return plotting coordinates matching the Willow device patch."""
    if n_qubits < 0 or n_qubits > 105:
        raise ValueError("Google Willow positions are defined for 0 <= n_qubits <= 105")

    coords = _google_willow_coordinates()[:n_qubits]
    return {
        index: (float(col), float(-row))
        for index, (row, col) in enumerate(coords)
    }


def _google_willow_coordinates() -> List[GridCoord]:
    return [
        (0, 6), (0, 7), (0, 8),
        (1, 5), (1, 6), (1, 7), (1, 8),
        (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10),
        (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10),
        (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (4, 10), (4, 11), (4, 12),
        (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8), (5, 9), (5, 10), (5, 11), (5, 12),
        (6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6), (6, 7), (6, 8), (6, 9), (6, 10), (6, 11), (6, 12), (6, 13), (6, 14),
        (7, 2), (7, 3), (7, 4), (7, 5), (7, 6), (7, 7), (7, 8), (7, 9), (7, 10), (7, 11), (7, 12), (7, 13),
        (8, 2), (8, 3), (8, 4), (8, 5), (8, 6), (8, 7), (8, 8), (8, 9), (8, 10), (8, 11), (8, 12),
        (9, 4), (9, 5), (9, 6), (9, 7), (9, 8), (9, 9), (9, 10), (9, 11),
        (10, 4), (10, 5), (10, 6), (10, 7), (10, 8), (10, 9), (10, 10),
        (11, 6), (11, 7), (11, 8), (11, 9),
        (12, 6), (12, 7), (12, 8),
    ]


def oqc_toshiko(n_qubits: int) -> HardwareTarget:
    """Oxford Quantum Circuits (OQC) Toshiko sparse near-planar topology.

    Toshiko is modeled as a 32-qubit sparse lattice with a near-planar layout:
    four rows of eight qubits with full intra-row nearest-neighbour couplers
    and alternating inter-row couplers between adjacent rows. This preserves a
    sparse degree-3 maximum while producing a more realistic planar embedding
    than the older synthetic dual-ring approximation.

    True native gateset: Rz, SX (1Q) + ECR (2Q)
    2Q gate error: ~4.6% (ECR gate)
    """
    if n_qubits <= 2:
        return line_topology(n_qubits)

    total_qubits = 32

    if n_qubits > total_qubits:
        return None

    coords = _oqc_toshiko_coordinates()[:n_qubits]
    coord_to_index = {coord: index for index, coord in enumerate(coords)}
    present_coords = set(coords)

    edges: Coupling = []

    for row, col in coords:
        right = (row, col + 1)
        if right in present_coords:
            edges.append((coord_to_index[(row, col)], coord_to_index[right]))

        if row < 3:
            if row % 2 == 0 and col % 2 == 1:
                down = (row + 1, col)
                if down in present_coords:
                    edges.append((coord_to_index[(row, col)], coord_to_index[down]))
            if row % 2 == 1 and col % 2 == 0:
                down = (row + 1, col)
                if down in present_coords:
                    edges.append((coord_to_index[(row, col)], coord_to_index[down]))

    bidirectional = edges + [(b, a) for a, b in edges]
    return HardwareTarget(
        name=f"oqc_toshiko_{n_qubits}",
        description=f"OQC Toshiko sparse near-planar topology ({total_qubits}Q)",
        basis_gates=("rz", "sx", "ecr"),  # OQC native: ECR
        coupling_map=bidirectional,
        vendor="oqc",
    )


def oqc_toshiko_positions(n_qubits: int) -> dict[int, tuple[float, float]]:
    """Return plotting coordinates for the OQC Toshiko near-planar layout."""
    if n_qubits < 0 or n_qubits > 32:
        raise ValueError("OQC Toshiko positions are defined for 0 <= n_qubits <= 32")

    coords = _oqc_toshiko_coordinates()[:n_qubits]
    return {
        index: (float(col), float(-row))
        for index, (row, col) in enumerate(coords)
    }


def _oqc_toshiko_coordinates() -> List[GridCoord]:
    return [
        (row, col)
        for row in range(4)
        for col in range(8)
    ]


# Keep alias for backward compatibility
def oqc_lucy(n_qubits: int) -> HardwareTarget:
    """Deprecated: Use oqc_toshiko instead."""
    return oqc_toshiko(n_qubits)


def get_hardware_choices(n_qubits: int) -> List[HardwareTarget]:
    """Get all available hardware targets for benchmarking.
    
    Targets that return None or raise ValueError for the given n_qubits are skipped.
    """
    builders = [
        lambda n: fully_connected(n),
        lambda n: line_topology(n),
        lambda n: ring_topology(n),
        lambda n: ladder_topology(n),
        lambda n: google_sycamore_topology(n),
        lambda n: google_willow(n),
        lambda n: rigetti_ankaa(n),
        lambda n: ionq_forte(n),
        lambda n: quantinuum_helios(n),
        lambda n: oqc_toshiko(n),
        lambda n: ibm_washington(n),
    ]
    targets: List[HardwareTarget] = []
    for builder in builders:
        try:
            t = builder(n_qubits)
        except (ValueError, TypeError):
            continue
        if t is not None:
            targets.append(t)
    return targets


def select_targets(names: Sequence[str], n_qubits: int) -> List[HardwareTarget]:
    """Resolve an explicit list of target names for a given logical size."""
    lookup = {t.name: t for t in get_hardware_choices(n_qubits)}
    missing = [name for name in names if name not in lookup]
    if missing:
        available = ", ".join(sorted(lookup))
        raise KeyError(f"Unknown hardware targets: {missing}. Available: {available}")
    return [lookup[name] for name in names]
