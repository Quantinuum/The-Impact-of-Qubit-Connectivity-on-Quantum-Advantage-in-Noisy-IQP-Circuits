"""Generate gate patterns for IQP circuits inspired by IQPopt."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, List, Sequence, Tuple

import numpy as np

GatePattern = List[Tuple[int, ...]]


@dataclass(frozen=True)
class PatternConfig:
    """Configuration describing an IQP gate pattern."""

    kind: str
    n_qubits: int
    interaction_order: int = 2
    density: float = 0.5
    seed: int | None = None

    def validate(self) -> None:
        if self.n_qubits < 1:
            raise ValueError("n_qubits must be positive")
        if self.interaction_order < 1:
            raise ValueError("interaction_order must be positive")
        if self.kind == "sparse":
            if not (0.0 < self.density <= 1.0):
                raise ValueError("density must be in (0, 1] for sparse patterns")


def single_qubit_terms(n_qubits: int) -> GatePattern:
    """Return the 1-body terms present in every IQP pattern family here."""
    return [(i,) for i in range(n_qubits)]


def dense_terms(n_qubits: int, interaction_order: int) -> GatePattern:
    """Return all terms up to the requested order, including 1-body terms."""
    interaction_order = min(interaction_order, n_qubits)
    gates: GatePattern = []
    gates.extend(single_qubit_terms(n_qubits))
    for size in range(2, interaction_order + 1):
        gates.extend(tuple(combo) for combo in combinations(range(n_qubits), size))
    return gates


def local_terms(n_qubits: int, interaction_order: int) -> GatePattern:
    """Return contiguous local terms on a 1D line, plus 1-body terms."""
    interaction_order = min(interaction_order, n_qubits)
    gates: GatePattern = []
    gates.extend(single_qubit_terms(n_qubits))
    for size in range(2, interaction_order + 1):
        for start in range(0, n_qubits - size + 1):
            gates.append(tuple(range(start, start + size)))
    return gates


def _rhg_lattice_qubit_counts(lx: int, ly: int, lz: int) -> tuple[int, int, int]:
    """Return (n_edge_qubits, n_face_qubits, total) for an lx×ly×lz RHG lattice."""
    # Edge qubits: one per edge of the cubic lattice
    n_x_edges = lx * (ly + 1) * (lz + 1)
    n_y_edges = (lx + 1) * ly * (lz + 1)
    n_z_edges = (lx + 1) * (ly + 1) * lz
    n_edge = n_x_edges + n_y_edges + n_z_edges
    # Face qubits: one per face of the cubic lattice
    n_xy_faces = lx * ly * (lz + 1)
    n_xz_faces = lx * (ly + 1) * lz
    n_yz_faces = (lx + 1) * ly * lz
    n_face = n_xy_faces + n_xz_faces + n_yz_faces
    return n_edge, n_face, n_edge + n_face


def rhg_lattice_terms(n_qubits: int) -> GatePattern:
    """Interactions on a Raussendorf-Harrington-Goyal (RHG) lattice.

    The RHG lattice is the topology used for topological MBQC on a 3D
    cluster state.  Qubits live on the **edges** and **face centres** of a
    cubic lattice; each face-centre qubit is connected to its 4 bounding
    edge qubits by a two-qubit commuting gate (ZZ(π/4) in the original
    construction).

    The lattice dimensions are chosen so the total qubit count is as close
    to *n_qubits* as possible without exceeding it.  The function returns
    single-qubit Z terms on every qubit plus all face-to-edge 2-body
    interactions as the gate pattern.

    Reference: Fujii & Tamate, arXiv:1406.6932; Raussendorf, Harrington &
    Goyal, New J. Phys. 9, 199 (2007).
    """
    # Find best (lx, ly, lz) with total qubits <= n_qubits
    best = None
    best_total = 0
    for lx in range(1, n_qubits):
        _, _, tot_lx = _rhg_lattice_qubit_counts(lx, 1, 1)
        if tot_lx > n_qubits:
            break
        for ly in range(lx, n_qubits):
            _, _, tot_lxy = _rhg_lattice_qubit_counts(lx, ly, 1)
            if tot_lxy > n_qubits:
                break
            for lz in range(ly, n_qubits):
                _, _, tot = _rhg_lattice_qubit_counts(lx, ly, lz)
                if tot > n_qubits:
                    break
                if tot > best_total:
                    best_total = tot
                    best = (lx, ly, lz)
    if best is None:
        raise ValueError(f"Cannot build RHG lattice with n_qubits={n_qubits} (need at least 14)")

    lx, ly, lz = best
    actual_n = best_total

    # ---- Assign qubit indices ----
    # Edge qubits: indexed first
    # x-edges: edge between vertex (vx, vy, vz) and (vx+1, vy, vz)
    #   for vx in [0, lx-1], vy in [0, ly], vz in [0, lz]
    def x_edge_idx(vx: int, vy: int, vz: int) -> int:
        return vx * (ly + 1) * (lz + 1) + vy * (lz + 1) + vz

    n_x_edges = lx * (ly + 1) * (lz + 1)

    # y-edges: edge between vertex (vx, vy, vz) and (vx, vy+1, vz)
    def y_edge_idx(vx: int, vy: int, vz: int) -> int:
        return n_x_edges + vx * ly * (lz + 1) + vy * (lz + 1) + vz

    n_y_edges = (lx + 1) * ly * (lz + 1)

    # z-edges: edge between vertex (vx, vy, vz) and (vx, vy, vz+1)
    def z_edge_idx(vx: int, vy: int, vz: int) -> int:
        return n_x_edges + n_y_edges + vx * (ly + 1) * lz + vy * lz + vz

    n_edge = n_x_edges + n_y_edges + (lx + 1) * (ly + 1) * lz

    # Face qubits: indexed after all edge qubits
    # xy-faces: face at cell (cx, cy) on plane z=vz
    def xy_face_idx(cx: int, cy: int, vz: int) -> int:
        return n_edge + cx * ly * (lz + 1) + cy * (lz + 1) + vz

    n_xy_faces = lx * ly * (lz + 1)

    # xz-faces: face at cell (cx, cz) on plane y=vy
    def xz_face_idx(cx: int, vy: int, cz: int) -> int:
        return n_edge + n_xy_faces + cx * (ly + 1) * lz + vy * lz + cz

    n_xz_faces = lx * (ly + 1) * lz

    # yz-faces: face at cell (cy, cz) on plane x=vx
    def yz_face_idx(vx: int, cy: int, cz: int) -> int:
        return n_edge + n_xy_faces + n_xz_faces + vx * ly * lz + cy * lz + cz

    # ---- Build gate pattern ----
    gates: GatePattern = []
    gates.extend(single_qubit_terms(actual_n))

    # xy-faces (normal to z): 4 edges are the boundary of the square
    for cx in range(lx):
        for cy in range(ly):
            for vz in range(lz + 1):
                f = xy_face_idx(cx, cy, vz)
                # bottom x-edge: (cx, cy, vz)-(cx+1, cy, vz)
                gates.append((f, x_edge_idx(cx, cy, vz)))
                # top x-edge: (cx, cy+1, vz)-(cx+1, cy+1, vz)
                gates.append((f, x_edge_idx(cx, cy + 1, vz)))
                # left y-edge: (cx, cy, vz)-(cx, cy+1, vz)
                gates.append((f, y_edge_idx(cx, cy, vz)))
                # right y-edge: (cx+1, cy, vz)-(cx+1, cy+1, vz)
                gates.append((f, y_edge_idx(cx + 1, cy, vz)))

    # xz-faces (normal to y): boundary edges in x and z
    for cx in range(lx):
        for vy in range(ly + 1):
            for cz in range(lz):
                f = xz_face_idx(cx, vy, cz)
                # bottom x-edge: (cx, vy, cz)-(cx+1, vy, cz)
                gates.append((f, x_edge_idx(cx, vy, cz)))
                # top x-edge: (cx, vy, cz+1)-(cx+1, vy, cz+1)
                gates.append((f, x_edge_idx(cx, vy, cz + 1)))
                # left z-edge: (cx, vy, cz)-(cx, vy, cz+1)
                gates.append((f, z_edge_idx(cx, vy, cz)))
                # right z-edge: (cx+1, vy, cz)-(cx+1, vy, cz+1)
                gates.append((f, z_edge_idx(cx + 1, vy, cz)))

    # yz-faces (normal to x): boundary edges in y and z
    for vx in range(lx + 1):
        for cy in range(ly):
            for cz in range(lz):
                f = yz_face_idx(vx, cy, cz)
                # bottom y-edge: (vx, cy, cz)-(vx, cy+1, cz)
                gates.append((f, y_edge_idx(vx, cy, cz)))
                # top y-edge: (vx, cy, cz+1)-(vx, cy+1, cz+1)
                gates.append((f, y_edge_idx(vx, cy, cz + 1)))
                # left z-edge: (vx, cy, cz)-(vx, cy, cz+1)
                gates.append((f, z_edge_idx(vx, cy, cz)))
                # right z-edge: (vx, cy+1, cz)-(vx, cy+1, cz+1)
                gates.append((f, z_edge_idx(vx, cy + 1, cz)))

    return gates


def rhg_lattice_dims(n_qubits: int) -> tuple[int, int, int, int]:
    """Return (lx, ly, lz, actual_n_qubits) for the best-fitting RHG lattice."""
    best = None
    best_total = 0
    for lx in range(1, n_qubits):
        _, _, tot_lx = _rhg_lattice_qubit_counts(lx, 1, 1)
        if tot_lx > n_qubits:
            break
        for ly in range(lx, n_qubits):
            _, _, tot_lxy = _rhg_lattice_qubit_counts(lx, ly, 1)
            if tot_lxy > n_qubits:
                break
            for lz in range(ly, n_qubits):
                _, _, tot = _rhg_lattice_qubit_counts(lx, ly, lz)
                if tot > n_qubits:
                    break
                if tot > best_total:
                    best_total = tot
                    best = (lx, ly, lz)
    if best is None:
        raise ValueError(f"Cannot build RHG lattice with n_qubits={n_qubits}")
    return best[0], best[1], best[2], best_total


def sparse_terms(
    n_qubits: int,
    interaction_order: int,
    density: float,
    seed: int | None = None,
) -> GatePattern:
    """Subsample the complete dense term list at the requested density.

    The sampling population includes one-body terms as well as all requested
    multi-qubit terms; one-body terms are not retained separately.
    """
    interaction_order = min(interaction_order, n_qubits)
    all_terms = dense_terms(n_qubits, interaction_order)
    rng = np.random.default_rng(seed)
    target_size = max(1, int(np.ceil(len(all_terms) * density)))
    chosen_indices = rng.choice(len(all_terms), size=target_size, replace=False)
    return [all_terms[idx] for idx in sorted(chosen_indices)]


def build_gate_pattern(config: PatternConfig) -> GatePattern:
    """Dispatch to the requested interaction-pattern constructor."""
    config.validate()
    kind = config.kind.lower()
    if kind == "dense":
        return dense_terms(config.n_qubits, config.interaction_order)
    if kind == "local":
        return local_terms(config.n_qubits, config.interaction_order)
    if kind == "sparse":
        return sparse_terms(
            config.n_qubits, config.interaction_order, config.density, config.seed
        )
    if kind == "rhg":
        return rhg_lattice_terms(config.n_qubits)
    raise ValueError(f"Unknown gate pattern '{config.kind}'")


def required_parameter_count(pattern: GatePattern) -> int:
    """Return the number of angles needed for one IQP layer."""
    return len(pattern)


def pattern_summary(pattern: GatePattern) -> dict[str, int]:
    """Return a histogram keyed by interaction order."""
    order_hist: dict[int, int] = {}
    for term in pattern:
        order_hist[len(term)] = order_hist.get(len(term), 0) + 1
    return order_hist


def iter_terms_by_size(pattern: GatePattern, sizes: Iterable[int]) -> Iterable[Tuple[int, ...]]:
    """Yield only those terms whose arity is in `sizes`."""
    sizes = set(sizes)
    for term in pattern:
        if len(term) in sizes:
            yield term
