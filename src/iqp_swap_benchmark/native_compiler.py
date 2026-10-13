"""Native hardware compilation using pytket for accurate gate counts.

This module provides hardware-specific compilation using pytket's native
gate rebasing passes. This gives more accurate gate counts than using
Qiskit's generic transpiler with approximate basis gates.

Native Gatesets by Hardware:
============================

IonQ (Trapped-Ion):
    - GPI: General single-qubit rotation (angle θ, phase φ)
    - GPI2: π/2 rotation (phase φ) 
    - ZZPhase: Native entangling gate (Mølmer-Sørensen)
    Reference: https://ionq.com/docs/getting-started-with-native-gates

Quantinuum (Trapped-Ion, H-Series):
    - Rz: Z rotation
    - PhasedX: Phased X rotation (combines Rx, Ry)
    - ZZPhase: Native entangling gate
    - ZZMax: Maximum-entangling ZZ gate (θ=π/2)
    Reference: https://docs.quantinuum.com/h-series/

Google (Sycamore, Willow):
    - PhasedXPow: Phased X rotation
    - Rz: Z rotation (virtual, zero-cost)
    - SYC: Sycamore gate (native 2Q gate, iSWAP-like)
    - CZ: Available on Willow
    Reference: https://quantumai.google/cirq/hardware/devices

IBM (Washington/Eagle r3):
    - Rz: Z rotation (virtual)
    - SX: √X gate
    - X: Pauli-X
    - CX: Native 2Q gate on Washington/Eagle r3
    Reference: https://docs.quantum.ibm.com/

Rigetti (Ankaa-3):
    - Rz: Z rotation (virtual, zero-duration)
    - RX: X rotation (θ = ±π/2 or π)
    - ISWAP: Native 2Q entangling gate (median fidelity 98.31%)
    Reference: https://docs.rigetti.com/

OQC (Lucy, Toshiko):
    - Rz: Z rotation
    - SX: √X gate  
    - ECR: Echoed cross-resonance (native 2Q)
    Reference: https://docs.oqc.tech/

"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Tuple, Optional


class HardwareVendor(Enum):
    """Supported hardware vendors for native compilation."""
    IONQ = auto()
    QUANTINUUM = auto()
    GOOGLE = auto()
    IBM = auto()
    RIGETTI = auto()
    OQC = auto()
    GENERIC = auto()  # Fallback to Qiskit transpiler


@dataclass
class NativeCompilationResult:
    """Results from native hardware compilation."""
    vendor: HardwareVendor
    single_qubit_gates: int
    two_qubit_gates: int
    total_gates: int
    depth: int
    entangling_depth: int
    swap_count: int
    native_gateset: Tuple[str, ...]
    circuit_repr: str  # String representation of compiled circuit
    used_pytket: bool
    fallback_reason: Optional[str] = None


def _detect_vendor(hardware_name: str) -> HardwareVendor:
    """Detect hardware vendor from target name."""
    name_lower = hardware_name.lower()
    if "ionq" in name_lower:
        return HardwareVendor.IONQ
    elif "quantinuum" in name_lower:
        return HardwareVendor.QUANTINUUM
    elif "google" in name_lower or "sycamore" in name_lower or "willow" in name_lower:
        return HardwareVendor.GOOGLE
    elif "ibm" in name_lower or "heavy_hex" in name_lower or "fake_" in name_lower:
        return HardwareVendor.IBM
    elif "rigetti" in name_lower or "aspen" in name_lower or "ankaa" in name_lower:
        return HardwareVendor.RIGETTI
    elif "oqc" in name_lower or "toshiko" in name_lower:
        return HardwareVendor.OQC
    else:
        return HardwareVendor.GENERIC


# Native gateset definitions with descriptive comments
NATIVE_GATESETS = {
    HardwareVendor.IONQ: (
        "GPI",      # General single-qubit gate: U(θ,φ) = [[cos(θ/2), -i*e^{-iφ}*sin(θ/2)], [-i*e^{iφ}*sin(θ/2), cos(θ/2)]]
        "GPI2",     # π/2 rotation: GPI2(φ) = GPI(π/2, φ)
        "ZZPhase",  # Mølmer-Sørensen: exp(-i * θ/2 * ZZ)
    ),
    HardwareVendor.QUANTINUUM: (
        "Rz",       # Z rotation: Rz(θ) = exp(-i * θ/2 * Z)
        "PhasedX",  # Phased X: PhasedX(θ,φ) = Rz(-φ) @ Rx(θ) @ Rz(φ)
        "ZZPhase",  # Parametric ZZ: exp(-i * θ/2 * ZZ)
        "ZZMax",    # Fixed ZZ(π/2): exp(-i * π/4 * ZZ)
    ),
    HardwareVendor.GOOGLE: (
        "PhasedXPow",  # Phased X power gate
        "Rz",          # Z rotation (virtual, zero-duration)
        "SYC",         # Sycamore gate: iSWAP-like with CZ component
        "CZ",          # Controlled-Z (Willow native)
    ),
    HardwareVendor.IBM: (
        "Rz",   # Z rotation (virtual)
        "SX",   # √X gate
        "X",    # Pauli-X
        "CX",   # CNOT (Washington/Eagle r3 native)
    ),
    HardwareVendor.RIGETTI: (
        "Rz",     # Z rotation (virtual, zero-duration)
        "RX",     # X rotation (θ = ±π/2 or π)
        "ISWAP",  # Native 2Q entangling gate
    ),
    HardwareVendor.OQC: (
        "Rz",   # Z rotation
        "SX",   # √X gate
        "ECR",  # Echoed cross-resonance gate
    ),
}


def compile_native_pytket(
    circuit,
    hardware_name: str,
    coupling_map: Optional[list] = None,
    seed: Optional[int] = None,
    placement_strategy: str = "default",
) -> NativeCompilationResult:
    """Compile circuit to native gates using pytket.
    
    Parameters
    ----------
    circuit
        Input circuit to compile. May be either a Qiskit `QuantumCircuit` or a
        pytket `Circuit`.
    hardware_name : str
        Target hardware name (e.g., 'ionq_10', 'quantinuum_16').
    coupling_map : list | None
        Coupling map for routing. None for all-to-all connectivity.
    seed : int | None
        Accepted for API compatibility with higher-level orchestration. The
        current pytket mapping passes do not expose or consume this seed, so
        native mapping/routing remains heuristic.
    placement_strategy : str
        Routing/placement strategy for constrained topologies:
        - ``"default"``: pytket ``DefaultMappingPass`` (recommended paper mode).
        - ``"graph_first"``: GraphPlacement followed by RoutingPass.
    
    Returns
    -------
    NativeCompilationResult
        Compilation results with native gate counts.
    """
    try:
        from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
        from pytket.passes import (
            DefaultMappingPass,
            DecomposeBoxes,
            RemoveRedundancies,
        )
        try:
            from pytket.passes import auto_rebase_pass
            _rebase_factory = auto_rebase_pass
        except ImportError:
            from pytket.passes import AutoRebase
            _rebase_factory = AutoRebase
        from pytket.architecture import Architecture
        from pytket.circuit import Circuit, OpType
        from pytket.transform import Transform
        
        PYTKET_AVAILABLE = True
    except ImportError:
        PYTKET_AVAILABLE = False
    
    vendor = _detect_vendor(hardware_name)
    native_gateset = NATIVE_GATESETS.get(vendor, ("Rz", "SX", "CX"))
    
    if not PYTKET_AVAILABLE:
        # Fallback to Qiskit-only compilation
        return _compile_qiskit_fallback(
            circuit,
            vendor,
            native_gateset,
            coupling_map,
            fallback_reason="pytket is not installed",
        )

    # Accept either a pytket Circuit directly or a Qiskit circuit that needs
    # conversion. This avoids a pytket -> qiskit -> pytket round-trip when the
    # logical circuit was already defined with pytket.
    if isinstance(circuit, Circuit):
        tk_circuit = circuit.copy()
        qiskit_source = None
    else:
        tk_circuit = qiskit_to_tk(circuit)
        qiskit_source = circuit
    
    # Decompose any boxes first
    DecomposeBoxes().apply(tk_circuit)

    swap_count = 0
    if coupling_map:
        # Route inside pytket so the hardware-native and FC-native paths stay
        # in the same compiler stack before native rebasing.
        try:
            architecture = Architecture(coupling_map)
            from pytket.passes import RoutingPass
            
            def _apply_seeded_or_default_mapping() -> None:
                """Apply default algorithmic mapping/routing behavior."""
                # In pytket 2.15 this pass gives placement a 100 ms search limit.
                # The circuit seed does not control that search, so repeated
                # compilation can give different placements and depths.
                DefaultMappingPass(architecture).apply(tk_circuit)

            if placement_strategy == "graph_first":
                from pytket.placement import GraphPlacement
                try:
                    GraphPlacement(architecture).place(tk_circuit)
                    RoutingPass(architecture).apply(tk_circuit)
                except Exception:
                    # GraphPlacement can time out on some large/irregular instances.
                    # Fall back to the existing default pytket mapping path.
                    _apply_seeded_or_default_mapping()
            else:
                _apply_seeded_or_default_mapping()
            swap_count = tk_circuit.n_gates_of_type(OpType.SWAP)
        except Exception as exc:
            if qiskit_source is None:
                return NativeCompilationResult(
                    vendor=vendor,
                    single_qubit_gates=0,
                    two_qubit_gates=0,
                    total_gates=0,
                    depth=0,
                    entangling_depth=0,
                    swap_count=0,
                    native_gateset=native_gateset,
                    circuit_repr="",
                    used_pytket=False,
                    fallback_reason=f"pytket routing failed: {exc}",
                )
            return _compile_qiskit_fallback(
                qiskit_source,
                vendor,
                native_gateset,
                coupling_map,
                fallback_reason=f"pytket routing failed: {exc}",
            )
    
    # Define target gateset based on vendor
    if vendor == HardwareVendor.IONQ:
        # IonQ: GPI, GPI2, ZZPhase
        target_gates = {OpType.GPI, OpType.GPI2, OpType.ZZPhase}
    elif vendor == HardwareVendor.QUANTINUUM:
        # Quantinuum: Rz, PhasedX, ZZPhase
        target_gates = {OpType.Rz, OpType.PhasedX, OpType.ZZPhase}
    elif vendor == HardwareVendor.GOOGLE:
        # Shared Google approximation for Sycamore and Willow: allow the
        # available phased-X operation together with both SYC (when exposed)
        # and CZ.
        phased_x_op = getattr(OpType, "PhasedXPow", None) or getattr(OpType, "PhasedX", None)
        sycamore_op = getattr(OpType, "SYC", None) or getattr(OpType, "Sycamore", None)
        target_gates = {OpType.Rz, OpType.CZ}
        if phased_x_op is not None:
            target_gates.add(phased_x_op)
        if sycamore_op is not None:
            target_gates.add(sycamore_op)
    elif vendor == HardwareVendor.IBM:
        # IBM Washington/Eagle r3: Rz, SX, X, CX
        target_gates = {OpType.Rz, OpType.SX, OpType.X, OpType.CX}
    elif vendor == HardwareVendor.RIGETTI:
        # Rigetti: Rz, Rx, ISWAP
        iswap_op = getattr(OpType, "ISWAP", None)
        if iswap_op is None:
            if qiskit_source is None:
                return NativeCompilationResult(
                    vendor=vendor,
                    single_qubit_gates=0,
                    two_qubit_gates=0,
                    total_gates=0,
                    depth=0,
                    entangling_depth=0,
                    swap_count=swap_count,
                    native_gateset=native_gateset,
                    circuit_repr=str(tk_circuit),
                    used_pytket=False,
                    fallback_reason="pytket build does not expose OpType.ISWAP",
                )
            return _compile_qiskit_fallback(
                qiskit_source,
                vendor,
                native_gateset,
                coupling_map,
                fallback_reason="pytket build does not expose OpType.ISWAP",
            )
        target_gates = {OpType.Rz, OpType.Rx, iswap_op}
    elif vendor == HardwareVendor.OQC:
        # OQC: Rz, SX, ECR
        target_gates = {OpType.Rz, OpType.SX, OpType.ECR}
    else:
        # Generic fallback
        target_gates = {OpType.Rz, OpType.SX, OpType.CX}
    
    # Routing, when requested, is handled inside this function before rebasing.
    # We intentionally stop after native rebasing plus light cleanup instead of
    # applying stronger resynthesis passes that would erase vendor differences.
    
    # Apply native gate rebasing
    try:
        rebase = _rebase_factory(target_gates)
        rebase.apply(tk_circuit)
    except Exception:
        # If auto_rebase fails, try manual decomposition
        Transform.RebaseToTket().apply(tk_circuit)
    
    # Keep only light cleanup here. FullPeepholeOptimise resynthesizes into
    # TK1/CX-like bases and destroys the vendor-native distinction we need for
    # meaningful cross-hardware comparisons.
    try:
        RemoveRedundancies().apply(tk_circuit)
    except Exception:
        pass
    
    # Count gates
    single_q = 0
    two_q = 0
    for cmd in tk_circuit.get_commands():
        n_qubits = cmd.op.n_qubits
        if n_qubits == 1:
            single_q += 1
        elif n_qubits == 2:
            two_q += 1
    
    total = single_q + two_q
    depth = tk_circuit.depth()
    entangling_depth = _qiskit_entangling_depth(tk_to_qiskit(tk_circuit))
    
    return NativeCompilationResult(
        vendor=vendor,
        single_qubit_gates=single_q,
        two_qubit_gates=two_q,
        total_gates=total,
        depth=depth,
        entangling_depth=entangling_depth,
        swap_count=swap_count,
        native_gateset=native_gateset,
        circuit_repr=str(tk_circuit),
        used_pytket=True,
    )


def _compile_qiskit_fallback(
    qiskit_circuit,
    vendor: HardwareVendor,
    native_gateset: Tuple[str, ...],
    coupling_map: Optional[list],
    fallback_reason: Optional[str] = None,
) -> NativeCompilationResult:
    """Fallback compilation using Qiskit when pytket is not available."""
    from qiskit import transpile
    from qiskit.transpiler import CouplingMap
    
    # Map vendor to Qiskit basis gates (approximation)
    basis_map = {
        HardwareVendor.IONQ: ["rx", "ry", "rz", "rxx"],
        HardwareVendor.QUANTINUUM: ["rz", "rx", "ry", "rzz"],
        HardwareVendor.GOOGLE: ["rz", "rx", "ry", "cz"],
        HardwareVendor.IBM: ["rz", "sx", "x", "cx"],  # Washington/Eagle r3 uses CX
        HardwareVendor.RIGETTI: ["rz", "rx", "iswap"],  # Ankaa uses ISWAP
        HardwareVendor.OQC: ["rz", "sx", "ecr"],
        HardwareVendor.GENERIC: ["rz", "sx", "cx"],
    }
    
    basis = basis_map.get(vendor, basis_map[HardwareVendor.GENERIC])
    cm = CouplingMap(coupling_map) if coupling_map else None
    
    compiled = transpile(
        qiskit_circuit,
        basis_gates=basis,
        coupling_map=cm,
        optimization_level=2,
    )
    
    # Count gates
    ops = compiled.count_ops()
    single_q = sum(v for k, v in ops.items() if k in {"rx", "ry", "rz", "sx", "x", "h", "s", "t"})
    two_q = sum(v for k, v in ops.items() if k in {"cx", "cz", "rxx", "rzz", "ecr", "swap", "iswap"})
    
    return NativeCompilationResult(
        vendor=vendor,
        single_qubit_gates=single_q,
        two_qubit_gates=two_q,
        total_gates=single_q + two_q,
        depth=compiled.depth(),
        entangling_depth=_qiskit_entangling_depth(compiled),
        swap_count=int(ops.get("swap", 0)),
        native_gateset=native_gateset,
        circuit_repr=str(compiled),
        used_pytket=False,
        fallback_reason=fallback_reason,
    )


def get_native_gateset_description(vendor: HardwareVendor) -> str:
    """Get human-readable description of native gateset for a vendor."""
    descriptions = {
        HardwareVendor.IONQ: "IonQ: GPI, GPI2 (1Q) + ZZPhase/MS (2Q)",
        HardwareVendor.QUANTINUUM: "Quantinuum: Rz, PhasedX (1Q) + ZZPhase (2Q)",
        HardwareVendor.GOOGLE: "Google: Rz, PhasedXPow (1Q) + SYC/CZ (2Q)",
        HardwareVendor.IBM: "IBM Washington: Rz, SX, X (1Q) + CX (2Q)",
        HardwareVendor.RIGETTI: "Rigetti Ankaa: Rz, RX (1Q) + ISWAP (2Q)",
        HardwareVendor.OQC: "OQC: Rz, SX (1Q) + ECR (2Q)",
        HardwareVendor.GENERIC: "Generic: Rz, SX (1Q) + CX (2Q)",
    }
    return descriptions.get(vendor, descriptions[HardwareVendor.GENERIC])


def _qiskit_entangling_depth(qiskit_circuit) -> int:
    """Return the layer depth counting only two-qubit operations."""
    return qiskit_circuit.depth(
        filter_function=lambda instruction: instruction.operation.num_qubits == 2
    )
