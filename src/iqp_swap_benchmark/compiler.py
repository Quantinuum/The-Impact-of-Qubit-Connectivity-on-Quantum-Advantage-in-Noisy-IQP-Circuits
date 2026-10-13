"""Compilation helpers that convert IQP gate patterns into logical circuits."""

from __future__ import annotations

import re
import sys
import warnings
from dataclasses import dataclass, field
from typing import Iterable, List, Literal, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap

from .circuit_patterns import GatePattern
from .topologies import (
    HardwareTarget,
    google_sycamore_topology,
    google_willow,
    ibm_washington,
    ionq_forte,
    oqc_toshiko,
    quantinuum_helios,
    rigetti_ankaa,
)

# Track if we've already warned about pytket fallback
_PYTKET_FALLBACK_WARNED = False


@dataclass
class CompilationResult:
    """Compiler outputs for one logical circuit / hardware-target pairing."""

    target: HardwareTarget
    swap_count: int
    single_qubit_gate_count: int
    two_qubit_gate_count: int
    depth: int
    entangling_layer_count: int
    gate_count: int
    baseline_single_qubit_gate_count: int
    baseline_gate_count: int
    baseline_depth: int
    baseline_entangling_layer_count: int
    compiled_with_swaps: object
    compiled_native: object
    used_native_compiler: bool = False  # True if pytket was used
    native_gateset: Tuple[str, ...] = field(default_factory=tuple)


def build_qiskit_iqp_circuit(
    n_qubits: int,
    params: Sequence[float],
    pattern: GatePattern,
    n_layers: int = 1,
) -> QuantumCircuit:
    """Build an IQP circuit in canonical ``H -> D_Z -> H`` form.
    
    Args:
        n_qubits: Number of qubits
        params: Parameters for all layers (length = len(pattern) * n_layers)
        pattern: Gate pattern defining the interactions
        n_layers: Number of times to repeat the diagonal unitary layer
    """
    terms_per_layer = len(pattern)
    expected_params = terms_per_layer * n_layers
    if len(params) != expected_params:
        raise ValueError(
            f"Parameter count {len(params)} does not match expected "
            f"{expected_params} ({terms_per_layer} terms × {n_layers} layers)"
        )
    
    qc = QuantumCircuit(n_qubits, name="iqp")
    # Initial Hadamards
    for wire in range(n_qubits):
        qc.h(wire)
    
    # Apply n_layers of the diagonal unitary
    for layer_idx in range(n_layers):
        start_idx = layer_idx * terms_per_layer
        layer_params = params[start_idx : start_idx + terms_per_layer]
        for theta, wires in zip(layer_params, pattern):
            _append_z_exponential(qc, float(theta), wires)

    # Final Hadamards convert the computational-basis measurement into the
    # X-basis measurement associated with the standard IQP sampling task.
    for wire in range(n_qubits):
        qc.h(wire)
    
    return qc


def build_pytket_iqp_circuit(
    n_qubits: int,
    params: Sequence[float],
    pattern: GatePattern,
    n_layers: int = 1,
):
    """Build an IQP circuit in pytket using canonical ``H -> D_Z -> H`` form."""
    from pytket.circuit import Circuit

    terms_per_layer = len(pattern)
    expected_params = terms_per_layer * n_layers
    if len(params) != expected_params:
        raise ValueError(
            f"Parameter count {len(params)} does not match expected "
            f"{expected_params} ({terms_per_layer} terms × {n_layers} layers)"
        )

    tk_circuit = Circuit(n_qubits)
    for wire in range(n_qubits):
        tk_circuit.H(wire)

    for layer_idx in range(n_layers):
        start_idx = layer_idx * terms_per_layer
        layer_params = params[start_idx : start_idx + terms_per_layer]
        for theta, wires in zip(layer_params, pattern):
            _append_z_exponential_tket(tk_circuit, float(theta), wires)

    for wire in range(n_qubits):
        tk_circuit.H(wire)

    return tk_circuit


def build_iqp_circuit(
    n_qubits: int,
    params: Sequence[float],
    pattern: GatePattern,
    n_layers: int = 1,
    backend: Literal["qiskit", "pytket"] = "qiskit",
):
    """Build a logical IQP circuit using the requested frontend backend."""
    if backend == "qiskit":
        return build_qiskit_iqp_circuit(
            n_qubits=n_qubits,
            params=params,
            pattern=pattern,
            n_layers=n_layers,
        )
    if backend == "pytket":
        return build_pytket_iqp_circuit(
            n_qubits=n_qubits,
            params=params,
            pattern=pattern,
            n_layers=n_layers,
        )
    raise ValueError(f"Unknown IQP circuit backend: {backend}")


def compile_for_target(
    circuit,
    target: HardwareTarget,
    optimization_level: int = 2,
    routing_method: str = "sabre",
    seed: int | None = None,
    use_native_compiler: bool = True,
    require_native_compiler: bool = True,
    full_device_placement: bool = True,
    placement_strategy: str = "default",
) -> CompilationResult:
    """Compile a circuit for a hardware target.
    
    Parameters
    ----------
    circuit
        Input logical circuit to compile. May be a Qiskit `QuantumCircuit` or
        a pytket `Circuit` when the native path is used.
    target : HardwareTarget
        Hardware target with topology and basis gates.
    optimization_level : int
        Qiskit optimization level (0-3), used by the Qiskit fallback path.
    routing_method : str
        Routing algorithm (default: 'sabre').
    seed : int | None
        Seed forwarded for API compatibility and for the Qiskit fallback.
        The current pytket default mapping path does not expose a routing seed.
    use_native_compiler : bool
        If True, try pytket native compilation first.
    require_native_compiler : bool
        If True, raise an error instead of silently falling back when pytket
        native compilation is unavailable or fails.
    full_device_placement : bool
        If True, route/layout on full-device connectivity for real hardware
        targets instead of reduced n-qubit subgraphs.
    placement_strategy : str
        Placement strategy for native pytket path. Supported:
        - "default" (existing behavior)
        - "graph_first" (GraphPlacement then RoutingPass)
    
    Returns
    -------
    CompilationResult
        Compilation results including gate counts and depths.
    """
    global _PYTKET_FALLBACK_WARNED
    
    used_native = False
    native_gateset: Tuple[str, ...] = ()
    if placement_strategy not in {"default", "graph_first"}:
        raise ValueError(
            f"Unknown placement_strategy '{placement_strategy}'. "
            "Use 'default' or 'graph_first'."
        )
    
    # Choose routing/placement topology: full device or requested subgraph
    placement_target = _resolve_placement_target(target, full_device_placement)

    # Try native compilation with pytket for all targets so constrained
    # topologies and FC baselines use the same compiler stack.
    if use_native_compiler:
        try:
            from .native_compiler import compile_native_pytket, _detect_vendor, NATIVE_GATESETS

            # Keep the native path entirely inside pytket so the hardware and
            # FC baselines are both compiled from the same raw logical circuit.
            native_result = compile_native_pytket(
                circuit,
                target.name,
                placement_target.coupling_map,
                seed=seed,
                placement_strategy=placement_strategy,
            )
            swap_count = native_result.swap_count
            
            # Check if pytket was actually used (not fallback)
            vendor = _detect_vendor(target.name)
            native_gateset = NATIVE_GATESETS.get(vendor, ("Rz", "SX", "CX"))
            
            # Keep the FC baseline as the raw logical circuit compiled directly
            # to the native gateset. This is the intended "ideal fully connected"
            # reference, so we do not pre-transpile it through Qiskit first.
            baseline_result = compile_native_pytket(
                circuit,
                target.name,
                None,  # No coupling for baseline
                seed=seed,
                placement_strategy=placement_strategy,
            )

            if not native_result.used_pytket or not baseline_result.used_pytket:
                reason = (
                    native_result.fallback_reason
                    or baseline_result.fallback_reason
                    or "pytket native rebasing unavailable"
                )
                if require_native_compiler:
                    _raise_pytket_required_error(reason)
                if not _PYTKET_FALLBACK_WARNED:
                    _print_pytket_fallback_warning(reason)
                    _PYTKET_FALLBACK_WARNED = True

            used_native = native_result.used_pytket and baseline_result.used_pytket
            
            return CompilationResult(
                target=target,
                swap_count=swap_count,
                single_qubit_gate_count=native_result.single_qubit_gates,
                two_qubit_gate_count=native_result.two_qubit_gates,
                depth=native_result.depth,
                entangling_layer_count=native_result.entangling_depth,
                gate_count=native_result.total_gates,
                baseline_single_qubit_gate_count=baseline_result.single_qubit_gates,
                baseline_gate_count=baseline_result.total_gates,
                baseline_depth=baseline_result.depth,
                baseline_entangling_layer_count=baseline_result.entangling_depth,
                compiled_with_swaps=circuit,
                compiled_native=circuit,
                used_native_compiler=used_native,
                native_gateset=native_gateset,
            )
            
        except ImportError as e:
            if require_native_compiler:
                _raise_pytket_required_error(str(e))
            if not _PYTKET_FALLBACK_WARNED:
                _print_pytket_fallback_warning(str(e))
                _PYTKET_FALLBACK_WARNED = True
        except RuntimeError:
            # Preserve the original explicit native-compiler error message.
            raise
        except Exception as e:
            if require_native_compiler:
                _raise_pytket_required_error(f"Error: {e}")
            if not _PYTKET_FALLBACK_WARNED:
                _print_pytket_fallback_warning(f"Error: {e}")
                _PYTKET_FALLBACK_WARNED = True
    
    # Fallback to Qiskit-only compilation
    if not isinstance(circuit, QuantumCircuit):
        raise TypeError(
            "Qiskit-only fallback requires a Qiskit QuantumCircuit input. "
            "Use require_native_compiler=True with pytket logical circuits."
        )
    basis_for_count = list(target.basis_with_swap)
    coupling = _to_coupling_map(placement_target.coupling_map)
    # Keep the FC baseline as the raw logical circuit with no connectivity
    # constraint so it remains the ideal fully connected reference.
    baseline_native = transpile(
        circuit,
        basis_gates=list(target.basis_gates),
        coupling_map=None,
        optimization_level=optimization_level,
        routing_method=routing_method,
        seed_transpiler=seed,
    )
    baseline_gate_count = int(baseline_native.size())
    baseline_depth = baseline_native.depth()
    compiled_swaps = transpile(
        circuit,
        basis_gates=basis_for_count,
        coupling_map=coupling,
        optimization_level=optimization_level,
        routing_method=routing_method,
        seed_transpiler=seed,
    )
    swap_count = int(compiled_swaps.count_ops().get("swap", 0))
    compiled_native = transpile(
        compiled_swaps,
        basis_gates=list(target.basis_gates),
        coupling_map=coupling,
        optimization_level=optimization_level,
        routing_method=routing_method,
        seed_transpiler=seed,
    )
    two_qubit_count = _count_two_qubit_ops(compiled_native)
    gate_count = int(compiled_native.size())
    return CompilationResult(
        target=target,
        swap_count=swap_count,
        single_qubit_gate_count=gate_count - two_qubit_count,
        two_qubit_gate_count=two_qubit_count,
        depth=compiled_native.depth(),
        entangling_layer_count=_qiskit_entangling_depth(compiled_native),
        gate_count=gate_count,
        baseline_single_qubit_gate_count=baseline_gate_count - _count_two_qubit_ops(baseline_native),
        baseline_gate_count=baseline_gate_count,
        baseline_depth=baseline_depth,
        baseline_entangling_layer_count=_qiskit_entangling_depth(baseline_native),
        compiled_with_swaps=compiled_swaps,
        compiled_native=compiled_native,
        used_native_compiler=False,
        native_gateset=target.basis_gates,
    )


def _print_pytket_fallback_warning(reason: str) -> None:
    """Print a big warning when falling back to Qiskit compilation."""
    msg = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        ⚠️  NATIVE COMPILATION FAILED  ⚠️                       ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  pytket native compilation is NOT available. Falling back to Qiskit.         ║
║                                                                              ║
║  Reason: {reason[:60]:<60} ║
║                                                                              ║
║  This means gate counts may not reflect true native hardware gatesets:       ║
║    • IonQ: GPI, GPI2, ZZPhase will be approximated as rx, ry, rxx           ║
║    • Quantinuum: Rz, PhasedX, ZZPhase will be approximated as rz, rzz       ║
║    • Google: SYC gate will be approximated as CZ                             ║
║                                                                              ║
║  To enable native compilation:                                               ║
║    pip install pytket pytket-qiskit                                          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(msg, file=sys.stderr)


def _raise_pytket_required_error(reason: str) -> None:
    """Raise a loud error when native compilation is required."""
    raise RuntimeError(
        "\n"
        "================================ PYTKET REQUIRED ================================\n"
        "Native compilation was required for this experiment, but the run would have\n"
        "fallen back to approximate Qiskit compilation instead.\n\n"
        f"Reason: {reason}\n\n"
        "This repository's paper-facing benchmark path is intended to use pytket-native\n"
        "routing/rebasing for both the hardware-constrained circuit and the native-FC\n"
        "baseline. To continue only for debugging, call:\n\n"
        "    compile_for_target(..., require_native_compiler=False)\n\n"
        "To fix the environment, install pytket and pytket-qiskit in the active env.\n"
        "================================================================================\n"
    )


def _append_z_exponential(qc: QuantumCircuit, theta: float, wires: Tuple[int, ...]) -> None:
    """Append e^{+i * theta * Z_1 ⊗ Z_2 ⊗ ... ⊗ Z_k} to the circuit.

    For k qubits, this is implemented as:
    1. Direct ``RZ`` for 1-body terms
    2. Direct ``RZZ`` for 2-body terms
    3. CNOT ladder + ``RZ`` + reverse ladder for higher-body terms
    """
    if len(wires) == 1:
        qc.rz(-2.0 * theta, wires[0])
        return

    # Multi-qubit Z rotation: CNOT ladder → RZ → reverse CNOT ladder
    # With Qiskit's rotation convention, -2*theta implements e^{+i theta Z...Z}.
    if len(wires) == 2:
        # Optimized 2-qubit case using diagonal RZZ directly
        qc.rzz(-2.0 * theta, wires[0], wires[1])
    else:
        # General k-qubit case: CNOT ladder decomposition
        # Forward CNOT ladder
        for i in range(len(wires) - 1):
            qc.cx(wires[i], wires[i + 1])
        # RZ on the last qubit
        qc.rz(-2.0 * theta, wires[-1])
        # Reverse CNOT ladder
        for i in range(len(wires) - 2, -1, -1):
            qc.cx(wires[i], wires[i + 1])


def _append_z_exponential_tket(tk_circuit, theta: float, wires: Tuple[int, ...]) -> None:
    """Append e^{+i * theta * Z_1 ⊗ ... ⊗ Z_k} using pytket operations.

    pytket parameterized rotation gates use angles in units of pi, so we divide
    radian-valued `theta` by `np.pi` before inserting `Rz`/`ZZPhase`.
    """
    angle = -2.0 * theta / np.pi
    if len(wires) == 1:
        tk_circuit.Rz(angle, wires[0])
        return
    if len(wires) == 2:
        tk_circuit.ZZPhase(angle, wires[0], wires[1])
        return
    for i in range(len(wires) - 1):
        tk_circuit.CX(wires[i], wires[i + 1])
    tk_circuit.Rz(angle, wires[-1])
    for i in range(len(wires) - 2, -1, -1):
        tk_circuit.CX(wires[i], wires[i + 1])


def _count_two_qubit_ops(qc: QuantumCircuit) -> int:
    """Count non-measurement two-qubit operations in a compiled circuit."""
    return int(
        sum(
            1
            for instr, qargs, _ in qc.data
            if len(qargs) == 2 and instr.name not in {"measure", "barrier"}
        )
    )


def _qiskit_entangling_depth(qc: QuantumCircuit) -> int:
    """Return depth after filtering to only two-qubit instructions."""
    return qc.depth(
        filter_function=lambda instruction: instruction.operation.num_qubits == 2
    )


def _to_coupling_map(coupling) -> CouplingMap | None:
    """Normalize a coupling-map-like input to Qiskit's `CouplingMap`."""
    if coupling is None:
        return None
    if isinstance(coupling, CouplingMap):
        return coupling
    return CouplingMap(coupling)


def _fully_connected_coupling(n_qubits: int) -> list[tuple[int, int]]:
    """Return a bidirectional all-to-all coupling map for n qubits."""
    return [(i, j) for i in range(n_qubits) for j in range(n_qubits) if i != j]


_FULL_DEVICE_QUBITS = {
    "ionq_forte": 36,
    "quantinuum_helios": 98,
    "google_sycamore": 53,
    "google_willow": 105,
    "ibm_washington": 127,
    "rigetti_ankaa": 82,
    "oqc_toshiko": 32,
}


def _extract_hardware_base(name: str) -> str:
    """Strip qubit/count suffixes from hardware names."""
    base = re.sub(r"_\d+x\d+", "", name)
    base = re.sub(r"_\d+$", "", base)
    return base


def _resolve_placement_target(target: HardwareTarget, full_device_placement: bool) -> HardwareTarget:
    """Return a target used for routing/layout.

    For real hardware, this can upgrade an n-qubit subgraph target to the full
    device connectivity to allow global layout optimization. This affects only
    placement/routing; the logical circuit size is unchanged.
    """
    if not full_device_placement:
        return target

    base = _extract_hardware_base(target.name)
    full_n = _FULL_DEVICE_QUBITS.get(base)
    if full_n is None:
        return target

    # If already full (or larger), keep as-is.
    if target.coupling_map is not None:
        max_idx = max((max(a, b) for a, b in target.coupling_map), default=-1)
        if max_idx + 1 >= full_n:
            return target

    full_target: HardwareTarget | None
    if base == "ionq_forte":
        full_target = ionq_forte(full_n)
    elif base == "quantinuum_helios":
        full_target = quantinuum_helios(full_n)
    elif base == "google_sycamore":
        full_target = google_sycamore_topology(full_n)
    elif base == "google_willow":
        full_target = google_willow(full_n)
    elif base == "ibm_washington":
        full_target = ibm_washington(full_n)
    elif base == "rigetti_ankaa":
        full_target = rigetti_ankaa(full_n)
    elif base == "oqc_toshiko":
        full_target = oqc_toshiko(full_n)
    else:
        return target

    if full_target is None or full_target.coupling_map is None:
        return target
    return full_target
