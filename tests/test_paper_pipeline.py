"""Exercise the graph builders and native compiler."""

import numpy as np
import pytest

from iqp_swap_benchmark.circuit_patterns import PatternConfig, build_gate_pattern
from iqp_swap_benchmark.compiler import build_iqp_circuit, compile_for_target
from iqp_swap_benchmark.topologies import get_hardware_choices


@pytest.mark.parametrize("pattern,n,edges", [
    ("dense", 32, 496), ("local", 32, 31),
    ("rhg", 18, 24), ("rhg", 44, 64), ("rhg", 90, 144), ("rhg", 127, 208),
])
def test_paper_interaction_graphs(pattern, n, edges):
    terms = build_gate_pattern(PatternConfig(kind=pattern, n_qubits=n, interaction_order=2))
    assert sum(len(term) == 2 for term in terms) == edges
    assert all(0 <= wire < n for term in terms for wire in term)


def test_sparse_sampling_is_reproducible():
    config = PatternConfig(kind="sparse", n_qubits=32, interaction_order=2, density=0.4, seed=42)
    terms = build_gate_pattern(config)
    assert terms == build_gate_pattern(config)
    assert len(terms) == 212  # ceil(0.4 * (32 one-body + 496 pairwise terms))


@pytest.mark.parametrize("target", get_hardware_choices(4), ids=lambda target: target.name)
def test_native_compilation_on_every_target(target):
    terms = build_gate_pattern(PatternConfig(kind="dense", n_qubits=4, interaction_order=2))
    circuit = build_iqp_circuit(4, np.linspace(-0.4, 0.4, len(terms)), terms)
    result = compile_for_target(circuit, target, require_native_compiler=True)
    assert result.used_native_compiler
    assert result.depth > 0
    assert result.baseline_depth > 0
    assert result.gate_count == result.single_qubit_gate_count + result.two_qubit_gate_count
    if target.coupling_map is None:
        assert result.depth == result.baseline_depth
