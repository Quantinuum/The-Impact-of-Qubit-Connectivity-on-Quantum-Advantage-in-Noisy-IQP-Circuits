"""Check archived experiments against the numerical claims in the paper."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PHASE = ROOT / "paper_images/phase_diagram"
TAG = "samples20_mean_standard_rerun"
DEVICES = ["quantinuum_helios", "ionq_forte", "google_willow", "google_sycamore",
           "ibm_washington", "rigetti_ankaa", "oqc_toshiko"]
TABLE = {
    "dense": ([66, 79, 1839, 3357, 1028, 922, 3148], [1, 1, 3.7, 6.8, 5.4, 5.0, 7.3]),
    "sparse": ([47, 60, 1298, 1501, 679, 542, 1837], [1, 1, 3.7, 4.3, 5.1, 4.1, 6.0]),
    "local": ([36, 49, 254, 254, 99, 96, 254], [1] * 7),
}


def phase(pattern, n):
    return pd.read_csv(PHASE / f"depth_by_k2_{pattern}_n{n}_layers1_{TAG}.csv")


def test_archived_artifact_checksums():
    manifest = json.loads((ROOT / "results/artifacts.json").read_text())
    for item in manifest["artifacts"]:
        assert hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest() == item["sha256"], item["path"]
    assert len([a for a in manifest["artifacts"] if a["role"] == "paper figure"]) == 23


@pytest.mark.parametrize("pattern", TABLE)
def test_paper_table(pattern):
    means = phase(pattern, 32).groupby("hardware")[["depth", "native_fc_depth"]].mean()
    depths, overheads = TABLE[pattern]
    for device, depth, overhead in zip(DEVICES, depths, overheads):
        row = means.loc[f"{device}_32"]
        assert round(row.depth) == depth
        assert round(row.depth / row.native_fc_depth, 1) == overhead


@pytest.mark.parametrize("n,device,depth,overhead", [
    (90, "quantinuum_helios", 19, 1.0),
    (90, "google_willow", 614, 5.3),
    (90, "ibm_washington", 550, 11.5),
    (127, "ibm_washington", 1128, 20.9),
])
def test_rhg_paper_values(n, device, depth, overhead):
    means = phase("rhg", n).groupby("hardware")[["depth", "native_fc_depth"]].mean()
    row = means.loc[f"{device}_{n}"]
    assert round(row.depth) == depth
    assert round(row.depth / row.native_fc_depth, 1) == overhead


@pytest.mark.parametrize("path", sorted((ROOT / "paper_images").rglob("*.csv")))
def test_complete_samples_and_ratios(path):
    df = pd.read_csv(path)
    keys = ["pattern", "hardware"] + (["n_qubits"] if "n_qubits" in df else [])
    samples = 10 if "compilation_overhead" in str(path) else 20
    assert df.groupby(keys).size().eq(samples).all()
    for _, group in df.groupby(keys):
        assert sorted(group.circuit_index) == list(range(samples))
    assert df.logical_circuit_backend.eq("qiskit").all()
    assert df.n_layers.eq(1).all()
    assert np.isfinite(df[["depth", "native_fc_depth", "depth_ratio_native_fc"]]).all().all()
    assert (df[["depth", "native_fc_depth"]] > 0).all().all()
    np.testing.assert_allclose(df.depth_ratio_native_fc, df.native_fc_depth / df.depth)
    np.testing.assert_allclose(df.gate_count, df.single_qubit_gate_count + df.two_qubit_gate_count)


def test_scaling_grid():
    base = ROOT / "paper_images/compilation_overhead"
    standard = pd.read_csv(base / "results_default.csv")
    rhg = pd.read_csv(base / "results_rhg_default.csv")
    assert len(standard) == 3960
    assert len(rhg) == 360
    assert set(standard.pattern) == {"dense", "sparse", "local"}
    assert sorted(standard.n_qubits.unique()) == [4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32]
    assert sorted(rhg.n_qubits.unique()) == [18, 44, 90, 127]
    assert standard[standard.pattern == "local"].depth_ratio_native_fc.eq(1).all()


def test_sc3_splice_preserves_existing_reference_columns():
    from scripts.rerun_sc3_compilation_overhead import _splice_rows

    original = pd.read_csv(ROOT / "paper_images/compilation_overhead/results_default.csv")
    replacement = original[original.hardware.str.startswith("ibm_washington_")].copy()
    reference_columns = [c for c in replacement if c.startswith("fully_connected_")
                         or c.endswith("_ratio_fc") or c.endswith("_diff_fc")]
    replacement[reference_columns] = np.nan
    updated = _splice_rows(original, replacement)
    pd.testing.assert_frame_equal(original, updated, check_dtype=False)


def test_sc3_splice_rejects_missing_circuits():
    from scripts.rerun_sc3_compilation_overhead import _splice_rows

    original = pd.read_csv(ROOT / "paper_images/compilation_overhead/results_default.csv")
    replacement = original[original.hardware.str.startswith("ibm_washington_")].iloc[1:]
    with pytest.raises(ValueError, match="row-count mismatch"):
        _splice_rows(original, replacement)


@pytest.mark.parametrize("n", [4, 32])
def test_original_standard_sampling_uses_seed_42(n):
    from iqp_swap_benchmark.analysis import ExperimentConfig, run_swap_experiment

    original = pd.read_csv(ROOT / "paper_images/compilation_overhead/results_default.csv")
    hardware = f"fully_connected_{n}"
    original = original[original.hardware == hardware]
    fresh = run_swap_experiment(ExperimentConfig(
        n_qubits=n, pattern_kinds=["dense", "local", "sparse"],
        circuits_per_pattern=10, hardware_names=[hardware], seed=42,
    ))
    keys = ["pattern", "circuit_index"]
    columns = ["depth", "gate_count", "native_fc_depth", "native_fc_gate_count", "parameter_count"]
    pd.testing.assert_frame_equal(
        original.set_index(keys).sort_index()[columns],
        fresh.set_index(keys).sort_index()[columns], check_exact=True,
    )
