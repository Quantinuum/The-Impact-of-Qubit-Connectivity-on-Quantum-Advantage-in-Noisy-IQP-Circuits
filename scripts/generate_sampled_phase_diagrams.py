#!/usr/bin/env python3
"""Generate phase-diagram CSVs/PDFs using multi-sample averaging.

This is the paper's multi-sample phase study. It compiles random circuits
for each pattern and plots each hardware point using:
  - mean native FC depth (D_FC)
  - mean compiled depth (D_H)
with error bars from sample standard deviation.

Outputs are tag-suffixed so baseline files are not overwritten.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Force script execution to use this repo's local `src/` package tree first.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from iqp_swap_benchmark.analysis import ExperimentConfig, plot_phase_diagram, run_swap_experiment
from iqp_swap_benchmark.circuit_patterns import rhg_lattice_dims
from iqp_swap_benchmark.colors import set_device_label_mode
from iqp_swap_benchmark.noise_model import critical_depth

P_MIN = 1e-6
P_MAX = 0.055
P_CRITICAL_MIN = 1e-6


_RHG_HARDWARE_LIMITS = {
    "fully_connected": None,
    "quantinuum_helios": 98,
    "ionq_forte": 36,
    "ibm_washington": 127,
    "google_willow": 105,
    "rigetti_ankaa": 82,
    "google_sycamore": 53,
    "oqc_toshiko": 32,
}


def _suffix_parts(backend: str, tag: str | None) -> str:
    parts: list[str] = []
    if backend != "qiskit":
        parts.append(backend)
    if tag:
        parts.append(tag)
    return "".join(f"_{p}" for p in parts)


def _hardware_names(n: int) -> list[str]:
    return [
        f"fully_connected_{n}",
        f"quantinuum_helios_{n}",
        f"ionq_forte_{n}",
        f"google_sycamore_{n}",
        f"google_willow_{n}",
        f"ibm_washington_{n}",
        f"rigetti_ankaa_{n}",
        f"oqc_toshiko_{n}",
        f"line_{n}",
        f"ring_{n}",
        f"ladder_{n}",
    ]


def _rhg_hardware_names(n: int) -> list[str]:
    names: list[str] = []
    for hw, max_q in _RHG_HARDWARE_LIMITS.items():
        if max_q is not None and n > max_q:
            continue
        names.append(f"{hw}_{n}")
    return names


def _build_hw_depths(df: pd.DataFrame) -> dict[str, tuple[float, float, float, float]]:
    hw_depths: dict[str, tuple[float, float, float, float]] = {}
    for hw in sorted(df["hardware"].unique()):
        hw_df = df[df["hardware"] == hw]
        native_d = hw_df["native_fc_depth"].mean()
        native_std = hw_df["native_fc_depth"].std()
        compiled_d = hw_df["depth"].mean()
        compiled_std = hw_df["depth"].std()
        hw_depths[hw] = (
            float(native_d),
            0.0 if pd.isna(native_std) else float(native_std),
            float(compiled_d),
            0.0 if pd.isna(compiled_std) else float(compiled_std),
        )
    return hw_depths


def _plot_from_df(
    df: pd.DataFrame,
    output_path: Path,
    n_qubits: int,
    n_layers: int,
    use_log_scale: bool,
) -> None:
    hw_depths = _build_hw_depths(df)
    max_compiled = max(v[2] + v[3] for v in hw_depths.values())
    max_crit = critical_depth(P_CRITICAL_MIN, 2, c=1.0)
    if use_log_scale:
        y_max = max(max_compiled, max_crit) * 1.1
    else:
        y_max = max_compiled * 1.15

    fig = plot_phase_diagram(
        output_path=output_path,
        k=2,
        c=1.0,
        p_range=(P_MIN, P_MAX),
        p_critical_min=P_CRITICAL_MIN,
        depth_range=(1, y_max),
        use_log_scale=use_log_scale,
        x_log_scale=True,
        hardware_depths=hw_depths,
        n_qubits=n_qubits,
        n_layers=n_layers,
    )
    plt.close(fig)


def _run_pattern(
    pattern: str,
    n_qubits: int,
    n_layers: int,
    circuits_per_pattern: int,
    backend: str,
    placement_strategy: str,
    seed: int,
    tag: str | None,
    use_log_scale: bool,
) -> tuple[Path, Path]:
    config = ExperimentConfig(
        n_qubits=n_qubits,
        pattern_kinds=[pattern],
        circuits_per_pattern=circuits_per_pattern,
        hardware_names=_hardware_names(n_qubits),
        interaction_order=2,
        density=0.4,
        seed=seed,
        optimization_level=2,
        n_layers=n_layers,
        logical_circuit_backend=backend,
        placement_strategy=placement_strategy,
    )
    df = run_swap_experiment(config)

    extra = _suffix_parts(backend, tag)
    csv_path = Path(
        f"paper_images/phase_diagram/depth_by_k2_{pattern}_n{n_qubits}_layers{n_layers}{extra}.csv"
    )
    pdf_path = Path(
        f"paper_images/phase_diagram/phase_diagram_{pattern}_k2_n{n_qubits}_layers{n_layers}{extra}.pdf"
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    _plot_from_df(df, pdf_path, n_qubits=n_qubits, n_layers=n_layers, use_log_scale=use_log_scale)
    return csv_path, pdf_path


def _run_rhg(
    target_n: int,
    n_layers: int,
    circuits_per_pattern: int,
    backend: str,
    placement_strategy: str,
    seed: int,
    tag: str | None,
    use_log_scale: bool,
) -> tuple[Path, Path]:
    _, _, _, actual_n = rhg_lattice_dims(target_n)
    config = ExperimentConfig(
        n_qubits=actual_n,
        pattern_kinds=["rhg"],
        circuits_per_pattern=circuits_per_pattern,
        hardware_names=_rhg_hardware_names(actual_n),
        interaction_order=2,
        density=1.0,
        seed=seed,
        optimization_level=2,
        n_layers=n_layers,
        logical_circuit_backend=backend,
        placement_strategy=placement_strategy,
    )
    df = run_swap_experiment(config)

    extra = _suffix_parts(backend, tag)
    csv_path = Path(
        f"paper_images/phase_diagram/depth_by_k2_rhg_n{actual_n}_layers{n_layers}{extra}.csv"
    )
    pdf_path = Path(
        f"paper_images/phase_diagram/phase_diagram_rhg_k2_n{actual_n}_layers{n_layers}{extra}.pdf"
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    _plot_from_df(df, pdf_path, n_qubits=actual_n, n_layers=n_layers, use_log_scale=use_log_scale)
    return csv_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate averaged phase-diagram data/plots with multi-sample circuits."
    )
    parser.add_argument(
        "--n-qubits",
        type=int,
        default=16,
        help="Qubit count for dense/local/sparse patterns (default: 16).",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=1,
        help="Number of IQP layers (default: 1).",
    )
    parser.add_argument(
        "--circuits-per-pattern",
        type=int,
        default=5,
        help="Number of random circuits per pattern family (default: 5).",
    )
    parser.add_argument(
        "--logical-circuit-backend",
        choices=["qiskit", "pytket"],
        default="qiskit",
        help="Frontend used to define the logical IQP circuit.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Experiment RNG seed (default: 42).",
    )
    parser.add_argument(
        "--placement",
        choices=["default", "graph_first"],
        default="default",
        help="Placement/routing strategy for constrained topologies.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="samples5_mean",
        help="Extra suffix tag for output filenames (default: samples5_mean).",
    )
    parser.add_argument(
        "--linear-scale",
        action="store_true",
        help="Use linear y-axis (default is logarithmic).",
    )
    parser.add_argument(
        "--skip-rhg",
        action="store_true",
        help="Skip RHG phase-diagram generation.",
    )
    parser.add_argument(
        "--rename",
        action="store_true",
        help="Use anonymized hardware labels (FC_1, FC_2, SC_1, ...).",
    )
    args = parser.parse_args()
    set_device_label_mode("anon" if args.rename else "full")

    use_log_scale = not args.linear_scale
    tag = args.tag.strip() if args.tag is not None else ""
    if tag and not re.fullmatch(r"[A-Za-z0-9_.-]+", tag):
        raise ValueError("Tag may contain only letters, digits, underscore, dash, or dot.")

    print("=" * 80)
    print("Generating averaged phase diagrams")
    print(
        f"backend={args.logical_circuit_backend} "
        f"circuits_per_pattern={args.circuits_per_pattern} "
        f"n_layers={args.n_layers} "
        f"placement={args.placement} "
        f"seed={args.seed} "
        f"tag={tag or '(none)'}"
    )
    print("=" * 80)

    for pattern in ["dense", "sparse", "local"]:
        csv_path, pdf_path = _run_pattern(
            pattern=pattern,
            n_qubits=args.n_qubits,
            n_layers=args.n_layers,
            circuits_per_pattern=args.circuits_per_pattern,
            backend=args.logical_circuit_backend,
            placement_strategy=args.placement,
            seed=args.seed,
            tag=tag,
            use_log_scale=use_log_scale,
        )
        print(f"[{pattern}] csv={csv_path}")
        print(f"[{pattern}] pdf={pdf_path}")

    if not args.skip_rhg:
        for rhg_target in [90, 127]:
            csv_path, pdf_path = _run_rhg(
                target_n=rhg_target,
                n_layers=args.n_layers,
                circuits_per_pattern=args.circuits_per_pattern,
                backend=args.logical_circuit_backend,
                placement_strategy=args.placement,
                seed=args.seed,
                tag=tag,
                use_log_scale=use_log_scale,
            )
            print(f"[rhg-{rhg_target}] csv={csv_path}")
            print(f"[rhg-{rhg_target}] pdf={pdf_path}")


if __name__ == "__main__":
    main()
