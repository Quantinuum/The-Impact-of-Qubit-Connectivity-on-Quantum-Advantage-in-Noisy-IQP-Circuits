#!/usr/bin/env python3
"""Regenerate sampled phase-diagram PDFs from existing CSV data only.

This script does not run compilation experiments. It only:
1) reads existing phase-diagram CSV files,
2) aggregates mean/std per hardware,
3) redraws the PDF phase diagrams.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Ensure local src/ is importable when script is run from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from iqp_swap_benchmark.analysis import plot_phase_diagram
from iqp_swap_benchmark.colors import set_device_label_mode
from iqp_swap_benchmark.noise_model import critical_depth

P_MIN = 1e-6
P_MAX = 0.055
P_CRITICAL_MIN = 1e-6


def _suffix_parts(backend: str, tag: str | None) -> str:
    parts: list[str] = []
    if backend != "qiskit":
        parts.append(backend)
    if tag:
        parts.append(tag)
    return "".join(f"_{p}" for p in parts)


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


def _plot_from_csv(
    csv_path: Path,
    pdf_path: Path,
    *,
    n_qubits: int,
    n_layers: int,
    use_log_scale: bool,
    y_max: float | None = None,
    y_min: float | None = None,
) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    hw_depths = _build_hw_depths(df)

    if y_max is None:
        max_compiled = max(v[2] + v[3] for v in hw_depths.values())
        max_crit = critical_depth(P_CRITICAL_MIN, 2, c=1.0)
        y_max = (
            max(max_compiled, max_crit) * 1.1
            if use_log_scale
            else max_compiled * 1.15
        )

    fig = plot_phase_diagram(
        output_path=pdf_path,
        k=2,
        c=1.0,
        p_range=(P_MIN, P_MAX),
        p_critical_min=P_CRITICAL_MIN,
        depth_range=(1 if y_min is None else y_min, y_max),
        use_log_scale=use_log_scale,
        x_log_scale=True,
        hardware_depths=hw_depths,
        n_qubits=n_qubits,
        n_layers=n_layers,
    )
    plt.close(fig)


def _required_y_max(csv_path: Path, *, use_log_scale: bool) -> float:
    """Return the normal per-plot y maximum before sharing it across plots."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")

    hw_depths = _build_hw_depths(pd.read_csv(csv_path))
    max_compiled = max(v[2] + v[3] for v in hw_depths.values())
    max_crit = critical_depth(P_CRITICAL_MIN, 2, c=1.0)
    return (
        max(max_compiled, max_crit) * 1.1
        if use_log_scale
        else max_compiled * 1.15
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate sampled phase-diagram plots from existing CSV files only."
    )
    parser.add_argument("--n-qubits", type=int, default=16, help="Qubits for dense/sparse/local.")
    parser.add_argument("--n-layers", type=int, default=1, help="IQP layer count.")
    parser.add_argument(
        "--logical-circuit-backend",
        choices=["qiskit", "pytket"],
        default="qiskit",
        help="Frontend suffix used in filenames.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="samples10_mean",
        help="Suffix tag used in filenames (default: samples10_mean).",
    )
    parser.add_argument(
        "--linear-scale",
        action="store_true",
        help="Use linear y-axis (default is logarithmic).",
    )
    parser.add_argument(
        "--skip-rhg",
        action="store_true",
        help="Skip RHG plots.",
    )
    parser.add_argument(
        "--include-rhg127",
        action="store_true",
        help="Also regenerate RHG n=127 plot.",
    )
    parser.add_argument(
        "--rename",
        action="store_true",
        help="Use anonymized hardware labels (FC_1, FC_2, SC_1, ...).",
    )
    parser.add_argument(
        "--shared-y-scale",
        action="store_true",
        help="Use one common y-axis maximum across every selected plot.",
    )
    parser.add_argument(
        "--linear-bottom-padding-fraction",
        type=float,
        default=0.0,
        help=(
            "Reserve this fraction of the shared linear y maximum below zero "
            "so shallow points remain visible (requires --linear-scale and "
            "--shared-y-scale)."
        ),
    )
    args = parser.parse_args()

    tag = args.tag.strip() if args.tag is not None else ""
    if tag and not re.fullmatch(r"[A-Za-z0-9_.-]+", tag):
        raise ValueError("Tag may contain only letters, digits, underscore, dash, or dot.")

    set_device_label_mode("anon" if args.rename else "full")
    use_log_scale = not args.linear_scale
    if not 0.0 <= args.linear_bottom_padding_fraction <= 0.25:
        parser.error("--linear-bottom-padding-fraction must be between 0 and 0.25")
    if args.linear_bottom_padding_fraction and (
        use_log_scale or not args.shared_y_scale
    ):
        parser.error(
            "--linear-bottom-padding-fraction requires --linear-scale "
            "and --shared-y-scale"
        )
    extra = _suffix_parts(args.logical_circuit_backend, tag)

    jobs: list[tuple[Path, Path, int]] = []
    for pattern in ["dense", "sparse", "local"]:
        csv_path = Path(
            f"paper_images/phase_diagram/depth_by_k2_{pattern}_n{args.n_qubits}_layers{args.n_layers}{extra}.csv"
        )
        pdf_path = Path(
            f"paper_images/phase_diagram/phase_diagram_{pattern}_k2_n{args.n_qubits}_layers{args.n_layers}{extra}.pdf"
        )
        jobs.append((csv_path, pdf_path, args.n_qubits))

    if not args.skip_rhg:
        rhg_sizes = [90]
        if args.include_rhg127:
            rhg_sizes.append(127)
        for actual_n in rhg_sizes:
            csv_path = Path(
                f"paper_images/phase_diagram/depth_by_k2_rhg_n{actual_n}_layers{args.n_layers}{extra}.csv"
            )
            pdf_path = Path(
                f"paper_images/phase_diagram/phase_diagram_rhg_k2_n{actual_n}_layers{args.n_layers}{extra}.pdf"
            )
            jobs.append((csv_path, pdf_path, actual_n))

    shared_y_max = None
    if args.shared_y_scale:
        shared_y_max = max(
            _required_y_max(csv_path, use_log_scale=use_log_scale)
            for csv_path, _, _ in jobs
        )
        scale_name = "log" if use_log_scale else "linear"
        print(f"Using shared {scale_name} y maximum: {shared_y_max:.6g}")

    shared_y_min = None
    if args.linear_bottom_padding_fraction:
        assert shared_y_max is not None
        shared_y_min = -shared_y_max * args.linear_bottom_padding_fraction
        print(
            "Using shared linear bottom padding: "
            f"{args.linear_bottom_padding_fraction:.1%} "
            f"(axis minimum {shared_y_min:.6g}; lowest labeled tick remains 0)"
        )

    for csv_path, pdf_path, n_qubits in jobs:
        _plot_from_csv(
            csv_path,
            pdf_path,
            n_qubits=n_qubits,
            n_layers=args.n_layers,
            use_log_scale=use_log_scale,
            y_max=shared_y_max,
            y_min=shared_y_min,
        )
        print(f"Saved {pdf_path}")


if __name__ == "__main__":
    main()
