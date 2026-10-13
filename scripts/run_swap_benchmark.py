#!/usr/bin/env python3
"""CLI entry point for generating and plotting IQP SWAP statistics."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd

from iqp_swap_benchmark.analysis import (
    ExperimentConfig,
    plot_swap_results,
    plot_ratio_vs_qubits,
    plot_ratio_vs_qubits_per_pattern,
    plot_phase_diagram,
    run_swap_experiment,
    save_results,
)
from iqp_swap_benchmark.topologies import get_hardware_choices


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-qubits",
        type=int,
        nargs="+",
        default=[6],
        help="Number of qubits to benchmark (space-separated list)",
    )
    parser.add_argument(
        "--circuits-per-pattern",
        type=int,
        default=5,
        help="How many random parameter sets to benchmark per pattern",
    )
    parser.add_argument(
        "--patterns",
        nargs="*",
        default=["dense", "local", "sparse"],
        help="Gate pattern types to include",
    )
    parser.add_argument(
        "--hardware",
        nargs="*",
        default=None,
        help="Specific hardware targets to evaluate. Defaults adapt per n_qubits.",
    )
    parser.add_argument("--density", type=float, default=0.4, help="Density for sparse patterns")
    parser.add_argument(
        "--interaction-order",
        type=int,
        default=2,
        help="Maximum interaction order for IQP terms (2=two-qubit, 3=three-qubit, etc.)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/iqp_swap_counts.csv"),
        help="Path to write the CSV results",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        help="Optional path to save a bar plot of average SWAP counts",
    )
    parser.add_argument(
        "--metric",
        choices=[
            "swap_count",
            "two_qubit_gate_count",
            "gate_count",
            "entangling_layer_count",
            "gate_count_ratio_fc",
            "gate_count_diff_fc",
            "gate_count_ratio_native_fc",
            "gate_count_diff_native_fc",
            "depth",
            "depth_ratio_fc",
            "depth_diff_fc",
            "depth_ratio_native_fc",
            "depth_diff_native_fc",
            "entangling_layer_ratio_fc",
            "entangling_layer_diff_fc",
            "entangling_layer_ratio_native_fc",
            "entangling_layer_diff_native_fc",
        ],
        default="gate_count_ratio_native_fc",
        help="Metric to aggregate in the plot (default: gate_count_ratio_native_fc for same-gateset comparison)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="RNG seed shared across pattern sampling and transpilation",
    )
    parser.add_argument(
        "--optimization-level",
        type=int,
        choices=range(4),
        default=2,
        help="Qiskit transpile optimization level",
    )
    parser.add_argument(
        "--scaling-plot",
        type=Path,
        default=None,
        help="Path prefix for scaling plots (generates one plot per pattern, e.g., prefix_dense.png)",
    )
    parser.add_argument(
        "--extrapolate-to",
        type=int,
        default=70,
        help="Maximum qubit count to extrapolate scaling plots to (default: 70)",
    )
    parser.add_argument(
        "--log-scale",
        action="store_true",
        help="Use logarithmic scale for y-axis in scaling plots",
    )
    parser.add_argument(
        "--phase-diagram",
        type=Path,
        default=None,
        help="Path to save phase diagram plot showing simulatability boundary in (p, D) space",
    )
    parser.add_argument(
        "--logical-circuit-backend",
        choices=["qiskit", "pytket"],
        default="qiskit",
        help="Frontend used to define the logical IQP circuit before compilation",
    )
    parser.add_argument(
        "--placement",
        choices=["default", "graph_first"],
        default="default",
        help="Placement/routing strategy for constrained topologies.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    n_values = sorted(set(args.n_qubits))
    combined_frames = []
    multi = len(n_values) > 1
    output_base = args.output
    plot_base = args.plot

    for n in n_values:
        hardware = args.hardware
        if hardware is None:
            hardware = [t.name for t in get_hardware_choices(n)]

        print(
            f"[run] starting n={n} patterns={args.patterns} hardware={len(hardware)}",
            flush=True,
        )

        config = ExperimentConfig(
            n_qubits=n,
            pattern_kinds=args.patterns,
            circuits_per_pattern=args.circuits_per_pattern,
            hardware_names=hardware,
            interaction_order=args.interaction_order,
            density=args.density,
            seed=args.seed,
            optimization_level=args.optimization_level,
            logical_circuit_backend=args.logical_circuit_backend,
            placement_strategy=args.placement,
        )

        df = run_swap_experiment(config)
        print(f"[run] finished n={n}", flush=True)
        df["n_qubits"] = n
        combined_frames.append(df)
        
        # Only save intermediate results for single-n runs
        if not multi:
            save_results(df, output_base)
            print(f"Saved results for n={n} to {output_base}")

        if plot_base is not None and not multi:
            fig = plot_swap_results(df, plot_base, metric=args.metric, n_qubits=n)
            print(f"Saved plot for n={n} metric '{args.metric}' to {plot_base}")
            plt.close(fig)

    if multi:
        combined = pd.concat(combined_frames, ignore_index=True)
        aggregate_output = output_base
        if not aggregate_output.suffix:
            aggregate_output = aggregate_output / "iqp_swap_counts.csv"
        save_results(combined, aggregate_output)
        print(f"Saved aggregated results to {aggregate_output}")

        # Generate scaling plots (one per pattern) if requested
        if args.scaling_plot is not None:
            # Use the scaling_plot path as directory for per-pattern plots
            output_dir = args.scaling_plot.parent if args.scaling_plot.suffix else args.scaling_plot
            prefix = args.scaling_plot.stem if args.scaling_plot.suffix else "ratio_vs_qubits"
            figs = plot_ratio_vs_qubits_per_pattern(
                combined,
                output_dir,
                metric=args.metric,
                filename_prefix=prefix,
                extrapolate_to=args.extrapolate_to,
                use_log_scale=args.log_scale,
            )
            for fig in figs:
                plt.close(fig)

        # Generate phase diagram if requested
        if args.phase_diagram is not None:
            # Use maximum n_qubits from the experiments for k
            k = max(n_values)
            fig = plot_phase_diagram(
                df=combined,
                output_path=args.phase_diagram,
                k=k,
            )
            plt.close(fig)


if __name__ == "__main__":
    main()
