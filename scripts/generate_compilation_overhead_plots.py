#!/usr/bin/env python3
"""Generate compilation-overhead plots from an existing results CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd

from iqp_swap_benchmark.analysis import plot_ratio_vs_qubits, plot_ratio_vs_qubits_per_pattern
from iqp_swap_benchmark.colors import set_device_label_mode


def parse_pattern_columns(value: str) -> tuple[str, int]:
    """Parse a PATTERN=COLUMNS legend override."""
    try:
        pattern, columns_text = value.split("=", maxsplit=1)
        columns = int(columns_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Legend column overrides must use PATTERN=COLUMNS, e.g. local=3"
        ) from exc
    if not pattern or columns < 1:
        raise argparse.ArgumentTypeError(
            "Legend column overrides require a pattern and at least one column"
        )
    return pattern, columns


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("paper_images/compilation_overhead/results.csv"),
        help="Path to an existing benchmark results CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper_images/compilation_overhead/ratio_vs_qubits"),
        help="Directory where the per-pattern plots will be written.",
    )
    parser.add_argument(
        "--metric",
        default="gate_count_ratio_native_fc",
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
        help="Metric to plot on the y-axis.",
    )
    parser.add_argument(
        "--filename-prefix",
        default="ratio_vs_qubits",
        help="Filename prefix for the per-pattern plots.",
    )
    parser.add_argument(
        "--extrapolate-to",
        type=int,
        default=70,
        help="Maximum qubit count shown on the x-axis, including extrapolation.",
    )
    parser.add_argument(
        "--extrapolation-fit-points",
        type=int,
        default=4,
        help=(
            "Maximum number of the latest measured points used for each "
            "per-hardware extrapolation."
        ),
    )
    parser.add_argument(
        "--extrapolation-model",
        choices=["linear", "power_law", "exponential"],
        default="linear",
        help="Regression model used for per-hardware extrapolation.",
    )
    parser.add_argument(
        "--extra-extrapolation-model",
        choices=["linear", "power_law", "exponential"],
        default=None,
        help=(
            "Optional second extrapolation model, drawn with small x markers "
            "for comparison."
        ),
    )
    parser.add_argument(
        "--log-scale",
        action="store_true",
        help="Use a logarithmic y-axis.",
    )
    parser.add_argument(
        "--aggregate-plot",
        type=Path,
        default=None,
        help="Optional output path for a single aggregate plot across patterns.",
    )
    parser.add_argument(
        "--rename",
        action="store_true",
        help="Use anonymized hardware labels (FC_1, FC_2, SC_1, ...).",
    )
    parser.add_argument(
        "--no-title",
        action="store_true",
        help="Omit plot titles.",
    )
    parser.add_argument(
        "--axis-label-size",
        type=float,
        default=None,
        help="Override x/y axis label font size (e.g., 34).",
    )
    parser.add_argument(
        "--legend-size",
        type=float,
        default=None,
        help="Override the legend font size.",
    )
    parser.add_argument(
        "--legend-columns",
        type=int,
        default=1,
        help="Number of columns in the per-pattern legend.",
    )
    parser.add_argument(
        "--legend-columns-override",
        action="append",
        type=parse_pattern_columns,
        default=[],
        metavar="PATTERN=COLUMNS",
        help=(
            "Override the legend column count for one pattern; may be repeated "
            "(e.g. --legend-columns-override local=3)."
        ),
    )
    parser.add_argument(
        "--legend-anchor-y",
        type=float,
        default=None,
        help=(
            "Optional vertical anchor for the upper-right legend in axes "
            "coordinates; smaller values move it downward."
        ),
    )
    parser.add_argument(
        "--legend-anchor-patterns",
        nargs="+",
        default=None,
        help=(
            "Patterns to which --legend-anchor-y applies. By default, the "
            "anchor applies to every pattern."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    set_device_label_mode("anon" if args.rename else "full")
    df = pd.read_csv(args.input)

    figures = plot_ratio_vs_qubits_per_pattern(
        df,
        args.output_dir,
        metric=args.metric,
        filename_prefix=args.filename_prefix,
        extrapolate_to=args.extrapolate_to,
        extrapolation_fit_points=args.extrapolation_fit_points,
        extrapolation_model=args.extrapolation_model,
        extra_extrapolation_model=args.extra_extrapolation_model,
        use_log_scale=args.log_scale,
        show_title=not args.no_title,
        axis_label_size=args.axis_label_size,
        legend_size=args.legend_size,
        legend_columns=args.legend_columns,
        legend_columns_by_pattern=dict(args.legend_columns_override),
        legend_anchor_y=args.legend_anchor_y,
        legend_anchor_patterns=args.legend_anchor_patterns,
    )
    for figure in figures:
        plt.close(figure)

    print(f"Saved per-pattern plots to {args.output_dir}")

    if args.aggregate_plot is not None:
        aggregate_figure = plot_ratio_vs_qubits(
            df,
            output_path=args.aggregate_plot,
            metric=args.metric,
            aggregate_patterns=True,
            show_title=not args.no_title,
            axis_label_size=args.axis_label_size,
        )
        plt.close(aggregate_figure)
        print(f"Saved aggregate plot to {args.aggregate_plot}")


if __name__ == "__main__":
    main()
