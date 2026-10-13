"""High-level orchestration for SWAP counting experiments."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .circuit_patterns import PatternConfig, build_gate_pattern, required_parameter_count
from .compiler import CompilationResult, build_iqp_circuit, compile_for_target
from .topologies import HardwareTarget, select_targets
from .colors import (
    PAPER_COLORS,
    apply_paper_style,
    get_categorical_colors,
    get_hardware_colors,
    get_hardware_markers,
    get_hardware_style,
    get_figure_defaults,
    get_legend_defaults,
    get_plot_style_preset,
)


@dataclass
class ExperimentConfig:
    """Configuration for one batch of routed IQP compilation experiments."""

    n_qubits: int
    pattern_kinds: Sequence[str]
    circuits_per_pattern: int
    hardware_names: Sequence[str]
    interaction_order: int = 2
    density: float = 0.4
    seed: int = 1234
    optimization_level: int = 2
    n_layers: int = 1  # Number of times to repeat the IQP diagonal layer
    logical_circuit_backend: str = "qiskit"
    placement_strategy: str = "default"


def run_swap_experiment(config: ExperimentConfig) -> pd.DataFrame:
    """Run the compile-and-measure loop for one experiment configuration.

    The same logical IQP instance is compiled against every requested target so
    per-target depth and gate-count comparisons are made on matched circuits.
    """
    # One RNG is shared across patterns: their order changes the later samples.
    rng = np.random.default_rng(config.seed)
    targets = select_targets(config.hardware_names, config.n_qubits)
    compiler_seed_env = os.getenv("IQP_COMPILER_SEED")

    print(
        f"[run_swap_experiment] n_qubits={config.n_qubits} "
        f"patterns={list(config.pattern_kinds)} "
        f"circuits_per_pattern={config.circuits_per_pattern} "
        f"hardware={len(targets)} "
        f"k={config.interaction_order} "
        f"layers={config.n_layers} "
        f"placement={config.placement_strategy}",
        flush=True,
    )

    records: List[dict] = []
    for pattern_kind in config.pattern_kinds:
        print(f"[pattern] {pattern_kind}", flush=True)
        for circuit_idx in range(config.circuits_per_pattern):
            print(
                f"[circuit] pattern={pattern_kind} "
                f"index={circuit_idx + 1}/{config.circuits_per_pattern}",
                flush=True,
            )
            pattern_seed = rng.integers(0, 2**32 - 1)
            pattern = build_gate_pattern(
                PatternConfig(
                    kind=pattern_kind,
                    n_qubits=config.n_qubits,
                    interaction_order=config.interaction_order,
                    density=config.density,
                    seed=int(pattern_seed),
                )
            )
            param_count = required_parameter_count(pattern)
            # Generate params for all layers (each layer gets fresh random params)
            total_params = param_count * config.n_layers
            params = rng.uniform(-0.5 * np.pi, 0.5 * np.pi, size=total_params)
            logical_circuit = build_iqp_circuit(
                config.n_qubits,
                params.tolist(),
                pattern,
                n_layers=config.n_layers,
                backend=config.logical_circuit_backend,
            )
            for target in targets:
                print(
                    f"[compile] n={config.n_qubits} "
                    f"pattern={pattern_kind} "
                    f"circuit={circuit_idx + 1}/{config.circuits_per_pattern} "
                    f"target={target.name}",
                    flush=True,
                )
                compiled = compile_for_target(
                    logical_circuit,
                    target,
                    optimization_level=config.optimization_level,
                    placement_strategy=config.placement_strategy,
                    seed=(
                        None
                        if compiler_seed_env is not None
                        and compiler_seed_env.strip().lower() in {"none", "null"}
                        else (
                            int(compiler_seed_env)
                            if compiler_seed_env is not None and compiler_seed_env.strip() != ""
                            else int(pattern_seed % (2**32 - 1))
                        )
                    ),
                )
                records.append(
                    {
                        "pattern": pattern_kind,
                        "circuit_index": circuit_idx,
                        "hardware": target.name,
                        "swap_count": compiled.swap_count,
                        "single_qubit_gate_count": compiled.single_qubit_gate_count,
                        "two_qubit_gate_count": compiled.two_qubit_gate_count,
                        "depth": compiled.depth,
                        "entangling_layer_count": compiled.entangling_layer_count,
                        "gate_count": compiled.gate_count,
                        "native_fc_single_qubit_gate_count": compiled.baseline_single_qubit_gate_count,
                        "native_fc_gate_count": compiled.baseline_gate_count,
                        "native_fc_depth": compiled.baseline_depth,
                        "native_fc_entangling_layer_count": compiled.baseline_entangling_layer_count,
                        "parameter_count": param_count,
                        "n_layers": config.n_layers,
                        "logical_circuit_backend": config.logical_circuit_backend,
                    }
                )
    df = pd.DataFrame.from_records(records)
    df = _attach_fully_connected_ratio(df)
    return df


def save_results(df: pd.DataFrame, path: Path) -> None:
    """Persist an experiment DataFrame to CSV, creating parent folders first."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# Canonical hardware order for consistent plotting
# FC, trapped-ion (all-to-all) first, then superconducting by connectivity density
HARDWARE_ORDER = [
    "fully_connected",
    "quantinuum_helios",
    "ionq_forte",
    "google_willow",
    "google_sycamore",
    "ibm_washington",
    "rigetti_ankaa",
    "oqc_toshiko",
    "line",
    "ring",
    "ladder",
]


def _sort_hardware(hardware_list, strip_suffix: bool = False):
    """Sort hardware names according to canonical order."""
    import re

    def sort_key(name):
        # Strip both grid dimensions (e.g., _4x3) and qubit count (e.g., _10)
        base = re.sub(r"_\d+x\d+", "", name)  # Remove grid dims
        base = re.sub(r"_\d+$", "", base)      # Remove trailing qubit count
        try:
            idx = HARDWARE_ORDER.index(base)
        except ValueError:
            idx = len(HARDWARE_ORDER)
        return (idx, name)

    return sorted(hardware_list, key=sort_key)


def _extract_hardware_base(name: str) -> str:
    """Extract base hardware name, stripping qubit counts and grid dimensions."""
    import re
    # Strip grid dimensions like _4x3 and trailing qubit count like _10
    base = re.sub(r"_\d+x\d+", "", name)
    base = re.sub(r"_\d+$", "", base)
    return base


def plot_swap_results(
    df: pd.DataFrame,
    output_path: Path | None = None,
    metric: str = "swap_count",
    n_qubits: int | None = None,
) -> plt.Figure:
    raise_if_missing_metric(df, metric)
    style = get_plot_style_preset("paper")

    summary = (
        df.groupby(["hardware", "pattern"], as_index=False)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=style["figsize"])
    hardware = _sort_hardware(summary["hardware"].unique())
    patterns = sorted(summary["pattern"].unique())
    x = np.arange(len(hardware))
    width = 0.8 / max(len(patterns), 1)
    pattern_colors = get_categorical_colors(len(patterns))

    for idx, pattern in enumerate(patterns):
        subset = summary[summary["pattern"] == pattern]
        # Reorder subset to match hardware order
        subset = subset.set_index("hardware").reindex(hardware).reset_index()
        means = subset["mean"].to_numpy()
        stds = subset["std"].fillna(0.0).to_numpy()
        _pattern_labels = {"rhg": "RHG", "cubic_3d": "Cubic 3D"}
        offset = (idx - (len(patterns) - 1) / 2) * width
        ax.bar(
            x + offset,
            means,
            width=width,
            label=_pattern_labels.get(pattern, pattern.title()),
            color=pattern_colors[idx % len(pattern_colors)],
            edgecolor=PAPER_COLORS["white"],
            linewidth=0.8,
        )
        ax.errorbar(
            x + offset,
            means,
            yerr=stds,
            fmt="none",
            ecolor=PAPER_COLORS["black"],
            capsize=style["error_capsize"],
            linewidth=1.1,
        )

    # Strip qubit count suffix from hardware names for x-axis labels
    hardware_labels = [get_hardware_style(h)["label"] for h in hardware]
    ax.set_xticks(x)
    ax.set_xticklabels(hardware_labels, rotation=25, ha="right")
    ax.set_ylabel(metric_label(metric), fontsize=style["label_size"], fontweight="bold")
    ax.set_xlabel("Hardware target", fontsize=style["label_size"], fontweight="bold")
    if n_qubits is not None:
        ax.set_title(f"Compilation Overhead at n = {n_qubits}", fontsize=style["title_size"], fontweight="bold")
    apply_paper_style(ax, grid_axis="y")
    ax.tick_params(labelsize=style["tick_size"])
    legend_defaults = get_legend_defaults()
    legend_defaults.update({"title": "Pattern", "fontsize": style["legend_size"]})
    ax.legend(**legend_defaults)
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=style["save_dpi"], bbox_inches="tight", facecolor=PAPER_COLORS["white"])
    return fig


def _attach_fully_connected_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Attach FC-relative and native-FC-relative comparison columns.

    The `*_ratio_*` columns use the repo's convention
    `reference_metric / hardware_metric`, so values closer to `1.0` indicate
    less topology overhead.
    """
    fc_mask = df["hardware"].str.contains("fully_connected", case=False, na=False)
    if fc_mask.any():
        baseline = (
            df.loc[
                fc_mask,
                [
                    "pattern",
                    "circuit_index",
                    "gate_count",
                    "depth",
                    "entangling_layer_count",
                ],
            ]
            .rename(
                columns={
                    "gate_count": "fc_gate_count",
                    "depth": "fc_depth",
                    "entangling_layer_count": "fc_entangling_layer_count",
                }
            )
            .drop_duplicates(subset=["pattern", "circuit_index"])
            .set_index(["pattern", "circuit_index"])
        )
        df = df.join(baseline, on=["pattern", "circuit_index"], how="left")
        df.rename(
            columns={
                "fc_gate_count": "fully_connected_gate_count",
                "fc_depth": "fully_connected_depth",
                "fc_entangling_layer_count": "fully_connected_entangling_layer_count",
            },
            inplace=True,
        )
        gate_denom = df["gate_count"].where(
            df["gate_count"] != 0
        )
        depth_denom = df["depth"].where(
            df["depth"] != 0
        )
        df["gate_count_ratio_fc"] = df["fully_connected_gate_count"] / gate_denom
        df["gate_count_diff_fc"] = (
            df["fully_connected_gate_count"] - df["gate_count"]
        )
        df["depth_ratio_fc"] = df["fully_connected_depth"] / depth_denom
        df["depth_diff_fc"] = df["fully_connected_depth"] - df["depth"]
        entangling_denom = df["entangling_layer_count"].where(
            df["entangling_layer_count"] != 0
        )
        df["entangling_layer_ratio_fc"] = (
            df["fully_connected_entangling_layer_count"] / entangling_denom
        )
        df["entangling_layer_diff_fc"] = (
            df["fully_connected_entangling_layer_count"] - df["entangling_layer_count"]
        )
    else:
        df["fully_connected_gate_count"] = np.nan
        df["fully_connected_depth"] = np.nan
        df["fully_connected_entangling_layer_count"] = np.nan
        df["gate_count_ratio_fc"] = np.nan
        df["gate_count_diff_fc"] = np.nan
        df["depth_ratio_fc"] = np.nan
        df["depth_diff_fc"] = np.nan
        df["entangling_layer_ratio_fc"] = np.nan
        df["entangling_layer_diff_fc"] = np.nan

    native_gate_denom = df["gate_count"].where(df["gate_count"] != 0)
    native_depth_denom = df["depth"].where(df["depth"] != 0)
    native_entangling_denom = df["entangling_layer_count"].where(
        df["entangling_layer_count"] != 0
    )
    df["gate_count_ratio_native_fc"] = df["native_fc_gate_count"] / native_gate_denom
    df["gate_count_diff_native_fc"] = df["native_fc_gate_count"] - df["gate_count"]
    df["depth_ratio_native_fc"] = df["native_fc_depth"] / native_depth_denom
    df["depth_diff_native_fc"] = df["native_fc_depth"] - df["depth"]
    df["entangling_layer_ratio_native_fc"] = (
        df["native_fc_entangling_layer_count"] / native_entangling_denom
    )
    df["entangling_layer_diff_native_fc"] = (
        df["native_fc_entangling_layer_count"] - df["entangling_layer_count"]
    )
    return df


def plot_ratio_vs_qubits(
    df: pd.DataFrame,
    output_path: Path | None = None,
    metric: str = "gate_count_ratio_fc",
    aggregate_patterns: bool = True,
    show_title: bool = True,
    axis_label_size: float | None = None,
) -> plt.Figure:
    """Plot a metric as a function of qubit count, with lines for each hardware topology.

    Parameters
    ----------
    df : pd.DataFrame
        Combined results DataFrame with 'n_qubits' column.
    output_path : Path | None
        Optional path to save the figure.
    metric : str
        Which metric column to plot on the y-axis.
    aggregate_patterns : bool
        If True, average across patterns. If False, show separate lines per pattern.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    import re

    raise_if_missing_metric(df, metric)
    if "n_qubits" not in df.columns:
        raise KeyError("DataFrame must contain 'n_qubits' column for this plot")

    # Exclude fully connected from the comparison (it will always be 1.0)
    plot_df = df[~df["hardware"].str.contains("fully_connected", case=False, na=False)].copy()

    # Extract base hardware name (strip _N suffix and grid dimensions for grouping)
    plot_df["hardware_base"] = plot_df["hardware"].apply(_extract_hardware_base)

    if aggregate_patterns:
        summary = (
            plot_df.groupby(["hardware_base", "n_qubits"], as_index=False)[metric]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
    else:
        summary = (
            plot_df.groupby(["hardware_base", "pattern", "n_qubits"], as_index=False)[metric]
            .agg(["mean", "std", "count"])
            .reset_index()
        )

    style = get_plot_style_preset("compilation")
    fig, ax = plt.subplots(figsize=style["figsize"])

    hardware_list = _sort_hardware(summary["hardware_base"].unique(), strip_suffix=True)

    # Use colors module for consistent paper styling
    hw_colors = get_hardware_colors()
    hw_markers = get_hardware_markers()
    
    # Fallback for unknown hardware
    default_markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]
    default_colors = get_categorical_colors(10)

    for idx, hw in enumerate(hardware_list):
        hw_data = summary[summary["hardware_base"] == hw].sort_values("n_qubits")
        color = hw_colors.get(hw, default_colors[idx % len(default_colors)])
        marker = hw_markers.get(hw, default_markers[idx % len(default_markers)])

        if aggregate_patterns:
            ax.errorbar(
                hw_data["n_qubits"],
                hw_data["mean"],
                yerr=hw_data["std"].fillna(0),
                marker=marker,
                color=color,
                label=get_hardware_style(hw)["label"],
                capsize=style["error_capsize"],
                linewidth=style["line_width"],
                markersize=style["marker_size"],
                markeredgewidth=style["marker_edge_width"],
                linestyle="-",
            )
        else:
            linestyles = ["-", "--", "-.", ":"]
            patterns = hw_data["pattern"].unique()
            for pidx, pattern in enumerate(patterns):
                pdata = hw_data[hw_data["pattern"] == pattern]
                ls = linestyles[pidx % len(linestyles)]
                ax.errorbar(
                    pdata["n_qubits"],
                    pdata["mean"],
                    yerr=pdata["std"].fillna(0),
                    marker=marker,
                    color=color,
                    linestyle=ls,
                    label=f"{get_hardware_style(hw)['label']} ({pattern})",
                    capsize=style["error_capsize"],
                    linewidth=style["line_width"],
                    markersize=style["marker_size"],
                    markeredgewidth=style["marker_edge_width"],
                )

    label_size = axis_label_size if axis_label_size is not None else style["label_size"]
    ax.set_xlabel("Number of qubits", fontsize=label_size, fontweight='bold')
    ax.set_ylabel(metric_label(metric), fontsize=label_size, fontweight='bold')
    ax.axhline(
        y=1.0,
        color=PAPER_COLORS["teal_darkest"],
        linestyle="--",
        alpha=0.7,
        linewidth=style["baseline_width"],
        label="FC baseline",
    )
    apply_paper_style(ax, grid_axis="both")
    ax.legend(
        fontsize=style["legend_size"],
        frameon=True,
        framealpha=style["legend_framealpha"],
        edgecolor=PAPER_COLORS["teal_light"],
        fancybox=True,
        loc="lower right",
        borderpad=0.8,
        labelspacing=0.6,
        handlelength=2.2,
        markerscale=1.2,
    )
    computed_ticks = sorted(summary["n_qubits"].unique())
    if len(computed_ticks) > 5:
        step = max(1, len(computed_ticks) // 4)
        computed_ticks = [computed_ticks[i] for i in range(0, len(computed_ticks), step)]
        if computed_ticks[-1] != summary["n_qubits"].max():
            computed_ticks.append(summary["n_qubits"].max())
    ax.set_xticks(computed_ticks)
    ax.tick_params(labelsize=style["tick_size"])
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
    if show_title:
        ax.set_title("Compilation Overhead Scaling", fontsize=style["title_size"], fontweight='bold')
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=style["save_dpi"], bbox_inches="tight", facecolor=PAPER_COLORS["white"])
    return fig


def plot_ratio_vs_qubits_per_pattern(
    df: pd.DataFrame,
    output_dir: Path,
    metric: str = "gate_count_ratio_native_fc",
    filename_prefix: str = "ratio_vs_qubits",
    extrapolate_to: int = 70,
    extrapolation_fit_points: int = 4,
    extrapolation_model: str = "linear",
    extra_extrapolation_model: str | None = None,
    use_log_scale: bool = False,
    show_title: bool = True,
    axis_label_size: float | None = None,
    legend_size: float | None = None,
    legend_columns: int = 1,
    legend_columns_by_pattern: Mapping[str, int] | None = None,
    legend_anchor_y: float | None = None,
    legend_anchor_patterns: Sequence[str] | None = None,
) -> List[plt.Figure]:
    """Generate separate scaling plots for each pattern with extrapolation.

    Parameters
    ----------
    df : pd.DataFrame
        Combined results DataFrame with 'n_qubits' and 'pattern' columns.
    output_dir : Path
        Directory to save the plots.
    metric : str
        Which metric column to plot on the y-axis.
    filename_prefix : str
        Prefix for output filenames.
    extrapolate_to : int
        Maximum qubit count to extrapolate to (default: 70).
        Extrapolated region is shown with dotted lines.
    extrapolation_fit_points : int
        Maximum number of the latest measured points used for each
        per-hardware extrapolation.
    extrapolation_model : {"linear", "power_law", "exponential"}
        Regression model used for extrapolation. Power-law and exponential
        fits are anchored at the last measured point.
    extra_extrapolation_model : {"linear", "power_law", "exponential"}, optional
        Optional comparison model, anchored at the last measured point and
        drawn with small ``x`` markers instead of a line.
    use_log_scale : bool
        If True, use logarithmic scale for y-axis.
    legend_size : float, optional
        Override the preset legend font size.
    legend_columns : int
        Number of columns in the legend.
    legend_columns_by_pattern : mapping, optional
        Per-pattern overrides for ``legend_columns``.
    legend_anchor_y : float, optional
        Vertical anchor for the upper-right legend, in axes coordinates.
    legend_anchor_patterns : sequence of str, optional
        Patterns to which ``legend_anchor_y`` applies. If omitted, the anchor
        applies to every pattern.

    Returns
    -------
    List[plt.Figure]
        List of matplotlib figures (one per pattern).
    """
    from scipy import stats
    from matplotlib.lines import Line2D
    
    raise_if_missing_metric(df, metric)
    if "n_qubits" not in df.columns:
        raise KeyError("DataFrame must contain 'n_qubits' column for this plot")
    if "pattern" not in df.columns:
        raise KeyError("DataFrame must contain 'pattern' column for this plot")
    if extrapolation_fit_points < 2:
        raise ValueError("extrapolation_fit_points must be at least 2")
    if extrapolation_model not in {"linear", "power_law", "exponential"}:
        raise ValueError(
            "extrapolation_model must be 'linear', 'power_law', or 'exponential'"
        )
    if extra_extrapolation_model not in {
        None,
        "linear",
        "power_law",
        "exponential",
    }:
        raise ValueError(
            "extra_extrapolation_model must be None, 'linear', 'power_law', "
            "or 'exponential'"
        )
    if extra_extrapolation_model == extrapolation_model:
        raise ValueError("The primary and extra extrapolation models must differ")
    if legend_columns < 1:
        raise ValueError("legend_columns must be at least 1")
    if legend_columns_by_pattern is not None and any(
        columns < 1 for columns in legend_columns_by_pattern.values()
    ):
        raise ValueError("Every per-pattern legend column count must be at least 1")

    output_dir.mkdir(parents=True, exist_ok=True)
    patterns = sorted(df["pattern"].unique())
    figures = []
    style = get_plot_style_preset("compilation_per_pattern")
    rc_params = {
        "font.family": style["font_family"],
        "mathtext.fontset": "stix",
        "axes.titleweight": "semibold",
        "axes.labelweight": "semibold",
    }

    for pattern in patterns:
        pattern_df = df[df["pattern"] == pattern].copy()
        output_path = output_dir / f"{filename_prefix}_{pattern}.pdf"

        # Exclude fully connected from the comparison
        plot_df = pattern_df[~pattern_df["hardware"].str.contains("fully_connected", case=False, na=False)].copy()
        plot_df["hardware_base"] = plot_df["hardware"].apply(_extract_hardware_base)

        summary = (
            plot_df.groupby(["hardware_base", "n_qubits"], as_index=False)[metric]
            .agg(["mean", "std", "count"])
            .reset_index()
        )

        with plt.rc_context(rc_params):
            fig, ax = plt.subplots(figsize=style["figsize"])

            hardware_list = _sort_hardware(summary["hardware_base"].unique(), strip_suffix=True)

            # Use colors module for consistent paper styling
            hw_colors = get_hardware_colors()
            hw_markers = get_hardware_markers()
            default_markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]
            default_colors = get_categorical_colors(10)
        
            # Determine max computed n_qubits (global, for axis limits)
            max_computed_n = int(summary["n_qubits"].max())
            plot_xlim_max = max(max_computed_n, extrapolate_to)
            visible_y_values: list[float] = [1.0]
            any_hardware_extrapolated = False

            for idx, hw in enumerate(hardware_list):
                hw_data = summary[summary["hardware_base"] == hw].sort_values("n_qubits")
                color = hw_colors.get(hw, default_colors[idx % len(default_colors)])
                marker = hw_markers.get(hw, default_markers[idx % len(default_markers)])

                is_trapped_ion = hw in ("ionq_forte", "quantinuum_helios")
                linestyle = "--" if is_trapped_ion else "-"
                fillstyle = "none" if is_trapped_ion else "full"

                y_values = hw_data["mean"].values
                x_values = hw_data["n_qubits"].values
                y_errors = hw_data["std"].fillna(0).values
                visible_y_values.extend((y_values - y_errors).tolist())
                visible_y_values.extend((y_values + y_errors).tolist())

                ax.errorbar(
                    x_values,
                    y_values,
                    yerr=y_errors,
                    marker=marker,
                    color=color,
                    linestyle=linestyle,
                    fillstyle=fillstyle,
                    markeredgecolor=color,
                    markeredgewidth=style["marker_edge_width"],
                    label=get_hardware_style(hw)["label"],
                    capsize=style["error_capsize"],
                    linewidth=style["line_width"],
                    markersize=style["marker_size"],
                )

                # Per-hardware extrapolation: need >= 2 data points, and only
                # extrapolate beyond that hardware's last measured n.
                hw_max_n = int(x_values[-1])
                if len(x_values) >= 2 and plot_xlim_max > hw_max_n:
                    any_hardware_extrapolated = True
                    hw_extrap_x_future = np.arange(hw_max_n + 2, plot_xlim_max + 2, 2)
                    fit_n = min(extrapolation_fit_points, len(x_values))
                    fit_x = x_values[-fit_n:]
                    fit_y = y_values[-fit_n:]
                    def extrapolate(model: str) -> np.ndarray:
                        if model == "power_law":
                            if np.any(fit_x <= 0) or np.any(fit_y <= 0):
                                raise ValueError(
                                    "Power-law extrapolation requires positive x and y values"
                                )
                            slope, _intercept, _r_value, _p_value, _std_err = (
                                stats.linregress(np.log(fit_x), np.log(fit_y))
                            )
                            slope = min(float(slope), 0.0)
                            result = y_values[-1] * (
                                hw_extrap_x_future / hw_max_n
                            ) ** slope
                        elif model == "exponential":
                            if np.any(fit_y <= 0):
                                raise ValueError(
                                    "Exponential extrapolation requires positive y values"
                                )
                            slope, _intercept, _r_value, _p_value, _std_err = (
                                stats.linregress(fit_x, np.log(fit_y))
                            )
                            # Closed-form constrained least-squares solution
                            # for a non-increasing log-linear trend.
                            slope = min(float(slope), 0.0)
                            result = y_values[-1] * np.exp(
                                slope * (hw_extrap_x_future - hw_max_n)
                            )
                        else:
                            slope, _intercept, _r_value, _p_value, _std_err = (
                                stats.linregress(fit_x, fit_y)
                            )
                            result = y_values[-1] + slope * (
                                hw_extrap_x_future - hw_max_n
                            )

                        if model in {"power_law", "exponential"}:
                            # Positive models remain within the physical
                            # efficiency interval without a positive floor.
                            return np.clip(result, 0.0, 1.0)
                        if is_trapped_ion:
                            return np.clip(result, 0.95, 1.05)
                        return np.clip(result, 0.01, None)

                    hw_extrap_y_future = extrapolate(extrapolation_model)

                    # Anchor the dashed segment to the actual last measured point
                    # so the extrapolation visually starts from the real marker.
                    hw_extrap_x = np.concatenate(([hw_max_n], hw_extrap_x_future))
                    hw_extrap_y = np.concatenate(([y_values[-1]], hw_extrap_y_future))

                    mask = hw_extrap_x <= plot_xlim_max
                    visible_y_values.extend(hw_extrap_y[mask].tolist())

                    ax.plot(
                        hw_extrap_x,
                        hw_extrap_y,
                        color=color,
                        linestyle=(0, (1.2, 2.0)),
                        linewidth=style["baseline_width"],
                        alpha=0.58,
                    )

                    if extra_extrapolation_model is not None:
                        extra_y_future = extrapolate(extra_extrapolation_model)
                        extra_x = np.concatenate(([hw_max_n], hw_extrap_x_future))
                        extra_y = np.concatenate(([y_values[-1]], extra_y_future))
                        extra_mask = extra_x <= plot_xlim_max
                        visible_y_values.extend(extra_y[extra_mask].tolist())
                        ax.plot(
                            extra_x,
                            extra_y,
                            color=color,
                            linestyle="none",
                            marker="x",
                            markersize=style["marker_size"] * 0.55,
                            markeredgewidth=style["marker_edge_width"],
                            alpha=0.72,
                        )

            label_size = axis_label_size if axis_label_size is not None else style["label_size"]
            ax.set_xlabel("Number of qubits", fontsize=label_size, fontweight="bold")
            ax.set_ylabel(metric_label(metric), fontsize=label_size, fontweight="black")

            if show_title:
                pattern_titles = {"dense": "Dense IQP", "local": "Local IQP", "sparse": "Sparse IQP", "rhg": "RHG IQP", "cubic_3d": "Cubic 3D IQP"}
                title_text = pattern_titles.get(pattern, f"{pattern.title()} IQP")
                ax.set_title(f"Compilation Overhead: {title_text}", fontsize=style["title_size"], fontweight="black", pad=12)

            ax.axhline(
                y=1.0,
                color=PAPER_COLORS["teal_darkest"],
                linestyle=(0, (4.0, 2.4)),
                alpha=0.75,
                linewidth=style["baseline_width"],
                label="FC baseline",
            )

            if use_log_scale:
                ax.set_yscale("log")
                ax.set_ylabel(metric_label(metric) + " (log scale)", fontsize=label_size, fontweight="semibold")

            apply_paper_style(ax, grid_axis="y")
            ax.grid(False, axis="x")

            min_tick = int(summary["n_qubits"].min())
            computed_ticks = sorted(summary["n_qubits"].unique())
            extrap_ticks = list(range(max_computed_n + 10, plot_xlim_max + 1, 10))
            all_ticks = sorted(set(computed_ticks + extrap_ticks))
            ax.set_xticks(all_ticks)
            ax.set_xlim(min_tick - 1, plot_xlim_max + 1)

            visible_y_array = np.asarray([value for value in visible_y_values if np.isfinite(value)], dtype=float)
            if visible_y_array.size:
                y_min = float(visible_y_array.min())
                y_max = float(visible_y_array.max())
                y_span = max(0.02, y_max - y_min)
                pad = 0.08 * y_span
                ax.set_ylim(max(0.0, y_min - pad), y_max + pad)

            ax.tick_params(labelsize=style["tick_size"])
            for label in ax.get_xticklabels():
                label.set_fontfamily(style["font_family"])
                label.set_fontweight("bold")
                label.set_rotation(0)
                label.set_horizontalalignment("center")
            for label in ax.get_yticklabels():
                label.set_fontfamily(style["font_family"])
                label.set_fontweight("bold")

            extrapolation_label = (
                "Extrapolation"
                if extra_extrapolation_model is None
                else extrapolation_model.replace("_", " ").title()
            )
            extrapolation_handle = Line2D(
                [0],
                [0],
                color="#7F7F7F",
                linestyle=(0, (1.2, 2.0)),
                linewidth=style["baseline_width"],
                alpha=0.85,
                label=extrapolation_label,
            )

            handles, labels = ax.get_legend_handles_labels()
            if any_hardware_extrapolated:
                handles.append(extrapolation_handle)
                labels.append(extrapolation_label)
                if extra_extrapolation_model is not None:
                    extra_label = extra_extrapolation_model.replace("_", " ").title()
                    handles.append(
                        Line2D(
                            [0],
                            [0],
                            color="#7F7F7F",
                            linestyle="none",
                            marker="x",
                            markersize=style["marker_size"] * 0.65,
                            markeredgewidth=style["marker_edge_width"],
                            alpha=0.9,
                            label=extra_label,
                        )
                    )
                    labels.append(extra_label)

            legend_kwargs: dict[str, object] = {}
            anchor_this_pattern = (
                legend_anchor_patterns is None or pattern in legend_anchor_patterns
            )
            if legend_anchor_y is not None and anchor_this_pattern:
                legend_kwargs["bbox_to_anchor"] = (1.0, legend_anchor_y)

            pattern_legend_columns = (
                legend_columns_by_pattern.get(pattern, legend_columns)
                if legend_columns_by_pattern is not None
                else legend_columns
            )
            legend = ax.legend(
                handles,
                labels,
                fontsize=legend_size if legend_size is not None else style["legend_size"],
                frameon=True,
                framealpha=style["legend_framealpha"],
                edgecolor=PAPER_COLORS["teal_dark"],
                fancybox=True,
                loc="upper right",
                borderpad=0.72,
                labelspacing=0.36,
                handlelength=1.75,
                handletextpad=0.45,
                borderaxespad=0.4,
                markerscale=1.18,
                ncol=pattern_legend_columns,
                columnspacing=0.8,
                **legend_kwargs,
            )
            legend.get_frame().set_linewidth(1.5)
            legend.get_frame().set_facecolor("white")
            for text in legend.get_texts():
                text.set_fontfamily(style["font_family"])
                text.set_fontweight("bold")
                text.set_color(PAPER_COLORS["teal_darkest"])

            fig.tight_layout()
            fig.savefig(output_path, dpi=style["save_dpi"], bbox_inches="tight", facecolor=PAPER_COLORS["white"])
        figures.append(fig)
        print(f"Saved scaling plot for pattern '{pattern}' to {output_path}")

    return figures


def raise_if_missing_metric(df: pd.DataFrame, metric: str) -> None:
    if metric not in df.columns:
        available = ", ".join(sorted(df.columns))
        raise KeyError(f"Metric '{metric}' not found. Available columns: {available}")


# Convenience alias for plotting helpers.
PLOT_COLORS = PAPER_COLORS


def plot_phase_diagram(
    df: pd.DataFrame | None = None,
    output_path: Path | str = "paper_images/phase_diagram/phase_diagram.pdf",
    k: int = 2,
    c: float = 1.0,
    p_range: tuple[float, float] = (0.005, 0.1),
    p_critical_min: float | None = None,  # Min p for critical line (can be smaller than p_range[0])
    depth_range: tuple[float, float] = (0, 500),
    n_points: int = 500,
    use_log_scale: bool = False,
    x_log_scale: bool | None = None,
    hardware_depths: dict[str, tuple[float, float, float, float]] | None = None,
    hardware_p_eff: dict[str, float] | None = None,
    compiled_depths: dict[str, tuple[float, float]] | None = None,  # Legacy param
    n_qubits: int | None = None,  # Optional qubit count for title
    n_layers: int | None = None,
    y_axis_label: str = r"Circuit depth $D$",
) -> plt.Figure:
    """Generate phase diagram in (p, D) space showing simulatability boundary.

    Plots the critical depth curve D*(p,k) = c * log(k/p) / p which separates
    the simulatable region (above the curve, where noise dominates) from the
    potentially hard region (below the curve).

    Parameters
    ----------
    df : pd.DataFrame | None
        Optional DataFrame containing compilation results. If provided, will
        plot hardware operating points at (p_eff, D_H) for each hardware target.
    output_path : Path | str
        Path to save the figure.
    k : int
        Interaction locality (max qubits per gate term). Default k=2 for
        standard 2-qubit IQP interactions. This is NOT the circuit size.
    c : float
        Constant prefactor in the critical depth formula (default 1.0).
    p_range : tuple[float, float]
        Range of error rates (x-axis) as (p_min, p_max).
    depth_range : tuple[float, float]
        Range of circuit depths (y-axis) as (D_min, D_max).
    n_points : int
        Number of points for plotting the critical depth curve.
    use_log_scale : bool
        If True, use log scaling on the y-axis.
    x_log_scale : bool | None
        If True, use log scaling on the x-axis. Defaults to the same value as
        use_log_scale when not provided.
    hardware_depths : dict[str, tuple[float, float, float, float]] | None
        Dict mapping hardware names to (native_depth, native_std, compiled_depth, compiled_std).
        Native depth is the circuit depth without routing (FC topology).
        Compiled depth is after routing with SWAPs for the target hardware.
    compiled_depths : dict[str, tuple[float, float]] | None
        Legacy parameter - use hardware_depths instead.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    from .noise_model import HARDWARE_ERROR_RATES, critical_depth

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if x_log_scale is None:
        x_log_scale = use_log_scale

    plot_style = get_plot_style_preset("phase_diagram")
    fig, ax = plt.subplots(figsize=plot_style["figsize"])
    visible_ymax = depth_range[1]

    # Use p_critical_min for critical line calculation if provided
    # This allows the critical line to extend further left than the display x-axis
    p_crit_start = p_critical_min if p_critical_min is not None else p_range[0]
    
    # Plot critical depth curve
    if x_log_scale:
        p_values = np.logspace(np.log10(p_crit_start), np.log10(p_range[1]), n_points)
    else:
        p_values = np.linspace(p_crit_start, p_range[1], n_points)
    # Filter valid p values (need p < k for log(k/p) > 0)
    valid_mask = p_values < k
    p_valid = p_values[valid_mask]
    d_critical = np.array([critical_depth(p, k, c) for p in p_valid])

    # Use a clipped copy only for region fills so they extend continuously to
    # the visible axis boundary. Plot the true, unclipped critical curve and
    # let Matplotlib clip it at the axes; otherwise values above the y-limit
    # appear as a misleading horizontal plateau.
    d_critical_clipped = np.clip(d_critical, depth_range[0], visible_ymax)

    # Plot the boundary with black
    if x_log_scale and use_log_scale:
        ax.loglog(
            p_valid,
            d_critical,
            color=PAPER_COLORS["black"],
            linewidth=plot_style["line_width"],
            label=r"$D^*(p,k) = \frac{c\,\ln(k/p)}{p}$",
        )
    elif x_log_scale:
        ax.semilogx(
            p_valid,
            d_critical,
            color=PAPER_COLORS["black"],
            linewidth=plot_style["line_width"],
            label=r"$D^*(p,k) = \frac{c\,\ln(k/p)}{p}$",
        )
    else:
        ax.plot(
            p_valid,
            d_critical,
            color=PAPER_COLORS["black"],
            linewidth=plot_style["line_width"],
            label=r"$D^*(p,k) = \frac{c\,\ln(k/p)}{p}$",
        )

    # Fill regions using paper color palette
    # Simulatable region (above the curve) - light teal
    ax.fill_between(
        p_valid,
        d_critical_clipped,
        visible_ymax,
        alpha=0.25,
        color=PAPER_COLORS["teal_light"],
        label="Simulatable (noise-dominated)",
    )
    # Potentially hard region (below the curve) - coral
    ax.fill_between(
        p_valid,
        depth_range[0],
        d_critical_clipped,
        alpha=0.18,
        color=PAPER_COLORS["coral"],
        label="Potentially hard",
    )

    # Hardware point styles using paper color palette
    # Only real hardware platforms (no simulated line/ring/ladder)
    hw_colors = get_hardware_colors()
    hw_markers = get_hardware_markers()
    hardware_styles = {
        "quantinuum_helios": {"color": hw_colors["quantinuum_helios"], "marker": hw_markers["quantinuum_helios"]},
        "ionq_forte": {"color": hw_colors["ionq_forte"], "marker": hw_markers["ionq_forte"]},
        "google_sycamore": {"color": hw_colors["google_sycamore"], "marker": hw_markers["google_sycamore"]},
        "google_willow": {"color": hw_colors["google_willow"], "marker": hw_markers["google_willow"]},
        "ibm_washington": {"color": hw_colors["ibm_washington"], "marker": hw_markers["ibm_washington"]},
        "rigetti_ankaa": {"color": hw_colors["rigetti_ankaa"], "marker": hw_markers["rigetti_ankaa"]},
        "oqc_toshiko": {"color": hw_colors["oqc_toshiko"], "marker": hw_markers["oqc_toshiko"]},
    }

    # Map hardware names from CSV to canonical names
    hw_name_map = {
        "google_sycamore": "google_sycamore",
        "google_willow": "google_willow",
        "rigetti_ankaa": "rigetti_ankaa",
        "ibm_washington": "ibm_washington",
        "oqc_toshiko": "oqc_toshiko",
        "ionq_forte": "ionq_forte",
        "quantinuum_helios": "quantinuum_helios",
    }
    
    # Skip simulated topologies (line, ring, ladder) - they're exemplary only
    SKIP_HARDWARE = {"line", "ring", "ladder", "fully_connected"}

    # Plot hardware from hardware_depths dict (new format with native + compiled)
    if hardware_depths is not None:
        ordered_hw_keys = _sort_hardware(list(hardware_depths.keys()))
        for hw_key in ordered_hw_keys:
            native_depth, native_std, compiled_depth, compiled_std = hardware_depths[hw_key]
            # Extract base hardware name (strip qubit count suffix like _10, _16)
            import re
            base_name = re.sub(r"_\d+$", "", hw_key)
            
            # Skip simulated topologies and fully_connected
            if base_name in SKIP_HARDWARE:
                continue
            
            # Get error rate using direct mapping or base name
            mapped_name = hw_name_map.get(base_name, base_name)
            if mapped_name not in HARDWARE_ERROR_RATES and (
                hardware_p_eff is None or hw_key not in hardware_p_eff
            ):
                continue
            p_eff = (
                hardware_p_eff[hw_key]
                if hardware_p_eff is not None and hw_key in hardware_p_eff
                else HARDWARE_ERROR_RATES[mapped_name]
            )
            
            # Skip if outside p_range
            if p_eff < p_range[0] or p_eff > p_range[1]:
                continue
            
            hw_style = hardware_styles.get(mapped_name, {"color": "gray", "marker": "o"})
            hw_label = get_hardware_style(mapped_name)["label"]
            
            # Calculate critical depth at this hardware's error rate
            d_crit_at_p = critical_depth(p_eff, k, c)
            
            # Draw vertical dashed line from bottom to critical depth (simulatability threshold)
            ax.vlines(
                p_eff,
                ymin=depth_range[0],
                ymax=min(d_crit_at_p, visible_ymax),
                colors=hw_style["color"],
                linestyles="--",
                linewidths=1.0,
                alpha=0.7,
                zorder=3,
            )
            
            # Check if FC hardware (no routing overhead: native ~= compiled)
            is_fc = abs(compiled_depth - native_depth) < native_depth * 0.01  # <1% difference
            
            if is_fc:
                # FC hardware (trapped-ion): just plot one marker, compact legend
                ax.scatter(
                    p_eff,
                    native_depth,
                    s=255,
                    color=hw_style["color"],
                    marker=hw_style["marker"],
                    edgecolors="black",
                    linewidths=plot_style["marker_edge_width"],
                    label=f"{hw_label} ({native_depth:.0f})",
                    zorder=5,
                )
                
                # Horizontal line extending left to show where it crosses boundary
                ax.hlines(
                    native_depth,
                    xmin=p_range[0],
                    xmax=p_eff,
                    colors=hw_style["color"],
                    linestyles="-",
                    linewidths=1.0,
                    alpha=0.5,
                    zorder=2,
                )
            else:
                # Sparse hardware: plot native + compiled with single compact legend entry
                
                # 1. Plot native depth marker (ideal circuit, before routing)
                ax.scatter(
                    p_eff,
                    native_depth,
                    s=285,
                    color=hw_style["color"],
                    marker=hw_style["marker"],
                    edgecolors="black",
                    linewidths=plot_style["marker_edge_width"],
                    zorder=5,
                )
                
                # 2. Draw vertical line from native to compiled depth
                ax.vlines(
                    p_eff,
                    ymin=native_depth,
                    ymax=compiled_depth,
                    colors=hw_style["color"],
                    linestyles="-",
                    linewidths=1.8,
                    alpha=0.8,
                    zorder=4,
                )
                
                # 3. Plot compiled depth marker (square, with error bar)
                ax.errorbar(
                    p_eff,
                    compiled_depth,
                    yerr=compiled_std,
                    marker="s",
                    markersize=plot_style["marker_size"],
                    color=hw_style["color"],
                    markeredgecolor="black",
                    markeredgewidth=plot_style["marker_edge_width"],
                    capsize=plot_style["error_capsize"],
                    capthick=1.8,
                    linestyle="none",
                    zorder=6,
                )
                
                # Horizontal line extending left from compiled depth to show boundary crossing
                ax.hlines(
                    compiled_depth,
                    xmin=p_range[0],
                    xmax=p_eff,
                    colors=hw_style["color"],
                    linestyles="-",
                    linewidths=1.0,
                    alpha=0.5,
                    zorder=2,
                )
                
                # Single compact legend entry: "Hardware (ideal/compiled)"
                ax.plot([], [], color=hw_style["color"], marker=hw_style["marker"], 
                    linestyle="-", linewidth=plot_style["line_width"], markersize=plot_style["marker_size"],
                    label=f"{hw_label} ({native_depth:.0f}/{compiled_depth:.0f})")
                
                # 4. Add annotation showing the routing overhead
                ratio = compiled_depth / native_depth
                if use_log_scale:
                    annotation_y = compiled_depth * 1.36
                else:
                    annotation_y = compiled_depth + 0.045 * (depth_range[1] - depth_range[0])
                ax.annotate(
                    f"{ratio:.1f}×",
                    xy=(p_eff, compiled_depth),
                    xytext=(p_eff, annotation_y),
                    fontsize=plot_style["annotation_size"],
                    color=hw_style["color"],
                    ha="center",
                    fontweight="bold",
                )

    # Legacy: Plot from compiled_depths if provided (old format)
    elif compiled_depths is not None:
        for hw_name in hardware_styles.keys():
            if hw_name not in HARDWARE_ERROR_RATES:
                continue
            p_eff = HARDWARE_ERROR_RATES[hw_name]
            if p_eff < p_range[0] or p_eff > p_range[1]:
                continue
            hw_style = hardware_styles[hw_name]
            hw_label = get_hardware_style(hw_name)["label"]
            d_crit_at_p = critical_depth(p_eff, k, c)
            
            ax.vlines(p_eff, ymin=depth_range[0], ymax=min(d_crit_at_p, visible_ymax),
                      colors=hw_style["color"], linestyles="--", linewidths=1.0, alpha=0.7, zorder=3)
            
            # Find matching hardware in compiled_depths
            ref_depth = d_crit_at_p * 0.3
            for hw_key, (mean_d, _) in compiled_depths.items():
                if hw_name in hw_key.lower():
                    ref_depth = mean_d
                    break

            ax.scatter(p_eff, ref_depth, s=110, color=hw_style["color"], marker=hw_style["marker"],
                       edgecolors="black", linewidths=plot_style["marker_edge_width"], label=f"{hw_label} ($p$={p_eff})", zorder=5)

    apply_paper_style(ax, grid_axis="both", show_minor_grid=use_log_scale)
    x_axis_label = (
        r"Depth-normalized effective error rate $p_{\mathrm{eff}}$"
        if hardware_p_eff is not None
        else r"Effective-noise $p$ (approx. 2Q error)"
    )
    ax.set_xlabel(x_axis_label, fontsize=plot_style["label_size"], fontweight="bold")
    ax.set_ylabel(y_axis_label, fontsize=plot_style["label_size"], fontweight="bold")
    title_parts = [f"k={k}"]
    if n_qubits is not None:
        title_parts.append(f"n={n_qubits}")
    ax.set_title(
        ", ".join(title_parts),
        fontsize=plot_style["title_size"],
        fontweight="bold",
    )
    if use_log_scale:
        ax.set_xlim(p_range)
    else:
        ax.set_xlim(p_range)
    ax.set_ylim(depth_range[0], visible_ymax)
    if not use_log_scale and depth_range[0] < 0:
        # Allow visual padding below zero for shallow markers without displaying
        # physically meaningless negative circuit-depth tick labels.
        nonnegative_ticks = [tick for tick in ax.get_yticks() if tick >= 0]
        if not any(np.isclose(tick, 0.0) for tick in nonnegative_ticks):
            nonnegative_ticks.insert(0, 0.0)
        ax.set_yticks(nonnegative_ticks)
    ax.tick_params(labelsize=plot_style["tick_size"])
    
    legend_defaults = get_legend_defaults()
    legend_defaults.update({
        "loc": "lower left",
        "fontsize": plot_style["legend_size"],
        "markerscale": 1.55,
        "handlelength": 2.05,
        "handletextpad": 0.54,
        "borderpad": 0.76,
        "borderaxespad": 0.45,
        "labelspacing": 0.36,
        "edgecolor": PAPER_COLORS["teal_dark"],
    })
    if not use_log_scale:
        legend_defaults["bbox_to_anchor"] = (0.02, 0.02)
    legend = ax.legend(**legend_defaults)
    legend.get_frame().set_linewidth(1.5)
    legend.get_frame().set_facecolor("white")
    for text in legend.get_texts():
        text.set_fontweight("bold")
        text.set_color(PAPER_COLORS["teal_darkest"])

    fig.tight_layout()
    fig.savefig(output_path, dpi=plot_style["save_dpi"], bbox_inches="tight", facecolor=PAPER_COLORS["white"])
    print(f"Saved phase diagram to {output_path}")

    return fig


def metric_label(metric: str) -> str:
    """Return human-readable label for metric, using paper notation where applicable."""
    if metric == "swap_count":
        return "Average SWAP count"
    if metric == "two_qubit_gate_count":
        return "Average two-qubit gate count"
    if metric == "depth":
        return r"Circuit depth $D_H$"
    if metric == "entangling_layer_count":
        return r"Entangling-layer depth $D_H^{(2q)}$"
    if metric == "gate_count":
        return "Circuit gate count"
    if metric == "gate_count_ratio_fc":
        return r"$D_{\mathrm{FC}} / D_H$ (gate count)"
    if metric == "gate_count_diff_fc":
        return r"$D_{\mathrm{FC}} - D_H$ (gate count)"
    if metric == "depth_ratio_fc":
        return r"$D_{\mathrm{FC}} / D_H$ (depth)"
    if metric == "depth_diff_fc":
        return r"$D_{\mathrm{FC}} - D_H$ (depth)"
    if metric == "entangling_layer_ratio_fc":
        return r"$D_{\mathrm{FC}}^{(2q)} / D_H^{(2q)}$"
    if metric == "entangling_layer_diff_fc":
        return r"$D_{\mathrm{FC}}^{(2q)} - D_H^{(2q)}$"
    if metric == "gate_count_ratio_native_fc":
        return r"$D_{\mathrm{FC}} / D_H$"
    if metric == "gate_count_diff_native_fc":
        return r"$\Delta D = D_{\mathrm{FC}} - D_H$"
    if metric == "depth_ratio_native_fc":
        return r"$D_{\mathrm{FC}} / D_H$"
    if metric == "depth_diff_native_fc":
        return r"$\Delta D = D_{\mathrm{FC}} - D_H$ (depth)"
    if metric == "entangling_layer_ratio_native_fc":
        return r"$D_{\mathrm{FC}}^{(2q)} / D_H^{(2q)}$"
    if metric == "entangling_layer_diff_native_fc":
        return r"$\Delta D^{(2q)} = D_{\mathrm{FC}}^{(2q)} - D_H^{(2q)}$"
    return f"Average {metric}"
