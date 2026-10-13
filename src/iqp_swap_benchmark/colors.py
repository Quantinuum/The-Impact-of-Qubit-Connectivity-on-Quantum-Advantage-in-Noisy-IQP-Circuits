"""Colors and plotting styles used in the paper.

The values below keep regenerated figures consistent with the paper:
- Primary colors: White, Black, Quantum Teal
- Supporting tones: Teal gradients for accessibility and data viz
- Accent colors: Purple, Coral, Orange for differentiation

Usage in matplotlib:
    from iqp_swap_benchmark.colors import PAPER_COLORS, get_hardware_colors

    plt.plot(x, y, color=PAPER_COLORS['quantum_teal'])
    hw_colors = get_hardware_colors()
    plt.bar(x, y, color=hw_colors['ionq_forte'])
"""

import os
from typing import Any, Dict, List, Tuple, cast

# =============================================================================
# PAPER COLOR PALETTE
# =============================================================================

PAPER_COLORS: Dict[str, str] = {
    # Primary colors (backgrounds, graphical elements)
    "white": "#FFFFFF",           # RGB: 255, 255, 255
    "black": "#000000",           # RGB: 0, 0, 0
    "quantum_teal": "#69D3BE",    # RGB: 105, 211, 190 - Main plot color
    
    # Supporting teal tones (accessibility, data viz)
    "teal_lightest": "#E1F6F2",   # RGB: 225, 246, 242
    "teal_light": "#A5E5D7",      # RGB: 165, 229, 215
    "teal_medium": "#69D3BE",     # RGB: 105, 211, 190
    "teal_dark": "#30A08E",       # RGB: 48, 160, 142
    "teal_darkest": "#1D605B",    # RGB: 29, 96, 91
    
    # Accent colors (data differentiation)
    "purple": "#8064A2",          # RGB: 128, 100, 162
    "coral": "#E75D72",           # RGB: 231, 93, 114
    "orange": "#FF9A56",          # RGB: 255, 154, 86
}

# Convenience aliases
TEAL = PAPER_COLORS["quantum_teal"]
BLACK = PAPER_COLORS["black"]
WHITE = PAPER_COLORS["white"]


# =============================================================================
# DEVICE LABEL MODES (FULL VS ANONYMIZED)
# =============================================================================

_DEVICE_LABEL_MODE = os.getenv("IQP_DEVICE_LABEL_MODE", "full").strip().lower()


def set_device_label_mode(mode: str) -> None:
    """Set hardware label mode for plotting.

    Parameters
    ----------
    mode : str
        Either ``"full"`` (default full vendor names) or ``"anon"``
        (short anonymized labels).
    """
    normalized = mode.strip().lower()
    if normalized not in {"full", "anon"}:
        raise ValueError(f"Unknown device label mode '{mode}'. Use 'full' or 'anon'.")
    global _DEVICE_LABEL_MODE
    _DEVICE_LABEL_MODE = normalized


def get_device_label_mode() -> str:
    """Return current device label mode."""
    return _DEVICE_LABEL_MODE


# =============================================================================
# HARDWARE-SPECIFIC COLOR ASSIGNMENTS
# =============================================================================

def get_hardware_colors() -> Dict[str, str]:
    """Get color assignments for each hardware platform.
    
    Returns a dict mapping hardware base names to paper palette colors.
    Trapped-ion platforms get teal variants, superconducting get accents.
    """
    return {
        # Trapped-ion (all-to-all connectivity) - Teal variants
        "quantinuum_helios": PAPER_COLORS["quantum_teal"],  # Quantinuum Helios 98Q
        "ionq_forte": PAPER_COLORS["teal_dark"],            # IonQ Forte 36Q
        "fully_connected": PAPER_COLORS["teal_light"],     # Reference/ideal
        
        # Superconducting - Accent colors
        "google_sycamore": PAPER_COLORS["coral"],           # Google Sycamore (tilted grid)
        "google_willow": "#C04050",                       # Google Willow (2D grid) - darker coral
        "ibm_washington": PAPER_COLORS["purple"],           # IBM Washington 127Q Eagle r3
        "rigetti_ankaa": PAPER_COLORS["orange"],            # Rigetti Ankaa-3
        "oqc_toshiko": PAPER_COLORS["teal_darkest"],        # OQC Toshiko 32Q
        
        # Simulated topologies - Gray/neutral
        "line": "#7F7F7F",         # Medium gray
        "ring": "#A0A0A0",         # Light gray
        "ladder": "#5F5F5F",       # Dark gray
    }


def get_hardware_markers() -> Dict[str, str]:
    """Get marker assignments for each hardware platform."""
    return {
        "quantinuum_helios": "p",  # Pentagon
        "ionq_forte": ">",         # Triangle right
        "fully_connected": "o",    # Circle
        "google_sycamore": "v",    # Triangle down
        "google_willow": "^",      # Triangle up
        "ibm_washington": "h",     # Hexagon
        "rigetti_ankaa": "<",      # Triangle left
        "oqc_toshiko": "D",        # Diamond
        "line": "s",               # Square
        "ring": "^",               # Triangle up
        "ladder": "d",             # Thin diamond
    }


def get_hardware_style(hardware_name: str) -> Dict[str, str]:
    """Get full plotting style for a hardware platform.
    
    Parameters
    ----------
    hardware_name : str
        Hardware name (may include suffix like '_10')
    
    Returns
    -------
    dict
        Style dict with 'color', 'marker', 'label' keys
    """
    import re
    # Extract base name (strip _N suffix and grid dimensions)
    base = re.sub(r"_\d+x\d+", "", hardware_name)
    base = re.sub(r"_\d+$", "", base)
    
    colors = get_hardware_colors()
    markers = get_hardware_markers()
    
    return {
        "color": colors.get(base, "#7F7F7F"),
        "marker": markers.get(base, "o"),
        "label": _format_hardware_label(base),
    }


def _format_hardware_label(base_name: str) -> str:
    """Format hardware name for plot labels."""
    full_label_map = {
        "quantinuum_helios": "Quantinuum Helios",
        "ionq_forte": "IonQ Forte",
        "fully_connected": "Ideal (FC)",
        "google_sycamore": "Google Sycamore",
        "google_willow": "Google Willow",
        "ibm_washington": "IBM Washington",
        "rigetti_ankaa": "Rigetti Ankaa-3",
        "oqc_toshiko": "OQC Toshiko",
        "line": "Line",
        "ring": "Ring",
        "ladder": "Ladder",
    }
    anon_label_map = {
        "quantinuum_helios": "FC_1",
        "ionq_forte": "FC_2",
        "fully_connected": "FC_REF",
        "google_willow": "SC_1",
        "google_sycamore": "SC_2",
        "ibm_washington": "SC_3",
        "rigetti_ankaa": "SC_4",
        "oqc_toshiko": "SC_5",
        "line": "SIM_L",
        "ring": "SIM_R",
        "ladder": "SIM_D",
    }
    label_map = anon_label_map if _DEVICE_LABEL_MODE == "anon" else full_label_map
    return label_map.get(base_name, base_name.replace("_", " ").title())


# =============================================================================
# COLOR SEQUENCES FOR DATA VISUALIZATION
# =============================================================================

def get_categorical_colors(n: int) -> List[str]:
    """Get n distinct colors from the paper palette.
    
    For up to 4 colors, uses the main accent palette.
    For more, cycles through teal gradients + accents.
    """
    base_colors = [
        PAPER_COLORS["quantum_teal"],
        PAPER_COLORS["purple"],
        PAPER_COLORS["coral"],
        PAPER_COLORS["orange"],
        PAPER_COLORS["teal_dark"],
        PAPER_COLORS["teal_darkest"],
        PAPER_COLORS["teal_light"],
    ]
    
    if n <= len(base_colors):
        return base_colors[:n]
    
    # Cycle if more colors needed
    return [base_colors[i % len(base_colors)] for i in range(n)]


def get_sequential_colors(n: int, palette: str = "teal") -> List[str]:
    """Get n colors in a sequential gradient.
    
    Parameters
    ----------
    n : int
        Number of colors needed
    palette : str
        'teal' for teal gradient, 'gray' for neutral
    
    Returns
    -------
    list
        List of hex color strings
    """
    if palette == "teal":
        endpoints = [
            PAPER_COLORS["teal_lightest"],
            PAPER_COLORS["teal_light"],
            PAPER_COLORS["quantum_teal"],
            PAPER_COLORS["teal_dark"],
            PAPER_COLORS["teal_darkest"],
        ]
    else:
        endpoints = ["#F0F0F0", "#C0C0C0", "#808080", "#404040", "#202020"]
    
    if n <= len(endpoints):
        step = max(1, len(endpoints) // n)
        return [endpoints[i * step] for i in range(n)]
    
    # Interpolate if more colors needed
    return _interpolate_colors(endpoints, n)


def _interpolate_colors(colors: List[str], n: int) -> List[str]:
    """Interpolate between a list of colors to get n colors."""
    import numpy as np
    
    def hex_to_rgb(h: str) -> Tuple[int, int, int]:
        h = h.lstrip('#')
        return cast(Tuple[int, int, int], tuple(int(h[i:i+2], 16) for i in (0, 2, 4)))
    
    def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
        return '#{:02x}{:02x}{:02x}'.format(*rgb)
    
    rgb_colors = [hex_to_rgb(c) for c in colors]
    result = []
    
    for i in range(n):
        t = i / max(1, n - 1) * (len(colors) - 1)
        idx = int(t)
        frac = t - idx
        
        if idx >= len(colors) - 1:
            result.append(colors[-1])
        else:
            c1 = rgb_colors[idx]
            c2 = rgb_colors[idx + 1]
            interp = cast(Tuple[int, int, int], tuple(int(c1[j] + frac * (c2[j] - c1[j])) for j in range(3)))
            result.append(rgb_to_hex(interp))
    
    return result


# =============================================================================
# MATPLOTLIB STYLE CONFIGURATION
# =============================================================================

def apply_paper_style(ax=None, *, grid_axis: str = "both", show_minor_grid: bool = False):
    """Apply the paper styling to matplotlib axes.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes, optional
        Axes to style. If None, styles the current axes.
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        ax = plt.gca()
    
    ax.set_facecolor(PAPER_COLORS["white"])

    # Axis styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(PAPER_COLORS["teal_darkest"])
    ax.spines['bottom'].set_color(PAPER_COLORS["teal_darkest"])
    ax.spines['left'].set_linewidth(1.1)
    ax.spines['bottom'].set_linewidth(1.1)
    
    # Grid
    ax.grid(True, axis=grid_axis, alpha=0.2, color=PAPER_COLORS["teal_light"], linestyle='-')
    if show_minor_grid:
        ax.minorticks_on()
        ax.grid(True, which='minor', axis=grid_axis, alpha=0.08, color=PAPER_COLORS["teal_light"])
    ax.set_axisbelow(True)
    
    # Tick styling
    ax.tick_params(colors=PAPER_COLORS["teal_darkest"], which='major', length=5, width=1.0)
    ax.tick_params(colors=PAPER_COLORS["teal_darkest"], which='minor', length=3, width=0.8)


def get_plot_style_preset(kind: str = "paper") -> Dict[str, Any]:
    """Return consistent plotting presets for publication-quality figures.

    The presets preserve the paper palette but normalize typography,
    figure sizing, line weights, and export quality.
    """
    base: Dict[str, Any] = {
        "font_family": "DejaVu Sans",
        "title_size": 19,
        "label_size": 15,
        "tick_size": 12,
        "legend_size": 11,
        "annotation_size": 11,
        "line_width": 2.3,
        "baseline_width": 1.8,
        "marker_size": 7,
        "marker_edge_width": 1.2,
        "error_capsize": 3,
        "legend_framealpha": 0.94,
        "save_dpi": 300,
    }

    if kind == "compilation":
        base.update({
            "figsize": (13.5, 8.0),
            "title_size": 20,
            "label_size": 15,
            "tick_size": 12,
            "legend_size": 11,
        })
    elif kind == "compilation_per_pattern":
        base.update({
            "font_family": "STIXGeneral",
            "figsize": (13.0, 7.2),
            "title_size": 28,
            "label_size": 27,
            "tick_size": 25,
            "legend_size": 20,
            "line_width": 3.0,
            "baseline_width": 2.9,
            "marker_size": 10.2,
            "marker_edge_width": 1.6,
            "error_capsize": 3.5,
            "legend_framealpha": 0.97,
        })
    elif kind == "phase_diagram":
        base.update({
            "figsize": (12.8, 9.2),
            "title_size": 32,
            "label_size": 30,
            "tick_size": 24,
            "legend_size": 18,
            "annotation_size": 16,
            "marker_size": 13.5,
            "marker_edge_width": 1.8,
            "error_capsize": 5,
        })
    else:
        base.update({"figsize": (10.0, 7.0)})

    return base


def get_figure_defaults() -> Dict:
    """Get recommended figure defaults for publication-quality plots."""
    return {
        "figsize": (10, 7),
        "dpi": 300,
        "facecolor": PAPER_COLORS["white"],
        "edgecolor": "none",
    }


def get_legend_defaults() -> Dict:
    """Get recommended legend defaults for clear, readable legends."""
    return {
        "fontsize": 11,
        "frameon": True,
        "framealpha": 0.94,
        "edgecolor": PAPER_COLORS["teal_light"],
        "fancybox": True,
        "loc": "best",
    }
