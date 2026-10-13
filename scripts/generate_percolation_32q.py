#!/usr/bin/env python3
"""Generate the 32-qubit percolation figure set and its RHG companion.

Dense, sparse and local patterns use 32 qubits, k=2 and 50 trials. The
RHG summary uses 90 qubits and 100 trials. Fixed seeds keep the graph examples
and the transition statistics reproducible.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.generate_percolation_figure import (
    export_percolation_singles,
    plot_percolation_combined,
    plot_transition_combined,
)
from scripts.generate_percolation_pattern_panels import (
    export_pattern_percolation_panels,
)
from scripts.generate_percolation_pattern_summaries import export_pattern_summaries
from scripts.generate_rhg_percolation import plot_rhg_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the paper percolation figures."
    )
    parser.add_argument("--n-qubits", type=int, default=32)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--rhg-n-qubits", type=int, default=90)
    parser.add_argument("--rhg-n-trials", type=int, default=100)
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("paper_images/percolation_32_q_alternative"),
    )
    args = parser.parse_args()

    noise_levels = [0.0, 0.25, 0.5]
    args.base_dir.mkdir(parents=True, exist_ok=True)

    combined = plot_percolation_combined(
        output_path=str(args.base_dir / "percolation_combined.pdf"),
        n_qubits=args.n_qubits,
        k=args.k,
        noise_levels=noise_levels,
    )
    plt.close(combined)

    transition = plot_transition_combined(
        output_path=str(args.base_dir / "transition_combined.pdf"),
        n_qubits=args.n_qubits,
        k=args.k,
        n_trials=args.n_trials,
    )
    plt.close(transition)

    export_percolation_singles(
        base_dir=args.base_dir / "singles",
        n_qubits=args.n_qubits,
        k=args.k,
        noise_levels=noise_levels,
        n_trials=args.n_trials,
    )
    export_pattern_percolation_panels(
        base_dir=args.base_dir / "pattern_panels",
        n_qubits=args.n_qubits,
        k=args.k,
        noise_levels=noise_levels,
    )
    export_pattern_summaries(
        base_dir=args.base_dir / "pattern_summaries",
        n_qubits=args.n_qubits,
        k=args.k,
        noise_levels=noise_levels,
        n_trials=args.n_trials,
        show_pattern_title=False,
    )
    plot_rhg_summary(
        output_path=args.base_dir / "pattern_summaries" / "rhg.pdf",
        n_qubits=args.rhg_n_qubits,
        noise_levels=noise_levels,
        n_trials=args.rhg_n_trials,
        show_pattern_title=False,
    )

    print(f"Saved 32-qubit percolation figure set to {args.base_dir}")


if __name__ == "__main__":
    main()
