#!/usr/bin/env python3
"""Rerun and splice only SC_3 into the paper compilation-overhead CSVs.

The SC_3 correction uses the following experiment settings:
10 circuits per pattern, seed 1234, Qiskit logical circuits, pytket native
compilation, full-device placement, and pytket's default mapping strategy.
The RHG correction uses the same settings. Other standard-sweep rows were
sampled with seed 42; see REPRODUCING.md for the two-stage reconstruction.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from importlib.metadata import version
from pathlib import Path

import pandas as pd
import qiskit
import pytket

from iqp_swap_benchmark.analysis import ExperimentConfig, run_swap_experiment


STANDARD_QUBITS = [4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32]
RHG_QUBITS = [18, 44, 90, 127]
SC3_PREFIX = "ibm_washington_"


def _run_grid(
    qubit_counts: list[int],
    patterns: list[str],
    *,
    samples: int,
    seed: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for n_qubits in qubit_counts:
        config = ExperimentConfig(
            n_qubits=n_qubits,
            pattern_kinds=patterns,
            circuits_per_pattern=samples,
            hardware_names=[f"{SC3_PREFIX}{n_qubits}"],
            interaction_order=2,
            density=0.4 if "sparse" in patterns else 1.0,
            seed=seed,
            optimization_level=2,
            n_layers=1,
            logical_circuit_backend="qiskit",
            placement_strategy="default",
        )
        frame = run_swap_experiment(config)
        frame["n_qubits"] = n_qubits
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _splice_rows(original: pd.DataFrame, replacement: pd.DataFrame) -> pd.DataFrame:
    keys = ["pattern", "circuit_index", "hardware", "n_qubits"]
    old_sc3 = original[original["hardware"].str.startswith(SC3_PREFIX)].copy()
    if len(old_sc3) != len(replacement):
        raise ValueError(
            f"SC_3 row-count mismatch: existing={len(old_sc3)}, "
            f"replacement={len(replacement)}"
        )

    old_keys = set(map(tuple, old_sc3[keys].itertuples(index=False, name=None)))
    new_keys = set(map(tuple, replacement[keys].itertuples(index=False, name=None)))
    if old_keys != new_keys:
        raise ValueError("Replacement SC_3 keys do not match the existing CSV keys")

    updated = original.set_index(keys)
    fresh = replacement.set_index(keys)
    common_columns = [column for column in fresh.columns if column in updated.columns]
    existing = updated.loc[fresh.index, common_columns]
    fresh_values = fresh[common_columns]
    # A targeted hardware-only run cannot populate columns that require an
    # explicit fully-connected target in the same run. Retain those existing
    # values instead of replacing them with NaN.
    updated.loc[fresh.index, common_columns] = fresh_values.where(
        fresh_values.notna(), existing
    )
    return updated.reset_index()[original.columns]


def _splice_csv_text(
    original_path: Path,
    replacement_path: Path,
    output_path: Path,
) -> None:
    """Replace SC_3 CSV cells while preserving all unrelated source text."""
    keys = ["pattern", "circuit_index", "hardware", "n_qubits"]

    with replacement_path.open(newline="") as handle:
        replacement_reader = csv.reader(handle)
        replacement_header = next(replacement_reader)
        replacement_index = {name: i for i, name in enumerate(replacement_header)}
        replacements = {
            tuple(row[replacement_index[name]] for name in keys): row
            for row in replacement_reader
        }

    with original_path.open(newline="") as handle:
        original_reader = csv.reader(handle)
        original_header = next(original_reader)
        if original_header != replacement_header:
            raise ValueError("Original and replacement CSV columns do not match")
        original_index = {name: i for i, name in enumerate(original_header)}
        rows = list(original_reader)

    replaced_keys: set[tuple[str, ...]] = set()
    for index, row in enumerate(rows):
        if not row[original_index["hardware"]].startswith(SC3_PREFIX):
            continue
        key = tuple(row[original_index[name]] for name in keys)
        fresh = replacements.get(key)
        if fresh is None:
            raise ValueError(f"Missing replacement row for key {key}")
        # Empty fields in the targeted run are reference-dependent quantities;
        # retain their values from the original multi-hardware experiment.
        rows[index] = [new if new != "" else old for old, new in zip(row, fresh)]
        replaced_keys.add(key)

    if replaced_keys != set(replacements):
        raise ValueError("Replacement CSV contains keys absent from the original CSV")

    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(original_header)
        writer.writerows(rows)


def _backup_and_write(
    source_path: Path,
    replacement: pd.DataFrame,
    replacement_path: Path,
    backup_path: Path,
) -> None:
    if backup_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing backup: {backup_path}"
        )
    original = pd.read_csv(source_path)
    _splice_rows(original, replacement)  # Validate row keys and replacement data.
    shutil.copy2(source_path, backup_path)
    _splice_csv_text(backup_path, replacement_path, source_path)
    print(f"Updated {source_path}; backup saved to {backup_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("paper_images/compilation_overhead"),
    )
    args = parser.parse_args()

    standard = _run_grid(
        STANDARD_QUBITS,
        ["dense", "local", "sparse"],
        samples=args.samples,
        seed=args.seed,
    )
    rhg = _run_grid(
        RHG_QUBITS,
        ["rhg"],
        samples=args.samples,
        seed=args.seed,
    )

    standard_raw = args.base_dir / "results_sc3_default_rerun.csv"
    rhg_raw = args.base_dir / "results_rhg_sc3_default_rerun.csv"
    standard.to_csv(standard_raw, index=False)
    rhg.to_csv(rhg_raw, index=False)

    _backup_and_write(
        args.base_dir / "results_default.csv",
        standard,
        standard_raw,
        args.base_dir / "results_default_before_sc3_rerun.csv",
    )
    _backup_and_write(
        args.base_dir / "results_rhg_default.csv",
        rhg,
        rhg_raw,
        args.base_dir / "results_rhg_default_before_sc3_rerun.csv",
    )

    manifest = {
        "hardware_label": "SC_3",
        "hardware_base": "ibm_washington",
        "standard_qubits": STANDARD_QUBITS,
        "rhg_qubits": RHG_QUBITS,
        "samples": args.samples,
        "seed": args.seed,
        "logical_circuit_backend": "qiskit",
        "native_compiler": "pytket",
        "placement_strategy": "default",
        "full_device_placement": True,
        "optimization_level": 2,
        "n_layers": 1,
        "qiskit_version": qiskit.__version__,
        "pytket_version": pytket.__version__,
        "pytket_qiskit_version": version("pytket-qiskit"),
    }
    manifest_path = args.base_dir / "sc3_default_rerun_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved rerun manifest to {manifest_path}")


if __name__ == "__main__":
    main()
