# Reproduce the paper results

Activate the environment described in the README and run from the repository
root. [Generated examples and image previews](reproduced/reference/README.md)
are already included under `reproduced/reference/`. Commands below write your
own outputs to other folders under `reproduced/`, which Git ignores, preserving
both the examples and the archived paper results.

```bash
export MPLBACKEND=Agg
mkdir -p reproduced/compilation reproduced/phase
```

## Scaling panels from saved circuit results

```bash
python scripts/generate_compilation_overhead_plots.py \
  --input paper_images/compilation_overhead/results_default.csv \
  --output-dir reproduced/compilation --metric depth_ratio_native_fc \
  --extrapolation-fit-points 10 --extrapolation-model exponential \
  --extra-extrapolation-model power_law --extrapolate-to 70 \
  --rename --no-title --legend-size 24 --legend-columns 2 \
  --legend-columns-override local=3 \
  --legend-anchor-y 0.84 --legend-anchor-patterns dense sparse

python scripts/generate_compilation_overhead_plots.py \
  --input paper_images/compilation_overhead/results_rhg_default.csv \
  --output-dir reproduced/compilation --metric depth_ratio_native_fc \
  --extrapolation-fit-points 10 --extrapolation-model exponential \
  --extra-extrapolation-model power_law --extrapolate-to 127 \
  --rename --no-title --legend-size 24 --legend-columns 2 \
  --legend-anchor-y 0.84
```

Each measured point is the mean of per-instance `D_FC / D_H`; error bars use
the sample standard deviation. Dotted and cross-marked extensions are fitted
extrapolations, not additional compiled circuits.

## Phase panels from saved circuit results

The submitted linear panels have a common maximum of 5,000 and a minimum of
-300 for visual padding, with zero as the lowest labeled tick. The existing
CLI's automatically chosen shared limit differs, so use its existing plotting
function with the submitted limits explicitly:

```bash
python - <<'PY'
from pathlib import Path
from scripts.regenerate_sampled_phase_plots import _plot_from_csv
from iqp_swap_benchmark.colors import set_device_label_mode

set_device_label_mode("anon")
for pattern, n in [("dense", 32), ("sparse", 32), ("local", 32),
                   ("rhg", 90), ("rhg", 127)]:
    csv = Path(f"paper_images/phase_diagram/depth_by_k2_{pattern}_n{n}_layers1_samples20_mean_standard_rerun.csv")
    pdf = Path(f"reproduced/phase/{pattern}_{n}.pdf")
    _plot_from_csv(csv, pdf, n_qubits=n, n_layers=1,
                   use_log_scale=False, y_max=5000, y_min=-300)
PY
```

The n=127 data support the numerical result in the manuscript. Its panel is
included as an accompanying plot in [phase_diagram_qce/](paper_images/phase_diagram_qce/README.md),
alongside the four submitted phase panels. The loop above redraws all five.

## Percolation and topology panels

```bash
python scripts/generate_percolation_32q.py --base-dir reproduced/percolation

python - <<'PY'
from scripts.generate_topology_figure import export_device_topology_panels
export_device_topology_panels("reproduced/topologies")
PY
```

The percolation entry point produces combined views, individual panels and
final summaries. The summaries are in `reproduced/percolation/pattern_summaries/`. Dense/sparse/local use 32
qubits and 50 trials; RHG uses 90 qubits and 100 trials. The three exemplar
removal probabilities are 0, 0.25 and 0.5.

Topology output filenames use the public device identifiers:

| Paper filename stem | Generator filename stem |
| --- | --- |
| `fc_1` | `quantinuum_helios` |
| `fc_2` | `ionq_forte` |
| `sc_1` | `google_willow` |
| `sc_2` | `google_sycamore` |
| `sc_3` | `ibm_washington` |
| `sc_4` | `rigetti_ankaa` |
| `sc_5` | `oqc_toshiko` |
| `line`, `ring`, `ladder` | unchanged |

## Fresh compilation experiments

Run the standard and RHG scaling sweeps with seed 42, then calculate the SC_3
scaling samples with seed 1234. The phase study uses seed 42. Keep the pattern
order shown below because it determines the random sampling sequence. The RHG
seed follows the generator default; the saved baseline counts alone cannot
independently identify its original angles.

```bash
python scripts/run_swap_benchmark.py \
  --n-qubits 4 6 8 10 12 14 16 18 20 24 28 32 \
  --patterns dense local sparse --circuits-per-pattern 10 \
  --seed 42 --logical-circuit-backend qiskit --placement default \
  --output reproduced/compilation/results_default.csv

python scripts/run_swap_benchmark.py \
  --n-qubits 18 44 90 127 --patterns rhg --circuits-per-pattern 10 \
  --seed 42 --logical-circuit-backend qiskit --placement default \
  --output reproduced/compilation/results_rhg_default.csv

python scripts/rerun_sc3_compilation_overhead.py \
  --samples 10 --seed 1234 --base-dir reproduced/compilation

mkdir -p reproduced/phase-run
cd reproduced/phase-run
python ../../scripts/generate_sampled_phase_diagrams.py \
  --n-qubits 32 --n-layers 1 --circuits-per-pattern 20 \
  --seed 42 --logical-circuit-backend qiskit --placement default \
  --tag samples20_mean_standard_rerun --rename
cd ../..
```

The scaling CSVs retain the original generator's all-to-all reference rows
above the nominal FC_1/FC_2 device capacities (FC_1: 98, FC_2: 36). These rows
and their plotted continuations are part of the submitted scaling artifacts;
they are mathematical all-to-all models, not evidence of larger physical
devices. The phase-study generator enforces these capacities. SC_5 is a
32-qubit proxy graph, as stated in the paper.

The correction runner preserves pre-correction backups and refuses to overwrite
existing backups. Use a fresh output directory for each reconstruction.
Because the correction uses a different seed, sparse samples are not matched
between SC_3 and the other targets. Its legacy `*_ratio_fc`, `*_diff_fc` and
`fully_connected_*` columns retain values from the original run. The paper uses
the internally consistent `native_fc_*` measurements and
`depth_ratio_native_fc`; do not substitute the legacy FC-relative columns.

A full run with these settings was compared with the saved paper results.
Row comparisons and differences at the paper's rounding precision are in
[validation.json](results/validation.json).

All 5,100 native-FC baseline measurements match the archive, including baseline
depth and gate counts. Fresh routed depths match in 4,854 records; 246 differ.

| Dataset | Records | Exact routed-depth matches |
| --- | ---: | ---: |
| Standard scaling, after SC_3 correction | 3,960 | 3,766 |
| RHG scaling, after SC_3 correction | 360 | 323 |
| Phase study, all five configurations | 780 | 765 |

### Why identical inputs can produce different routed depths

The pinned pytket 2.15.0 `DefaultMappingPass(...).to_dict()` records GraphPlacement
with `timeout=100` milliseconds, `matches=2000`, and maximum pattern depth and
gate count both 100. This is the configuration of the actual default pass,
which differs from the standalone GraphPlacement constructor defaults.
GraphPlacement's timeout limits its placement search, as described in the
[public API documentation](https://docs.quantinuum.com/tket/api-docs/placement.html).

Repeated compilation of the same 32-qubit sparse circuit produced depths
1,668 and 1,355 with identical circuit inputs and compiler settings. Its native
baseline depth stayed at 346. A fixed sampling seed therefore does not guarantee
identical routing outcomes. System load can also affect a time-limited search.

Use the saved CSVs to reproduce the reported numbers and plots. Compare freshly
compiled results against those CSVs when evaluating a new run.

In the phase comparison, 765 of 780 routed depths matched exactly and all
780 native-FC baseline depths matched. All local and RHG phase rows matched.
Five of the 25 reported hardware/pattern mean depths changed at integer
rounding precision; two overhead values changed at one decimal place. The
per-entry comparisons are included in the validation report. Passing the test
suite validates the archived paper values and the compilation pipeline; it does
not imply that fresh routed results match the paper exactly.

The phase study restarts the sampling RNG for each pattern; the standard
scaling study traverses patterns in `dense local sparse` order. Changing this
order changes the sampled circuits. Keep the archived CSVs as the reference
when evaluating fresh runs, especially across compiler/platform changes.
