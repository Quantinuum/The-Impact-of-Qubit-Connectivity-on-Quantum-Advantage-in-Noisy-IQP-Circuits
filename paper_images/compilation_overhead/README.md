# Compilation depth as circuit size increases

- `results_default.csv`: 3,960 records for dense, local and sparse patterns,
  with 10 circuits at each of 12 sizes from 4 to 32 qubits.
- `results_rhg_default.csv`: 360 records for RHG circuits at 18, 44, 90 and
  127 qubits, with 10 circuits per size and hardware model.
- [default/](default/README.md): the four scaling panels used in the paper.

The saved CSVs and PDFs match the paper release exactly. To redraw them, use
the [scaling commands](../../REPRODUCING.md#scaling-panels-from-saved-circuit-results)
with these CSVs as inputs.

## Why a fresh run can differ

The original standard run used seed 42. A later SC_3-only correction used
seed 1234. Repeating a single seed for every target does not reproduce this
sequence. The full-run guide includes both steps and the required pattern
order: `dense local sparse`.

Even with these settings, pytket's default placement search has a 100 ms time
limit. Identical circuits can receive different placements and routed depths.
The corrected full rerun matched:

| Data | Baseline depths | Routed depths |
| --- | ---: | ---: |
| Standard patterns | 3,960 / 3,960 | 3,766 / 3,960 |
| RHG | 360 / 360 | 323 / 360 |

The largest routed-depth differences were 582 layers for the standard data and
95 for RHG. See the [validation report](../../results/validation.json) for
the numerical comparisons.

## Which columns to use

The paper's plotted efficiency is `native_fc_depth / depth`, stored in
`depth_ratio_native_fc`. Each point averages the individual circuit ratios;
error bars show their sample standard deviation. The table uses the ratio of
mean depths instead.

Use the `native_fc_*` columns for the baseline in the same gate set. The older
`fully_connected_*`, `*_ratio_fc` and `*_diff_fc` columns were kept during the
SC_3 correction and should not be substituted for the paper's native baseline.
SC_3 sparse samples use a different seed from the other targets.

The RHG scaling data retain all-to-all model rows beyond the named devices'
physical sizes. Those are mathematical connectivity models. The phase study
applies the device-size limits. The RHG initial seed is taken from the old
generator's default of 42; the original angles were not saved separately.
