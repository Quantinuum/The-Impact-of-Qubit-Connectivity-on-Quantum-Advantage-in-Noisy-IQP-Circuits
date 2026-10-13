# Phase-study data and paper table

The five CSVs contain the 780 circuit records used for the phase study:
dense, sparse and local at 32 qubits, and RHG at 90 and 127 qubits.
Each hardware/pattern configuration has 20 circuits. These files are unchanged
from the paper results. The plotted PDFs are in
[phase_diagram_qce/](../phase_diagram_qce/README.md).

To redraw the plots, use the saved CSVs with the
[phase plotting commands](../../REPRODUCING.md#phase-panels-from-saved-circuit-results).
The numerical table uses the mean of `depth`, rounded to an integer. Its
overhead is `mean(depth) / mean(native_fc_depth)`, rounded to one decimal;
it is not the mean of per-circuit ratios.

## Differences in a fresh run

The full rerun used seed 42, restarting the random-number generator for each
pattern, just as the phase script does. All 780 baseline measurements matched.
Routed depths matched in 218/220 dense, 207/220 sparse, 220/220 local,
80/80 RHG-90 and 40/40 RHG-127 records.

The following paper table entries changed in that fresh run:

| Pattern / model | Paper mean depth | Fresh mean depth, rounded | Paper overhead | Fresh overhead |
| --- | ---: | ---: | ---: | ---: |
| dense / SC_3 | 1028 | 999 | 5.4 | 5.3 |
| dense / SC_5 | 3148 | 3143 | 7.3 | 7.2 |
| sparse / SC_1 | 1298 | 1305 | 3.7 | 3.7 |
| sparse / SC_3 | 679 | 680 | 5.1 | 5.1 |
| sparse / SC_5 | 1837 | 1844 | 6.0 | 6.0 |

The compiler's default placement search is limited to 100 ms. Repeated calls
with identical circuits can choose different placements, changing the circuit
depth and the plotted mean or error bar. This was checked directly; it is not
just a PDF formatting issue. Use these saved CSVs as the reference for
comparisons with newly compiled results.

The other 20 rounded mean depths and 23 of the 25 rounded overheads matched.
See [validation.json](../../results/validation.json) for every comparison and
[REPRODUCING.md](../../REPRODUCING.md#fresh-compilation-experiments) for full-run
commands. The 127-qubit RHG data support the paper's numerical result, although
its separate plot was not included in the paper. This release includes that
[accompanying plot](../phase_diagram_qce/phase_diagram_rhg_k2_n127_layers1_samples20_mean_standard_rerun_linear_anon.pdf)
using these saved data and the same settings as the other phase panels.
