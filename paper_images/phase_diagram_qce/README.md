# Phase panels

This folder contains five PDFs: the four submitted phase panels (dense,
sparse and local at 32 qubits, and RHG at 90 qubits) plus the
[127-qubit RHG phase plot](phase_diagram_rhg_k2_n127_layers1_samples20_mean_standard_rerun_linear_anon.pdf).
The larger RHG result is discussed numerically in the paper; its plot is
included here as an accompanying figure. All five use the saved data in
[phase_diagram/](../phase_diagram/README.md).

Use the [verified plotting commands](../../REPRODUCING.md#phase-panels-from-saved-circuit-results)
to redraw all five panels. They use the short FC/SC labels and a linear vertical
range from -300 to 5,000. The negative margin is visual padding; zero is the
lowest labeled tick. The script's automatic limits produce a different layout,
so the guide supplies these limits explicitly.

The four submitted PDFs are unchanged and redraw exactly from the saved data.
The 127-qubit panel was generated with the same existing plotting function;
there is no submitted PDF counterpart to compare against. Its SC_3 mean depth
is 1,128, its native fully connected baseline depth is 54, and its depth overhead
rounds to 20.9. All 40 RHG-127 records matched in the full compilation rerun.

Fresh compilation can shift other plotted means and error bars because of the
compiler's time-limited placement search. The [data-folder README](../phase_diagram/README.md)
lists the observed changes. The included plots use the saved paper results.
