# Graph-fragmentation panels used in the paper

`dense.pdf`, `sparse.pdf`, `local.pdf` and `rhg.pdf` are unchanged copies of
the four submitted panels. Despite the folder name, RHG uses **90 qubits**;
the other patterns use 32.

Run from the repository root:

```bash
python scripts/generate_percolation_32q.py --base-dir reproduced/percolation
```

The final panels are written to `reproduced/percolation/pattern_summaries/`.
The command also creates intermediate views. Standard patterns use 50 trials;
RHG uses 100. The example removal probabilities are 0, 0.25 and 0.5. Seeds are
fixed by the existing code.

All four regenerated panels matched the paper images pixel for pixel at
72 dpi. There is no known numerical or visible difference with these settings.
PDF metadata, such as timestamps, can change file bytes without changing
the rendered image.
