# Results and reproducibility

The [paper figures and circuit data](../paper_images/README.md) and
[generated figure gallery](../reproduced/reference/README.md) are ready to view.

| File | Contents |
| --- | --- |
| `artifacts.json` | SHA-256 checksums for the 24 reference PDFs and seven CSVs |
| `reproduced_artifacts.json` | Checksums and image comparisons for 23 generated PDFs and PNG previews |
| `sc3_rerun_provenance.json` | Compiler versions, seed and settings for the SC_3 scaling samples |
| `validation.json` | Test results and numerical comparisons with a fresh compilation run |

All 23 generated plots match their reference images at 72 dpi. Scaling and
phase plots use the saved CSVs; percolation and topology plots use their
generators. The circuit schematic is a static figure.

A full compilation run matched all 5,100 baseline depths and 4,854 routed depths.
The compiler's time-limited placement search can change routed depths, including
five rounded mean depths and two rounded overheads in the paper's table.
The [scaling](../paper_images/compilation_overhead/README.md) and
[phase](../paper_images/phase_diagram/README.md) notes give the comparisons.

See [REPRODUCING.md](../REPRODUCING.md) for commands and parameter settings.
