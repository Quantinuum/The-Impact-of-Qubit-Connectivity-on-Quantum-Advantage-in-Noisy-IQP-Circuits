# Paper figures and data

This folder contains **24 PDFs and seven CSVs**: the 23 unchanged submitted
figures, one accompanying 127-qubit RHG phase plot, and the unchanged saved
experiment data.
The paper is available on [arXiv](https://arxiv.org/abs/2604.12635).

| Folder | Contents |
| --- | --- |
| [compilation_overhead/](compilation_overhead/README.md) | Two scaling CSVs and four scaling panels |
| [phase_diagram/](phase_diagram/README.md) | Five CSVs used for the phase study and numerical table |
| [phase_diagram_qce/](phase_diagram_qce/README.md) | Four submitted phase panels and the 127-qubit RHG phase plot |
| [percolation/](percolation/README.md) | Four graph-fragmentation panels |
| [topologies/](topologies/README.md) | Ten connectivity diagrams |
| [iqp_schematics/](iqp_schematics/README.md) | The circuit schematic |

The 22 generated panels were redrawn and matched the paper images pixel for
pixel when rendered at 72 dpi. New PDF files may have different timestamps or
file bytes even when they look identical. The schematic is supplied as a
static PDF. The additional 127-qubit RHG panel was drawn from its saved CSV
using the same plotting code and axis settings; it has no submitted PDF
counterpart to compare against.

Fresh circuit compilation can change plotted points because the compiler's
placement search has a time limit. This is a numerical difference, not just a
formatting change. The scaling and phase folder READMEs give the measured
differences and explain the required settings. New runs should write to
`reproduced/`, keeping these paper files intact.

See the [generated output gallery](../reproduced/reference/README.md) for
regenerated PDFs and image previews.

See [REPRODUCING.md](../REPRODUCING.md) for commands and
[artifacts.json](../results/artifacts.json) for file checksums.
