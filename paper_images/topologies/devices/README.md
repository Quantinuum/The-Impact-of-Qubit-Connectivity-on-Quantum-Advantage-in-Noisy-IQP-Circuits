# Paper connectivity diagrams

The ten PDFs are unchanged copies of the paper's connectivity diagrams.
All ten regenerated diagrams matched the paper images pixel for pixel at
72 dpi. The generator uses public model names in filenames:

| Paper PDF | Generated filename |
| --- | --- |
| `fc_1.pdf` | `quantinuum_helios.pdf` |
| `fc_2.pdf` | `ionq_forte.pdf` |
| `sc_1.pdf` | `google_willow.pdf` |
| `sc_2.pdf` | `google_sycamore.pdf` |
| `sc_3.pdf` | `ibm_washington.pdf` |
| `sc_4.pdf` | `rigetti_ankaa.pdf` |
| `sc_5.pdf` | `oqc_toshiko.pdf` |
| `line.pdf`, `ring.pdf`, `ladder.pdf` | Same names |

Run from the repository root:

```bash
python - <<'PY'
from scripts.generate_topology_figure import export_device_topology_panels
export_device_topology_panels("reproduced/topologies")
PY
```

The filename differences do not change the diagrams. New PDF timestamps can
also change file hashes without changing the image. SC_5 is the paper's
32-qubit proxy graph. These diagrams show the connectivity models used in
the paper.
