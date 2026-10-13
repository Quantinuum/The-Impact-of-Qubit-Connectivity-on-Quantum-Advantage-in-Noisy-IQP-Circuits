"""Hardware-specific error rates for effective noise parameter p_eff.

These values are representative 2-qubit gate error rates from literature and
vendor specifications. They are used to compute the effective noise parameter
p_eff for each hardware topology when evaluating the simulatability margin.

References:
- IonQ: ~0.4% two-qubit error (Forte, MS gate)
- Quantinuum: ~0.079% two-qubit error (Helios/H2, ZZPhase gate)
- Google Sycamore: ~0.6% two-qubit error (SYC gate)
- Google Willow: ~0.33% two-qubit error (CZ gate)
- IBM Washington: ~1.15% two-qubit error (CX gate, median from FakeWashingtonV2)
- Rigetti Ankaa-3: 1.94% adopted effective-error proxy for this study
- OQC Toshiko: ~4.6% two-qubit error (ECR gate)

For simulated topologies (line, ring, ladder, rect_grid, fully_connected),
we use an idealized low error rate to isolate the effect of connectivity.

Paper notation:
- p_eff(H): effective per-layer noise parameter for hardware H
- Used in: D*(p,k) = c * log(k/p) / p critical depth calculation
"""

from __future__ import annotations

from typing import Dict

# Two-qubit gate error rates (literature values)
HARDWARE_ERROR_RATES: Dict[str, float] = {

    # trapped-ion systems (all-to-all connectivity)
    "quantinuum_helios": 0.00079, # Quantinuum Helios (H2), ZZPhase gate
    "ionq_forte": 0.004,           # IonQ Forte 36Q, MS gate

    # superconducting architectures
    "google_willow": 0.0033,      # Google Willow 105Q, CZ gate
    "google_sycamore": 0.006,     # Google Sycamore 53Q, SYC gate
    "oqc_toshiko": 0.046,         # OQC Toshiko 32Q, ECR gate (4.6% error)
    "ibm_washington": 0.01146,    # IBM Washington 127Q Eagle r3, CX gate (median from FakeWashington)
    "rigetti_ankaa": 0.0194,       # Adopted ISWAP effective-error proxy

    # simulated connectivity (idealized error rate)
    "fully_connected": 0.001,
    "line": 0.001,
    "ring": 0.001,
    "ladder": 0.001,
    "rect_grid": 0.001,
}

def get_error_rate(hardware_name: str) -> float:
    """Get the effective 2-qubit error rate for a hardware type.
    
    Parameters
    ----------
    hardware_name : str
        Hardware name (may include qubit suffix like 'ionq_10')
    
    Returns
    -------
    float
        Two-qubit gate error rate p_eff
    """
    import re
    # Strip qubit count and grid dimensions
    base = re.sub(r"_\d+x\d+", "", hardware_name)
    base = re.sub(r"_\d+$", "", base)
    
    if base in HARDWARE_ERROR_RATES:
        return HARDWARE_ERROR_RATES[base]
    
    # Default for unknown hardware
    return 0.01  # 1% default


def critical_depth(p: float, k: int, c: float = 1.0) -> float:
    """Compute critical depth D*(p, k) from noisy IQP simulatability theory.
    
    The critical depth is the threshold above which a noisy IQP circuit
    becomes efficiently classically simulatable.
    
    D*(p, k) = c * log(k/p) / p
    
    Parameters
    ----------
    p : float
        Effective per-layer noise parameter (0 < p < 1)
    k : int
        Interaction locality (max qubits per term)
    c : float
        Proportionality constant (default 1.0 for normalized scale)
    
    Returns
    -------
    float
        Critical depth threshold
    
    References
    ----------
    Eq. (3) in paper: d* ~ O(p^{-1} log(k/p))
    """
    import numpy as np
    if p <= 0 or p >= 1:
        raise ValueError(f"Noise parameter p must be in (0, 1), got {p}")
    if k < 1:
        raise ValueError(f"Locality k must be >= 1, got {k}")
    
    return c * np.log(k / p) / p


def simulatability_margin(depth_H: float, p_eff: float, k: int, c: float = 1.0) -> float:
    """Compute simulatability margin m(H) = D*(p_eff, k) - D_H.
    
    Parameters
    ----------
    depth_H : float
        Physical circuit depth on hardware H
    p_eff : float
        Effective noise parameter for hardware H
    k : int
        Interaction locality
    c : float
        Proportionality constant for D*
    
    Returns
    -------
    float
        Margin m(H): positive = potentially hard, negative = simulatable
    
    References
    ----------
    Eq. (7) in paper: m(H) = D*(p_eff(H), k) - D_H
    """
    d_star = critical_depth(p_eff, k, c)
    return d_star - depth_H
