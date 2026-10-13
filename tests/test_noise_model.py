import numpy as np
import pytest

from iqp_swap_benchmark.noise_model import critical_depth


def test_critical_depth_uses_logarithm_in_numerator() -> None:
    p = 0.01
    k = 2
    c = 1.5

    assert critical_depth(p, k, c) == pytest.approx(c * np.log(k / p) / p)


@pytest.mark.parametrize("p", [0.0, 1.0, -0.1, 1.1])
def test_critical_depth_rejects_invalid_noise(p: float) -> None:
    with pytest.raises(ValueError, match="Noise parameter p"):
        critical_depth(p, 2)


def test_critical_depth_rejects_invalid_locality() -> None:
    with pytest.raises(ValueError, match="Locality k"):
        critical_depth(0.01, 0)
