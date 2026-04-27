import numpy as np
from numpy.random import default_rng
from syrupy.assertion import SnapshotAssertion

from rock_physics_open.equinor_utilities.various_utilities import hs_average


def test_hashin_shtrikman_average(snapshot: SnapshotAssertion):
    rg = default_rng(42)
    f = rg.random(10)
    k1 = 36.8e9 * np.ones_like(f)
    mu1 = 44.0e9 * np.ones_like(f)
    rho1 = 2650.0 * np.ones_like(f)
    k2 = 71.2e9 * np.ones_like(f)
    mu2 = 32.0e9 * np.ones_like(f)
    rho2 = 2710.0 * np.ones_like(f)
    assert snapshot == hs_average(k1, mu1, rho1, k2, mu2, rho2, f)
