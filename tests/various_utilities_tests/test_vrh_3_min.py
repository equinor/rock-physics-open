import numpy as np
from numpy.random import default_rng
from syrupy.assertion import SnapshotAssertion

from rock_physics_open.equinor_utilities.various_utilities import min_3_voigt_reuss_hill


def test_3_mineral_voigt_reuss_hill(snapshot: SnapshotAssertion):
    rg = default_rng(260363)
    f1 = 0.33 * rg.random(11)
    f2 = 0.33 * rg.random(11)
    f3 = 0.33 * rg.random(11)
    vp1 = 6050.0 * np.ones(11)
    vs1 = 4090.0 * np.ones(11)
    rho1 = 2650.0 * np.ones(11)
    vp2 = 6640.0 * np.ones(11)
    vs2 = 3440.0 * np.ones(11)
    rho2 = 2710.0 * np.ones(11)
    vp3 = 7340.0 * np.ones(11)
    vs3 = 3960.0 * np.ones(11)
    rho3 = 2870.0 * np.ones(11)
    assert snapshot == min_3_voigt_reuss_hill(
        vp1, vs1, rho1, f1, vp2, vs2, rho2, f2, vp3, vs3, rho3, f3
    )
