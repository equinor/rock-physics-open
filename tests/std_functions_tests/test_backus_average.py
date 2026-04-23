import numpy as np
from syrupy.assertion import SnapshotAssertion

from rock_physics_open.equinor_utilities.std_functions import backus_average


def test_backus_average(snapshot: SnapshotAssertion):
    vp1i = np.ones(10) * 3500.0
    vp2i = np.ones(10) * 2800.0
    vs1i = np.ones(10) * 1500.0
    vs2i = np.ones(10) * 1100.0
    rho1i = np.ones(10) * 2560.0
    rho2i = np.ones(10) * 2580.0
    f1 = np.linspace(0.0, 1.0, 10)

    assert snapshot == backus_average(vp1i, vs1i, rho1i, vp2i, vs2i, rho2i, f1)
