import numpy as np
from syrupy.assertion import SnapshotAssertion

from rock_physics_open.equinor_utilities.std_functions import hertz_mindlin


def test_hertz_mindlin(snapshot: SnapshotAssertion):
    k1 = np.ones(11) * 36.6e9
    mu1 = np.ones(11) * 44.0e9
    phi_c = np.ones(11) * 0.4
    p = np.ones(11) * 30e6
    shear_red = np.linspace(0.0, 1.0, 11)
    n = np.ones(11) * 7.5
    assert snapshot == hertz_mindlin(k1, mu1, phi_c, p, shear_red, n)
