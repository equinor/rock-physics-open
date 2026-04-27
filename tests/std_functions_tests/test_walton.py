import numpy as np
from syrupy.assertion import SnapshotAssertion

from rock_physics_open.equinor_utilities.std_functions import walton_smooth


def test_walton_smooth(snapshot: SnapshotAssertion):
    k = np.ones(11) * 36.8e9
    mu = np.ones(11) * 44.0e9
    phi = np.linspace(0.1, 0.36, 11)
    p_eff = np.ones(11) * 15.0e6
    assert snapshot == walton_smooth(k, mu, phi, p_eff)


def test_walton_smooth_n(snapshot: SnapshotAssertion):
    k = np.ones(11) * 36.8e9
    mu = np.ones(11) * 44.0e9
    phi = np.linspace(0.1, 0.36, 11)
    p_eff = np.ones(11) * 15.0e6
    n = 8.5
    assert snapshot == walton_smooth(k, mu, phi, p_eff, coord=n)
