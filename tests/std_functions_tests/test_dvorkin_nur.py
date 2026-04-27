import numpy as np
from syrupy.assertion import SnapshotAssertion

from rock_physics_open.equinor_utilities.std_functions import dvorkin_contact_cement


def test_dvorkin_nur(snapshot: SnapshotAssertion):
    frac_cem = np.linspace(0.01, 0.1, 10)
    por0_sst = 0.4 * np.ones(10)
    mu0_sst = 44e9 * np.ones(10)
    k0_sst = 36.8e9 * np.ones(10)
    mu0_cem = 32e9 * np.ones(10)
    k0_cem = 71e9 * np.ones(10)
    vs_red = 0.25 * np.ones(10)
    c = 9.0
    assert snapshot == dvorkin_contact_cement(
        frac_cem, por0_sst, mu0_sst, k0_sst, mu0_cem, k0_cem, vs_red, c
    )
