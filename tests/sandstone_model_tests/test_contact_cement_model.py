import os

import numpy as np

from rock_physics_open.equinor_utilities.snapshot_test_utilities import (
    INITIATE,
    compare_snapshots,
    get_snapshot_name,
    read_snapshot,
    store_snapshot,
)
from rock_physics_open.sandstone_models import contact_cement_model

k_min = 36.8e9 * np.ones(20)
mu_min = 44.0e9 * np.ones(20)
rho_min = 2650 * np.ones(20)
k_cem = 36.8e9 * np.ones(20)
mu_cem = 44.0e9 * np.ones(20)
rho_cem = 2650 * np.ones(20)
k_fl = 2.7e9 * np.ones(20)
rho_fl = 1005 * np.ones(20)
phi = np.linspace(0.0, 0.45, 20)
frac_cem = 0.05
phi_c = 0.45
n = 8.0
shear_red = 0.25


def test_contact_cement_model():
    args = contact_cement_model(
        k_min,
        mu_min,
        rho_min,
        k_cem,
        mu_cem,
        rho_cem,
        k_fl,
        rho_fl,
        phi,
        frac_cem,
        phi_c,
        n,
        shear_red,
    )

    if not os.path.isfile(get_snapshot_name()) or INITIATE:
        store_snapshot(get_snapshot_name(), *args)
    else:
        assert compare_snapshots(args, read_snapshot(get_snapshot_name()))
