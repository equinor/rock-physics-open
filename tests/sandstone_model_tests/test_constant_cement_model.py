import os

import numpy as np

from rock_physics_open.equinor_utilities.snapshot_test_utilities import (
    INITIATE,
    compare_snapshots,
    get_snapshot_name,
    read_snapshot,
    store_snapshot,
)
from rock_physics_open.sandstone_models import (
    constant_cement_model,
    constant_cement_model_dry,
)

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


def test_constant_cement_model_dry():
    args = constant_cement_model_dry(
        k_min, mu_min, k_cem, mu_cem, phi, frac_cem, phi_c, n, shear_red
    )

    if not os.path.isfile(get_snapshot_name()) or INITIATE:
        store_snapshot(get_snapshot_name(), *args)
    else:
        assert compare_snapshots(args, read_snapshot(get_snapshot_name()))


def test_constant_cement_model():
    args = constant_cement_model(
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


new_phi = np.linspace(0.0, 0.40, 20)
low_phi_c = 0.40
high_frac_cem = 0.10


def test_constant_cement_model_high_phi():
    args = constant_cement_model(
        k_min,
        mu_min,
        rho_min,
        k_cem,
        mu_cem,
        rho_cem,
        k_fl,
        rho_fl,
        new_phi,
        high_frac_cem,
        low_phi_c,
        n,
        shear_red,
    )
    # assert that there are NaN values in the output at the correct indices
    expected_idx_nan = new_phi > low_phi_c - high_frac_cem
    assert np.all(expected_idx_nan == np.isnan(args[0]))

    if not os.path.isfile(get_snapshot_name()) or INITIATE:
        store_snapshot(get_snapshot_name(), *args)
    else:
        assert compare_snapshots(args, read_snapshot(get_snapshot_name()))


def test_constant_cement_model_high_phi_extrapolate():
    # Set flag for extrapolation to True and assert that there are no NaN values in the output
    args = constant_cement_model(
        k_min,
        mu_min,
        rho_min,
        k_cem,
        mu_cem,
        rho_cem,
        k_fl,
        rho_fl,
        new_phi,
        high_frac_cem,
        low_phi_c,
        n,
        shear_red,
        extrapolate_to_max_phi=True,
    )
    assert not np.any(np.isnan(args[0]))

    if not os.path.isfile(get_snapshot_name()) or INITIATE:
        store_snapshot(get_snapshot_name(), *args)
    else:
        assert compare_snapshots(args, read_snapshot(get_snapshot_name()))
