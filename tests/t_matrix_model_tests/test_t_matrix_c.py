import os

import numpy as np

from rock_physics_open.equinor_utilities.snapshot_test_utilities import (
    INITIATE,
    compare_snapshots,
    get_snapshot_name,
    read_snapshot,
    store_snapshot,
)
from rock_physics_open.t_matrix_models import (
    t_matrix_porosity_c_alpha_v,
    t_matrix_porosity_vectorised,
)

k_min = np.ones(21) * 71.0e9
mu_min = np.ones(21) * 32.0e9
rho_min = np.ones(21) * 2710.0
k_fl = np.ones(21) * 2.7e9
rho_fl = np.ones(21) * 1005.0
phi = np.linspace(0.15, 0.25, 21)
perm = np.ones(21) * 100
visco = np.ones(21) * 100
alpha = np.array([0.9, 0.1])
v = np.array([0.9, 0.1])
tau = 1.0e-7 * np.ones_like(alpha)
frequency = 1000
angle = 90.0
frac_inc_con = 0.5
frac_inc_ani = 0.5


def test_t_matrix_c():
    args = t_matrix_porosity_c_alpha_v(
        k_min,
        mu_min,
        rho_min,
        k_fl,
        rho_fl,
        phi,
        perm,
        visco,
        alpha,
        v,
        float(tau[0]),
        frequency,
        angle,
        frac_inc_con,
        frac_inc_ani,
    )

    if not os.path.isfile(get_snapshot_name()) or INITIATE:
        store_snapshot(get_snapshot_name(), *args)
    else:
        assert compare_snapshots(args, read_snapshot(get_snapshot_name()))


def test_t_matrix_vectorised():
    args = t_matrix_porosity_vectorised(
        k_min,
        mu_min,
        rho_min,
        k_fl,
        rho_fl,
        phi,
        perm,
        visco,
        alpha,
        v,
        tau,
        frequency,
        angle,
        frac_inc_con,
        frac_inc_ani,
    )

    if not os.path.isfile(get_snapshot_name()) or INITIATE:
        store_snapshot(get_snapshot_name(), *args)
    else:
        assert compare_snapshots(args, read_snapshot(get_snapshot_name()))
