import os

import numpy as np

from rock_physics_open.equinor_utilities.snapshot_test_utilities import (
    INITIATE,
    compare_snapshots,
    get_snapshot_name,
    read_snapshot,
    store_snapshot,
)
from rock_physics_open.t_matrix_models import run_t_matrix, t_matrix_porosity_vectorised

k_min = np.ones(11) * 71.0e9
mu_min = np.ones(11) * 32.0e9
rho_min = np.ones(11) * 2710.0
k_fl = np.ones(11) * 2.7e9
rho_fl = np.ones(11) * 1005.0
phi = np.linspace(0.15, 0.35, 11)
perm = 100.0
visco = 100.0
alpha = np.array([0.8, 0.1, 0.01])
v = np.array([0.89, 0.1, 0.01])
tau = 1e-7 * np.ones_like(alpha)
frequency = 1000
angle = 90.0
frac_inc_con = 0.5
frac_inc_ani = 0.5
pressure = np.array([18.0e6, 28.0e6])


def test_run_t_matrix():
    args = run_t_matrix(
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
        pressure=pressure,
    )

    if not os.path.isfile(get_snapshot_name()) or INITIATE:
        store_snapshot(get_snapshot_name(), *args)
    else:
        assert compare_snapshots(args, read_snapshot(get_snapshot_name()))


def test_run_t_matrix_porosity_vectorised():
    args = t_matrix_porosity_vectorised(
        k_min,
        mu_min,
        rho_min,
        k_fl,
        rho_fl,
        phi,
        perm * np.ones_like(phi),
        visco * np.ones_like(phi),
        alpha,
        v,
        tau,
        frequency,
        angle,
        frac_inc_con,
        frac_inc_ani,
        pressure=pressure,
    )

    if not os.path.isfile(get_snapshot_name()) or INITIATE:
        store_snapshot(get_snapshot_name(), *args)
    else:
        assert compare_snapshots(args, read_snapshot(get_snapshot_name()))
