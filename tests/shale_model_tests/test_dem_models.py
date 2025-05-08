import os

import numpy as np

from rock_physics_open.equinor_utilities.snapshot_test_utilities import (
    INITIATE,
    compare_snapshots,
    get_snapshot_name,
    read_snapshot,
    store_snapshot,
)
from rock_physics_open.shale_models import (
    dem_model,
    dem_model_dual_por,
    shale_4_min_dem_overlay,
    shale_model_4_mineral_dem,
)

k1 = 36.8e9 * np.ones(20)
mu1 = 44.0e9 * np.ones(20)
rho1 = 2650 * np.ones(20)
k2 = 15.0e9 * np.ones(20)
mu2 = 7.5e9 * np.ones(20)
rho2 = 2680 * np.ones(20)
k3 = 71.0e9 * np.ones(20)
mu3 = 32.5e9 * np.ones(20)
rho3 = 2720 * np.ones(20)
k4 = 94.9e9 * np.ones(20)
mu4 = 45.0e9 * np.ones(20)
rho4 = 2870 * np.ones(20)
k_fl = 2.7e9 * np.ones(20)
rho_fl = 1005 * np.ones(20)
phi = np.linspace(0.0, 0.20, 20)
frac2 = np.linspace(0.0, 1.0, 20)
asp_1 = 0.85 * np.ones(20)
asp_2 = 0.15 * np.ones(20)
f1 = 0.5 * np.linspace(0.0, 1.0, 20)
f2 = 0.25 * np.linspace(1.0, 0.0, 20)
f3 = 0.25 * np.linspace(1.0, 0.0, 20)
frac_inc = np.linspace(0.0, 0.4, 20)
frac_inc_1 = np.linspace(0.3, 0.2, 20)
asp1 = 0.5 * np.ones(20)
rhob_inp = 2700 * np.ones(20)
asp2 = 0.85 * np.ones(20)
asp3 = 0.25 * np.ones(20)
asp4 = 0.05 * np.ones(20)
asp = 0.85 * np.ones(20)
tol = 1.0e-6
prop_clay = 0.33


def test_dem_model():
    args = dem_model(k1, mu1, rho1, k2, mu2, rho2, frac2, asp_2, tol)

    if not os.path.isfile(get_snapshot_name()) or INITIATE:
        store_snapshot(get_snapshot_name(), *args)
    else:
        assert compare_snapshots(args, read_snapshot(get_snapshot_name()))


def test_dem_dual_por_model():
    args = dem_model_dual_por(
        k1,
        mu1,
        rho1,
        k2,
        mu2,
        rho2,
        k3,
        mu3,
        rho3,
        frac_inc,
        frac_inc_1,
        asp_1,
        asp_2,
        tol,
    )

    if not os.path.isfile(get_snapshot_name()) or INITIATE:
        store_snapshot(get_snapshot_name(), *args)
    else:
        assert compare_snapshots(args, read_snapshot(get_snapshot_name()))


def test_dem_4_min_model():
    args = shale_model_4_mineral_dem(
        k1,
        mu1,
        rho1,
        k2,
        mu2,
        rho2,
        k3,
        mu3,
        rho3,
        k4,
        mu4,
        rho4,
        k_fl,
        rho_fl,
        phi,
        f1,
        f2,
        f3,
        rhob_inp,
        asp1,
        asp2,
        asp3,
        asp4,
        asp,
        mod_type="SCA",
    )

    if not os.path.isfile(get_snapshot_name()) or INITIATE:
        store_snapshot(get_snapshot_name(), *args)
    else:
        assert compare_snapshots(args, read_snapshot(get_snapshot_name()))


def test_dem_4_min_overlay_model():
    args = shale_4_min_dem_overlay(
        k1,
        mu1,
        rho1,
        k2,
        mu2,
        rho2,
        k3,
        mu3,
        rho3,
        k4,
        mu4,
        rho4,
        k_fl,
        rho_fl,
        phi,
        f1,
        f2,
        prop_clay,
        asp,
    )

    if not os.path.isfile(get_snapshot_name()) or INITIATE:
        store_snapshot(get_snapshot_name(), *args)
    else:
        assert compare_snapshots(args, read_snapshot(get_snapshot_name()))
