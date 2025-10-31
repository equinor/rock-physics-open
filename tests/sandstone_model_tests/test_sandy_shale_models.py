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
    cemented_shaly_sand_sandy_shale_model,
    friable_shaly_sand_sandy_shale_model,
)
from rock_physics_open.sandstone_models.friable_models import CoordinateNumberFunction

k_sst = 36.8e9 * np.ones(20)
mu_sst = 44.0e9 * np.ones(20)
rho_sst = 2650 * np.ones(20)
k_mud = 15.0e9 * np.ones(20)
mu_mud = 7.5e9 * np.ones(20)
rho_mud = 2680 * np.ones(20)
k_cem = 36.8e9 * np.ones(20)
mu_cem = 44.0e9 * np.ones(20)
rho_cem = 2650 * np.ones(20)
k_fl_sst = 0.8e9 * np.ones(20)
rho_fl_sst = 850 * np.ones(20)
k_fl_mud = 2.7e9 * np.ones(20)
rho_fl_mud = 1005 * np.ones(20)
phi = np.linspace(0.40, 0.07, 20)
p_eff_mud = 20.0e6 * np.ones(20)
p_eff_sst = 20.0e6 * np.ones(20)
shale_frac = np.linspace(0.0, 1.0, 20)
frac_cem = 0.05 * np.ones(20)
phi_c_sst = 0.45
phi_c_mud = 0.7
n_sst = 8.0
n_mud = 15.0
coord_num_func_mud: CoordinateNumberFunction = "PorBased"
coord_num_func_sst: CoordinateNumberFunction = "PorBased"
shear_red_sst = 0.25
shear_red_mud = 1.0
phi_intr_mud = 0.05


def test_cemented_sandy_shale_model():
    args = cemented_shaly_sand_sandy_shale_model(
        k_sst=k_sst,
        mu_sst=mu_sst,
        rho_sst=rho_sst,
        k_cem=k_cem,
        mu_cem=mu_cem,
        rho_cem=rho_cem,
        k_mud=k_mud,
        mu_mud=mu_mud,
        rho_mud=rho_mud,
        k_fl_sst=k_fl_sst,
        rho_fl_sst=rho_fl_sst,
        k_fl_mud=k_fl_mud,
        rho_fl_mud=rho_fl_mud,
        phi=phi,
        p_eff_mud=p_eff_mud,
        shale_frac=shale_frac,
        frac_cem=frac_cem,
        phi_c_sst=phi_c_sst,
        n_sst=n_sst,
        shear_red_sst=shear_red_sst,
        phi_c_mud=phi_c_mud,
        phi_intr_mud=phi_intr_mud,
        coord_num_func_mud=coord_num_func_mud,
        n_mud=n_mud,
        shear_red_mud=shear_red_mud,
    )

    if not os.path.isfile(get_snapshot_name()) or INITIATE:
        _ = store_snapshot(get_snapshot_name(), *args)
    else:
        assert compare_snapshots(args, read_snapshot(get_snapshot_name()))


def test_friable_sandy_shale_model():
    args = friable_shaly_sand_sandy_shale_model(
        k_sst=k_sst,
        mu_sst=mu_sst,
        rho_sst=rho_sst,
        k_mud=k_mud,
        mu_mud=mu_mud,
        rho_mud=rho_mud,
        k_fl_sst=k_fl_sst,
        rho_fl_sst=rho_fl_sst,
        k_fl_mud=k_fl_mud,
        rho_fl_mud=rho_fl_mud,
        phi=phi,
        p_eff_sst=p_eff_sst,
        p_eff_mud=p_eff_mud,
        shale_frac=shale_frac,
        phi_c_sst=phi_c_sst,
        phi_c_mud=phi_c_mud,
        phi_intr_mud=phi_intr_mud,
        coord_num_func_sst=coord_num_func_sst,
        n_sst=n_sst,
        coord_num_func_mud=coord_num_func_mud,
        n_mud=n_mud,
        shear_red_sst=shear_red_sst,
        shear_red_mud=shear_red_mud,
    )

    if not os.path.isfile(get_snapshot_name()) or INITIATE:
        _ = store_snapshot(get_snapshot_name(), *args)
    else:
        assert compare_snapshots(args, read_snapshot(get_snapshot_name()))
