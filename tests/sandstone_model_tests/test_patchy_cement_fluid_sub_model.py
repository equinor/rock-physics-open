import os

import numpy as np
import pandas as pd

from rock_physics_open.equinor_utilities.snapshot_test_utilities import (
    INITIATE,
    compare_snapshots,
    get_snapshot_name,
    read_snapshot,
    store_snapshot,
)
from rock_physics_open.sandstone_models import (
    patchy_cement_model_cem_frac,
    patchy_cement_model_weight,
    patchy_cement_pressure_fluid_substitution,
)
from tests import config

TEST_DIR = config.TESTDATA_DIR
os.chdir(TEST_DIR)

dataset = pd.read_csv("test_well.csv")

vp_old = dataset["VP"].to_numpy()
vs_old = dataset["VS"].to_numpy()
rho_b_old = dataset["RHOB"].to_numpy() * 1000.0
phi = dataset["PHIT"].to_numpy()
idx_high_phi = phi > 0.1
phi = phi[idx_high_phi]
vp_old = vp_old[idx_high_phi]
vs_old = vs_old[idx_high_phi]
rho_b_old = rho_b_old[idx_high_phi]
k_min = 136.8e9 * np.ones_like(phi)
mu_min = 44.0e9 * np.ones_like(phi)
rho_min = 2650 * np.ones_like(phi)
k_cem = 36.8e9 * np.ones_like(phi)
mu_cem = 44.0e9 * np.ones_like(phi)
rho_cem = 2650 * np.ones_like(phi)
k_fl_old = 0.8e9 * np.ones_like(phi)
rho_fl_old = 850 * np.ones_like(phi)
k_fl_new = 2.7e9 * np.ones_like(phi)
rho_fl_new = 1005 * np.ones_like(phi)
p_eff_old = 20.0e6 * np.ones_like(phi)
p_eff_new = 25.0e6 * np.ones_like(phi)
p_eff_low = 20.0e6 * np.ones_like(phi)
frac_cem_up = 0.10
frac_cem = 0.03
phi_c = 0.45
coord_num_func = "PorBased"
n = 8.0
shear_red = 0.3
weight_k = 0.6
weight_mu = 0.4


def test_patchy_cement_fluid_sub_model_no_change():
    args = patchy_cement_pressure_fluid_substitution(
        k_min,
        mu_min,
        rho_min,
        k_cem,
        mu_cem,
        rho_cem,
        k_fl_old,
        rho_fl_old,
        k_fl_old,
        rho_fl_old,
        phi,
        p_eff_old,
        p_eff_old,
        vp_old,
        vs_old,
        rho_b_old,
        p_eff_low,
        frac_cem_up,
        None,
        shear_red,
        phi_c,
        coord_num_func,
        n,
        model_type="weight",
        phi_below_zero="disregard",
        phi_above_phi_c="disregard",
        k_sat_above_k_min="disregard",
        above_upper_bound="disregard",
        below_lower_bound="disregard",
    )
    if not os.path.isfile(get_snapshot_name()) or INITIATE:
        store_snapshot(get_snapshot_name(), *args)
    else:
        assert compare_snapshots(args, read_snapshot(get_snapshot_name()))


def test_patchy_cement_fluid_sub_model_weight():
    args = patchy_cement_pressure_fluid_substitution(
        k_min,
        mu_min,
        rho_min,
        k_cem,
        mu_cem,
        rho_cem,
        k_fl_old,
        rho_fl_old,
        k_fl_new,
        rho_fl_new,
        phi,
        p_eff_old,
        p_eff_new,
        vp_old,
        vs_old,
        rho_b_old,
        p_eff_low,
        frac_cem_up,
        None,
        shear_red,
        phi_c,
        coord_num_func,
        n,
        model_type="weight",
        phi_below_zero="disregard",
        phi_above_phi_c="disregard",
        k_sat_above_k_min="disregard",
        above_upper_bound="disregard",
        below_lower_bound="disregard",
    )
    if not os.path.isfile(get_snapshot_name()) or INITIATE:
        store_snapshot(get_snapshot_name(), *args)
    else:
        assert compare_snapshots(args, read_snapshot(get_snapshot_name()))


def test_patchy_cement_fluid_sub_model_weight_snap():
    args = patchy_cement_pressure_fluid_substitution(
        k_min,
        mu_min,
        rho_min,
        k_cem,
        mu_cem,
        rho_cem,
        k_fl_old,
        rho_fl_old,
        k_fl_new,
        rho_fl_new,
        phi,
        p_eff_old,
        p_eff_new,
        vp_old,
        vs_old,
        rho_b_old,
        p_eff_low,
        frac_cem_up,
        None,
        shear_red,
        phi_c,
        coord_num_func,
        n,
        model_type="weight",
        phi_below_zero="snap",
        phi_above_phi_c="snap",
        k_sat_above_k_min="snap",
        above_upper_bound="snap",
        below_lower_bound="snap",
    )
    if not os.path.isfile(get_snapshot_name()) or INITIATE:
        store_snapshot(get_snapshot_name(), *args)
    else:
        assert compare_snapshots(args, read_snapshot(get_snapshot_name()))


def test_patchy_cement_fluid_sub_model_cem_frac():
    args = patchy_cement_pressure_fluid_substitution(
        k_min,
        mu_min,
        rho_min,
        k_cem,
        mu_cem,
        rho_cem,
        k_fl_old,
        rho_fl_old,
        k_fl_new,
        rho_fl_new,
        phi,
        p_eff_old,
        p_eff_new,
        vp_old,
        vs_old,
        rho_b_old,
        p_eff_low,
        frac_cem_up,
        frac_cem,
        shear_red,
        phi_c,
        coord_num_func,
        n,
        model_type="cem_frac",
        phi_below_zero="disregard",
        phi_above_phi_c="disregard",
        k_sat_above_k_min="disregard",
        above_upper_bound="disregard",
        below_lower_bound="disregard",
    )
    if not os.path.isfile(get_snapshot_name()) or INITIATE:
        store_snapshot(get_snapshot_name(), *args)
    else:
        assert compare_snapshots(args, read_snapshot(get_snapshot_name()))


def test_patchy_cement_fluid_sub_model_cem_frac_snap():
    args = patchy_cement_pressure_fluid_substitution(
        k_min,
        mu_min,
        rho_min,
        k_cem,
        mu_cem,
        rho_cem,
        k_fl_old,
        rho_fl_old,
        k_fl_new,
        rho_fl_new,
        phi,
        p_eff_old,
        p_eff_new,
        vp_old,
        vs_old,
        rho_b_old,
        p_eff_low,
        frac_cem_up,
        frac_cem,
        shear_red,
        phi_c,
        coord_num_func,
        n,
        model_type="cem_frac",
        phi_below_zero="snap",
        phi_above_phi_c="snap",
        k_sat_above_k_min="snap",
        above_upper_bound="snap",
        below_lower_bound="snap",
    )
    if not os.path.isfile(get_snapshot_name()) or INITIATE:
        store_snapshot(get_snapshot_name(), *args)
    else:
        assert compare_snapshots(args, read_snapshot(get_snapshot_name()))


def test_patchy_cement_model_weight():
    args = patchy_cement_model_weight(
        k_min,
        mu_min,
        rho_min,
        k_cem,
        mu_cem,
        rho_cem,
        k_fl_old,
        rho_fl_old,
        phi,
        p_eff_old,
        frac_cem,
        phi_c,
        coord_num_func,
        n,
        shear_red,
        weight_k,
        weight_mu,
    )
    if not os.path.isfile(get_snapshot_name()) or INITIATE:
        store_snapshot(get_snapshot_name(), *args)
    else:
        assert compare_snapshots(args, read_snapshot(get_snapshot_name()))


def test_patchy_cement_model_cem_frac():
    args = patchy_cement_model_cem_frac(
        k_min,
        mu_min,
        rho_min,
        k_cem,
        mu_cem,
        rho_cem,
        k_fl_old,
        rho_fl_old,
        phi,
        p_eff_old,
        frac_cem,
        phi_c,
        coord_num_func,
        n,
        shear_red,
    )
    if not os.path.isfile(get_snapshot_name()) or INITIATE:
        store_snapshot(get_snapshot_name(), *args)
    else:
        assert compare_snapshots(args, read_snapshot(get_snapshot_name()))


def test_patchy_cement_model_exceed_phi_extrapolate():
    frac_cem = 0.10
    phi_c = 0.40
    args = patchy_cement_model_cem_frac(
        k_min,
        mu_min,
        rho_min,
        k_cem,
        mu_cem,
        rho_cem,
        k_fl_old,
        rho_fl_old,
        phi,
        p_eff_old,
        frac_cem,
        phi_c,
        coord_num_func,
        n,
        shear_red,
    )
    # Set flag for extrapolation to True and assert that there are no NaN values in the output
    args = patchy_cement_model_cem_frac(
        k_min,
        mu_min,
        rho_min,
        k_cem,
        mu_cem,
        rho_cem,
        k_fl_old,
        rho_fl_old,
        phi,
        p_eff_old,
        frac_cem,
        phi_c,
        coord_num_func,
        n,
        shear_red,
    )
    assert not np.any(np.isnan(args[0]))

    if not os.path.isfile(get_snapshot_name()) or INITIATE:
        store_snapshot(get_snapshot_name(), *args)
    else:
        assert compare_snapshots(args, read_snapshot(get_snapshot_name()))
