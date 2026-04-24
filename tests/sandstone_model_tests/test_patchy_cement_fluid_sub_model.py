from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from syrupy.assertion import SnapshotAssertion

from rock_physics_open.equinor_utilities.units import g_cc_to_kg_m3
from rock_physics_open.sandstone_models import (
    patchy_cement_model_cem_frac,
    patchy_cement_model_weight,
    patchy_cement_pressure_fluid_substitution,
)
from rock_physics_open.sandstone_models.friable_models import CoordinateNumberFunction


@dataclass(frozen=True)
class PatchyCementParams:
    vp_old: npt.NDArray[np.float64]
    vs_old: npt.NDArray[np.float64]
    rho_b_old: npt.NDArray[np.float64]
    phi: npt.NDArray[np.float64]
    k_min: npt.NDArray[np.float64]
    mu_min: npt.NDArray[np.float64]
    rho_min: npt.NDArray[np.float64]
    k_cem: npt.NDArray[np.float64]
    mu_cem: npt.NDArray[np.float64]
    rho_cem: npt.NDArray[np.float64]
    k_fl_old: npt.NDArray[np.float64]
    rho_fl_old: npt.NDArray[np.float64]
    k_fl_new: npt.NDArray[np.float64]
    rho_fl_new: npt.NDArray[np.float64]
    p_eff_old: npt.NDArray[np.float64]
    p_eff_new: npt.NDArray[np.float64]
    p_eff_low: npt.NDArray[np.float64]
    frac_cem_up: float
    frac_cem: float
    phi_c: float
    coord_num_func: CoordinateNumberFunction
    n: float
    shear_red: float
    weight_k: float
    weight_mu: float


@pytest.fixture
def params(testdata: Path) -> PatchyCementParams:
    dataset = pd.read_csv(testdata / "test_well.csv")

    vp_old_df = dataset["VP"].to_numpy()
    vs_old_df = dataset["VS"].to_numpy()
    rho_b_old_df = dataset["RHOB"].to_numpy()
    phi_df = dataset["PHIT"].to_numpy()
    idx_high_phi = phi_df > 0.1

    return PatchyCementParams(
        vp_old=vp_old_df[idx_high_phi],
        vs_old=vs_old_df[idx_high_phi],
        rho_b_old=g_cc_to_kg_m3(rho_b_old_df[idx_high_phi]),
        phi=phi_df[idx_high_phi],
        k_min=136.8e9 * np.ones_like(phi_df[idx_high_phi]),
        mu_min=44.0e9 * np.ones_like(phi_df[idx_high_phi]),
        rho_min=2650 * np.ones_like(phi_df[idx_high_phi]),
        k_cem=36.8e9 * np.ones_like(phi_df[idx_high_phi]),
        mu_cem=44.0e9 * np.ones_like(phi_df[idx_high_phi]),
        rho_cem=2650 * np.ones_like(phi_df[idx_high_phi]),
        k_fl_old=0.8e9 * np.ones_like(phi_df[idx_high_phi]),
        rho_fl_old=850 * np.ones_like(phi_df[idx_high_phi]),
        k_fl_new=2.7e9 * np.ones_like(phi_df[idx_high_phi]),
        rho_fl_new=1005 * np.ones_like(phi_df[idx_high_phi]),
        p_eff_old=20.0e6 * np.ones_like(phi_df[idx_high_phi]),
        p_eff_new=25.0e6 * np.ones_like(phi_df[idx_high_phi]),
        p_eff_low=20.0e6 * np.ones_like(phi_df[idx_high_phi]),
        frac_cem_up=0.10,
        frac_cem=0.03,
        phi_c=0.45,
        coord_num_func="PorBased",
        n=8.0,
        shear_red=0.3,
        weight_k=0.6,
        weight_mu=0.4,
    )


def test_patchy_cement_fluid_sub_model_no_change(
    snapshot: SnapshotAssertion, params: PatchyCementParams
):
    assert snapshot == patchy_cement_pressure_fluid_substitution(
        k_min=params.k_min,
        mu_min=params.mu_min,
        rho_min=params.rho_min,
        k_cem=params.k_cem,
        mu_cem=params.mu_cem,
        rho_cem=params.rho_cem,
        k_fl_old=params.k_fl_old,
        rho_fl_old=params.rho_fl_old,
        k_fl_new=params.k_fl_old,
        rho_fl_new=params.rho_fl_old,
        phi=params.phi,
        p_eff_old=params.p_eff_old,
        p_eff_new=params.p_eff_old,
        vp_old=params.vp_old,
        vs_old=params.vs_old,
        rho_b_old=params.rho_b_old,
        p_eff_low=params.p_eff_low,
        frac_cem_up=params.frac_cem_up,
        frac_cem=params.frac_cem,
        shear_red=params.shear_red,
        phi_c=params.phi_c,
        coord_num_func=params.coord_num_func,
        n=params.n,
        model_type="weight",
        phi_below_zero="disregard",
        phi_above_phi_c="disregard",
        k_sat_above_k_min="disregard",
        above_upper_bound="disregard",
        below_lower_bound="disregard",
    )


def test_patchy_cement_fluid_sub_model_weight(
    snapshot: SnapshotAssertion, params: PatchyCementParams
):
    assert snapshot == patchy_cement_pressure_fluid_substitution(
        k_min=params.k_min,
        mu_min=params.mu_min,
        rho_min=params.rho_min,
        k_cem=params.k_cem,
        mu_cem=params.mu_cem,
        rho_cem=params.rho_cem,
        k_fl_old=params.k_fl_old,
        rho_fl_old=params.rho_fl_old,
        k_fl_new=params.k_fl_new,
        rho_fl_new=params.rho_fl_new,
        phi=params.phi,
        p_eff_old=params.p_eff_old,
        p_eff_new=params.p_eff_new,
        vp_old=params.vp_old,
        vs_old=params.vs_old,
        rho_b_old=params.rho_b_old,
        p_eff_low=params.p_eff_low,
        frac_cem_up=params.frac_cem_up,
        frac_cem=params.frac_cem,
        shear_red=params.shear_red,
        phi_c=params.phi_c,
        coord_num_func=params.coord_num_func,
        n=params.n,
        model_type="weight",
        phi_below_zero="disregard",
        phi_above_phi_c="disregard",
        k_sat_above_k_min="disregard",
        above_upper_bound="disregard",
        below_lower_bound="disregard",
    )


def test_patchy_cement_fluid_sub_model_weight_snap(
    snapshot: SnapshotAssertion, params: PatchyCementParams
):
    assert snapshot == patchy_cement_pressure_fluid_substitution(
        k_min=params.k_min,
        mu_min=params.mu_min,
        rho_min=params.rho_min,
        k_cem=params.k_cem,
        mu_cem=params.mu_cem,
        rho_cem=params.rho_cem,
        k_fl_old=params.k_fl_old,
        rho_fl_old=params.rho_fl_old,
        k_fl_new=params.k_fl_new,
        rho_fl_new=params.rho_fl_new,
        phi=params.phi,
        p_eff_old=params.p_eff_old,
        p_eff_new=params.p_eff_new,
        vp_old=params.vp_old,
        vs_old=params.vs_old,
        rho_b_old=params.rho_b_old,
        p_eff_low=params.p_eff_low,
        frac_cem_up=params.frac_cem_up,
        frac_cem=params.frac_cem,
        shear_red=params.shear_red,
        phi_c=params.phi_c,
        coord_num_func=params.coord_num_func,
        n=params.n,
        model_type="weight",
        phi_below_zero="snap",
        phi_above_phi_c="snap",
        k_sat_above_k_min="snap",
        above_upper_bound="snap",
        below_lower_bound="snap",
    )


def test_patchy_cement_fluid_sub_model_cem_frac(
    snapshot: SnapshotAssertion, params: PatchyCementParams
):
    assert snapshot == patchy_cement_pressure_fluid_substitution(
        k_min=params.k_min,
        mu_min=params.mu_min,
        rho_min=params.rho_min,
        k_cem=params.k_cem,
        mu_cem=params.mu_cem,
        rho_cem=params.rho_cem,
        k_fl_old=params.k_fl_old,
        rho_fl_old=params.rho_fl_old,
        k_fl_new=params.k_fl_new,
        rho_fl_new=params.rho_fl_new,
        phi=params.phi,
        p_eff_old=params.p_eff_old,
        p_eff_new=params.p_eff_new,
        vp_old=params.vp_old,
        vs_old=params.vs_old,
        rho_b_old=params.rho_b_old,
        p_eff_low=params.p_eff_low,
        frac_cem_up=params.frac_cem_up,
        frac_cem=params.frac_cem,
        shear_red=params.shear_red,
        phi_c=params.phi_c,
        coord_num_func=params.coord_num_func,
        n=params.n,
        model_type="cement_fraction",
        phi_below_zero="disregard",
        phi_above_phi_c="disregard",
        k_sat_above_k_min="disregard",
        above_upper_bound="disregard",
        below_lower_bound="disregard",
    )


def test_patchy_cement_fluid_sub_model_cem_frac_snap(
    snapshot: SnapshotAssertion, params: PatchyCementParams
):
    assert snapshot == patchy_cement_pressure_fluid_substitution(
        k_min=params.k_min,
        mu_min=params.mu_min,
        rho_min=params.rho_min,
        k_cem=params.k_cem,
        mu_cem=params.mu_cem,
        rho_cem=params.rho_cem,
        k_fl_old=params.k_fl_old,
        rho_fl_old=params.rho_fl_old,
        k_fl_new=params.k_fl_new,
        rho_fl_new=params.rho_fl_new,
        phi=params.phi,
        p_eff_old=params.p_eff_old,
        p_eff_new=params.p_eff_new,
        vp_old=params.vp_old,
        vs_old=params.vs_old,
        rho_b_old=params.rho_b_old,
        p_eff_low=params.p_eff_low,
        frac_cem_up=params.frac_cem_up,
        frac_cem=params.frac_cem,
        shear_red=params.shear_red,
        phi_c=params.phi_c,
        coord_num_func=params.coord_num_func,
        n=params.n,
        model_type="cement_fraction",
        phi_below_zero="snap",
        phi_above_phi_c="snap",
        k_sat_above_k_min="snap",
        above_upper_bound="snap",
        below_lower_bound="snap",
    )


def test_patchy_cement_model_weight(
    snapshot: SnapshotAssertion, params: PatchyCementParams
):
    assert snapshot == patchy_cement_model_weight(
        k_min=params.k_min,
        mu_min=params.mu_min,
        rho_min=params.rho_min,
        k_cem=params.k_cem,
        mu_cem=params.mu_cem,
        rho_cem=params.rho_cem,
        k_fl=params.k_fl_old,
        rho_fl=params.rho_fl_old,
        phi=params.phi,
        p_eff=params.p_eff_old,
        frac_cem=params.frac_cem,
        phi_c=params.phi_c,
        coord_num_func=params.coord_num_func,
        n=params.n,
        shear_red=params.shear_red,
        weight_k=params.weight_k,
        weight_mu=params.weight_mu,
    )


def test_patchy_cement_model_cem_frac(
    snapshot: SnapshotAssertion, params: PatchyCementParams
):
    assert snapshot == patchy_cement_model_cem_frac(
        k_min=params.k_min,
        mu_min=params.mu_min,
        rho_min=params.rho_min,
        k_cem=params.k_cem,
        mu_cem=params.mu_cem,
        rho_cem=params.rho_cem,
        k_fl=params.k_fl_old,
        rho_fl=params.rho_fl_old,
        phi=params.phi,
        p_eff=params.p_eff_old,
        frac_cem=params.frac_cem,
        phi_c=params.phi_c,
        coord_num_func=params.coord_num_func,
        n=params.n,
        shear_red=params.shear_red,
    )


def test_patchy_cement_model_exceed_phi_extrapolate(
    snapshot: SnapshotAssertion, params: PatchyCementParams
):
    output = patchy_cement_model_cem_frac(
        k_min=params.k_min,
        mu_min=params.mu_min,
        rho_min=params.rho_min,
        k_cem=params.k_cem,
        mu_cem=params.mu_cem,
        rho_cem=params.rho_cem,
        k_fl=params.k_fl_old,
        rho_fl=params.rho_fl_old,
        phi=params.phi,
        p_eff=params.p_eff_old,
        frac_cem=0.10,
        phi_c=0.40,
        coord_num_func=params.coord_num_func,
        n=params.n,
        shear_red=params.shear_red,
    )
    assert not np.any(np.isnan(output))
    assert snapshot == output
