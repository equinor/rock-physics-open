from pathlib import Path

import numpy as np
from syrupy.assertion import SnapshotAssertion

from rock_physics_open.t_matrix_models import (
    run_t_matrix_forward_model_with_opt_params_exp,
    run_t_matrix_forward_model_with_opt_params_petec,
    run_t_matrix_with_opt_params_exp,
    run_t_matrix_with_opt_params_petec,
)

k_min = np.ones(101) * 71.0e9
mu_min = np.ones(101) * 32.0e9
rho_min = np.ones(101) * 2710.0
k_fl_orig = np.ones(101) * 2.7e9
rho_fl_orig = np.ones(101) * 1005.0
k_fl_sub = np.ones(101) * 0.8e9
rho_fl_sub = np.ones(101) * 800.0
vp = 4500.0 * np.linspace(1.1, 0.9, 101)
vs = 3000.0 * np.linspace(1.1, 0.9, 101)
rho = 2500 * np.linspace(1.1, 0.9, 101)
phi = np.linspace(0.15, 0.25, 101)
vsh = 0.1 * np.ones(101)
angle = 45.0
perm = 100.0
visco = 100.0
tau = 1e-7
freq = 1000
pressure = np.array([20.0e6, 22.0e6])


def test_run_t_matrix_with_opt_params_petec(
    snapshot: SnapshotAssertion,
    testdata: Path,
):
    fname = testdata.joinpath("petec_opt_param.pkl")
    assert snapshot == run_t_matrix_with_opt_params_petec(
        min_k=k_min,
        min_mu=mu_min,
        min_rho=rho_min,
        fl_k_orig=k_fl_orig,
        fl_rho_orig=rho_fl_orig,
        fl_k_sub=k_fl_sub,
        fl_rho_sub=rho_fl_sub,
        vp=vp,
        vs=vs,
        rhob=rho,
        phi=phi,
        angle=angle,
        perm=perm,
        visco=visco,
        tau=tau,
        freq=freq,
        f_name=fname,
        fluid_sub=True,
    )


def test_run_t_matrix_with_opt_params_exp(snapshot: SnapshotAssertion, testdata: Path):
    fname = testdata.joinpath("exp_opt_param.pkl")
    assert snapshot == run_t_matrix_with_opt_params_exp(
        fl_k_orig=k_fl_orig,
        fl_rho_orig=rho_fl_orig,
        fl_k_sub=k_fl_sub,
        fl_rho_sub=rho_fl_sub,
        vp=vp,
        vs=vs,
        rhob=rho,
        phi=phi,
        vsh=vsh,
        angle=angle,
        perm=perm,
        visco=visco,
        tau=tau,
        freq=freq,
        f_name=fname,
        fluid_sub=True,
    )


def test_run_t_matrix_opt_forward_model_petec(
    snapshot: SnapshotAssertion, testdata: Path
):
    fname = testdata.joinpath("petec_opt_param.pkl")
    assert snapshot == run_t_matrix_forward_model_with_opt_params_petec(
        min_k=k_min,
        min_mu=mu_min,
        min_rho=rho_min,
        fl_k=k_fl_orig,
        fl_rho=rho_fl_orig,
        phi=phi,
        angle=angle,
        perm=perm,
        visco=visco,
        tau=tau,
        freq=freq,
        f_name=fname,
    )


def test_run_t_matrix_opt_forward_model_exp(
    snapshot: SnapshotAssertion, testdata: Path
):
    fname = testdata.joinpath("exp_opt_param.pkl")
    assert snapshot == run_t_matrix_forward_model_with_opt_params_exp(
        fl_k=k_fl_orig,
        fl_rho=rho_fl_orig,
        phi=phi,
        vsh=vsh,
        angle=angle,
        perm=perm,
        visco=visco,
        tau=tau,
        freq=freq,
        f_name=fname,
    )
