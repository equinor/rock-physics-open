import os

import numpy as np

from rock_physics_open.equinor_utilities.various_utilities import vp_vs_rho_stats


def test_vp_vs_rho_stats():
    rng = np.random.default_rng(123456)
    vp_obs = rng.uniform(low=0.0, high=1.0, size=100)
    vs_obs = rng.uniform(low=0.0, high=1.0, size=100)
    rho_obs = rng.uniform(low=0.0, high=1.0, size=100)
    vp_est = rng.uniform(low=0.0, high=1.0, size=100)
    vs_est = rng.uniform(low=0.0, high=1.0, size=100)
    rho_est = rng.uniform(low=0.0, high=1.0, size=100)
    fname = os.path.join(os.path.dirname(__file__), "snapshots", "vp_vs_rho_stats.csv")
    file_open_mode = "w"
    disp_res = False
    vp_vs_rho_stats(
        vp_obs,
        vs_obs,
        rho_obs,
        vp_est,
        vs_est,
        rho_est,
        fname,
        "pytest",
        "pytest_well",
        file_mode=file_open_mode,
        disp_results=disp_res,
    )


def test_multi_vp_vs_rho_stats():
    rng = np.random.default_rng(123456)
    vp_obs = [rng.uniform(low=0.0, high=1.0, size=100)] * 3
    vs_obs = [rng.uniform(low=0.0, high=1.0, size=100)] * 3
    rho_obs = [rng.uniform(low=0.0, high=1.0, size=100)] * 3
    vp_est = [rng.uniform(low=0.0, high=1.0, size=100)] * 3
    vs_est = [rng.uniform(low=0.0, high=1.0, size=100)] * 3
    rho_est = [rng.uniform(low=0.0, high=1.0, size=100)] * 3
    set_names = ["pytest_1", "pytest_2", "pytest_3"]
    well_names = ["pytest_well_1", "pytest_well_2", "pytest_well_3"]
    fname = os.path.join(
        os.path.dirname(__file__), "snapshots", "multi_vp_vs_rho_stats.csv"
    )
    file_open_mode = "w"
    disp_res = False
    vp_vs_rho_stats(
        vp_obs,
        vs_obs,
        rho_obs,
        vp_est,
        vs_est,
        rho_est,
        fname,
        set_names,
        well_names,
        file_mode=file_open_mode,
        disp_results=disp_res,
    )
