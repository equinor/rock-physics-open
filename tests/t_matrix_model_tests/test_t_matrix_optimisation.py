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
from rock_physics_open.t_matrix_models.curvefit_t_matrix_exp import (
    curvefit_t_matrix_exp,
)
from rock_physics_open.t_matrix_models.curvefit_t_matrix_min import (
    curve_fit_2_inclusion_sets,
)
from rock_physics_open.t_matrix_models.opt_subst_utilities import gen_opt_routine
from tests import config

TEST_DIR = config.TESTDATA_DIR
os.chdir(TEST_DIR)
inp_df = pd.read_csv("tmatrix_test_data.csv")
k_min = inp_df["K_MIN"].to_numpy() * 1.0e9
mu_min = inp_df["MU_MIN"].to_numpy() * 1.0e9
rho_min = inp_df["RHO_MIN"].to_numpy() * 1000.0
k_fl_orig = inp_df["K_FL"].to_numpy() * 1.0e9
rho_fl_orig = inp_df["RHO_FL"].to_numpy() * 1000.0
vp = inp_df["VP"].to_numpy()
vs = inp_df["VS"].to_numpy()
rho = inp_df["RHOB"].to_numpy() * 1000.0
phi = inp_df["PHIT"].to_numpy()
vsh = inp_df["VSH"].to_numpy()
tau = 1e-7 * np.ones_like(phi)
angle = 45 * np.ones_like(phi)
perm = 100.0 * np.ones_like(phi)
visco = 100.0 * np.ones_like(phi)
freq = 1000.0 * np.ones_like(phi)
def_vp_vs_ratio = 1.8 * np.ones_like(phi)


"""
Direct testing of the T Matrix optimisation has proven to be difficult due to numerical differences when running
on different versions of Python and different architectures. Even if the number of significant digits are reduced,
there can be instances where the least significant digit will vary. The solution to this is to split the testing
in two parts: one for the optimisation in general and one for a single run of the T Matrix part of it. A simple
polynomial function is used for testing the optimisation.
"""


def poly_function(x, a, b, c, d, e):
    """
    Simple function to test the scipy optimisation function curve_fit

    Parameters
    ----------
    x : np.ndarray
        independent variable in polynomial
    a : float
        constant
    b : float
        first order coefficient
    c : float
        second order coefficient
    d : float
        third order coefficient
    e : float
        fourth order coefficient

    Returns
    -------
    y : np.ndarray
        polynomial value
    """
    return a + b * x + c * x**2 + d * x**3 + e * x**4


def test_optimisation_part():
    a = 0.25398574328
    b = 0.28536214953
    c = 0.98235978612
    d = 0.23987512467
    e = 0.45897987345

    x = np.linspace(0, 1, 18)
    par_init = 0.5 * np.ones(5)
    lower_bound = np.zeros(5)  # Number of parameters in estimation
    upper_bound = np.ones(5)
    y = poly_function(x, a, b, c, d, e)
    _, _, args = gen_opt_routine(
        poly_function, x, y, par_init, lower_bound, upper_bound
    )

    if not os.path.isfile(get_snapshot_name()) or INITIATE:
        store_snapshot(get_snapshot_name(), args)
    else:
        assert compare_snapshots(args, read_snapshot(get_snapshot_name()))


def test_t_matrix_opt_params_petec():
    x_data = np.stack(
        (
            phi,
            k_min,
            mu_min,
            rho_min,
            k_fl_orig,
            rho_fl_orig,
            angle,
            perm,
            visco,
            tau,
            freq,
            def_vp_vs_ratio,
        ),
        axis=1,
    )
    # Arbitrary setting for T Matrix parameters
    par = [0.5, 0.5, 0.5, 0.2, 0.2]
    args = curve_fit_2_inclusion_sets(x_data, *par)

    if not os.path.isfile(get_snapshot_name()) or INITIATE:
        store_snapshot(get_snapshot_name(), args)
    else:
        assert compare_snapshots(args, read_snapshot(get_snapshot_name()))


def test_t_matrix_opt_params_exp():
    x_data = np.stack(
        (
            phi,
            vsh,
            k_fl_orig,
            rho_fl_orig,
            angle,
            perm,
            visco,
            tau,
            freq,
            def_vp_vs_ratio,
        ),
        axis=1,
    )
    # Arbitrary setting for T Matrix parameters
    par = [0.5, 0.5, 0.5, 0.30, 0.7, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    args = curvefit_t_matrix_exp(x_data, *par)

    if not os.path.isfile(get_snapshot_name()) or INITIATE:
        store_snapshot(get_snapshot_name(), args)
    else:
        assert compare_snapshots(args, read_snapshot(get_snapshot_name()))
