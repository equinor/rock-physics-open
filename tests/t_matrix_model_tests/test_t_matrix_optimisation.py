import numpy as np
import numpy.typing as npt
import pandas as pd
from syrupy.assertion import SnapshotAssertion

from rock_physics_open.equinor_utilities.optimisation_utilities import gen_opt_routine
from rock_physics_open.equinor_utilities.units import g_cc_to_kg_m3
from rock_physics_open.t_matrix_models.curvefit_t_matrix_exp import (
    curvefit_t_matrix_exp,
)
from rock_physics_open.t_matrix_models.curvefit_t_matrix_min import (
    curve_fit_2_inclusion_sets,
)

from ..conftest import TESTDATA_DIR

inp_df = pd.read_csv(TESTDATA_DIR / "tmatrix_test_data.csv")

k_min = inp_df["K_MIN"].to_numpy() * 1.0e9
mu_min = inp_df["MU_MIN"].to_numpy() * 1.0e9
rho_min = g_cc_to_kg_m3(inp_df["RHO_MIN"].to_numpy())
k_fl_orig = inp_df["K_FL"].to_numpy() * 1.0e9
rho_fl_orig = g_cc_to_kg_m3(inp_df["RHO_FL"].to_numpy())
vp = inp_df["VP"].to_numpy()
vs = inp_df["VS"].to_numpy()
rho = g_cc_to_kg_m3(inp_df["RHOB"].to_numpy())
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


def poly_function(
    x_data: npt.NDArray[np.float64],
    a: float,
    b: float,
    c: float,
    d: float,
    e: float,
) -> npt.NDArray[np.float64]:
    """
    Simple function to test the scipy optimisation function curve_fit.

    Parameters
    ----------
    x_data
        independent variable in polynomial
    a
        constant
    b
        first order coefficient
    c
        second order coefficient
    d
        third order coefficient
    e
        fourth order coefficient

    Returns
    -------
    polynomial value
    """
    return a + b * x_data + c * x_data**2 + d * x_data**3 + e * x_data**4


def test_optimisation_part(snapshot_approx: SnapshotAssertion):
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

    _, _, opt_params = gen_opt_routine(
        opt_function=poly_function,
        x_data_orig=x,
        y_data=y,
        x_init=par_init,
        low_bound=lower_bound,
        high_bound=upper_bound,
    )
    assert snapshot_approx == opt_params


def test_t_matrix_opt_params_petec(snapshot: SnapshotAssertion):
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
    assert snapshot == curve_fit_2_inclusion_sets(x_data, *par)


def test_t_matrix_opt_params_exp(snapshot: SnapshotAssertion):
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
    assert snapshot == curvefit_t_matrix_exp(x_data, *par)
