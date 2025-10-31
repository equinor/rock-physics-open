import numpy as np
import numpy.typing as npt

from rock_physics_open.equinor_utilities import std_functions
from rock_physics_open.sandstone_models.constant_cement_models import (
    constant_cement_model,
)
from rock_physics_open.sandstone_models.friable_models import (
    CoordinateNumberFunction,
    friable_model,
)


def unresolved_constant_cement_sand_shale_model(
    k_min_sst: npt.NDArray[np.float64],
    mu_min_sst: npt.NDArray[np.float64],
    rho_min_sst: npt.NDArray[np.float64],
    k_cem: npt.NDArray[np.float64],
    mu_cem: npt.NDArray[np.float64],
    rho_cem: npt.NDArray[np.float64],
    k_mud: npt.NDArray[np.float64],
    mu_mud: npt.NDArray[np.float64],
    rho_mud: npt.NDArray[np.float64],
    k_fl_sst: npt.NDArray[np.float64],
    rho_fl_sst: npt.NDArray[np.float64],
    k_fl_mud: npt.NDArray[np.float64],
    rho_fl_mud: npt.NDArray[np.float64],
    phi_sst: npt.NDArray[np.float64],
    phi_mud: npt.NDArray[np.float64],
    p_eff_mud: npt.NDArray[np.float64],
    shale_frac: npt.NDArray[np.float64],
    frac_cem: float,
    phi_c_sst: float,
    phi_c_mud: float,
    n_sst: float,
    coord_num_func_mud: CoordinateNumberFunction,
    n_mud: float,
    shear_red_sst: float,
    shear_red_mud: float,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """
    Model for silisiclastic rocks with alternating layers of cemented sand and friable shale, and in which the layers
    are not resolved by the investigating signal. Backus average is used to calculate the anisotropic effect of the
    alternating layers.

    Parameters
    ----------
    k_min_sst : np.ndarray
        Sandstone matrix bulk modulus [Pa].
    mu_min_sst : np.ndarray
        Sandstone matrix shear modulus [Pa].
    rho_min_sst : np.ndarray
        Sandstone matrix bulk density [kg/m^3].
    k_cem : np.ndarray
        Sandstone cement bulk modulus [Pa].
    mu_cem : np.ndarray
        Sandstone cement shear modulus [Pa].
    rho_cem : np.ndarray
        Sandstone cement bulk density [kg/m^3].
    k_mud : np.ndarray
        Shale bulk modulus [Pa].
    mu_mud : np.ndarray
        Shale shear modulus [Pa].
    rho_mud : np.ndarray
        Shale bulk density [kg/m^3].
    k_fl_sst : np.ndarray
        Fluid bulk modulus for sandstone fluid [Pa].
    rho_fl_sst : np.ndarray
        Fluid bulk density for sandstone fluid [kg/m^3].
    k_fl_mud : np.ndarray
        Fluid bulk modulus for shale fluid [Pa].
    rho_fl_mud : np.ndarray
        Fluid bulk density for shale fluid[kg/m^3].
    phi_sst : np.ndarray
        Sandstone porosity [fraction].
    phi_mud : np.ndarray
        Shale porosity [fraction].
    p_eff_mud : np.ndarray
        Effective pressure in mud [Pa].
    shale_frac : np.ndarray
        Shale fraction [fraction].
    frac_cem : float
        Cement volume fraction [fraction].
    phi_c_sst : float
        Critical porosity for sandstone [fraction].
    phi_c_mud : float
        Critical porosity for mud [fraction].
    n_sst : float
        Coordination number for sandstone [unitless].
    n_mud : float
        Coordination number for shale [unitless].
    coord_num_func_mud : str
        Indication if coordination number should be calculated from porosity or kept constant for shale.
    shear_red_sst : float
        Shear reduction factor for sandstone [fraction].
    shear_red_mud : float
        Shear reduction factor for mud [fraction].

    Returns
    -------
    tuple
        vpv, vsv, vph, vsh, rho : np.ndarray
        vertical p-wave velocity, vertical shear-wave velocity, horizontal p-wave velocity, horizontal shear-wave
        velocity (all [m/s]), bulk density [kg/m^3].
    """
    # Estimate the sand end member through the constant cement model
    vp_sst, vs_sst, rho_b_sst, _, _ = constant_cement_model(
        k_min=k_min_sst,
        mu_min=mu_min_sst,
        rho_min=rho_min_sst,
        k_cem=k_cem,
        mu_cem=mu_cem,
        rho_cem=rho_cem,
        k_fl=k_fl_sst,
        rho_fl=rho_fl_sst,
        phi=phi_sst,
        frac_cem=frac_cem,
        phi_c=phi_c_sst,
        n=n_sst,
        shear_red=shear_red_sst,
    )

    # Estimate the shale end member through the friable model
    vp_mud, vs_mud, rho_b_mud, _, _ = friable_model(
        k_min=k_mud,
        mu_min=mu_mud,
        rho_min=rho_mud,
        k_fl=k_fl_mud,
        rho_fl=rho_fl_mud,
        phi=phi_mud,
        p_eff=p_eff_mud,
        phi_c=phi_c_mud,
        coord_num_func=coord_num_func_mud,
        n=n_mud,
        shear_red=shear_red_mud,
    )

    # Calculate Backus average for the effective medium
    vpv, vsv, vph, vsh, rho = std_functions.backus_average(
        vp1=vp_sst,
        vs1=vs_sst,
        rho1=rho_b_sst,
        vp2=vp_mud,
        vs2=vs_mud,
        rho2=rho_b_mud,
        f1=1.0 - shale_frac,
    )

    return vpv, vsv, vph, vsh, rho
