import numpy as np
import numpy.typing as npt

from rock_physics_open.equinor_utilities import std_functions


def gassmann_sub_model(
    k_min: npt.NDArray[np.float64],
    k_fl_orig: npt.NDArray[np.float64],
    rho_fl_orig: npt.NDArray[np.float64],
    k_fl_sub: npt.NDArray[np.float64],
    rho_fl_sub: npt.NDArray[np.float64],
    k_sat_orig: npt.NDArray[np.float64],
    mu: npt.NDArray[np.float64],
    rho_sat_orig: npt.NDArray[np.float64],
    por: npt.NDArray[np.float64],
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """
    Gassmann model to go from one saturated state to another.

    Parameters
    ----------
    k_min
        Mineral bulk modulus [Pa].
    k_fl_orig
        Original fluid  bulk modulus [Pa].
    rho_fl_orig
        Original fluid density [lg/m^3].
    k_fl_sub
        Substituted fluid  bulk modulus [Pa].
    rho_fl_sub
        Substituted fluid density [lg/m^3].
    k_sat_orig
        Saturated rock bulk modulus with original fluid  [Pa].
    mu
        Rock shear modulus [Pa].
    rho_sat_orig
        Saturated rock density with original fluid [kg/m^3].
    por
        Porosity [fraction].

    Returns
    -------
    vp_sat
        Saturated compressional velocity [m/s].
    vs_sat
        Saturated shear velocity [m/s].
    rho_sat
        Saturated density [kg/m^3].
    ai_sat
        Saturated acoustic impedance [kg/m^3 x m/s].
    vpvs_sat
        Saturated velocity ratio [unitless].
    k_sat
        Saturated bulk modulus [Pa].
    mu
        Shear modulus [Pa] (unchanged from dry state).
    """
    rho_sat_sub = rho_sat_orig + por * (rho_fl_sub - rho_fl_orig)
    k_sat_sub = std_functions.gassmann2(k_sat_orig, k_fl_orig, k_fl_sub, por, k_min)
    vp_sat_sub, vs_sat_sub, ai_sat_sub, vpvs_sat_sub = std_functions.velocity(
        k_sat_sub, mu, rho_sat_sub
    )

    return vp_sat_sub, vs_sat_sub, rho_sat_sub, ai_sat_sub, vpvs_sat_sub, k_sat_sub, mu
