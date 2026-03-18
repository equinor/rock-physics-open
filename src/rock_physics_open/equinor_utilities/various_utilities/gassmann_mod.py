import numpy as np
import numpy.typing as npt

from rock_physics_open.equinor_utilities import std_functions


def gassmann_model(
    k_min: npt.NDArray[np.float64],
    k_fl: npt.NDArray[np.float64],
    rho_fl: npt.NDArray[np.float64],
    k_dry: npt.NDArray[np.float64],
    mu: npt.NDArray[np.float64],
    rho_dry: npt.NDArray[np.float64],
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
    Gassmann model to go from dry rock to saturated state.

    Parameters
    ----------
    k_min
        Mineral bulk modulus [Pa].
    k_fl
        Fluid  bulk modulus [Pa].
    rho_fl
        Fluid density [lg/m^3].
    k_dry
        Dry rock bulk modulus [Pa].
    mu
        Dry rock shear modulus [Pa].
    rho_dry
        Dry rock density [kg/m^3].
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
    rho_sat = rho_dry + por * rho_fl
    k_sat = std_functions.gassmann(k_dry, por, k_fl, k_min)
    vp_sat, vs_sat, ai_sat, vpvs_sat = std_functions.velocity(k_sat, mu, rho_sat)

    return vp_sat, vs_sat, rho_sat, ai_sat, vpvs_sat, k_sat, mu
