import numpy as np
import numpy.typing as npt

from rock_physics_open.equinor_utilities import std_functions


def gassmann_dry_model(
    k_min: npt.NDArray[np.float64],
    k_fl: npt.NDArray[np.float64],
    rho_fl: npt.NDArray[np.float64],
    k_sat: npt.NDArray[np.float64],
    mu: npt.NDArray[np.float64],
    rho_sat: npt.NDArray[np.float64],
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
    Gassmann model to go from saturated rock to dry state.

    Parameters
    ----------
    k_min
        Mineral bulk modulus [Pa].
    k_fl
        Fluid  bulk modulus [Pa].
    rho_fl
        Fluid density [lg/m^3].
    k_sat
        Saturated rock bulk modulus [Pa].
    mu
        Saturated rock shear modulus [Pa].
    rho_sat
        Saturated rock density [kg/m^3].
    por
        Porosity [fraction].

    Returns
    -------
    vp_dry
        Dry compressional velocity [m/s].
    vs_dry
        Dry shear velocity [m/s].
    rho_dry
        Dry density [kg/m^3].
    ai_dry
        Dry acoustic impedance [kg/m^3 x m/s].
    vpvs_dry
        Dry velocity ratio [unitless].
    k_dry
        Dry bulk modulus [Pa].
    mu
        Shear modulus [Pa] (unchanged from saturated state).
    """
    rho_dry = rho_sat - por * rho_fl
    k_dry = std_functions.gassmann_dry(k_sat, por, k_fl, k_min)
    vp_dry, vs_dry, ai_dry, vpvs_dry = std_functions.velocity(k_dry, mu, rho_dry)

    return vp_dry, vs_dry, rho_dry, ai_dry, vpvs_dry, k_dry, mu
