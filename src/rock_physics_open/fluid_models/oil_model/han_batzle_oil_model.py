from warnings import warn

import numpy as np
import numpy.typing as npt

from rock_physics_open.fluid_models.oil_model.oil_utilities import (
    oil_density_to_api,
    oil_density_to_gcc,
    oil_density_to_kg_m_3,
)

from .live_oil_density import live_oil_density
from .oil_bubble_point import bp_standing


def gas_apprearant_density(
    oil_density: npt.NDArray[np.float64],
    gas_gravity: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    From Han and Batzle 2000: Equation 5
    The Rock Physics Handbook Ed. 3, Equation 6.22.42
    Appearant liquid density of natural gas at standard conditions

    Args:
        oil_density (npt.NDArray[np.float64]): oil density at standard conditions [kg/m^3]
        gas_gravity (npt.NDArray[np.float64]): gas gravity relative to air [unitless]

    Returns:
        npt.NDArray[np.float64]: gas liquid density [kg/m^3]
    """
    oil_api = oil_density_to_api(oil_density_to_gcc(oil_density))
    tmp = 0.61731 * (10.0 ** (-0.00326 * oil_api)) + (
        1.5177 - 0.54349 * np.log10(oil_api)
    ) * np.log10(gas_gravity)
    return oil_density_to_kg_m_3(tmp)


def pseudo_liquid_density(
    oil_density: npt.NDArray[np.float64],
    gor: npt.NDArray[np.float64],
    gas_gravity: npt.NDArray[np.float64],
    gas_coefficient: float = 0.113,
) -> npt.NDArray[np.float64]:
    """
    From Han and Batzle 2000: equation 7
    The Rock Physics Handbook Ed. 3, Equation 6.22.43-44
    Calculate the pseudo liquid density at standard conditions.
    With the default setting for the 'gas_coefficient', this is adapted to
    the Han and Batzle model for live oil velocity

    Args:
        gor (npt.NDArray[np.float64]): gas / oil volume ratio [unitless]
        oil_density (npt.NDArray[np.float64]): oil density at standard conditions [kg/m^3]
        gas_gravity (npt.NDArray[np.float64]): gas gravity relative to air [unitless]
        gas_coefficient: (float): empirical gas coefficient


    Returns:
        npt.NDArray[np.float64]: pseudu liquid density
    """
    # Calculate the apprearant volume fraction of the pseudo-liquid gas
    # at standard conditions
    gas_app_dens = gas_apprearant_density(
        oil_density=oil_density, gas_gravity=gas_gravity
    )
    vol_gas = (
        0.00123
        * gor
        * gas_gravity
        / (oil_density_to_gcc(gas_app_dens) + 0.00123 * gor * gas_gravity)
    )
    return oil_density * (1.0 - vol_gas) + gas_coefficient * gas_app_dens * vol_gas


def density_correction_vasquez_beggs_ahmed(
    gor: npt.NDArray[np.float64],
    temp: npt.NDArray[np.float64],
    gravity: npt.NDArray[np.float64],
    rho0: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Correction to Batzle and Wang 1992 oil density model for pressures above
    bubble point. This method is referred to as 'a more widely used correction'.
    The Rock Physics Handbook, 3rd ed, 2020: Equation 6.22.38-40

    Args:
        gor (float | np.ndarray): gas / oil ratio [l/l]
        temp (float | np.ndarray): temperature [°C]
        gravity (float | np.ndarray): gas gravity relative to air [unitless]
        rho0 (float | np.ndarray): oil density at standard conditions [kg/m^3]

    Returns:
        float | np.ndarray: exponent for correction of pressures above bubble point
    """
    rho0_api = oil_density_to_api(oil_density_to_gcc(rho0))
    return 1.0e-5 * (
        28.08 * gor + 30.96 * temp - 1180.0 * gravity + 12.61 * rho0_api - 882.6
    )


def han_batzle_live_oil_velocity(
    reference_density: npt.NDArray[np.float64],
    gas_oil_ratio: npt.NDArray[np.float64],
    gas_gravity: npt.NDArray[np.float64],
    temperature: npt.NDArray[np.float64],
    pressure: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Han and Batzle 2000: oil velocity model, equation 10-16
    The Rock Physics Handbook Ed. 3, Equation 6.22.52

    Args:
        reference_density (npt.NDArray[np.float64]): oil density at standard conditions [kg/m^3]
        gas_oil_ratio (npt.NDArray[np.float64]): gas / Oil volume ratio [l/l]
        gas_gravity (npt.NDArray[np.float64]): gas gravity relative to air [unitless]
        temperature (npt.NDArray[np.float64]): temperature [°C]
        pressure (npt.NDArray[np.float64]): formation pressure [Pa]

    Returns:
        npt.NDArray[np.float64]: oil velocity [m/s]
    """
    # Estimate the bubble point pressure and identify pressure
    # values above and below bubble point
    bubble_point_pres = bp_standing(
        density=reference_density,
        gas_oil_ratio=gas_oil_ratio,
        gas_gravity=gas_gravity,
        temperature=temperature,
    )
    idx_above_bp = pressure >= bubble_point_pres

    pseudo_liq_den_gcc = oil_density_to_gcc(
        pseudo_liquid_density(
            oil_density=reference_density,
            gor=gas_oil_ratio,
            gas_gravity=gas_gravity,
        )
    )
    api_liq_den = oil_density_to_api(pseudo_liq_den_gcc)
    pres_mpa = 1.0e-6 * pressure
    vel_live = (
        (1900.273 * pseudo_liq_den_gcc**0.64773 - 256.216)
        - ((3.044 + 0.012 * api_liq_den) * temperature)
        + ((3 + 0.031 * api_liq_den) * pres_mpa)
        + ((0.3356 * np.exp(-4.036 * pseudo_liq_den_gcc)) * pres_mpa * temperature)
    )
    vel_live[~idx_above_bp] = np.nan
    return vel_live


def han_batzle_live_oil_density(
    temperature: npt.NDArray[np.float64],
    pressure: npt.NDArray[np.float64],
    reference_density: npt.NDArray[np.float64],
    gas_oil_ratio: npt.NDArray[np.float64],
    gas_gravity: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Correction to the Batzle and Wang 1992 live oil density model
    for pressures above bubble point

    Args:
        temperature (npt.NDArray[np.float64]): temperature [°C]
        pressure (npt.NDArray[np.float64]): formation pressure [Pa]
        reference_density (npt.NDArray[np.float64]): oil density at standard conditions [kg/m^3]
        gas_oil_ratio (npt.NDArray[np.float64]): gas / Oil volume ratio [l/l]
        gas_gravity (npt.NDArray[np.float64]): gas gravity relative to air [unitless]


    Returns:
        npt.NDArray[np.float64]: oil density [kg/m^3]
    """

    # Estimate the bubble point pressure and identify pressure
    # values above and below bubble point
    bubble_point_pres = bp_standing(
        density=reference_density,
        gas_oil_ratio=gas_oil_ratio,
        gas_gravity=gas_gravity,
        temperature=temperature,
    )
    idx_above_bp = pressure > bubble_point_pres

    # Calculate oil density, add correction for pressure values
    # above bubble point, set values under bubble point to NaN
    # and issue a warning
    in_situ_oil_density = live_oil_density(
        temperature=temperature,
        pressure=bubble_point_pres,
        reference_density=reference_density,
        gas_oil_ratio=gas_oil_ratio,
        gas_gravity=gas_gravity,
    )
    if np.any(~idx_above_bp):
        in_situ_oil_density[~idx_above_bp] = np.nan
        warn(
            f"live oil density: {np.sum(~idx_above_bp)} samples under bubble point, set to NaN"
        )
    if np.any(idx_above_bp):
        a_factor = density_correction_vasquez_beggs_ahmed(
            gor=gas_oil_ratio[idx_above_bp],
            temp=temperature[idx_above_bp],
            gravity=gas_gravity[idx_above_bp],
            rho0=reference_density[idx_above_bp],
        )
        in_situ_oil_density[idx_above_bp] = (
            in_situ_oil_density[idx_above_bp]
            * (pressure[idx_above_bp] / bubble_point_pres[idx_above_bp]) ** a_factor
        )
    return in_situ_oil_density
