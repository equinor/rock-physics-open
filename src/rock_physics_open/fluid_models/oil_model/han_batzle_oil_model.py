from warnings import warn

import numpy as np

from rock_physics_open.fluid_models.oil_model.oil_utilities import (
    ArrayLikeFloat,
    as_float_array,
    inputs_are_scalar,
    oil_density_to_api,
    oil_density_to_gcc,
    oil_density_to_kg_m_3,
)

from .live_oil_density import live_oil_density
from .oil_bubble_point import bp_standing


def gas_appearant_density(
    oil_density: ArrayLikeFloat,
    gas_gravity: ArrayLikeFloat,
) -> ArrayLikeFloat:
    """
    From Han and Batzle 2000: Equation 5
    The Rock Physics Handbook Ed. 3, Equation 6.22.42
    Appearant liquid density of natural gas at standard conditions

    Args:
        oil_density (ArrayLikeFloat): oil density at standard conditions [kg/m^3]
        gas_gravity (ArrayLikeFloat): gas gravity relative to air [unitless]

    Returns:
        ArrayLikeFloat: gas liquid density [kg/m^3]
    """
    scalar_input = inputs_are_scalar(oil_density, gas_gravity)
    oil_density_arr = as_float_array(oil_density)
    gas_gravity_arr = as_float_array(gas_gravity)

    oil_api = oil_density_to_api(oil_density_to_gcc(oil_density_arr))
    tmp = 0.61731 * (10.0 ** (-0.00326 * oil_api)) + (
        1.5177 - 0.54349 * np.log10(oil_api)
    ) * np.log10(gas_gravity_arr)
    density = oil_density_to_kg_m_3(tmp)
    return density[0] if scalar_input else density


def pseudo_liquid_density(
    oil_density: ArrayLikeFloat,
    gor: ArrayLikeFloat,
    gas_gravity: ArrayLikeFloat,
    gas_coefficient: float = 0.113,
) -> ArrayLikeFloat:
    """
    From Han and Batzle 2000: equation 7
    The Rock Physics Handbook Ed. 3, Equation 6.22.43-44
    Calculate the pseudo liquid density at standard conditions.
    With the default setting for the 'gas_coefficient', this is adapted to
    the Han and Batzle model for live oil velocity

    Args:
        gor (ArrayLikeFloat): gas / oil volume ratio [unitless]
        oil_density (ArrayLikeFloat): oil density at standard conditions [kg/m^3]
        gas_gravity (ArrayLikeFloat): gas gravity relative to air [unitless]
        gas_coefficient: (float): empirical gas coefficient


    Returns:
        ArrayLikeFloat: pseudo liquid density
    """
    # Calculate the apprearant volume fraction of the pseudo-liquid gas
    # at standard conditions
    scalar_input = inputs_are_scalar(oil_density, gor, gas_gravity)
    oil_density_arr = as_float_array(oil_density)
    gor_arr = as_float_array(gor)
    gas_gravity_arr = as_float_array(gas_gravity)

    gas_app_dens = gas_appearant_density(
        oil_density=oil_density_arr, gas_gravity=gas_gravity_arr
    )
    vol_gas = (
        0.00123
        * gor_arr
        * gas_gravity_arr
        / (oil_density_to_gcc(gas_app_dens) + 0.00123 * gor_arr * gas_gravity_arr)
    )
    density = (
        oil_density_arr * (1.0 - vol_gas) + gas_coefficient * gas_app_dens * vol_gas
    )
    return density[0] if scalar_input else density


def density_correction_vasquez_beggs_ahmed(
    gor: ArrayLikeFloat,
    temp: ArrayLikeFloat,
    gravity: ArrayLikeFloat,
    rho0: ArrayLikeFloat,
) -> ArrayLikeFloat:
    """
    Correction to Batzle and Wang 1992 oil density model for pressures above
    bubble point. This method is referred to as 'a more widely used correction'.
    The Rock Physics Handbook, 3rd ed, 2020: Equation 6.22.38-40

    Args:
        gor (ArrayLikeFloat): gas / oil ratio [l/l]
        temp (ArrayLikeFloat): temperature [°C]
        gravity (ArrayLikeFloat): gas gravity relative to air [unitless]
        rho0 (ArrayLikeFloat): oil density at standard conditions [kg/m^3]

    Returns:
        ArrayLikeFloat: exponent for correction of pressures above bubble point
    """
    scalar_input = inputs_are_scalar(gor, temp, gravity, rho0)
    gor_arr = as_float_array(gor)
    temp_arr = as_float_array(temp)
    gravity_arr = as_float_array(gravity)
    rho0_arr = as_float_array(rho0)

    rho0_api = oil_density_to_api(oil_density_to_gcc(rho0_arr))
    correction = 1.0e-5 * (
        28.08 * gor_arr
        + 30.96 * temp_arr
        - 1180.0 * gravity_arr
        + 12.61 * rho0_api
        - 882.6
    )
    return correction[0] if scalar_input else correction


def han_batzle_live_oil_velocity(
    reference_density: ArrayLikeFloat,
    gas_oil_ratio: ArrayLikeFloat,
    gas_gravity: ArrayLikeFloat,
    temperature: ArrayLikeFloat,
    pressure: ArrayLikeFloat,
) -> ArrayLikeFloat:
    """
    Han and Batzle 2000: oil velocity model, equation 10-16
    The Rock Physics Handbook Ed. 3, Equation 6.22.52

    Args:
        reference_density (ArrayLikeFloat): oil density at standard conditions [kg/m^3]
        gas_oil_ratio (ArrayLikeFloat): gas / Oil volume ratio [l/l]
        gas_gravity (ArrayLikeFloat): gas gravity relative to air [unitless]
        temperature (ArrayLikeFloat): temperature [°C]
        pressure (ArrayLikeFloat): formation pressure [Pa]

    Returns:
        ArrayLikeFloat: oil velocity [m/s]
    """
    # Estimate the bubble point pressure and identify pressure
    # values above and below bubble point
    scalar_input = inputs_are_scalar(
        reference_density,
        gas_oil_ratio,
        gas_gravity,
        temperature,
        pressure,
    )
    reference_density_arr = as_float_array(reference_density)
    gas_oil_ratio_arr = as_float_array(gas_oil_ratio)
    gas_gravity_arr = as_float_array(gas_gravity)
    temperature_arr = as_float_array(temperature)
    pressure_arr = as_float_array(pressure)

    bubble_point_pres = bp_standing(
        density=reference_density_arr,
        gas_oil_ratio=gas_oil_ratio_arr,
        gas_gravity=gas_gravity_arr,
        temperature=temperature_arr,
    )
    idx_above_bp = pressure_arr > bubble_point_pres

    pseudo_liq_den_gcc = oil_density_to_gcc(
        pseudo_liquid_density(
            oil_density=reference_density_arr,
            gor=gas_oil_ratio_arr,
            gas_gravity=gas_gravity_arr,
        )
    )
    api_liq_den = oil_density_to_api(pseudo_liq_den_gcc)
    pres_mpa = 1.0e-6 * pressure_arr
    vel_live = (
        (1900.273 * pseudo_liq_den_gcc**0.64773 - 256.216)
        - ((3.044 + 0.012 * api_liq_den) * temperature_arr)
        + ((3 + 0.031 * api_liq_den) * pres_mpa)
        + ((0.3356 * np.exp(-4.036 * pseudo_liq_den_gcc)) * pres_mpa * temperature_arr)
    )
    vel_live[~idx_above_bp] = np.nan
    return vel_live[0] if scalar_input else vel_live


def han_batzle_live_oil_density(
    temperature: ArrayLikeFloat,
    pressure: ArrayLikeFloat,
    reference_density: ArrayLikeFloat,
    gas_oil_ratio: ArrayLikeFloat,
    gas_gravity: ArrayLikeFloat,
) -> ArrayLikeFloat:
    """
    Correction to the Batzle and Wang 1992 live oil density model
    for pressures above bubble point

    Args:
        temperature (ArrayLikeFloat): temperature [°C]
        pressure (ArrayLikeFloat): formation pressure [Pa]
        reference_density (ArrayLikeFloat): oil density at standard conditions [kg/m^3]
        gas_oil_ratio (ArrayLikeFloat): gas / Oil volume ratio [l/l]
        gas_gravity (ArrayLikeFloat): gas gravity relative to air [unitless]

    Returns:
        ArrayLikeFloat: oil density [kg/m^3]
    """

    # Estimate the bubble point pressure and identify pressure
    # values above and below bubble point
    scalar_input = inputs_are_scalar(
        temperature,
        pressure,
        reference_density,
        gas_oil_ratio,
        gas_gravity,
    )
    temperature_arr = as_float_array(temperature)
    pressure_arr = as_float_array(pressure)
    reference_density_arr = as_float_array(reference_density)
    gas_oil_ratio_arr = as_float_array(gas_oil_ratio)
    gas_gravity_arr = as_float_array(gas_gravity)

    bubble_point_pres = bp_standing(
        density=reference_density_arr,
        gas_oil_ratio=gas_oil_ratio_arr,
        gas_gravity=gas_gravity_arr,
        temperature=temperature_arr,
    )
    idx_above_bp = pressure_arr > bubble_point_pres

    # Calculate oil density, add correction for pressure values
    # above bubble point, set values under bubble point to NaN
    # and issue a warning
    in_situ_oil_density = live_oil_density(
        temperature=temperature_arr,
        pressure=bubble_point_pres,
        reference_density=reference_density_arr,
        gas_oil_ratio=gas_oil_ratio_arr,
        gas_gravity=gas_gravity_arr,
    )
    if np.any(~idx_above_bp):
        in_situ_oil_density[~idx_above_bp] = np.nan
        warn(
            f"live oil density: {np.sum(~idx_above_bp)} samples under bubble point, set to NaN"
        )
    if np.any(idx_above_bp):
        a_factor = density_correction_vasquez_beggs_ahmed(
            gor=gas_oil_ratio_arr[idx_above_bp],
            temp=temperature_arr[idx_above_bp],
            gravity=gas_gravity_arr[idx_above_bp],
            rho0=reference_density_arr[idx_above_bp],
        )
        in_situ_oil_density[idx_above_bp] = (
            in_situ_oil_density[idx_above_bp]
            * (pressure_arr[idx_above_bp] / bubble_point_pres[idx_above_bp]) ** a_factor
        )
    return in_situ_oil_density[0] if scalar_input else in_situ_oil_density
