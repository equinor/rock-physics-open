import numpy as np
import numpy.typing as npt

from rock_physics_open.fluid_models.oil_model.oil_utilities import (
    ArrayLikeFloat,
    as_float_array,
    inputs_are_scalar,
    oil_density_to_gcc,
    oil_density_to_kg_m_3,
)


def live_oil_density(
    temperature: ArrayLikeFloat,
    pressure: ArrayLikeFloat | None,  # pyright: ignore[reportUnusedParameter]
    reference_density: ArrayLikeFloat,
    gas_oil_ratio: ArrayLikeFloat,
    gas_gravity: ArrayLikeFloat,
) -> npt.NDArray[np.float64]:
    """
    Density of live oil at saturation.

    Equation 24 in Batzle & Wang [1].

    :param reference_density: Density of the oil without dissolved gas
        at 15.6 degrees Celsius and atmospheric pressure. [kg/m^3]
    :param pressure: Formation pressure [Pa] of oil (for future implementation only)
    :param gas_oil_ratio: The volume ratio of gas to oil [l/l]
    :param temperature: Temperature [°C] of oil.
    :param gas_gravity: molar mass of gas relative to air molar mas.
    :return: Density of live oil [kg/m^3].
    """
    scalar_input = inputs_are_scalar(
        temperature,
        reference_density,
        gas_oil_ratio,
        gas_gravity,
    )
    temperature_arr = as_float_array(temperature)
    reference_density_arr = as_float_array(reference_density)
    gas_oil_ratio_arr = as_float_array(gas_oil_ratio)
    gas_gravity_arr = as_float_array(gas_gravity)

    density_gcc = oil_density_to_gcc(reference_density_arr)
    b0 = live_oil_volume_factor(
        temperature=temperature_arr,
        reference_density=reference_density_arr,
        gas_oil_ratio=gas_oil_ratio_arr,
        gas_gravity=gas_gravity_arr,
    )
    density = (
        oil_density_to_kg_m_3(
            density_gcc + 0.0012 * gas_gravity_arr * gas_oil_ratio_arr
        )
        / b0
    )
    return density[0] if scalar_input else density


def live_oil_pseudo_density(
    temperature: ArrayLikeFloat,
    reference_density: ArrayLikeFloat,
    gas_oil_ratio: ArrayLikeFloat,
    gas_gravity: ArrayLikeFloat,
) -> npt.NDArray[np.float64]:
    """
    Pseudo density used to substitute reference density in dead_oil_wave_velocity
    for live oils.

    Equation 22 in Batzle & Wang [1].

    :param reference_density: Density of the oil without dissolved gas
        at 15.6 degrees Celsius and atmospheric pressure. [kg/m^3]
    :param gas_oil_ratio: The volume ratio of gas to oil [l/l]
    :param temperature: Temperature [°C] of oil.
    :param gas_gravity: molar mass of gas relative to air molar mas.
    :return: Pseudo-density of live oil [kg/m^3].
    """
    scalar_input = inputs_are_scalar(
        temperature,
        reference_density,
        gas_oil_ratio,
        gas_gravity,
    )
    temperature_arr = as_float_array(temperature)
    reference_density_arr = as_float_array(reference_density)
    gas_oil_ratio_arr = as_float_array(gas_oil_ratio)
    gas_gravity_arr = as_float_array(gas_gravity)

    density_gcc = oil_density_to_gcc(reference_density_arr)
    b0 = live_oil_volume_factor(
        temperature=temperature_arr,
        reference_density=reference_density_arr,
        gas_oil_ratio=gas_oil_ratio_arr,
        gas_gravity=gas_gravity_arr,
    )
    density = oil_density_to_kg_m_3(density_gcc / b0) / (1 + 0.001 * gas_oil_ratio_arr)
    return density[0] if scalar_input else density


def live_oil_volume_factor(
    temperature: ArrayLikeFloat,
    reference_density: ArrayLikeFloat,
    gas_oil_ratio: ArrayLikeFloat,
    gas_gravity: ArrayLikeFloat,
) -> npt.NDArray[np.float64]:
    """
    Volume factor derived by Standing (1962), equation 23 in Batzle & Wang [1].
    :param reference_density: Density of the oil without dissolved gas
        at 15.6 degrees Celsius and atmospheric pressure. [kg/m^3]
    :param gas_oil_ratio: The volume ratio of gas to oil [l/l]
    :param temperature: Temperature [°C] of oil.
    :param gas_gravity: molar mass of gas relative to air molar mas.
    :return: A volume factor in calculating pseudo-density of live oil [unitless].
    """
    scalar_input = inputs_are_scalar(
        temperature,
        reference_density,
        gas_oil_ratio,
        gas_gravity,
    )
    temperature_arr = as_float_array(temperature)
    reference_density_arr = as_float_array(reference_density)
    gas_oil_ratio_arr = as_float_array(gas_oil_ratio)
    gas_gravity_arr = as_float_array(gas_gravity)

    density_gcc = oil_density_to_gcc(reference_density_arr)
    volume_factor = (
        0.972
        + 0.00038
        * (
            2.4 * gas_oil_ratio_arr * np.sqrt(gas_gravity_arr / density_gcc)
            + temperature_arr
            + 17.8
        )
        ** 1.175
    )
    return volume_factor[0] if scalar_input else volume_factor
