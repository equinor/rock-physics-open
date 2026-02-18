import numpy as np
import numpy.typing as npt

from .dead_oil_velocity import dead_oil_velocity
from .live_oil_density import live_oil_pseudo_density
from .oil_utilities import ArrayLikeFloat, as_float_array, inputs_are_scalar


def live_oil_velocity(
    temperature: ArrayLikeFloat,
    pressure: ArrayLikeFloat,
    reference_density: ArrayLikeFloat,
    gas_oil_ratio: ArrayLikeFloat,
    gas_gravity: ArrayLikeFloat,
) -> npt.NDArray[np.float64]:
    """
    Primary wave velocity of live oil at saturation.

    Substitute Equation 22 in Equation 20 of Batzle & Wang [1].

    :param reference_density: Density of the oil without dissolved gas
        at 15.6 degrees Celsius and atmospheric pressure. [kg/m^3]
    :param pressure: Formation pressure [Pa] of oil
    :param gas_oil_ratio: The volume ratio of gas to oil [l/l]
    :param temperature: Temperature [°C] of oil.
    :param gas_gravity: molar mass of gas relative to air molar mas.
    :return: Primary wave velocity of live oil [m/s].
    """
    scalar_input = inputs_are_scalar(
        temperature, pressure, reference_density, gas_oil_ratio, gas_gravity
    )
    temperature_arr = as_float_array(temperature)
    pressure_arr = as_float_array(pressure)
    reference_density_arr = as_float_array(reference_density)
    gas_oil_ratio_arr = as_float_array(gas_oil_ratio)
    gas_gravity_arr = as_float_array(gas_gravity)

    rho_marked = live_oil_pseudo_density(
        temperature=temperature_arr,
        reference_density=reference_density_arr,
        gas_oil_ratio=gas_oil_ratio_arr,
        gas_gravity=gas_gravity_arr,
    )
    velocity = dead_oil_velocity(
        temperature=temperature_arr,
        pressure=pressure_arr,
        reference_density=rho_marked,
    )
    return velocity[0] if scalar_input else velocity
