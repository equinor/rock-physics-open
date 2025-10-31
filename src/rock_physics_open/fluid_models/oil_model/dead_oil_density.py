import numpy as np
import numpy.typing as npt


def pressure_adjusted_dead_oil_density(
    pressure: npt.NDArray[np.float64],
    reference_density: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Adjusts density of a dead oil (without dissolved gas) to a given pressure.

    Uses equation 18 from Batzle & Wang [1].

    :param reference_density: The density [kg/m^3] of the dead oil at 15.6 degrees Celsius
        and atmospheric pressure.
    :param pressure: Pressure [Pa] to adjust to.
    :return: Density of oil at given pressure and 21 degrees Celsius (~70 degrees
    Fahrenheit). [kg/m^3]
    """
    pressure_mpa = pressure / 1e6
    density_gcc = reference_density / 1000.0
    return 1000.0 * (
        density_gcc
        + (0.00277 * pressure_mpa - 1.71e-7 * pressure_mpa**3)
        * (density_gcc - 1.15) ** 2
        + 3.49e-4 * pressure_mpa
    )


def temperature_adjusted_dead_oil_density(
    temperature: npt.NDArray[np.float64],
    density_at_21c: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Adjusts density of a dead oil (without dissolved gas) to a given temperature.

    Uses equation 19 from Batzle & Wang [1].

    :param density_at_21c: The density [kg/m^3] of the dead oil at 21 degrees Celsius
    :param temperature: Temperature [°C] of oil.
    :return: Density of oil at given temperature. [kg/m^3]
    """
    density_at_21c_gcc = density_at_21c / 1000.0
    return (
        1000.0 * density_at_21c_gcc / (0.972 + 3.81e-4 * (temperature + 17.78) ** 1.175)
    )


def dead_oil_density(
    temperature: npt.NDArray[np.float64],
    pressure: npt.NDArray[np.float64],
    reference_density: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    The density of oil without dissolved gas (dead).

    Uses equation 18 & 19 from Batzle & Wang [1].

    :param reference_density: Density of oil at 15.6 degrees Celsius and atmospheric
        pressure [kg/m^3]
    :param pressure: Pressure [Pa] of oil
    :param temperature: Temperature [°C] of oil.
    :return: density of dead oil at given conditions (kg/m^3).
    """
    density_p = pressure_adjusted_dead_oil_density(
        pressure=pressure,
        reference_density=reference_density,
    )
    return temperature_adjusted_dead_oil_density(
        temperature=temperature,
        density_at_21c=density_p,
    )
