import numpy as np


def pressure_adjusted_dead_oil_density(
    pressure: np.ndarray | float,
    reference_density: np.ndarray | float,
) -> np.ndarray | float:
    """
    Adjusts density of a dead oil (without dissolved gas) to a given pressure.

    Uses equation 18 from Batzle & Wang [1].

    :param reference_density: The density (g/cc) of the dead oil at 15.6 degrees Celsius
        and atmospheric pressure.
    :param pressure: Pressure (MPa) to adjust to.
    :return: Density of oil at given pressure and 21 degrees Celsius (~70 degrees
    Farenheit).
    """
    return (
        reference_density
        + (0.00277 * pressure - 1.71 * 10**-7 * pressure**3)
        * (reference_density - 1.15) ** 2
        + 3.49 * 10**-4 * pressure
    )


def temperature_adjusted_dead_oil_density(
    temperature: np.ndarray | float,
    density_at_21c: np.ndarray,
) -> np.ndarray | float:
    """
    Adjusts density of a dead oil (without dissolved gas) to a given temperature.

    Uses equation 19 from Batzle & Wang [1].

    :param density_at_21c: The density (g/cc) of the dead oil at 21 degrees Celsius
    :param temperature: Temperature (Celsius) of oil.
    :return: Density of oil at given temperature.
    """
    return density_at_21c / (0.972 + 3.81 * (10**-4) * (temperature + 17.78) ** 1.175)


def dead_oil_density(
    temperature: np.ndarray | float,
    pressure: np.ndarray | float,
    reference_density: np.ndarray | float,
) -> np.ndarray | float:
    """
    The density of oil without dissolved gas (dead).

    Uses equation 18 & 19 from Batzle & Wang [1].

    :param reference_density: Density of oil at 15.6 degrees Celsius and atmospheric
        pressure. (g/cc)
    :param pressure: Pressure (MPa) of oil
    :param temperature: Temperature (Celsius) of oil.
    :return: density of dead oil at given conditions.
    """
    density_p = pressure_adjusted_dead_oil_density(pressure, reference_density)
    return temperature_adjusted_dead_oil_density(temperature, density_p)
