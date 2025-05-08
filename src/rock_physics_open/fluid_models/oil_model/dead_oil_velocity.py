import numpy as np


def dead_oil_velocity(
    temperature: np.ndarray | float,
    pressure: np.ndarray | float,
    reference_density: np.ndarray | float,
) -> np.ndarray | float:
    """
    The primary wave velocity in oil without dissolved gas (dead).

    Uses equation 20a from Batzle & Wang [1].

    :param reference_density: Density of oil at 15.6 degrees Celsius and atmospheric
        pressure (g/cc)
    :param pressure: Pressure (MPa) of oil
    :param temperature: Temperature (Celsius) of oil.
    :return: primary velocity of dead oil in m/s.
    """
    return (
        2096 * np.sqrt(reference_density / (2.6 - reference_density))
        - 3.7 * temperature
        + 4.64 * pressure
        + 0.0115
        * (4.12 * np.sqrt(1.08 * reference_density**-1 - 1) - 1)
        * temperature
        * pressure
    )
