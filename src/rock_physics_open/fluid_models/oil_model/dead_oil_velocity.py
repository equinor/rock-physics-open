import numpy as np
import numpy.typing as npt


def dead_oil_velocity(
    temperature: npt.NDArray[np.float64],
    pressure: npt.NDArray[np.float64],
    reference_density: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    The primary wave velocity in oil without dissolved gas (dead).

    Uses equation 20a from Batzle & Wang [1].

    :param reference_density: Density of oil at 15.6 degrees Celsius and atmospheric
        pressure [kg/m^3]
    :param pressure: Pressure [Pa] of oil
    :param temperature: Temperature [°C] of oil.
    :return: primary velocity of dead oil in m/s.
    """
    pressure_mpa = pressure * 1e-6
    density_gcc = reference_density / 1000.0
    return (
        2096 * np.sqrt(density_gcc / (2.6 - density_gcc))
        - 3.7 * temperature
        + 4.64 * pressure_mpa
        + 0.0115
        * (4.12 * np.sqrt(1.08 * density_gcc**-1 - 1) - 1)
        * temperature
        * pressure_mpa
    )
