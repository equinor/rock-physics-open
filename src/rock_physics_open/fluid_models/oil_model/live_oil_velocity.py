import numpy as np
import numpy.typing as npt

from .dead_oil_velocity import dead_oil_velocity
from .live_oil_density import live_oil_pseudo_density


def live_oil_velocity(
    temperature: npt.NDArray[np.float64],
    pressure: npt.NDArray[np.float64],
    reference_density: npt.NDArray[np.float64],
    gas_oil_ratio: npt.NDArray[np.float64],
    gas_gravity: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Primary wave velocity of live oil at saturation.

    Substitute Equation 22 in Equation 20 of Batzle & Wang [1].

    :param reference_density: Density of the oil without dissolved gas
        at 15.6 degrees Celsius and atmospheric pressure. [kg/m^3]
    :param pressure: Pressure [Pa] of oil
    :param gas_oil_ratio: The volume ratio of gas to oil [l/l]
    :param temperature: Temperature [°C] of oil.
    :param gas_gravity: molar mass of gas relative to air molar mas.
    :return: Primary wave velocity of live oil [m/s].
    """
    rho_marked = live_oil_pseudo_density(
        temperature=temperature,
        reference_density=reference_density,
        gas_oil_ratio=gas_oil_ratio,
        gas_gravity=gas_gravity,
    )
    return dead_oil_velocity(
        temperature=temperature,
        pressure=pressure,
        reference_density=rho_marked,
    )
