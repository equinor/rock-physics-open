from .dead_oil_velocity import dead_oil_velocity
from .live_oil_density import live_oil_pseudo_density


def live_oil_velocity(
    temperature, pressure, reference_density, gas_oil_ratio, gas_gravity
):
    """
    Primary wave velocity of live oil at saturation.

    Substitute Equation 22 in Equation 20 of Batzle & Wang [1].

    :param reference_density: Density of the oil without dissolved gas
        at 15.6 degrees Celsius and atmospheric pressure. (g/cc)
    :param pressure: Pressure (MPa) of oil
    :param gas_oil_ratio: The volume ratio of gas to oil [l/l]
    :param temperature: Temperature (Celsius) of oil.
    :param gas_gravity: molar mass of gas relative to air molar mas.
    :return: Primary wave velocity of live oil [m/s].
    """
    rho_marked = live_oil_pseudo_density(
        temperature, reference_density, gas_oil_ratio, gas_gravity
    )
    return dead_oil_velocity(temperature, pressure, rho_marked)
