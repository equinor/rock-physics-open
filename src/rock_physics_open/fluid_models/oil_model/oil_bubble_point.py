import numpy as np


def bp_standing(
    density: np.ndarray | float,
    gas_oil_ratio: np.ndarray | float,
    gas_gravity: np.ndarray | float,
    temperature: np.ndarray | float,
) -> np.ndarray | float:
    """
    Reservoir oils include some natural gas in solution. The pressure at which
    this natural gas begins to come out of solution and form bubbles is known
    as the bubble point pressure. See https://petrowiki.org/Oil_bubblepoint_pressure
    Based on the correlation from:
    Standing, M. B. "A pressure-volume-temperature correlation for mixtures of
    California oils and gases." Drilling and Production Practice. American
    Petroleum Institute, 1947.
    Uses refinment described here: https://petrowiki.org/Oil_bubblepoint_pressure
    :param density: density of oil at room conditions [kg/m^3]
    :param gas_oil_ratio: The volume ratio of gas to oil [l/l]
    :param gas_gravity: molar mass of gas relative to air molar mas.
    :param temperature: temperature of oil [°C]
    :return: bubble point pressure [Pa]
    """

    # Standing, M.B. (1947) uses:
    # * pressure in psi
    # * temperature in F
    # * gas oil ratio in cu ft per bbl
    # * bubble point in in absolute psi
    # * gravity_of_tank_oil in API
    #
    # The paper describes that bubble point is a function of
    # (gor / gr) ** 0.83
    #   * (10**(0.00091 * t)/10**(0.0125*gravity_of_tank_oil))
    #
    # and gives the following example:
    #   Bubble point pressure at 200° F
    #   of a liquid having gas-oil ratio
    #   350 CFB, a gas gravity 0.75, and
    #   a tank oil gravity of 30° API.
    #   The required pressure is found
    #   to be 1930 psia.
    #
    # We could scale to fit this example, however,
    # we use 1896 psia.
    #
    # For density in kg/m^3:
    # 10**(0.0125*gravity_of_tank_oil)=
    #        10**(1770.79/density - 1.64375)=
    #         e**(4072.69738323/density - 3.78487)
    # For temperature in Celsius:
    #    10**(0.00091 * Farenheit) =
    #    10**(0.001638*Celsius + 0.02912) =
    #     e**(0.00377163*Celsius + 0.0670513)
    #
    # Removing constants factors we get
    # (gor_ratio /
    #  gas_gravity)) ** 0.83 / ( np.exp(4072.69738323 / density -
    #  0.00377163438 * temperature_c)
    #
    # The equation occurs in this form as equation 21a in
    # Batzle, Michael, and Zhijing Wang. "Seismic properties of pore fluids."
    # Geophysics 57.11 (1992): 1396-1408.

    ratio = gas_oil_ratio / gas_gravity
    denominator = np.exp(4072 / density - 0.00377 * temperature)

    return 24469793.9134 * ratio**0.83 / denominator
