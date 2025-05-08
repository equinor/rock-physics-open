import numpy as np
from numpy import exp, sqrt
from scipy.constants import gas_constant

from rock_physics_open.equinor_utilities.conversions import celsius_to_kelvin

AIR_WEIGHT = 28.8  # g/mol


def gas_properties(
    temperature: np.ndarray | float,
    pressure: np.ndarray | float,
    gas_gravity: np.ndarray | float,
    model: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    :param gas_gravity: molar mass of gas relative to air molar mas.
    :param pressure: Confining pressure (Pa)
    :param temperature: Temperature (Celsius).
    :param model: for future use
    :return: vel_gas [m/s], den_gas [kg/m^3], k_gas [Pa], eta_gas [cP]
    """
    den_gas = gas_density(celsius_to_kelvin(temperature), pressure * 1e-6, gas_gravity)
    k_gas = gas_bulk_modulus(
        celsius_to_kelvin(temperature), pressure * 1e-6, gas_gravity
    )
    vel_gas = (k_gas / den_gas) ** 0.5

    eta_gas = lee_gas_viscosity(celsius_to_kelvin(temperature), pressure, gas_gravity)

    return vel_gas, den_gas, k_gas, eta_gas


def molecular_weight(gas_gravity: np.ndarray | float) -> np.ndarray | float:
    """
    calculates molecluar weight of a gas from gas gravity.
    :param gas_gravity: molar mass of gas relative to air molar mas.
    :return: The volume of the gas in g/mol.
    """
    return gas_gravity * AIR_WEIGHT


def molar_volume(
    absolute_temperature: np.ndarray | float,
    pressure: np.ndarray | float,
) -> np.ndarray | float:
    """
    calculates molar volume using the ideal gas law.
    :param absolute_temperature: The absolute temperature of the gas in kelvin.
    :param pressure: Confining pressure in MPa.
    :return: The volume of the gas in cc/mol.
    """
    return gas_constant * absolute_temperature / pressure


def ideal_gas_density(
    absolute_temperature: np.ndarray | float,
    pressure: np.ndarray | float,
    gas_gravity: np.ndarray | float,
) -> np.ndarray | float:
    """
    calculates molar volume using the ideal gas law.
    :param gas_gravity: molar mass of gas relative to air molar mas.
    :param absolute_temperature: The absolute temperature of the gas in kelvin.
    :param pressure: Confining pressure in MPa.
    :return: The density of the gas in g/cc
    """
    return molecular_weight(gas_gravity) / molar_volume(absolute_temperature, pressure)


def ideal_gas_primary_velocity(
    absolute_temperature: np.ndarray | float,
    gas_gravity: np.ndarray | float,
) -> np.ndarray | float:
    """
    :param gas_gravity: molar mass of gas relative to air molar mas.
    :param absolute_temperature: The absolute temperature of the gas in kelvin.
    :return: The compressional wave velocity of the gas in m/s.
    """
    return sqrt(gas_constant * absolute_temperature / molecular_weight(gas_gravity))


def ideal_gas(
    absolute_temperature: np.ndarray | float,
    pressure: np.ndarray | float,
    gas_gravity: np.ndarray | float,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    """
    :param gas_gravity: molar mass of gas relative to air molar mas.
    :param absolute_temperature: The absolute temperature of the gas in kelvin.
    :param pressure: Confining pressure in Pa.
    :return: ideal_gas_density, ideal_gas_velocity
    """
    ideal_gas_den = 1000 * ideal_gas_density(
        absolute_temperature, pressure * 1e6, gas_gravity
    )
    ideal_gas_vel = ideal_gas_primary_velocity(absolute_temperature, gas_gravity)
    return ideal_gas_den, ideal_gas_vel


def pseudoreduced_temperature(
    absolute_temperature: np.ndarray | float,
    gas_gravity: np.ndarray | float,
) -> np.ndarray | float:
    """
    calculates pseudoreduced temperature, equation 9a from Batzle & Wang [1].

    Uses relationship from

    Thomas, L. K., Hankinson, R. W., and Phillips, K. A., 1970,
    Determination of acoustic velocities for natural gas: J. Petr.
    Tech., 22, 889-892.

    :param gas_gravity: molar mass of gas relative to air molar mas.
    :param absolute_temperature: The absolute temperature of the gas in kelvin.
    :return: Pseudoreduced temperature in kelvin.
    """
    return absolute_temperature / (94.72 + 170.75 * gas_gravity)


def pseudoreduced_pressure(
    pressure: np.ndarray | float,
    gas_gravity: np.ndarray | float,
) -> np.ndarray | float:
    """
    calculates pseudoreduced pressure, equation 9a from Batzle & Wang [1].

    Uses relationship from

    Thomas, L. K., Hankinson, R. W., and Phillips, K. A., 1970,
    Determination of acoustic velocities for natural gas: J. Petr.
    Tech., 22, 889-892.

    :param gas_gravity: molar mass of gas relative to air molar mas.
    :param pressure: Confining pressure in MPa.
    :return: Pseudoreduced pressure in MPa.
    """
    return pressure / (4.892 - 0.4048 * gas_gravity)


def compressability_factor(
    absolute_temperature: np.ndarray | float,
    pressure: np.ndarray | float,
    gas_gravity: np.ndarray | float,
) -> np.ndarray | float:
    """
    calculates compressability hydro-carbon gas, equation 10b and 10c from
    Batzle & Wang [1].

    :param gas_gravity: molar mass of gas relative to air molar mas.
    :param absolute_temperature: The absolute temperature of the gas in kelvin.
    :param pressure: Confining pressure in MPa.
    :return: The density of the gas in g/cc
    """
    tpr = pseudoreduced_temperature(absolute_temperature, gas_gravity)
    ppr = pseudoreduced_pressure(pressure, gas_gravity)

    return (
        (0.03 + 0.00527 * (3.5 - tpr) ** 3) * ppr
        + 0.642 * tpr
        - 0.007 * tpr**4
        - 0.52
        + 0.109
        * (3.85 - tpr) ** 2
        / exp((0.45 + 8 * (0.56 - 1 / tpr) ** 2) * ppr**1.2 / tpr)
    )


def gas_density(
    absolute_temperature: np.ndarray | float,
    pressure: np.ndarray | float,
    gas_gravity: np.ndarray | float,
) -> np.ndarray | float:
    """
    The density of hydro-carbon gas, using equation 10 from Batzle & Wang [1].

    :param gas_gravity: molar mass of gas relative to air molar mas.
    :param absolute_temperature: The absolute temperature of the gas in kelvin.
    :param pressure: Confining pressure in MPa.
    :return: The density of the gas in g/cc
    """
    ideal_gas_den, ideal_gas_vel = ideal_gas(
        absolute_temperature, pressure * 1e-6, gas_gravity
    )
    return ideal_gas_den / compressability_factor(
        absolute_temperature, pressure, gas_gravity
    )


def compressability_rate_per_pseudoreduced_pressure(
    absolute_temperature: np.ndarray | float,
    pressure: np.ndarray | float,
    gas_gravity: np.ndarray | float,
) -> np.ndarray | float:
    """
    Derivate of compressability_factor with respect to pressure.

    :param gas_gravity: molar mass of gas relative to air molar mas.
    :param absolute_temperature: The absolute temperature of the gas in kelvin.
    :param pressure: Confining pressure in MPa.
    :return: The density of the gas in g/cc
    """
    tpr = pseudoreduced_temperature(absolute_temperature, gas_gravity)
    ppr = pseudoreduced_pressure(pressure, gas_gravity)

    return (
        0.03
        + 0.00527 * (3.5 - tpr) ** 3
        - (
            0.1308
            * (0.45 + 8 * (0.56 - tpr ** (-1)) ** 2)
            * (3.85 - tpr) ** 2
            * ppr**0.2
        )
        / (exp(((0.45 + 8 * (0.56 - tpr ** (-1)) ** 2) * ppr**1.2) / tpr) * tpr)
    )


def gas_bulk_modulus(
    absolute_temperature: np.ndarray | float,
    pressure: np.ndarray | float,
    gas_gravity: np.ndarray | float,
) -> np.ndarray | float:
    """
    The bulk modulus of hydro-carbon gas, using equation 11 from Batzle & Wang [1].

    :param gas_gravity: molar mass of gas relative to air molar mas.
    :param absolute_temperature: The absolute temperature of the gas in kelvin.
    :param pressure: Confining pressure in MPa.
    :return: The bulk modulus of the gas in MPa.
    """
    z = compressability_factor(absolute_temperature, pressure, gas_gravity)
    dz_dppr = compressability_rate_per_pseudoreduced_pressure(
        absolute_temperature, pressure, gas_gravity
    )

    ppr = pseudoreduced_pressure(pressure, gas_gravity)

    # Equation 11b
    gamma_0 = (
        0.85
        + 5.6 / (ppr + 2)
        + 27.1 / ((ppr + 3.5) ** 2)
        - 8.7 * exp(-0.65 * (ppr + 1))
    )

    return gamma_0 * pressure / (1 - dz_dppr * ppr / z)


def lee_gas_viscosity(
    absolute_temperature: np.ndarray | float,
    pressure: np.ndarray | float,
    gas_gravity: np.ndarray | float,
) -> np.ndarray:
    """
    :param absolute_temperature: Absolute temperature of the gas in kelvin.
    :param pressure: Confining pressure in Pa.
    :param gas_gravity: specific gravity of gas relative to air.
    :return: gas viscosity in cP

    Reference
    ---------
    Lee, J. D., et al. (1966). "Viscosity of Natural Gas." In The American Institute of
    Chemical Engineers Journal, Volume 12, Issue 6, pp. 1058-1062.

    Original equation is given in imperial units. Inputs are transformed to temperature
    in Farenheit and pressure in psi
    """
    temp_far = (absolute_temperature - 273.15) * 9.0 / 5.0 + 32.0
    pres_psi = pressure / 6894.757
    return (
        0.001
        * (temp_far + 459.67) ** 0.5
        / pres_psi
        * (0.7 + 1.5 * gas_gravity)
        / (gas_gravity + 1) ** 1.5
    )
