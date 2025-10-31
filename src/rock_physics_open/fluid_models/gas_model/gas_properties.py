import numpy as np
import numpy.typing as npt
from scipy.constants import gas_constant

from rock_physics_open.equinor_utilities.conversions import celsius_to_kelvin

AIR_WEIGHT = 28.8 * 1.0e-3  # kg/mol


def gas_properties(
    temperature: npt.NDArray[np.float64],
    pressure: npt.NDArray[np.float64],
    gas_gravity: npt.NDArray[np.float64],
    model: str | None = None,  # pyright: ignore[reportUnusedParameter]
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """
    :param gas_gravity: molar mass of gas relative to air molar mas.
    :param pressure: Confining pressure (Pa)
    :param temperature: Temperature (Celsius).
    :param model: for future use
    :return: vel_gas [m/s], den_gas [kg/m^3], k_gas [Pa], eta_gas [cP]
    """

    den_gas = gas_density(
        absolute_temperature=celsius_to_kelvin(temperature),
        pressure=pressure,
        gas_gravity=gas_gravity,
    )

    k_gas = gas_bulk_modulus(
        absolute_temperature=celsius_to_kelvin(temperature),
        pressure=pressure,
        gas_gravity=gas_gravity,
    )

    vel_gas = (k_gas / den_gas) ** 0.5

    eta_gas = lee_gas_viscosity(
        absolute_temperature=celsius_to_kelvin(temperature),
        pressure=pressure,
        gas_gravity=gas_gravity,
    )

    return vel_gas, den_gas, k_gas, eta_gas


def molecular_weight(gas_gravity: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    calculates molecular weight of a gas from gas gravity.
    :param gas_gravity: molar mass of gas relative to air molar mas.
    :return: The volume of the gas in kg/mol.
    """
    return gas_gravity * AIR_WEIGHT


def molar_volume(
    absolute_temperature: npt.NDArray[np.float64],
    pressure: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    calculates molar volume using the ideal gas law.
    :param absolute_temperature: The absolute temperature of the gas in kelvin.
    :param pressure: Confining pressure in Pa.
    :return: The volume of the gas in m^3/mol.
    """

    return gas_constant * absolute_temperature / pressure


def ideal_gas_density(
    absolute_temperature: npt.NDArray[np.float64],
    pressure: npt.NDArray[np.float64],
    gas_gravity: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    calculates molar volume using the ideal gas law.
    :param gas_gravity: molar mass of gas relative to air molar mas.
    :param absolute_temperature: The absolute temperature of the gas in kelvin.
    :param pressure: Confining pressure in Pa.
    :return: The density of the gas in kg/m^3
    """
    return molecular_weight(gas_gravity) / molar_volume(absolute_temperature, pressure)


def ideal_gas_primary_velocity(
    absolute_temperature: npt.NDArray[np.float64],
    gas_gravity: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    :param gas_gravity: molar mass of gas relative to air molar mas.
    :param absolute_temperature: The absolute temperature of the gas in kelvin.
    :return: The compressional wave velocity of the gas in m/s.
    """
    return np.sqrt(gas_constant * absolute_temperature / molecular_weight(gas_gravity))


def ideal_gas(
    absolute_temperature: npt.NDArray[np.float64],
    pressure: npt.NDArray[np.float64],
    gas_gravity: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    :param gas_gravity: molar mass of gas relative to air molar mas.
    :param absolute_temperature: The absolute temperature of the gas in kelvin.
    :param pressure: Confining pressure in Pa.
    :return: ideal_gas_velocity [m/s], ideal_gas_density [kg/m^3],
    """
    ideal_gas_den = ideal_gas_density(
        absolute_temperature=absolute_temperature,
        pressure=pressure,
        gas_gravity=gas_gravity,
    )
    ideal_gas_vel = ideal_gas_primary_velocity(
        absolute_temperature=absolute_temperature,
        gas_gravity=gas_gravity,
    )
    return ideal_gas_vel, ideal_gas_den


def pseudoreduced_temperature(
    absolute_temperature: npt.NDArray[np.float64],
    gas_gravity: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
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
    pressure: npt.NDArray[np.float64],
    gas_gravity: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    calculates pseudoreduced pressure, equation 9a from Batzle & Wang [1].

    Uses relationship from

    Thomas, L. K., Hankinson, R. W., and Phillips, K. A., 1970,
    Determination of acoustic velocities for natural gas: J. Petr.
    Tech., 22, 889-892.

    :param gas_gravity: molar mass of gas relative to air molar mas.
    :param pressure: Confining pressure in Pa.
    :return: Pseudoreduced pressure in Pa.
    """
    return pressure / (4.892 - 0.4048 * gas_gravity)


def compressibility_factor(
    absolute_temperature: npt.NDArray[np.float64],
    pressure: npt.NDArray[np.float64],
    gas_gravity: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    calculates compressibility hydro-carbon gas, equation 10b and 10c from
    Batzle & Wang [1].

    :param gas_gravity: molar mass of gas relative to air molar mas.
    :param absolute_temperature: The absolute temperature of the gas in kelvin.
    :param pressure: Confining pressure in Pa.
    :return: Gas compressibility - unitless
    """
    tpr = pseudoreduced_temperature(
        absolute_temperature=absolute_temperature,
        gas_gravity=gas_gravity,
    )

    # Pseudoreduced pressure has unit MPa in equation
    ppr = pseudoreduced_pressure(pressure, gas_gravity) * 1.0e-6

    return (
        (0.03 + 0.00527 * (3.5 - tpr) ** 3) * ppr
        + 0.642 * tpr
        - 0.007 * tpr**4
        - 0.52
        + 0.109
        * (3.85 - tpr) ** 2
        / np.exp((0.45 + 8.0 * (0.56 - 1 / tpr) ** 2) * ppr**1.2 / tpr)
    )


def gas_density(
    absolute_temperature: npt.NDArray[np.float64],
    pressure: npt.NDArray[np.float64],
    gas_gravity: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    The density of hydro-carbon gas, using equation 10 from Batzle & Wang [1].

    :param gas_gravity: molar mass of gas relative to air molar mas.
    :param absolute_temperature: The absolute temperature of the gas in kelvin.
    :param pressure: Confining pressure in Pa.
    :return: The density of the gas in kg/m^3
    """

    _, ideal_gas_den = ideal_gas(
        absolute_temperature=absolute_temperature,
        pressure=pressure,
        gas_gravity=gas_gravity,
    )
    return ideal_gas_den / compressibility_factor(
        absolute_temperature=absolute_temperature,
        pressure=pressure,
        gas_gravity=gas_gravity,
    )


def compressibility_rate_per_pseudoreduced_pressure(
    absolute_temperature: npt.NDArray[np.float64],
    pressure: npt.NDArray[np.float64],
    gas_gravity: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Derivate of compressibility_factor with respect to pressure.

    :param gas_gravity: molar mass of gas relative to air molar mas.
    :param absolute_temperature: The absolute temperature of the gas in kelvin.
    :param pressure: Confining pressure in MPa.
    :return: Derivative of the compressibility factor (unitless) with respect to pseudoreduced pressure
    """
    tpr = pseudoreduced_temperature(absolute_temperature, gas_gravity)

    # Pseudoreduced pressure is expected to be in MPa in the expression
    ppr = pseudoreduced_pressure(pressure, gas_gravity) * 1.0e-6

    return (
        0.03
        + 0.00527 * (3.5 - tpr) ** 3
        - (
            0.1308
            * (0.45 + 8 * (0.56 - tpr ** (-1)) ** 2)
            * (3.85 - tpr) ** 2
            * ppr**0.2
        )
        / (np.exp(((0.45 + 8 * (0.56 - tpr ** (-1)) ** 2) * ppr**1.2) / tpr) * tpr)
    )


def gas_bulk_modulus(
    absolute_temperature: npt.NDArray[np.float64],
    pressure: npt.NDArray[np.float64],
    gas_gravity: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    The bulk modulus of hydro-carbon gas, using equation 11 from Batzle & Wang [1].

    :param gas_gravity: molar mass of gas relative to air molar mas.
    :param absolute_temperature: The absolute temperature of the gas in kelvin.
    :param pressure: Confining pressure in Pa.
    :return: The bulk modulus of the gas in Pa.
    """
    z = compressibility_factor(
        absolute_temperature=absolute_temperature,
        pressure=pressure,
        gas_gravity=gas_gravity,
    )
    dz_dppr = compressibility_rate_per_pseudoreduced_pressure(
        absolute_temperature=absolute_temperature,
        pressure=pressure,
        gas_gravity=gas_gravity,
    )

    # Set ppr in unit MPa in order to use it in calculation of gamma_0
    ppr = pseudoreduced_pressure(pressure, gas_gravity) * 1.0e-6

    # Equation 11b
    gamma_0 = (
        0.85
        + 5.6 / (ppr + 2)
        + 27.1 / ((ppr + 3.5) ** 2)
        - 8.7 * np.exp(-0.65 * (ppr + 1))
    )

    return gamma_0 * pressure / (1 - dz_dppr * ppr / z)


def gas_viscosity(
    absolute_temperature: npt.NDArray[np.float64],
    pressure: npt.NDArray[np.float64],
    gas_gravity: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    The gas viscosity of hydrocarbon gas, using equations 12 and 13 of Batzle & Wang [1].

    :param absolute_temperature: The absolute temperature of the gas in kelvin.
    :param pressure: Confining pressure in Pa.
    :param gas_gravity: molar mass of gas relative to air mas.
    :return: The gas viscosity of the gas in cP.
    """
    temp_pr = pseudoreduced_temperature(absolute_temperature, gas_gravity)

    # Pseudoreduced pressure should be in unit MPa
    pres_pr = pseudoreduced_pressure(pressure, gas_gravity) * 1.0e-6

    eta_1 = 0.0001 * (
        temp_pr * (28.0 + 48.0 * gas_gravity - 5.0 * gas_gravity**2)
        - 6.47 * gas_gravity**-2
        + 35.0 * gas_gravity**-1
        + 1.14 * gas_gravity
        - 15.55
    )
    return eta_1 * (
        0.001
        * pres_pr
        * (
            (1057.0 - 8.08 * temp_pr) / pres_pr
            + (796.0 * pres_pr**0.5 - 704.0)
            / (((temp_pr - 1.0) ** 0.7) * (pres_pr + 1.0))
            - 3.24 * temp_pr
            - 38.0
        )
    )


def lee_gas_viscosity(
    absolute_temperature: npt.NDArray[np.float64],
    pressure: npt.NDArray[np.float64],
    gas_gravity: npt.NDArray[np.float64],
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
    in Fahrenheit and pressure in psi
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
