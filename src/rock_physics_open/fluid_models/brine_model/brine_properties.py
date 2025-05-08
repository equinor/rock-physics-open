import warnings

import numpy as np
from numpy import sqrt
from numpy.polynomial.polynomial import polyval2d, polyval3d


def brine_properties(
    temperature: np.ndarray | float,
    pressure: np.ndarray | float,
    salinity: np.ndarray | float,
    p_nacl: np.ndarray | float | None = None,
    p_kcl: np.ndarray | float | None = None,
    p_cacl: np.ndarray | float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    :param salinity: Salinity of solution as ppm of NaCl.
    :param pressure: Pressure (Pa) of oil
    :param temperature: Temperature (Celsius) of oil.
    :param p_nacl: NaCl percentage, for future use
    :param p_kcl: KCl percentage, for future use
    :param p_cacl: CaCl percentage, for future use
    :return: vel_b [m/s], den_b [kg/m^3], k_b [Pa]
    """
    vel_b = brine_primary_velocity(temperature, pressure * 1e-6, salinity * 1e-6)
    den_b = brine_density(temperature, pressure * 1e-6, salinity * 1e-6) * 1000
    k_b = vel_b**2 * den_b
    return vel_b, den_b, k_b


def brine_density(
    temperature: np.ndarray | float,
    pressure: np.ndarray | float,
    salinity: np.ndarray | float,
) -> np.ndarray | float:
    """
    density of sodium chloride solutions, equation 27 in Batzle & Wang [1].
    :param salinity: Salinity of solution as weight fraction (ppm/1000000) of
        sodium chloride.
    :param pressure: Pressure (MPa) of oil
    :param temperature: Temperature (Celsius) of oil.
    :return: density of solution in g/cc.
    """
    coefficients = [
        [[0.668, 3e-4], [8e-5, -13e-6], [3e-6, 0.0]],
        [[0.44, -24e-4], [-33e-4, 47e-6], [0.0, 0.0]],
    ]
    return water_density(temperature, pressure) + salinity * polyval3d(
        salinity, temperature, pressure, coefficients
    )


def brine_primary_velocity(
    temperature: np.ndarray | float,
    pressure: np.ndarray | float,
    salinity: np.ndarray | float,
) -> np.ndarray | float:
    """
    Primary wave velocity of sodium chloride solutions, equation 29 in Batzle & Wang [1]

    :param salinity: Salinity of solution as weight fraction (ppm/1000000) of
        sodium chloride.
    :param pressure: Pressure (MPa) of oil
    :param temperature: Temperature (Celsius) of oil.
    :return: velocity of solution in m/s.
    """
    coefficients = np.zeros((3, 4, 3))
    coefficients[0, 0, 0] = 1170
    coefficients[0, 1, 0] = -9.6
    coefficients[0, 2, 0] = 0.055
    coefficients[0, 3, 0] = -8.5e-5
    coefficients[0, 0, 1] = 2.6
    coefficients[0, 1, 1] = -29e-4
    coefficients[0, 0, 2] = -0.0476
    coefficients[1, 0, 0] = 780
    coefficients[1, 0, 1] = -10
    coefficients[1, 0, 2] = 0.16
    coefficients[2, 0, 0] = -820

    return water_primary_velocity(temperature, pressure) + salinity * polyval3d(
        sqrt(salinity), temperature, pressure, coefficients
    )


def water_density(
    temperature: np.ndarray | float,
    pressure: np.ndarray | float,
) -> np.ndarray | float:
    """
    Density of water,, equation 27a in Batzle & Wang [1].
    :param pressure: Pressure (MPa) of oil
    :param temperature: Temperature (Celsius) of oil.
    :return: Density of water in g/cc.
    """
    coefficients = [
        [1.0, 489e-6, -333e-9],
        [-8e-5, -2e-6, -2e-09],
        [-33e-7, 16e-9, 0.0],
        [1.75e-9, -13e-12, 0.0],
    ]
    return polyval2d(temperature, pressure, coefficients)


def water_primary_velocity(
    temperature: np.ndarray | float,
    pressure: np.ndarray | float,
) -> np.ndarray | float:
    """
    Primary wave velocity of water, table 1 and equation 28 in Batzle & Wang [1].
    :param pressure: Pressure (MPa) of oil
    :param temperature: Temperature (Celsius) of oil.
    :return: primary wave velocity of water in m/s.
    """
    if np.any(pressure > 100):
        warnings.warn(
            "Calculations for water velocity is not precise for\n"
            + "pressure outside [0,100]MPa"
            + f"pressure given: {pressure}MPa",
            stacklevel=1,
        )
    coefficients = [
        [1402.85, 1.524, 3.437e-3, -1.197e-5],
        [4.871, -1.11e-2, 1.739e-4, -1.628e-6],
        [-4.783e-2, 2.747e-4, -2.135e-6, 1.237e-8],
        [1.487e-4, -6.503e-7, -1.455e-8, 1.327e-10],
        [-2.197e-7, 7.987e-10, 5.23e-11, -4.614e-13],
    ]
    return polyval2d(temperature, pressure, coefficients)


def water(
    temperature: np.ndarray | float, pressure: np.ndarray | float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    :param pressure: Pressure (Pa) of oil
    :param temperature: Temperature (Celsius) of oil.
    :return: water_density [kg/m^3], water_velocity [m/s], water_bulk_modulus [Pa]
    """
    pressure_mpa = pressure * 1.0e-6
    water_den = water_density(temperature, pressure_mpa)
    water_vel = water_primary_velocity(temperature, pressure_mpa)
    water_k = water_vel**2 * water_den * 1000.0
    return water_den, water_vel, water_k
