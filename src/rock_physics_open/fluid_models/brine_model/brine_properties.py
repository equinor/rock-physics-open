import warnings
from typing import Any

import numpy as np
import numpy.typing as npt
from numpy.polynomial.polynomial import polyval2d, polyval3d


def brine_properties(
    temperature: npt.NDArray[np.float64],
    pressure: npt.NDArray[np.float64],
    salinity: npt.NDArray[np.float64],
    p_nacl: npt.NDArray[np.float64] | float | None = None,  # pyright: ignore[reportUnusedParameter]
    p_kcl: npt.NDArray[np.float64] | float | None = None,  # pyright: ignore[reportUnusedParameter]
    p_cacl: npt.NDArray[np.float64] | float | None = None,  # pyright: ignore[reportUnusedParameter]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    :param salinity: Salinity of solution as [ppm] of NaCl.
    :param pressure: Pressure [Pa]
    :param temperature: Temperature [°C]
    :param p_nacl: NaCl percentage, for future use
    :param p_kcl: KCl percentage, for future use
    :param p_cacl: CaCl percentage, for future use
    :return: Brine velocity vel_b [m/s], brine density den_b [kg/m^3], brine bulk modulus k_b [Pa]
    """
    vel_b = brine_primary_velocity(
        temperature=temperature,
        pressure=pressure,
        salinity=salinity,
    )
    den_b = brine_density(
        temperature=temperature,
        pressure=pressure,
        salinity=salinity,
    )
    k_b = vel_b**2 * den_b
    return vel_b, den_b, k_b


def brine_density(
    temperature: npt.NDArray[np.float64],
    pressure: npt.NDArray[np.float64],
    salinity: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    density of sodium chloride solutions, equation 27 in Batzle & Wang [1].
    :param salinity: Salinity of solution in ppm
    :param pressure: Pressure [Pa]
    :param temperature: Temperature [°C]
    :return: density of solution in [kg/m^3].
    """
    # Change unit of pressure to MPa
    pressure_mpa = pressure / 1.0e6
    # Change unit of salinity to fraction
    salinity_frac = salinity / 1.0e6

    coefficients = [
        [[0.668, 3e-4], [8e-5, -13e-6], [3e-6, 0.0]],
        [[0.44, -24e-4], [-33e-4, 47e-6], [0.0, 0.0]],
    ]
    water_den = water_density(temperature, pressure)
    brine_correction = (
        salinity_frac
        * polyval3d(salinity_frac, temperature, pressure_mpa, coefficients)
        * 1000.0
    )
    return water_den + brine_correction


def brine_primary_velocity(
    temperature: npt.NDArray[np.float64],
    pressure: npt.NDArray[np.float64],
    salinity: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Primary wave velocity of sodium chloride solutions, equation 29 in Batzle & Wang [1]

    :param salinity: Salinity of solution as [ppm] of sodium chloride
    :param pressure: Pressure [Pa]
    :param temperature: Temperature [°C]
    :return: velocity of solution in m/s.
    """
    # Change unit for salinity from ppm to fraction
    salinity_frac = salinity / 1.0e6
    # Change the unit for pressure from Pa to MPa
    pressure_mpa = pressure / 1.0e6

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

    return water_primary_velocity(temperature, pressure) + salinity_frac * polyval3d(
        np.sqrt(salinity_frac),
        temperature,
        pressure_mpa,
        coefficients,
    )


def water_density(
    temperature: npt.NDArray[np.float64],
    pressure: npt.NDArray[np.float64],
) -> npt.NDArray[Any]:
    """
    Density of water,, equation 27a in Batzle & Wang [1].
    :param pressure: Pressure [Pa]
    :param temperature: Temperature [°C]
    :return: Density of water in [kg/m^3].
    """
    # Change unit of pressure from Pa to MPa
    pressure_mpa = pressure / 1.0e6

    coefficients = [
        [1.0, 489e-6, -333e-9],
        [-8e-5, -2e-6, -2e-09],
        [-33e-7, 16e-9, 0.0],
        [1.75e-9, -13e-12, 0.0],
    ]
    return polyval2d(temperature, pressure_mpa, coefficients) * 1000.0


def water_primary_velocity(
    temperature: npt.NDArray[np.float64],
    pressure: npt.NDArray[np.float64],
) -> npt.NDArray[Any]:
    """
    Primary wave velocity of water, table 1 and equation 28 in Batzle & Wang [1].
    :param pressure: Pressure [Pa]
    :param temperature: Temperature [°C]
    :return: primary wave velocity of water in m/s.
    """
    # Change unit of pressure from Pa to MPa
    pressure_mpa = pressure / 1.0e6

    if np.any(pressure_mpa > 100):
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
    return polyval2d(temperature, pressure_mpa, coefficients)


def water(
    temperature: npt.NDArray[np.float64],
    pressure: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    :param pressure: Pressure [Pa]
    :param temperature: Temperature [°C]
    :return: water_velocity [m/s], water_density [kg/m^3], water_bulk_modulus [Pa]
    """
    water_den = water_density(temperature, pressure)
    water_vel = water_primary_velocity(temperature, pressure)
    water_k = water_vel**2 * water_den
    return water_vel, water_den, water_k


def brine_viscosity(
    temperature: npt.NDArray[np.float64],
    salinity: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Brine viscosity according to Batzle & Wang [1].

    Based on equation 32.
    """
    salinity_frac = salinity / 1.0e6
    return (
        0.1
        + 0.333 * salinity_frac
        + (1.65 + 91.9 * salinity_frac**3)
        * np.exp(-(0.42 * (salinity_frac**0.8 - 0.17) ** 2 + 0.045) * temperature**0.8)
    )
