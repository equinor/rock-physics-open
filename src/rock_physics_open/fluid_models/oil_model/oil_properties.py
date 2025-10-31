import warnings

import numpy as np
import numpy.typing as npt

from .dead_oil_density import dead_oil_density
from .dead_oil_velocity import dead_oil_velocity
from .live_oil_density import live_oil_density
from .live_oil_velocity import live_oil_velocity
from .oil_bubble_point import bp_standing


def oil_properties(
    temperature: npt.NDArray[np.float64],
    pressure: npt.NDArray[np.float64],
    rho0: npt.NDArray[np.float64],
    gas_oil_ratio: npt.NDArray[np.float64],
    gas_gravity: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    :param temperature: Temperature [°C] of oil.
    :param pressure: Pressure [Pa] of oil
    :param rho0: Density of the oil without dissolved gas at 15.6 degrees Celsius and
                 atmospheric pressure. [kg/m^3]
    :param gas_oil_ratio: The volume ratio of gas to oil [l/l]
    :param gas_gravity: Gas Gravity, molar mass of gas relative to air molar mas.
    :return: vel_oil [m/s], den_oil [kg/m^3], k_oil [Pa]
    """
    # Since live_oil with gas_oil_ratio=0.0 is not equal to dead oil
    # we use an apodization function to interpolate between the two

    def triangular_window(x: npt.NDArray[np.float64], length: int = 2):
        """
        A triangular window function around the origin, 1.0 at x=0.0, linear
        and 0.0 outside the window.
        :param length: total length of the window, ie., function is nonzero in
            [-length/2, length/2].
        :param x: numpy array containing x'es to evaluate the window at
        :return: value of window function at x.
        """
        x = np.asarray(x)  # Ensure x is a numpy array
        window = np.clip((np.abs(x) - length / 2) / (length / 2), 0, 1)
        return 1 - window

    loil_vel, loil_den = live_oil(
        temperature=temperature,
        pressure=pressure,
        reference_density=rho0,
        gas_oil_ratio=gas_oil_ratio,
        gas_gravity=gas_gravity,
    )
    doil_vel, doil_den = dead_oil(
        temperature=temperature,
        pressure=pressure,
        reference_density=rho0,
    )
    window = triangular_window(gas_oil_ratio)
    den_oil = doil_den * window + (1 - window) * loil_den
    vel_oil = doil_vel * window + (1 - window) * loil_vel
    k_oil = vel_oil**2 * den_oil
    return vel_oil, den_oil, k_oil


def dead_oil(
    temperature: npt.NDArray[np.float64],
    pressure: npt.NDArray[np.float64],
    reference_density: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    :param reference_density: Density of the oil without dissolved gas
        at 15.6 degrees Celsius and atmospheric pressure. [kg/m^3]
    :param gas_oil_ratio: The volume ratio of gas to oil [l/l]
    :param gas_gravity: molar mass of gas relative to air molar mas.
    :param pressure: Pressure [Pa] of oil
    :param temperature: Temperature [°C] of oil.
    :return: dead_oil_density [kg/m^3], dead_oil_velocity [m/s]
    """
    dead_oil_den = dead_oil_density(
        temperature=temperature,
        pressure=pressure,
        reference_density=reference_density,
    )
    dead_oil_vel = dead_oil_velocity(
        temperature=temperature,
        pressure=pressure,
        reference_density=reference_density,
    )
    return dead_oil_vel, dead_oil_den


def live_oil(
    temperature: npt.NDArray[np.float64],
    pressure: npt.NDArray[np.float64],
    reference_density: npt.NDArray[np.float64],
    gas_oil_ratio: npt.NDArray[np.float64],
    gas_gravity: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    :param reference_density: Density of the oil without dissolved gas
        at 15.6 degrees Celsius and atmospheric pressure. [kg/m^3]
    :param gas_oil_ratio: The volume ratio of gas to oil [l/l]
    :param gas_gravity: molar mass of gas relative to air molar mas.
    :param pressure: Pressure [Pa] of oil
    :param temperature: Temperature [°C] of oil.
    :return: live_oil_density , live_oil_velocity
    """
    if np.any(
        pressure
        < bp_standing(reference_density, gas_oil_ratio, gas_gravity, temperature)
    ):
        warnings.warn(
            "Pressure is below bubble point of oil, estimated elastic properties can be inaccurate",
            stacklevel=1,
        )
    live_oil_den = live_oil_density(
        temperature=temperature,
        pressure=pressure,
        reference_density=reference_density,
        gas_oil_ratio=gas_oil_ratio,
        gas_gravity=gas_gravity,
    )
    live_oil_vel = live_oil_velocity(
        temperature=temperature,
        pressure=pressure,
        reference_density=reference_density,
        gas_oil_ratio=gas_oil_ratio,
        gas_gravity=gas_gravity,
    )
    return (
        live_oil_vel,
        live_oil_den,
    )


def oil_viscosity(
    temperature: npt.NDArray[np.float64],
    pressure: npt.NDArray[np.float64],
    reference_density: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Calculate dead oil viscosity. If dissolved gas is present in the oil, the reference density
    should be substituted by live oil density.

    Equations 25a, 25b, 26a & 26b in Batzle and Wang 1992

    Based on Beggs and Robinson 1975

    :param temperature: Temperature [°C] of oil
    :param pressure: Pressure [Pa] of oil
    :param reference_density: Density of the oil without dissolved gas
    """
    # Change unit in pressure to MPa
    pressure_mpa = pressure / 1.0e6
    # Change unit in density to g/cc
    density_gcc = reference_density / 1000.0

    y_factor = 10 ** (5.693 - 2.863 / density_gcc)
    eta_t = -1.0 + 10 ** (0.505 * y_factor * (17.8 + temperature) ** -1.163)
    i_factor = 10 ** (
        18.6 * (0.1 * np.log10(eta_t) + (np.log10(eta_t) + 2) ** -0.1 - 0.985)
    )
    return eta_t + 0.145 * pressure_mpa * i_factor
