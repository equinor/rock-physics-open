import warnings

import numpy as np

from .dead_oil_density import dead_oil_density
from .dead_oil_velocity import dead_oil_velocity
from .live_oil_density import live_oil_density
from .live_oil_velocity import live_oil_velocity
from .oil_bubble_point import bp_standing


def oil_properties(
    temperature: np.ndarray | float,
    pressure: np.ndarray | float,
    rho0: np.ndarray | float,
    gas_oil_ratio: np.ndarray | float,
    gas_gravity: np.ndarray | float,
) -> np.ndarray | float:
    """
    :param temperature: Temperature (Celsius) of oil.
    :param pressure: Pressure (Pa) of oil
    :param rho0: Density of the oil without dissolved gas at 15.6 degrees Celsius and
                 atmospheric pressure. (kg/m^3)
    :param gas_oil_ratio: The volume ratio of gas to oil [l/l]
    :param gas_gravity: Gas Gravity, molar mass of gas relative to air molar mas.
    :return: vel_oil, den_oil, k_oil
    """
    # Since live_oil with gas_oil_ratio=0.0 is not equal to dead oil
    # we use an apodization function to interpolate between the two

    def triangular_window(x, length=2):
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

    loil_den, loil_vel = live_oil(
        temperature, pressure, rho0, gas_oil_ratio, gas_gravity
    )
    doil_den, doil_vel = dead_oil(temperature, pressure, rho0)
    window = triangular_window(gas_oil_ratio)
    den_oil = doil_den * window + (1 - window) * loil_den
    vel_oil = doil_vel * window + (1 - window) * loil_vel
    k_oil = vel_oil**2 * den_oil
    return vel_oil, den_oil, k_oil


def dead_oil(
    temperature: np.ndarray | float,
    pressure: np.ndarray | float,
    reference_density: np.ndarray | float,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    """
    :param reference_density: Density of the oil without dissolved gas
        at 15.6 degrees Celsius and atmospheric pressure. kg/m3
    :param gas_oil_ratio: The volume ratio of gas to oil [l/l]
    :param gas_gravity: molar mass of gas relative to air molar mas.
    :param pressure: Pressure (Pa) of oil
    :param temperature: Temperature (Celsius) of oil.
    :return: dead_oil_density, dead_oil_velocity
    """
    dead_oil_den = 1000 * dead_oil_density(
        temperature, pressure * 1e-6, reference_density / 1000
    )
    dead_oil_vel = dead_oil_velocity(
        temperature, pressure * 1e-6, reference_density / 1000
    )
    return dead_oil_den, dead_oil_vel


def live_oil(
    temperature: np.ndarray | float,
    pressure: np.ndarray | float,
    reference_density: np.ndarray | float,
    gas_oil_ratio: np.ndarray | float,
    gas_gravity: np.ndarray | float,
) -> tuple[np.ndarray | float, np.ndarray | float]:
    """
    :param reference_density: Density of the oil without dissolved gas
        at 15.6 degrees Celsius and atmospheric pressure. (kg/m^3)
    :param gas_oil_ratio: The volume ratio of gas to oil [l/l]
    :param gas_gravity: molar mass of gas relative to air molar mas.
    :param pressure: Pressure (Pa) of oil
    :param temperature: Temperature (Celsius) of oil.
    :return: live_oil_density, live_oil_velocity
    """
    if np.any(
        pressure
        < bp_standing(reference_density, gas_oil_ratio, gas_gravity, temperature)
    ):
        warnings.warn(
            "Pressure is below bubble point of oil, estimated elastic properties can be inaccurate",
            stacklevel=1,
        )
    live_oil_den = 1000 * live_oil_density(
        temperature,
        pressure * 1e-6,
        reference_density / 1000,
        gas_oil_ratio,
        gas_gravity,
    )
    live_oil_vel = live_oil_velocity(
        temperature,
        pressure * 1e-6,
        reference_density / 1000,
        gas_oil_ratio,
        gas_gravity,
    )
    return live_oil_den, live_oil_vel
