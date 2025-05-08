import numpy as np


def live_oil_density(
    temperature,
    pressure: np.ndarray | float | None,
    reference_density: np.ndarray | float,
    gas_oil_ratio: np.ndarray | float,
    gas_gravity: np.ndarray | float,
) -> np.ndarray | float:
    """
    Density of live oil at saturation.

    Equation 24 in Batzle & Wang [1].

    :param reference_density: Density of the oil without dissolved gas
        at 15.6 degrees Celsius and atmospheric pressure. (g/cc)
    :param pressure: Pressure (MPa) of oil (for future implementation only)
    :param gas_oil_ratio: The volume ratio of gas to oil [l/l]
    :param temperature: Temperature (Celsius) of oil.
    :param gas_gravity: molar mass of gas relative to air molar mas.
    :return: Density of live oil [g/cc].
    """
    b0 = live_oil_volume_factor(
        temperature, reference_density, gas_oil_ratio, gas_gravity
    )
    return (reference_density + 0.0012 * gas_gravity * gas_oil_ratio) / b0


def live_oil_pseudo_density(
    temperature: np.ndarray | float,
    reference_density: np.ndarray | float,
    gas_oil_ratio: np.ndarray | float,
    gas_gravity: np.ndarray | float,
) -> np.ndarray | float:
    """
    Pseudo density used to substitute reference density in dead_oil_wave_velocity
    for live oils.

    Equation 22 in Batzle & Wang [1].

    :param reference_density: Density of the oil without dissolved gas
        at 15.6 degrees Celsius and atmospheric pressure. (g/cc)
    :param gas_oil_ratio: The volume ratio of gas to oil [l/l]
    :param temperature: Temperature (Celsius) of oil.
    :param gas_gravity: molar mass of gas relative to air molar mas.
    :return: Pseudo-density of live oil.
    """
    b0 = live_oil_volume_factor(
        temperature, reference_density, gas_oil_ratio, gas_gravity
    )
    return (reference_density / b0) / (1 + 0.001 * gas_oil_ratio)


def live_oil_volume_factor(
    temperature: np.ndarray | float,
    reference_density: np.ndarray | float,
    gas_oil_ratio: np.ndarray | float,
    gas_gravity: np.ndarray | float,
) -> np.ndarray | float:
    """
    Volume factor derived by Standing (1962), equation 23 in Batzle & Wang [1].
    :param reference_density: Density of the oil without dissolved gas
        at 15.6 degrees Celsius and atmospheric pressure. (g/cc)
    :param gas_oil_ratio: The volume ratio of gas to oil [l/l]
    :param temperature: Temperature (Celsius) of oil.
    :param gas_gravity: molar mass of gas relative to air molar mas.
    :return: A volume factor in calculating pseudo-density of live oil.
    """
    return (
        0.972
        + 0.00038
        * (
            2.4 * gas_oil_ratio * np.sqrt(gas_gravity / reference_density)
            + temperature
            + 17.8
        )
        ** 1.175
    )
