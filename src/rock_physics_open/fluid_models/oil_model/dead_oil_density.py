import numpy as np
import numpy.typing as npt

from rock_physics_open.equinor_utilities.units import (
    g_cc_to_kg_m3,
    kg_m3_to_g_cc,
    pa_to_mpa,
)


def pressure_adjusted_dead_oil_density(
    pressure: npt.NDArray[np.float64],
    reference_density: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Adjusts density of a dead oil (without dissolved gas) to a given pressure.

    Uses equation 18 from Batzle & Wang [1].

    Parameters
    ----------
    pressure
        Formation pressure [Pa] to adjust to.
    reference_density
        The density [kg/m^3] of the dead oil at 15.6 degrees Celsius and atmospheric pressure.

    Returns
    -------
    Density of oil at given pressure and 21 degrees Celsius (~70 degrees Fahrenheit). [kg/m^3]
    """
    pressure_mpa = pa_to_mpa(pressure)
    density_gcc = kg_m3_to_g_cc(reference_density)
    return g_cc_to_kg_m3(
        density_gcc
        + (0.00277 * pressure_mpa - 1.71e-7 * pressure_mpa**3)
        * (density_gcc - 1.15) ** 2
        + 3.49e-4 * pressure_mpa
    )


def temperature_adjusted_dead_oil_density(
    temperature: npt.NDArray[np.float64],
    density_at_21c: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Adjusts density of a dead oil (without dissolved gas) to a given temperature.

    Uses equation 19 from Batzle & Wang [1].

    Parameters
    ----------
    temperature
        Temperature [°C] of oil.
    density_at_21c
        The density [kg/m^3] of the dead oil at 21 degrees Celsius

    Returns
    -------
    Density of oil at given temperature. [kg/m^3]
    """
    density_at_21c_gcc = kg_m3_to_g_cc(density_at_21c)
    return g_cc_to_kg_m3(
        density_at_21c_gcc / (0.972 + 3.81e-4 * (temperature + 17.78) ** 1.175)
    )


def dead_oil_density(
    temperature: npt.NDArray[np.float64],
    pressure: npt.NDArray[np.float64],
    reference_density: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    The density of oil without dissolved gas (dead).

    Uses equation 18 & 19 from Batzle & Wang [1].

    Parameters
    ----------
    temperature
        Temperature [°C] of oil.
    pressure
        Formation pressure [Pa] of oil
    reference_density
        Density of oil at 15.6 degrees Celsius and atmospheric pressure [kg/m^3]

    Returns
    -------
    density of dead oil at given conditions (kg/m^3).
    """
    density_p = pressure_adjusted_dead_oil_density(
        pressure=pressure,
        reference_density=reference_density,
    )
    return temperature_adjusted_dead_oil_density(
        temperature=temperature,
        density_at_21c=density_p,
    )
