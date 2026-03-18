import numpy as np
import numpy.typing as npt

from rock_physics_open.equinor_utilities.units import kg_m3_to_g_cc, pa_to_mpa


def dead_oil_velocity(
    temperature: npt.NDArray[np.float64],
    pressure: npt.NDArray[np.float64],
    reference_density: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    The primary wave velocity in oil without dissolved gas (dead).

    Uses equation 20a from Batzle & Wang [1].

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
    primary velocity of dead oil in m/s.
    """
    pressure_mpa = pa_to_mpa(pressure)
    density_gcc = kg_m3_to_g_cc(reference_density)
    return (
        2096 * np.sqrt(density_gcc / (2.6 - density_gcc))
        - 3.7 * temperature
        + 4.64 * pressure_mpa
        + 0.0115
        * (4.12 * np.sqrt(1.08 * density_gcc**-1 - 1) - 1)
        * temperature
        * pressure_mpa
    )
