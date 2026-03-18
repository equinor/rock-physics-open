from typing import Literal, assert_never

import numpy as np
import numpy.typing as npt

from rock_physics_open.equinor_utilities.units import kg_m3_to_g_cc, pa_to_mpa

from .dead_oil_density import dead_oil_density
from .dead_oil_velocity import dead_oil_velocity
from .han_batzle_oil_model import (
    han_batzle_live_oil_density,
    han_batzle_live_oil_velocity,
)
from .live_oil_density import live_oil_density
from .live_oil_velocity import live_oil_velocity
from .oil_utilities import (
    ArrayLikeFloat,
    as_float_array,
    inputs_are_scalar,
)

ModelVersionType = Literal["BW", "HB"]


def oil_properties(
    temperature: ArrayLikeFloat,
    pressure: ArrayLikeFloat,
    rho0: ArrayLikeFloat,
    gas_oil_ratio: ArrayLikeFloat,
    gas_gravity: ArrayLikeFloat,
    model_version: ModelVersionType = "HB",
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Default parameter setting is now to use the more correct Han & Batzle 2000 model for live oil properties above bubble point.

    Pressure values below bubble point will
    give NaN, as this is a non-physical state for the oil.

    Parameters
    ----------
    temperature
        Temperature [°C] of oil.
    pressure
        Formation pressure [Pa] of oil
    rho0
        Density of the oil without dissolved gas at 15.6 degrees Celsius and atmospheric pressure. [kg/m^3]
    gas_oil_ratio
        The volume ratio of gas to oil [l/l]
    gas_gravity
        Gas Gravity, molar mass of gas relative to air molar mas.
    model_version
        model_version: Model to use for live-oil properties. "BW" selects the Batzle-Wang (1992) model, which returns approximate values across all pressures and issues a warning when used below the bubble-point pressure. "HB" selects the Han-Batzle (2000) model, which includes corrections for pressures above the bubble point and returns NaN for pressures below the bubble-point pressure.

    Returns
    -------
    vel_oil
        Oil velocity [m/s].
    den_oil
        Oil density [kg/m^3].
    k_oil
        Oil bulk modulus [Pa].
    """
    # Since live_oil with gas_oil_ratio=0.0 is not equal to dead oil
    # we use an apodization function to interpolate between the two

    def triangular_window(
        x: npt.NDArray[np.float64], length: int = 2
    ) -> npt.NDArray[np.float64]:
        """
        A triangular window function around the origin, 1.0 at x=0.0, linear and 0.0 outside the window.

        Parameters
        ----------
        x
            numpy array containing x'es to evaluate the window at
        length
            total length of the window, ie., function is nonzero in [-length/2, length/2].

        Returns
        -------
        value of window function at x.
        """
        x = np.asarray(x)  # Ensure x is a numpy array
        window = np.clip((np.abs(x) - length / 2) / (length / 2), 0, 1)
        return 1 - window

    scalar_input = inputs_are_scalar(
        temperature,
        pressure,
        rho0,
        gas_oil_ratio,
        gas_gravity,
    )
    temperature_arr = as_float_array(temperature)
    pressure_arr = as_float_array(pressure)
    rho0_arr = as_float_array(rho0)
    gas_oil_ratio_arr = as_float_array(gas_oil_ratio)
    gas_gravity_arr = as_float_array(gas_gravity)

    loil_vel, loil_den = live_oil(
        temperature=temperature_arr,
        pressure=pressure_arr,
        reference_density=rho0_arr,
        gas_oil_ratio=gas_oil_ratio_arr,
        gas_gravity=gas_gravity_arr,
        model_version=model_version,
    )
    doil_vel, doil_den = dead_oil(
        temperature=temperature_arr,
        pressure=pressure_arr,
        reference_density=rho0_arr,
    )
    window = triangular_window(gas_oil_ratio_arr)
    den_oil = doil_den * window + (1 - window) * loil_den
    vel_oil = doil_vel * window + (1 - window) * loil_vel
    k_oil = vel_oil**2 * den_oil
    if scalar_input:
        return vel_oil[0], den_oil[0], k_oil[0]
    return vel_oil, den_oil, k_oil


def dead_oil(
    temperature: ArrayLikeFloat,
    pressure: ArrayLikeFloat,
    reference_density: ArrayLikeFloat,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Calculate the velocity and density of dead oil.

    Parameters
    ----------
    temperature
        Temperature [°C] of oil.
    pressure
        Formation pressure [Pa] of oil.
    reference_density
        Density of the oil without dissolved gas at 15.6 degrees Celsius and atmospheric pressure. [kg/m^3]

    Returns
    -------
    dead_oil_velocity
        Dead oil velocity [m/s].
    dead_oil_density
        Dead oil density [kg/m^3].
    """
    scalar_input = inputs_are_scalar(temperature, pressure, reference_density)
    temperature_arr = as_float_array(temperature)
    pressure_arr = as_float_array(pressure)
    reference_density_arr = as_float_array(reference_density)

    dead_oil_den = dead_oil_density(
        temperature=temperature_arr,
        pressure=pressure_arr,
        reference_density=reference_density_arr,
    )
    dead_oil_vel = dead_oil_velocity(
        temperature=temperature_arr,
        pressure=pressure_arr,
        reference_density=reference_density_arr,
    )
    if scalar_input:
        return dead_oil_vel[0], dead_oil_den[0]
    return dead_oil_vel, dead_oil_den


def live_oil(
    temperature: ArrayLikeFloat,
    pressure: ArrayLikeFloat,
    reference_density: ArrayLikeFloat,
    gas_oil_ratio: ArrayLikeFloat,
    gas_gravity: ArrayLikeFloat,
    model_version: ModelVersionType = "HB",
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    We introduce the correction to live oil properties above bubble point that is included in the Han and Batzle 2000 paper as a default.

    Parameters
    ----------
    temperature
        Temperature [°C] of oil.
    pressure
        Formation pressure [Pa] of oil
    reference_density
        Density of the oil without dissolved gas at 15.6 degrees Celsius and atmospheric pressure. [kg/m^3]
    gas_oil_ratio
        The volume ratio of gas to oil [l/l]
    gas_gravity
        molar mass of gas relative to air molar mas.
    model_version
        Batzle-Wang 1992 "BW" or Han-Batzle 2000 "HB"

    Returns
    -------
    live_oil_velocity
        Live oil velocity [m/s].
    live_oil_density
        Live oil density [kg/m^3].
    """
    temperature_arr = as_float_array(temperature)
    pressure_arr = as_float_array(pressure)
    reference_density_arr = as_float_array(reference_density)
    gas_oil_ratio_arr = as_float_array(gas_oil_ratio)
    gas_gravity_arr = as_float_array(gas_gravity)

    if model_version == "BW":
        live_oil_vel = live_oil_velocity(
            temperature=temperature_arr,
            pressure=pressure_arr,
            reference_density=reference_density_arr,
            gas_oil_ratio=gas_oil_ratio_arr,
            gas_gravity=gas_gravity_arr,
        )
        live_oil_den = live_oil_density(
            temperature=temperature_arr,
            reference_density=reference_density_arr,
            gas_oil_ratio=gas_oil_ratio_arr,
            gas_gravity=gas_gravity_arr,
        )
    elif model_version == "HB":
        live_oil_vel = han_batzle_live_oil_velocity(
            temperature=temperature_arr,
            pressure=pressure_arr,
            reference_density=reference_density_arr,
            gas_oil_ratio=gas_oil_ratio_arr,
            gas_gravity=gas_gravity_arr,
        )
        live_oil_den = han_batzle_live_oil_density(
            temperature=temperature_arr,
            pressure=pressure_arr,
            reference_density=reference_density_arr,
            gas_oil_ratio=gas_oil_ratio_arr,
            gas_gravity=gas_gravity_arr,
        )
    else:
        assert_never(model_version)

    if inputs_are_scalar(
        temperature,
        pressure,
        reference_density,
        gas_oil_ratio,
        gas_gravity,
    ):
        return live_oil_vel[0], live_oil_den[0]
    return live_oil_vel, live_oil_den


def oil_viscosity(
    temperature: ArrayLikeFloat,
    pressure: ArrayLikeFloat,
    reference_density: ArrayLikeFloat,
) -> npt.NDArray[np.float64]:
    """
    Calculate dead oil viscosity.

    If dissolved gas is present in the oil, the reference density should be substituted by live oil density.

    Equations 25a, 25b, 26a & 26b in Batzle and Wang 1992

    Based on Beggs and Robinson 1975

    Parameters
    ----------
    temperature
        Temperature [°C] of oil
    pressure
        Formation pressure [Pa] of oil
    reference_density
        Density of the oil without dissolved gas

    Returns
    -------
    Viscosity of oil.
    """
    scalar_input = inputs_are_scalar(temperature, pressure, reference_density)
    temperature_arr = as_float_array(temperature)
    pressure_arr = as_float_array(pressure)
    reference_density_arr = as_float_array(reference_density)

    pressure_mpa = pa_to_mpa(pressure_arr)
    density_gcc = kg_m3_to_g_cc(reference_density_arr)

    y_factor = 10 ** (5.693 - 2.863 / density_gcc)
    eta_t = -1.0 + 10 ** (0.505 * y_factor * (17.8 + temperature_arr) ** -1.163)
    i_factor = 10 ** (
        18.6 * (0.1 * np.log10(eta_t) + (np.log10(eta_t) + 2) ** -0.1 - 0.985)
    )
    viscosity = eta_t + 0.145 * pressure_mpa * i_factor
    return viscosity[0] if scalar_input else viscosity
