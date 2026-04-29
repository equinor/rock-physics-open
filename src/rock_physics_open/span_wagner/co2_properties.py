from importlib.resources import as_file, files
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import scipy.optimize
from scipy.interpolate import RegularGridInterpolator

from rock_physics_open.equinor_utilities.units import (
    celsius_to_kelvin,
    mpa_to_pa,
    pa_to_mpa,
)
from rock_physics_open.equinor_utilities.various_utilities.types import Array2D

from .coefficients import (
    a0,
    theta0,
)
from .equations import residual_helmholtz_energy
from .tables.lookup_table import (
    load_lookup_table_interpolator,
)

# Constants
CO2_GAS_CONSTANT = 0.1889241 * 1e3  # J / kg K
CO2_CRITICAL_TEMPERATURE = 304.1282  # K
CO2_CRITICAL_DENSITY = 467.6  # kg / m^3
CO2_CRITICAL_PRESSURE = 7.3773  # MPa
CO2_TRIPLE_TEMPERATURE = 216.592  # K
CO2_TRIPLE_PRESSURE = 0.51795  # MPa


def co2_properties(
    temp: npt.NDArray[np.float64],
    pres: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    CO2 properties are estimated according to Span & Wagner's equation of state model.

    Parameters
    ----------
    temp
        Temperature [°C]
    pres
        Pressure [Pa]

    Returns
    -------
    vel_co2
        CO2 velocity [m/s].
    den_co2
        CO2 density [kg/m^3].
    k_co2
        CO2 bulk modulus [Pa].

    References
    ----------
    R. Span and W. Wagner: A new equation of state for carbon dioxide covering the
    fluid region from the triple point temperature to 1100K at pressures up to
    800 MPa.
    J. Phys. Chem. Ref. Data, Vol. 25, No. 6, 1996, pp 1509 - 1596
    """
    abs_temp = celsius_to_kelvin(temp)
    pres_mpa = pres * 1.0e-6
    den_co2 = carbon_dioxide_density(abs_temp, pres_mpa)
    k_co2 = carbon_dioxide_bulk_modulus(abs_temp, den_co2)
    vel_co2 = (k_co2 / den_co2) ** 0.5

    return vel_co2, den_co2, k_co2


def co2_helmholtz_energy(
    delta: npt.NDArray[np.float64],
    tau: npt.NDArray[np.float64],
    dd: Literal[0, 1, 2],
    dt: Literal[0, 1, 2],
) -> npt.NDArray[np.float64] | float:
    """
    Helmholtz energy as defined by equation 6.1 in Span & Wagner [2].

    Parameters
    ----------
    delta
        Reduced density, unit-less. That is, density / CO2_CRITICAL_DENSITY.
    tau
        Inverse reduced temperature, unit-less. That is, CO2_CRITICAL_TEMPERATURE / (absolute) temperature
    dd
        Degree of derivation wrt. delta. Integer between 0 and 2.
    dt
        Degree of derivation wrt. tau. Integer between 0 and 2, as long as (dt + dd < 3)

    Returns
    -------
    Result.
    """
    return ideal_gas_helmholtz_energy(
        delta=delta,
        tau=tau,
        dd=dd,
        dt=dt,
    ) + co2_residual_helmholtz_energy(
        delta=delta,
        tau=tau,
        dd=dd,
        dt=dt,
    )


def ideal_gas_helmholtz_energy(
    delta: npt.NDArray[np.float64],
    tau: npt.NDArray[np.float64],
    dd: Literal[0, 1, 2],
    dt: Literal[0, 1, 2],
) -> npt.NDArray[np.float64] | float:
    """
    Helmholtz energy from ideal gas behavior as defined by equation 2.3 in Span & Wagner [2].

    See function `co2_helmholtz_energy` for argument documentation.

    Parameters
    ----------
    delta
        Reduced density.
    tau
        Inverse reduced temperature.
    dd
        Degree of derivation wrt. density.
    dt
        Degree of derivation wrt. temperature.

    Returns
    -------
    Result.

    """
    # Adjust array shapes
    tau = np.asarray(tau)
    delta = np.asarray(delta)
    return_scalar = tau.ndim == 0
    tau2 = tau.reshape(-1, 1)  # Needed for array-based sums

    if dt == dd == 0:
        _sum = np.sum(a0[0, 3:] * np.log(1 - np.exp(-theta0 * tau2)), axis=-1)
        result = (
            np.log(delta) + a0[0, 0] + a0[0, 1] * tau + a0[0, 2] * np.log(tau) + _sum
        )
    elif dt == 1 and dd == 0:
        _sum = np.sum(
            a0[0, 3:] * theta0 * ((1 - np.exp(-theta0 * tau2)) ** -1 - 1), axis=-1
        )
        result = a0[0, 1] + a0[0, 2] / tau + _sum
    elif dt == 0 and dd == 1:
        return 1 / delta
    elif dt == 2 and dd == 0:
        _sum = np.sum(
            a0[0, 3:]
            * theta0**2
            * np.exp(-theta0 * tau2)
            * (1 - np.exp(-theta0 * tau2)) ** -2,
            axis=-1,
        )
        result = -a0[0, 2] / tau**2 - _sum
    elif dt == 0 and dd == 2:
        return -1 / delta**2
    elif dt == 1 and dd == 1:
        return 0
    else:
        raise ValueError
    if return_scalar:
        return result[0]
    return result


def co2_residual_helmholtz_energy(
    delta: npt.NDArray[np.float64],
    tau: npt.NDArray[np.float64],
    dd: Literal[0, 1, 2],
    dt: Literal[0, 1, 2],
) -> npt.NDArray[np.float64] | float:
    """
    Residual part of Helmholtz energy as defined by the equation in Table 32 of Span & Wagner [2].

    See `co2_helmholtz_energy` for argument documentation.

    Parameters
    ----------
    delta
        Reduced density.
    tau
        Inverse reduced temperature.
    dd
        Degree of derivation wrt. density.
    dt
        Degree of derivation wrt. temperature.

    Returns
    -------
    Result.
    """
    tau = np.asarray(tau)
    delta = np.asarray(delta)
    return_scalar = (tau.ndim == 0) & (delta.ndim == 0)
    tau = tau.reshape(-1, 1)
    delta = delta.reshape(-1, 1)
    # tau == 1.0 or delta == 1.0 leads to numerically invalid results. The values are nudged to avoid nan output.
    tau[tau == 1.0] -= 1e-15
    delta[delta == 1.0] -= 1e-15

    res = residual_helmholtz_energy(
        delta_=delta,
        tau_=tau,
        diff_delta=dd,
        diff_tau=dt,
    )

    if return_scalar:
        return res[0]
    return res


def carbon_dioxide_pressure(
    absolute_temperature: npt.NDArray[np.float64],
    density: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    CO2 pressure as given by Table 3 of Span & Wagner [2].

    Parameters
    ----------
    absolute_temperature
        Temperature in K.
    density
        CO2 density (kg / m^3).

    Returns
    -------
    CO2 pressure [MPa]
    """
    tau = CO2_CRITICAL_TEMPERATURE / absolute_temperature
    delta = density / CO2_CRITICAL_DENSITY
    return pa_to_mpa(
        density
        * CO2_GAS_CONSTANT
        * absolute_temperature
        * (
            1
            + delta
            * co2_residual_helmholtz_energy(
                delta=delta,
                tau=tau,
                dd=1,
                dt=0,
            )
        )
    )


def carbon_dioxide_pressure_isentropic(
    absolute_temperature: npt.NDArray[np.float64],
    density: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    CO2 pressure as given by Table 3 of Span & Wagner [2]. Relevant primarily for isentropic bulk modulus calculations.

    Parameters
    ----------
    absolute_temperature
        Temperature in K.
    density
        CO2 density (kg / m^3).

    Returns
    -------
    CO2 pressure [MPa]
    """
    tau = CO2_CRITICAL_TEMPERATURE / absolute_temperature
    delta = density / CO2_CRITICAL_DENSITY

    first = (
        2
        * delta
        * co2_residual_helmholtz_energy(
            delta=delta,
            tau=tau,
            dd=1,
            dt=0,
        )
    )
    second = delta**2 * co2_residual_helmholtz_energy(
        delta=delta,
        tau=tau,
        dd=2,
        dt=0,
    )

    # See Table 3 of Span & Wagner (speed of sound)
    nom = (
        1
        + delta
        * co2_residual_helmholtz_energy(
            delta=delta,
            tau=tau,
            dd=1,
            dt=0,
        )
        - delta
        * tau
        * co2_residual_helmholtz_energy(
            delta=delta,
            tau=tau,
            dd=1,
            dt=1,
        )
    ) ** 2
    den = tau**2 * (
        co2_helmholtz_energy(
            delta=delta,
            tau=tau,
            dd=0,
            dt=2,
        )
    )
    third = -nom / den
    return pa_to_mpa(
        absolute_temperature * CO2_GAS_CONSTANT * (1 + first + second + third)
    )


def saturated_liquid_density(
    absolute_temperature: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Saturated liquid density as defined by equation 3.14 of Span & Wagner [2].

    Parameters
    ----------
    absolute_temperature
        Absolute temperature in K. Should satisfy: CO2_TRIPLE_TEMPERATURE < absolute_temperature < CO2_CRITICAL_TEMPERATURE

    Returns
    -------
    Result.
    """
    _a1 = 1.9245108
    _a2 = -0.62385555
    _a3 = -0.32731127
    _a4 = 0.39245142
    _t = 1 - absolute_temperature / CO2_CRITICAL_TEMPERATURE
    inner = _a1 * _t**0.34 + _a2 * _t**0.5 + _a3 * _t ** (10 / 6) + _a4 * _t ** (11 / 6)
    return CO2_CRITICAL_DENSITY * np.exp(inner)


def saturated_vapor_density(
    absolute_temperature: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Saturated vapor density as defined by equation 3.15 of Span & Wagner.

    Parameters
    ----------
    absolute_temperature
        Absolute temperature in K. Should satisfy: CO2_TRIPLE_TEMPERATURE < absolute_temperature < CO2_CRITICAL_TEMPERATURE

    Returns
    -------
    Result.
    """
    # Assert temp < critical
    _a1 = -1.7074879
    _a2 = -0.82274670
    _a3 = -4.6008549
    _a4 = -10.111178
    _a5 = -29.742252
    _t = 1 - absolute_temperature / CO2_CRITICAL_TEMPERATURE
    inner = (
        _a1 * _t**0.34
        + _a2 * _t**0.5
        + _a3 * _t
        + _a4 * _t ** (7 / 3)
        + _a5 * _t ** (14 / 3)
    )
    return CO2_CRITICAL_DENSITY * np.exp(inner)


def sublimation_pressure(
    absolute_temperature: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Sublimation pressure as defined by equation 3.12 of Span & Wagner [2].

    Parameters
    ----------
    absolute_temperature
        Absolute temperature in K. Should satisfy absolute_temperature < CO2_TRIPLE_TEMPERATURE

    Returns
    -------
    Result.
    """
    _a1 = -14.740846
    _a2 = 2.4327015
    _a3 = -5.3061778
    _t = 1 - absolute_temperature / CO2_TRIPLE_TEMPERATURE
    inner = _a1 * _t + _a2 * _t**1.9 + _a3 * _t**2.9
    return CO2_TRIPLE_PRESSURE * np.exp(inner / (1 - _t))


def vapor_pressure(
    absolute_temperature: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Vapor pressure as defined by equation 3.13 of Span & Wagner [2].

    Parameters
    ----------
    absolute_temperature
        Absolute temperature in K. Should satisfy: CO2_TRIPLE_TEMPERATURE < absolute_temperature < CO2_CRITICAL_TEMPERATURE

    Returns
    -------
    Result.
    """
    _a1 = -7.0602087
    _a2 = 1.9391218
    _a3 = -1.6463597
    _a4 = -3.2995634
    _t = 1 - absolute_temperature / CO2_CRITICAL_TEMPERATURE
    inner = _a1 * _t**1.0 + _a2 * _t**1.5 + _a3 * _t**2.0 + _a4 * _t**4.0
    return CO2_CRITICAL_PRESSURE * np.exp(inner / (1 - _t))


def melting_pressure(
    absolute_temperature: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Melting pressure as defined by equation 3.10 of Span & Wagner [2].

    Parameters
    ----------
    absolute_temperature
        Absolute temperature in K. Should satisfy CO2_TRIPLE_TEMPERATURE < absolute_temperature

    Returns
    -------
    Result.
    """
    _a1 = 1955.5390
    _a2 = 2055.4593
    _t = absolute_temperature / CO2_TRIPLE_TEMPERATURE - 1
    return CO2_TRIPLE_PRESSURE * (1 + _a1 * _t + _a2 * _t**2)


def _determine_density_bounds(
    absolute_temperature: npt.NDArray[np.float64],
    pressure: npt.NDArray[np.float64],
    force_vapor: bool | Literal["auto"],
) -> Array2D[np.float64]:
    """Calculate the upper and lower bound on density."""
    bounds = np.zeros((absolute_temperature.size, 2))
    bounds[:, 0] = 0.1
    bounds[:, 1] = 1500.0

    below_triple = absolute_temperature < CO2_TRIPLE_TEMPERATURE
    below_critical = ~below_triple & (absolute_temperature < CO2_CRITICAL_TEMPERATURE)

    bounds[below_triple, 1] = saturated_vapor_density(
        absolute_temperature[below_triple]
    )
    if force_vapor is True:
        bounds[below_critical, 1] = saturated_vapor_density(
            absolute_temperature[below_critical]
        )
    elif force_vapor is False:
        bounds[below_critical, 0] = saturated_liquid_density(
            absolute_temperature[below_critical]
        )
    else:  # force_vapor == 'auto'
        below_vapor_pressure = np.zeros(absolute_temperature.shape, dtype=bool)
        below_vapor_pressure[below_critical] = pressure[
            below_critical
        ] < vapor_pressure(absolute_temperature[below_critical])
        is_vapor = below_critical & below_vapor_pressure
        bounds[is_vapor, 1] = saturated_vapor_density(
            absolute_temperature=absolute_temperature[is_vapor]
        )
        is_liquid = below_critical & ~below_vapor_pressure
        bounds[is_liquid, 0] = saturated_liquid_density(absolute_temperature[is_liquid])
    return bounds


def _find_initial_density_values(
    bounds: Array2D[np.float64],
    absolute_temperature: npt.NDArray[np.float64],
    pressure: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Finds approximate density values for the provided temperature(s) and pressure(s).

    The result is only intended to be used by `array_carbon_dioxide_density`.

    Parameters
    ----------
    bounds
        Density bounds.
    absolute_temperature
        Absolute temperature (K).
    pressure
        Pressure (MPa).

    Returns
    -------
    Result.
    """
    temps = np.geomspace(
        np.min(absolute_temperature) * 0.99, np.max(absolute_temperature) * 1.01, 41
    )
    press = np.geomspace(
        np.min(pressure) * 0.99, np.max(absolute_temperature) * 1.01, 41
    )
    tt, pp = np.meshgrid(temps, press, indexing="ij")
    densi = carbon_dioxide_density(
        absolute_temperature=tt.flatten(),
        pressure=pp.flatten(),
        force_vapor="auto",
        raise_error=False,
    ).reshape(temps.size, press.size)
    points: Array2D[np.float64] = np.array((temps, press))
    rgi = RegularGridInterpolator(points, densi, method="linear")
    iv = rgi(np.array((absolute_temperature, pressure)).T)
    oob = (iv < bounds[:, 0]) | (iv > bounds[:, 1]) | np.isnan(iv)
    iv[oob] = np.mean(bounds[oob], axis=1)
    return iv


def array_carbon_dioxide_density(
    absolute_temperature: npt.NDArray[np.float64],
    pressure: npt.NDArray[np.float64],
    force_vapor: bool,
) -> npt.NDArray[np.float64]:
    """
    Alternative implementation of a vectorized carbon dioxide density function.

    Implemented primarily for demonstration purposes. For large arrays, a look-up-table approach should be preferred.

    Utilizes scipy.optimize.newton, which is the only root-finding method of scipy that supports a vectorized functions.

    For argument documentation, see carbon_dioxide_density.

    Parameters
    ----------
    absolute_temperature
        Absolute temperature (K).
    pressure
        Pressure (MPa).
    force_vapor
        Whether to force vapor phase.

    Returns
    -------
    Result.
    """
    absolute_temperature = np.asarray(absolute_temperature)
    pressure = np.asarray(pressure)
    bounds = _determine_density_bounds(absolute_temperature, pressure, force_vapor)
    iv = _find_initial_density_values(bounds, absolute_temperature, pressure)
    opt_root = scipy.optimize.newton(
        lambda x: carbon_dioxide_pressure(absolute_temperature, x) - pressure,
        x0=iv,
        maxiter=10,
    )
    # scipy.optimize.newton may not always converge. We need to determine which of the elements of the solution are
    #  invalid. The opt.converged variable does not seem to suffice, so we perform separate checks. First, check that
    #  the solution is a valid root
    invalid = ~np.isclose(
        carbon_dioxide_pressure(
            absolute_temperature=absolute_temperature,
            density=opt_root,
        ),
        pressure,
        atol=1e-5,
        rtol=0.0,
    )

    # Next, check if the solution is anywhere out of bounds (since newton does not support brackets), and check for nan
    # values.
    invalid |= (
        (opt_root < bounds[:, 0]) | (opt_root > bounds[:, 1]) | np.isnan(opt_root)
    )

    # Finally, use the robust density method to determine the invalid results
    sol = opt_root
    sol[invalid] = carbon_dioxide_density(
        absolute_temperature[invalid], pressure[invalid], force_vapor=force_vapor
    )
    return sol


def _calculate_carbon_dioxide_density(
    absolute_temperature: npt.NDArray[np.float64],
    pressure: npt.NDArray[np.float64],
    force_vapor: bool | Literal["auto"] = "auto",
    raise_error: bool = True,
) -> npt.NDArray[np.float64]:
    """
    Density of carbon dioxide.

    Found solving the Pressure equation of Table 3 in Span & Wagner [2] numerically for density using a vectorized bisection method.

    To ensure a single solution, the phase of the liquid must first be
    determined.

    Parameters
    ----------
    absolute_temperature
        Absolute temperature (K).
    pressure
        Pressure (MPa).
    force_vapor
        If 'auto', the phase of the fluid is automatically determined. However, along the vaporization line (assuming T_triple < absolute_temperature < T_critical), the fluid is in two-phase equilibrium and the phase cannot be uniquely determined. If force_vapor is set to True, vapor phase is always selected, if False, liquid phase is selected. Outside the temperature bounds, this argument has no effect. This argument should only be used close to the vaporization boundary, otherwise the behavior might not be as expected.
    raise_error
        If True, raises an error if density cannot be determined. Otherwise, returns np.nan.

    Returns
    -------
    Density (kg / m^3)

    """
    absolute_temperature = np.atleast_1d(np.asarray(absolute_temperature, dtype=float))
    pressure = np.atleast_1d(np.asarray(pressure, dtype=float))

    bounds = _determine_density_bounds(absolute_temperature, pressure, force_vapor)
    lo = bounds[:, 0] * 0.95
    hi = bounds[:, 1] * 1.05

    # Vectorized bisection: 200 iterations is far more than needed (bisection
    # halves the interval each step, so 200 iterations gives ~10^-60 relative
    # precision), but the loop breaks early once all elements converge to
    # within 1e-12 relative tolerance. Typically ~50 iterations suffice.
    valid = np.ones(lo.shape, dtype=bool)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f_mid = carbon_dioxide_pressure(absolute_temperature, mid) - pressure
        mask = f_mid[valid] > 0
        lo_valid = lo[valid].copy()
        hi_valid = hi[valid].copy()
        mid_valid = mid[valid]
        hi_valid[mask] = mid_valid[mask]
        lo_valid[~mask] = mid_valid[~mask]
        lo[valid] = lo_valid
        hi[valid] = hi_valid
        converged = (hi - lo) < 1e-12 * np.abs(mid)
        valid &= ~converged
        if not np.any(valid):
            break

    result = 0.5 * (lo + hi)

    f_result = carbon_dioxide_pressure(absolute_temperature, result) - pressure
    failed = np.abs(f_result) > 1e-5
    if np.any(failed):
        n_failed = int(np.sum(failed))
        if raise_error:
            msg = f"Density calculation did not converge for {n_failed} element(s)."
            raise ValueError(msg)
        result[failed] = np.nan

    return result


def carbon_dioxide_density(
    absolute_temperature: npt.NDArray[np.float64],
    pressure: npt.NDArray[np.float64],
    interpolate: bool = False,
    **kwargs: Any,
) -> npt.NDArray[np.float64]:
    """
    Density of carbon dioxide. Found either by direct calculation or interpolation.

    Any additional arguments are passed to `_calculate_carbon_dioxide_density`.

    Parameters
    ----------
    absolute_temperature
        Absolute temperature (K).
    pressure
        Pressure (MPa).
    interpolate
        Flag whether to interpolate data or not. If not, data is calculated directly. This is more accurate, but also more time-consuming. Data outside the bounds of the interpolator will be set to np.nan.
    **kwargs

    Returns
    -------
    Density (kg / m^3)
    """
    if interpolate is False:
        return _calculate_carbon_dioxide_density(
            absolute_temperature=absolute_temperature,
            pressure=pressure,
            **kwargs,
        )
    assert interpolate is True

    ref = files("rock_physics_open.span_wagner.tables") / "carbon_dioxide_density.npz"
    with as_file(ref) as fp:
        interpolator = load_lookup_table_interpolator(fp)

    return interpolator(absolute_temperature, pressure)


def carbon_dioxide_bulk_modulus(
    absolute_temperature: npt.NDArray[np.float64],
    density: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Isentropic bulk modulus, derived from the expression for speed of sound in Table 3 of Span & Wagner."""
    d_pressure = carbon_dioxide_pressure_isentropic(
        absolute_temperature=absolute_temperature,
        density=density,
    )
    return mpa_to_pa(density * d_pressure)
