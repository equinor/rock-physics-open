"""
Package for calculating the residual helmholtz energy for CO2.

Pre-computed numpy implementations of the Span & Wagner [2] equations and their
derivatives (up to 2nd order with respect to tau and delta). These were originally
generated via sympy symbolic differentiation and lambdify, and are now hardcoded
as pure numpy expressions to avoid the sympy runtime dependency.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from .coefficients import (
    A4,
    B4,
    C4,
    D4,
    a4,
    alpha3,
    b4,
    beta3,
    beta4,
    c2,
    d1,
    d2,
    d3,
    epsilon3,
    gamma3,
    n1,
    n2,
    n3,
    n4,
    t1,
    t2,
    t3,
)


def _dirac_delta(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Discrete Dirac delta: returns 1 where x == 0, else 0."""
    return (x == 0).astype(np.float64)


# ---------------------------------------------------------------------------
# S1: n1 * delta^d1 * tau^t1
# ---------------------------------------------------------------------------


def _s1_dd0_dt0(
    tau: npt.NDArray[np.float64], delta: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    return delta**d1 * n1 * tau**t1


def _s1_dd0_dt1(
    tau: npt.NDArray[np.float64], delta: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    return delta**d1 * n1 * t1 * tau ** (t1 - 1)


def _s1_dd0_dt2(
    tau: npt.NDArray[np.float64], delta: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    return delta**d1 * n1 * t1 * tau ** (t1 - 2) * (t1 - 1)


def _s1_dd1_dt0(
    tau: npt.NDArray[np.float64], delta: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    return d1 * delta ** (d1 - 1) * n1 * tau**t1


def _s1_dd1_dt1(
    tau: npt.NDArray[np.float64], delta: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    return d1 * delta ** (d1 - 1) * n1 * t1 * tau ** (t1 - 1)


def _s1_dd2_dt0(
    tau: npt.NDArray[np.float64], delta: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    return d1 * delta ** (d1 - 2) * n1 * tau**t1 * (d1 - 1)


# ---------------------------------------------------------------------------
# S2: n2 * delta^d2 * tau^t2 * exp(-delta^c2)
# ---------------------------------------------------------------------------


def _s2_dd0_dt0(
    tau: npt.NDArray[np.float64], delta: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    return delta**d2 * n2 * tau**t2 * np.exp(-(delta**c2))


def _s2_dd0_dt1(
    tau: npt.NDArray[np.float64], delta: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    return delta**d2 * n2 * t2 * tau ** (t2 - 1) * np.exp(-(delta**c2))


def _s2_dd0_dt2(
    tau: npt.NDArray[np.float64], delta: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    return delta**d2 * n2 * t2 * tau ** (t2 - 2) * (t2 - 1) * np.exp(-(delta**c2))


def _s2_dd1_dt0(
    tau: npt.NDArray[np.float64], delta: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    return -c2 * delta ** (c2 + d2 - 1) * n2 * tau**t2 * np.exp(
        -(delta**c2)
    ) + d2 * delta ** (d2 - 1) * n2 * tau**t2 * np.exp(-(delta**c2))


def _s2_dd1_dt1(
    tau: npt.NDArray[np.float64], delta: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    return (
        delta ** (d2 - 1)
        * n2
        * t2
        * tau ** (t2 - 1)
        * (-c2 * delta**c2 + d2)
        * np.exp(-(delta**c2))
    )


def _s2_dd2_dt0(
    tau: npt.NDArray[np.float64], delta: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    return (
        delta ** (d2 - 2)
        * n2
        * tau**t2
        * (
            -2 * c2 * d2 * delta**c2
            + c2 * delta**c2 * (c2 * delta**c2 - c2 + 1)
            + d2 * (d2 - 1)
        )
        * np.exp(-(delta**c2))
    )


# ---------------------------------------------------------------------------
# S3: n3 * delta^d3 * tau^t3 * exp(-alpha3*(delta-eps3)^2 - beta3*(tau-gamma3)^2)
# ---------------------------------------------------------------------------


def _s3_exp(
    delta: npt.NDArray[np.float64], tau: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Common exponential factor for s3 terms."""
    return np.exp(-alpha3 * (delta - epsilon3) ** 2 - beta3 * (-gamma3 + tau) ** 2)


def _s3_dd0_dt0(
    tau: npt.NDArray[np.float64], delta: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    return delta**d3 * n3 * tau**t3 * _s3_exp(delta, tau)


def _s3_dd0_dt1(
    tau: npt.NDArray[np.float64], delta: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    e = _s3_exp(delta, tau)
    return (
        -beta3 * delta**d3 * n3 * tau**t3 * (-2 * gamma3 + 2 * tau) * e
        + delta**d3 * n3 * t3 * tau ** (t3 - 1) * e
    )


def _s3_dd0_dt2(
    tau: npt.NDArray[np.float64], delta: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    return (
        delta**d3
        * n3
        * tau**t3
        * (
            4 * beta3 * t3 * (gamma3 - tau) / tau
            + 2 * beta3 * (2 * beta3 * (gamma3 - tau) ** 2 - 1)
            + t3 * (t3 - 1) / tau**2
        )
        * _s3_exp(delta, tau)
    )


def _s3_dd1_dt0(
    tau: npt.NDArray[np.float64], delta: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    e = _s3_exp(delta, tau)
    return (
        -alpha3 * delta**d3 * n3 * tau**t3 * (2 * delta - 2 * epsilon3) * e
        + d3 * delta ** (d3 - 1) * n3 * tau**t3 * e
    )


def _s3_dd1_dt1(
    tau: npt.NDArray[np.float64], delta: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    return (
        delta**d3
        * n3
        * tau**t3
        * (
            -4 * alpha3 * beta3 * (delta - epsilon3) * (gamma3 - tau)
            - 2 * alpha3 * t3 * (delta - epsilon3) / tau
            + 2 * beta3 * d3 * (gamma3 - tau) / delta
            + d3 * t3 / (delta * tau)
        )
        * _s3_exp(delta, tau)
    )


def _s3_dd2_dt0(
    tau: npt.NDArray[np.float64], delta: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    return (
        delta**d3
        * n3
        * tau**t3
        * (
            -4 * alpha3 * d3 * (delta - epsilon3) / delta
            + 2 * alpha3 * (2 * alpha3 * (delta - epsilon3) ** 2 - 1)
            + d3 * (d3 - 1) / delta**2
        )
        * _s3_exp(delta, tau)
    )


# ---------------------------------------------------------------------------
# S4: n4 * bigdelta^b4 * delta * exp(-C4*(delta-1)^2 - D4*(tau-1)^2)
# where bigdelta = theta^2 + B4*|delta-1|^(2*a4)
#       theta = (1 - tau) + A4*|delta-1|^(1/beta4)
# ---------------------------------------------------------------------------


def _s4_bigdelta(
    tau: npt.NDArray[np.float64], delta: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """bigdelta = theta^2 + B4*|delta-1|^(2*a4)"""
    return (
        B4 * np.abs(delta - 1) ** (2 * a4)
        + (A4 * np.abs(delta - 1) ** (beta4 ** (-1.0)) - tau + 1) ** 2
    )


def _s4_phi(
    delta: npt.NDArray[np.float64], tau: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """phi = exp(-C4*(delta-1)^2 - D4*(tau-1)^2)"""
    return np.exp(-C4 * (delta - 1) ** 2 - D4 * (tau - 1) ** 2)


def _s4_dd0_dt0(
    tau: npt.NDArray[np.float64], delta: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    return delta * n4 * _s4_bigdelta(tau, delta) ** b4 * _s4_phi(delta, tau)


def _s4_dd0_dt1(
    tau: npt.NDArray[np.float64], delta: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    bd = _s4_bigdelta(tau, delta)
    phi = _s4_phi(delta, tau)
    return (
        -D4 * delta * n4 * (2 * tau - 2) * bd**b4 * phi
        + b4
        * delta
        * n4
        * bd ** (b4 - 1)
        * (-2 * A4 * np.abs(delta - 1) ** (beta4 ** (-1.0)) + 2 * tau - 2)
        * phi
    )


def _s4_dd0_dt2(
    tau: npt.NDArray[np.float64], delta: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    bd = _s4_bigdelta(tau, delta)
    phi = _s4_phi(delta, tau)
    theta_val = A4 * np.abs(delta - 1) ** (beta4 ** (-1.0)) - tau + 1
    return (
        2
        * delta
        * n4
        * bd**b4
        * (
            4 * D4 * b4 * (tau - 1) * theta_val / bd
            + D4 * (2 * D4 * (tau - 1) ** 2 - 1)
            + b4 * (2 * b4 * theta_val**2 / bd + 1 - 2 * theta_val**2 / bd) / bd
        )
        * phi
    )


def _s4_dd1_dt0(
    tau: npt.NDArray[np.float64], delta: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    bd = _s4_bigdelta(tau, delta)
    phi = _s4_phi(delta, tau)
    theta_val = A4 * np.abs(delta - 1) ** (beta4 ** (-1.0)) - tau + 1
    abs_dm1 = np.abs(delta - 1)
    sgn = np.sign(delta - 1)
    return (
        -C4 * delta * n4 * (2 * delta - 2) * bd**b4 * phi
        + b4
        * delta
        * n4
        * bd ** (b4 - 1)
        * (
            2 * A4 * theta_val * abs_dm1 ** (-1 + beta4 ** (-1.0)) * sgn / beta4
            + 2 * B4 * a4 * abs_dm1 ** (2 * a4 - 1) * sgn
        )
        * phi
        + n4 * bd**b4 * phi
    )


def _s4_dd1_dt1(
    tau: npt.NDArray[np.float64], delta: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    bd = _s4_bigdelta(tau, delta)
    phi = _s4_phi(delta, tau)
    theta_val = A4 * np.abs(delta - 1) ** (beta4 ** (-1.0)) - tau + 1
    abs_dm1 = np.abs(delta - 1)
    sgn = np.sign(delta - 1)
    d_bigdelta = A4 * theta_val * abs_dm1 ** (
        beta4 ** (-1.0)
    ) / beta4 + B4 * a4 * abs_dm1 ** (2 * a4)
    return (
        2
        * n4
        * bd**b4
        * (
            -A4 * b4 * delta * abs_dm1 ** (-1 + beta4 ** (-1.0)) * sgn / (beta4 * bd)
            + 2 * C4 * D4 * delta * (delta - 1) * (tau - 1)
            + 2 * C4 * b4 * delta * (delta - 1) * theta_val / bd
            - 2 * D4 * b4 * delta * (tau - 1) * d_bigdelta * sgn / (bd * abs_dm1)
            - D4 * (tau - 1)
            - 2 * b4**2 * delta * d_bigdelta * theta_val * sgn / (bd**2 * abs_dm1)
            + 2 * b4 * delta * d_bigdelta * theta_val * sgn / (bd**2 * abs_dm1)
            - b4 * theta_val / bd
        )
        * phi
    )


def _s4_dd2_dt0(
    tau: npt.NDArray[np.float64], delta: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    bd = _s4_bigdelta(tau, delta)
    phi = _s4_phi(delta, tau)
    theta_val = A4 * np.abs(delta - 1) ** (beta4 ** (-1.0)) - tau + 1
    abs_dm1 = np.abs(delta - 1)
    sgn = np.sign(delta - 1)
    dm1 = delta - 1
    d_bigdelta = A4 * theta_val * abs_dm1 ** (
        beta4 ** (-1.0)
    ) / beta4 + B4 * a4 * abs_dm1 ** (2 * a4)
    return (
        2
        * n4
        * bd**b4
        * (
            -4 * C4 * b4 * delta * dm1 * d_bigdelta * sgn / (bd * abs_dm1)
            + C4 * delta * (2 * C4 * dm1**2 - 1)
            - 2 * C4 * dm1
            + b4
            * delta
            * (
                A4**2 * abs_dm1 ** (2 / beta4) * sgn**2 / (beta4**2 * dm1**2)
                + 2
                * A4
                * theta_val
                * abs_dm1 ** (-1 + beta4 ** (-1.0))
                * _dirac_delta(dm1)
                / beta4
                - A4
                * theta_val
                * abs_dm1 ** (beta4 ** (-1.0))
                * sgn**2
                / (beta4 * dm1**2)
                + A4
                * theta_val
                * abs_dm1 ** (beta4 ** (-1.0))
                * sgn**2
                / (beta4**2 * dm1**2)
                + 2 * B4 * a4**2 * abs_dm1 ** (2 * a4) * sgn**2 / dm1**2
                + 2 * B4 * a4 * abs_dm1 ** (2 * a4 - 1) * _dirac_delta(dm1)
                - B4 * a4 * abs_dm1 ** (2 * a4) * sgn**2 / dm1**2
                + 2 * b4 * d_bigdelta**2 * sgn**2 / (dm1**2 * bd)
                - 2 * d_bigdelta**2 * sgn**2 / (dm1**2 * bd)
            )
            / bd
            + 2 * b4 * d_bigdelta * sgn / (bd * abs_dm1)
        )
        * phi
    )


# ---------------------------------------------------------------------------
# Dispatch table: maps (expression_index, diff_delta, diff_tau) to functions
# expression_index: 1=s1, 2=s2, 3=s3, 4=s4
# ---------------------------------------------------------------------------

_EXPRESSIONS: dict[tuple[int, int, int], Callable[..., Any]] = {
    (1, 0, 0): _s1_dd0_dt0,
    (1, 0, 1): _s1_dd0_dt1,
    (1, 0, 2): _s1_dd0_dt2,
    (1, 1, 0): _s1_dd1_dt0,
    (1, 1, 1): _s1_dd1_dt1,
    (1, 2, 0): _s1_dd2_dt0,
    (2, 0, 0): _s2_dd0_dt0,
    (2, 0, 1): _s2_dd0_dt1,
    (2, 0, 2): _s2_dd0_dt2,
    (2, 1, 0): _s2_dd1_dt0,
    (2, 1, 1): _s2_dd1_dt1,
    (2, 2, 0): _s2_dd2_dt0,
    (3, 0, 0): _s3_dd0_dt0,
    (3, 0, 1): _s3_dd0_dt1,
    (3, 0, 2): _s3_dd0_dt2,
    (3, 1, 0): _s3_dd1_dt0,
    (3, 1, 1): _s3_dd1_dt1,
    (3, 2, 0): _s3_dd2_dt0,
    (4, 0, 0): _s4_dd0_dt0,
    (4, 0, 1): _s4_dd0_dt1,
    (4, 0, 2): _s4_dd0_dt2,
    (4, 1, 0): _s4_dd1_dt0,
    (4, 1, 1): _s4_dd1_dt1,
    (4, 2, 0): _s4_dd2_dt0,
}


def residual_helmholtz_energy(
    delta_: npt.NDArray[np.float64],
    tau_: npt.NDArray[np.float64],
    diff_delta: int,
    diff_tau: int,
) -> npt.NDArray[np.float64]:
    """
    Equation 6.1 from Span & Wagner [2]. Calculates the residual helmholtz energy of co2 or its derivatives. tau_ and
    delta_ must have be numpy arrays of shape (N, 1). This allows for vectorization.

    :param delta_: Reduced density. Unit-less. numpy.ndarray with shape (N, 1)
    :param tau_: Inverse reduced temperature. Unit-less. numpy.ndarray with shape (N, 1)
    :param diff_delta: Degree of derivation wrt. delta. Integer.
    :param diff_tau: Degree of derivation wrt. tau. Integer.

    :return: Helmholtz free energy. Unit-less. numpy.ndarray with shape (N,)
    """
    _s1 = np.sum(_EXPRESSIONS[(1, diff_delta, diff_tau)](tau_, delta_), axis=-1)
    _s2 = np.sum(_EXPRESSIONS[(2, diff_delta, diff_tau)](tau_, delta_), axis=-1)
    _s3 = np.sum(_EXPRESSIONS[(3, diff_delta, diff_tau)](tau_, delta_), axis=-1)
    _s4 = np.sum(_EXPRESSIONS[(4, diff_delta, diff_tau)](tau_, delta_), axis=-1)

    return _s1 + _s2 + _s3 + _s4
