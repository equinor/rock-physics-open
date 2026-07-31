# Transition: Removal of `sympy` Dependency

This document records the code changes made when removing the `sympy` runtime
dependency from `rock-physics-open` (PR #103). The `sympy` library was used in a
single module — `span_wagner/equations.py` — to symbolically define the Span &
Wagner residual Helmholtz energy equations, differentiate them, and `lambdify`
them into NumPy callables at **import time**. The new implementation replaces
this with pre-computed, hardcoded NumPy expressions that are mathematically
equivalent but carry no `sympy` overhead.

---

## Table of Contents

1. [Motivation](#motivation)
2. [Scope of Changes](#scope-of-changes)
3. [Original `sympy` Implementation](#original-sympy-implementation)
4. [New Pure-NumPy Implementation](#new-pure-numpy-implementation)
5. [Derivation Methodology](#derivation-methodology)
6. [One-to-One Mapping of Expressions](#one-to-one-mapping-of-expressions)
   - [S1 — Power-law terms](#s1--power-law-terms)
   - [S2 — Exponential terms](#s2--exponential-terms)
   - [S3 — Gaussian-bell terms](#s3--gaussian-bell-terms)
   - [S4 — Non-analytic terms](#s4--non-analytic-terms)
7. [Other Changes in This Transition](#other-changes-in-this-transition)
8. [Verification](#verification)

---

## Motivation

- **`sympy` was a heavy runtime dependency** used in only one place
  (`span_wagner/equations.py`). Removing it reduces install size, speeds up
  import time, and simplifies the dependency tree.
- The symbolic differentiation + `lambdify` step occurred **at import time**,
  adding ~2–5 seconds of startup latency on every import.
- The expressions and their derivatives are **fixed mathematical formulas** from
  Span & Wagner [2] — they never change at runtime and can therefore be
  pre-computed once and checked in as source code.

## Scope of Changes

| File | Change |
|------|--------|
| `src/rock_physics_open/span_wagner/equations.py` | Complete rewrite: sympy symbolic → hardcoded NumPy |
| `src/rock_physics_open/span_wagner/co2_properties.py` | Removed `float_vectorize` helper; replaced with direct `np.vectorize` |
| `pyproject.toml` | Removed `sympy` from `dependencies` and `allowedUntypedLibraries` |

Additional unrelated clean-ups were bundled in the same PR but are not related
to the sympy removal (ternary plot robustness, filter_input pandas fix, pytest
marker removal).

---

## Original `sympy` Implementation

The original code in `span_wagner/equations.py` (on the `main` branch) is
preserved below in its entirety:

```python
"""
Package for calculating the residual helmholtz energy for CO2. The module uses
sympy to evaluate the functions defined in Span & Wagner [2], as well as its
derivatives.
"""

from typing import Any, Callable, Literal

import numpy as np
import numpy.typing as npt
import sympy as sp

from .coefficients import coeff_vars


def residual_helmholtz_energy(
    delta_: npt.NDArray[np.float64],
    tau_: npt.NDArray[np.float64],
    diff_delta: int,
    diff_tau: int,
) -> npt.NDArray[np.float64]:
    """
    Equation 6.1 from Span & Wagner [2]. Calculates the residual helmholtz
    energy of co2 or its derivatives. tau_ and delta_ must have be numpy
    arrays of shape (N, 1). This allows for vectorization.

    :param delta_: Reduced density. Unit-less. numpy.ndarray with shape (N, 1)
    :param tau_: Inverse reduced temperature. Unit-less. numpy.ndarray with
        shape (N, 1)
    :param diff_delta: Degree of derivation wrt. delta. Integer.
    :param diff_tau: Degree of derivation wrt. tau. Integer.

    :return: Helmholtz free energy. Unit-less. numpy.ndarray with shape (N,)
    """
    _s1 = np.sum(
        _LAMBDIFIED_EXPRESSIONS[(s1, diff_delta, diff_tau)](tau_, delta_),
        axis=-1,
    )
    _s2 = np.sum(
        _LAMBDIFIED_EXPRESSIONS[(s2, diff_delta, diff_tau)](tau_, delta_),
        axis=-1,
    )
    _s3 = np.sum(
        _LAMBDIFIED_EXPRESSIONS[(s3, diff_delta, diff_tau)](tau_, delta_),
        axis=-1,
    )
    _s4 = np.sum(
        _LAMBDIFIED_EXPRESSIONS[(s4, diff_delta, diff_tau)](tau_, delta_),
        axis=-1,
    )

    return _s1 + _s2 + _s3 + _s4


# Define coefficients symbols. These should correspond one-to-one with variable
# names in the coefficient module
coeff_symbols = (
    n1,
    t1,
    d1,
    n2,
    d2,
    t2,
    c2,
    n3,
    d3,
    t3,
    alpha3,
    epsilon3,
    beta3,
    gamma3,
    n4,
    b4,
    a4,
    beta4,
    A4,
    B4,
    C4,
    D4,
) = sp.symbols(
    "n1 t1 d1 n2 d2 t2 c2 n3 d3 t3 alpha3 epsilon3 beta3 gamma3 "
    "n4 b4 a4 beta4 A4 B4 C4 D4"
)
tau, delta = sp.symbols("tau delta", real=True)
i = sp.symbols("i", integer=True)

# --- Symbolic expressions (Eq. 6.1 sub-sums, Span & Wagner [2]) ---
s1 = n1 * delta**d1 * tau**t1
s2 = n2 * delta**d2 * tau**t2 * sp.exp(-(delta**c2))
s3 = (
    n3
    * delta**d3
    * tau**t3
    * sp.exp(-alpha3 * (delta - epsilon3) ** 2 - beta3 * (tau - gamma3) ** 2)
)

theta_expr = (1 - tau) + A4 * ((delta - 1) ** 2) ** (1 / (2 * beta4))
bigdelta_expr = theta_expr**2 + B4 * ((delta - 1) ** 2) ** a4
bigphi_expr = sp.exp(-C4 * (delta - 1) ** 2 - D4 * (tau - 1) ** 2)
s4 = n4 * bigdelta_expr**b4 * delta * bigphi_expr


def _lambdify(
    expr: sp.Expr,
    diff_delta: Literal[0, 1, 2],
    diff_tau: Literal[0, 1, 2],
) -> Callable[..., Any]:
    diff = [delta] * diff_delta + [tau] * diff_tau
    if len(diff) > 0:
        expr = expr.diff(*diff)
    expr = expr.powsimp()
    _sympy_lambda = sp.utilities.lambdify(
        list(coeff_symbols) + [tau, delta],
        expr,
        modules=["numpy", {"DiracDelta": lambda x: x == 0}],
    )
    return lambda _tau, _delta: _sympy_lambda(*coeff_vars, _tau, _delta)


_LAMBDIFIED_EXPRESSIONS: dict[tuple[Any, int, int], Callable[..., Any]] = {
    (e, dd, dt): _lambdify(expr=e, diff_delta=dd, diff_tau=dt)
    for e in (s1, s2, s3, s4)
    for dd in (0, 1, 2)
    for dt in (0, 1, 2)
    if dd + dt <= 2
}
```

### How it worked

1. **Symbolic definition**: Four sub-expressions (`s1`–`s4`) were defined using
   `sympy` symbols corresponding to the coefficient arrays from
   `coefficients.py`.
2. **Differentiation**: At import time, the `_lambdify` helper called
   `expr.diff(delta, …, tau, …)` to take 0th, 1st, or 2nd derivatives with
   respect to `delta` and/or `tau`.
3. **Simplification & compilation**: `powsimp()` simplified the result,
   then `sp.utilities.lambdify(…, modules=["numpy", …])` compiled it to a
   NumPy-callable function.
4. **Mapping**: A `DiracDelta` replacement provided `lambda x: x == 0` for the
   Dirac delta that appears in 2nd-order derivatives of `s4`.
5. **Dispatch**: `_LAMBDIFIED_EXPRESSIONS` stored all 24 combinations
   `(expr, dd, dt)` where `dd + dt ≤ 2` — this dictionary was used by
   `residual_helmholtz_energy()`.

---

## New Pure-NumPy Implementation

The new code eliminates `sympy` entirely. Each of the 24 expression/derivative
combinations is now a standalone Python function that directly computes the
result using NumPy operations. The coefficients are imported as concrete NumPy
arrays from `coefficients.py` (instead of as sympy symbols).

A dispatch table `_EXPRESSIONS` replaces the old `_LAMBDIFIED_EXPRESSIONS`
dictionary:

```python
_EXPRESSIONS: dict[tuple[int, int, int], Callable[..., Any]] = {
    (1, 0, 0): _s1_dd0_dt0,
    (1, 0, 1): _s1_dd0_dt1,
    # … (24 entries total)
}
```

The key changed from `(sympy_expr, dd, dt)` to `(int_index, dd, dt)` where the
index 1–4 identifies the sub-expression.

`residual_helmholtz_energy()` was updated accordingly:

```python
# Old:
_s1 = np.sum(_LAMBDIFIED_EXPRESSIONS[(s1, diff_delta, diff_tau)](tau_, delta_), axis=-1)

# New:
_s1 = np.sum(_EXPRESSIONS[(1, diff_delta, diff_tau)](tau_, delta_), axis=-1)
```

---

## Derivation Methodology

The new hardcoded expressions were generated by:

1. Using the original `sympy` code to symbolically differentiate each
   sub-expression.
2. Calling `sympy.simplify()` / `powsimp()` on each derivative.
3. Converting the resulting symbolic expression to a NumPy-compatible Python
   function manually (equivalent to what `lambdify` would produce).
4. Replacing `((delta - 1)**2)**(1/(2*beta4))` with `|delta - 1|^(1/beta4)`
   (using `np.abs`) to match `sympy`'s treatment of real-valued symbols.
5. Verifying numerical equivalence against the original `lambdify` output for a
   range of test inputs.

---

## One-to-One Mapping of Expressions

Below is the detailed correspondence between each sympy-derived expression and
its hardcoded NumPy replacement. Notation:
- `dd` = number of derivatives w.r.t. δ (delta)
- `dt` = number of derivatives w.r.t. τ (tau)

### S1 — Power-law terms

**Symbolic definition:**

$$S_1 = n_1 \cdot \delta^{d_1} \cdot \tau^{t_1}$$

| Key | Derivative | Symbolic result | NumPy function |
|-----|-----------|-----------------|----------------|
| `(s1, 0, 0)` → `(1, 0, 0)` | $S_1$ | $n_1 \delta^{d_1} \tau^{t_1}$ | `_s1_dd0_dt0` |
| `(s1, 0, 1)` → `(1, 0, 1)` | $\partial S_1 / \partial \tau$ | $n_1 \delta^{d_1} t_1 \tau^{t_1 - 1}$ | `_s1_dd0_dt1` |
| `(s1, 0, 2)` → `(1, 0, 2)` | $\partial^2 S_1 / \partial \tau^2$ | $n_1 \delta^{d_1} t_1 (t_1 - 1) \tau^{t_1 - 2}$ | `_s1_dd0_dt2` |
| `(s1, 1, 0)` → `(1, 1, 0)` | $\partial S_1 / \partial \delta$ | $n_1 d_1 \delta^{d_1 - 1} \tau^{t_1}$ | `_s1_dd1_dt0` |
| `(s1, 1, 1)` → `(1, 1, 1)` | $\partial^2 S_1 / \partial \delta \partial \tau$ | $n_1 d_1 \delta^{d_1 - 1} t_1 \tau^{t_1 - 1}$ | `_s1_dd1_dt1` |
| `(s1, 2, 0)` → `(1, 2, 0)` | $\partial^2 S_1 / \partial \delta^2$ | $n_1 d_1 (d_1 - 1) \delta^{d_1 - 2} \tau^{t_1}$ | `_s1_dd2_dt0` |

### S2 — Exponential terms

**Symbolic definition:**

$$S_2 = n_2 \cdot \delta^{d_2} \cdot \tau^{t_2} \cdot e^{-\delta^{c_2}}$$

| Key | Derivative | NumPy function |
|-----|-----------|----------------|
| `(s2, 0, 0)` → `(2, 0, 0)` | $S_2$ | `_s2_dd0_dt0` |
| `(s2, 0, 1)` → `(2, 0, 1)` | $\partial S_2 / \partial \tau$ | `_s2_dd0_dt1` |
| `(s2, 0, 2)` → `(2, 0, 2)` | $\partial^2 S_2 / \partial \tau^2$ | `_s2_dd0_dt2` |
| `(s2, 1, 0)` → `(2, 1, 0)` | $\partial S_2 / \partial \delta$ | `_s2_dd1_dt0` |
| `(s2, 1, 1)` → `(2, 1, 1)` | $\partial^2 S_2 / \partial \delta \partial \tau$ | `_s2_dd1_dt1` |
| `(s2, 2, 0)` → `(2, 2, 0)` | $\partial^2 S_2 / \partial \delta^2$ | `_s2_dd2_dt0` |

**Notable derivative details for S2:**

The first δ-derivative introduces a product-rule term:

$$\frac{\partial S_2}{\partial \delta} = n_2 \tau^{t_2} e^{-\delta^{c_2}} \left( d_2 \delta^{d_2-1} - c_2 \delta^{c_2+d_2-1} \right)$$

The second δ-derivative expands further via the chain rule:

$$\frac{\partial^2 S_2}{\partial \delta^2} = n_2 \tau^{t_2} \delta^{d_2-2} e^{-\delta^{c_2}} \left( d_2(d_2-1) - 2c_2 d_2 \delta^{c_2} + c_2 \delta^{c_2}(c_2 \delta^{c_2} - c_2 + 1) \right)$$

### S3 — Gaussian-bell terms

**Symbolic definition:**

$$S_3 = n_3 \cdot \delta^{d_3} \cdot \tau^{t_3} \cdot \exp\!\left(-\alpha_3 (\delta - \epsilon_3)^2 - \beta_3 (\tau - \gamma_3)^2\right)$$

The exponential factor appears in every S3 derivative and is extracted into a
helper `_s3_exp(delta, tau)`.

| Key | Derivative | NumPy function |
|-----|-----------|----------------|
| `(s3, 0, 0)` → `(3, 0, 0)` | $S_3$ | `_s3_dd0_dt0` |
| `(s3, 0, 1)` → `(3, 0, 1)` | $\partial S_3 / \partial \tau$ | `_s3_dd0_dt1` |
| `(s3, 0, 2)` → `(3, 0, 2)` | $\partial^2 S_3 / \partial \tau^2$ | `_s3_dd0_dt2` |
| `(s3, 1, 0)` → `(3, 1, 0)` | $\partial S_3 / \partial \delta$ | `_s3_dd1_dt0` |
| `(s3, 1, 1)` → `(3, 1, 1)` | $\partial^2 S_3 / \partial \delta \partial \tau$ | `_s3_dd1_dt1` |
| `(s3, 2, 0)` → `(3, 2, 0)` | $\partial^2 S_3 / \partial \delta^2$ | `_s3_dd2_dt0` |

**Notable derivative details for S3:**

The second τ-derivative collects terms:

$$\frac{\partial^2 S_3}{\partial \tau^2} = S_3 \cdot \left[ \frac{t_3(t_3-1)}{\tau^2} + \frac{4\beta_3 t_3 (\gamma_3 - \tau)}{\tau} + 2\beta_3\!\left(2\beta_3(\gamma_3-\tau)^2 - 1\right) \right]$$

The mixed derivative (dd=1, dt=1):

$$\frac{\partial^2 S_3}{\partial \delta \partial \tau} = S_3 \cdot \left[ -4\alpha_3\beta_3(\delta-\epsilon_3)(\gamma_3-\tau) - \frac{2\alpha_3 t_3 (\delta-\epsilon_3)}{\tau} + \frac{2\beta_3 d_3 (\gamma_3-\tau)}{\delta} + \frac{d_3 t_3}{\delta\tau} \right]$$

### S4 — Non-analytic terms

**Symbolic definition:**

This is the most complex sub-expression (equations 6.5–6.7 from Span & Wagner [2]):

$$\theta = (1 - \tau) + A_4 \left|\delta - 1\right|^{1/\beta_4}$$

$$\Delta = \theta^2 + B_4 \left|\delta - 1\right|^{2a_4}$$

$$\Psi = \exp\!\left(-C_4 (\delta-1)^2 - D_4 (\tau-1)^2\right)$$

$$S_4 = n_4 \cdot \Delta^{b_4} \cdot \delta \cdot \Psi$$

The original sympy code used `((delta - 1)**2)**(1/(2*beta4))` which, for
real-valued symbols, is equivalent to `|delta - 1|^(1/beta4)`. In the NumPy
code, `np.abs(delta - 1)` is used explicitly.

Two helper functions extract common sub-expressions:
- `_s4_bigdelta(tau, delta)` → computes $\Delta$
- `_s4_phi(delta, tau)` → computes $\Psi$

| Key | Derivative | NumPy function |
|-----|-----------|----------------|
| `(s4, 0, 0)` → `(4, 0, 0)` | $S_4$ | `_s4_dd0_dt0` |
| `(s4, 0, 1)` → `(4, 0, 1)` | $\partial S_4 / \partial \tau$ | `_s4_dd0_dt1` |
| `(s4, 0, 2)` → `(4, 0, 2)` | $\partial^2 S_4 / \partial \tau^2$ | `_s4_dd0_dt2` |
| `(s4, 1, 0)` → `(4, 1, 0)` | $\partial S_4 / \partial \delta$ | `_s4_dd1_dt0` |
| `(s4, 1, 1)` → `(4, 1, 1)` | $\partial^2 S_4 / \partial \delta \partial \tau$ | `_s4_dd1_dt1` |
| `(s4, 2, 0)` → `(4, 2, 0)` | $\partial^2 S_4 / \partial \delta^2$ | `_s4_dd2_dt0` |

**Notable details for S4:**

- The `DiracDelta` function from sympy (mapped to `lambda x: x == 0`) appears
  in the second δ-derivative of S4. In the new code this is replaced by
  `_dirac_delta(x)` which returns `(x == 0).astype(np.float64)`.
- S4 derivatives involve `np.sign(delta - 1)` arising from differentiation of
  `|delta - 1|`.
- An intermediate quantity `d_bigdelta` (the δ-derivative of Δ, divided by
  2 and without the sign factor) is computed as a local variable in several
  functions to keep expressions manageable.

---

## Other Changes in This Transition

### `co2_properties.py` — Removal of `float_vectorize`

The `float_vectorize` decorator was a thin wrapper around `np.vectorize`:

**Old code:**

```python
from collections.abc import Callable
from typing import ParamSpec

_P = ParamSpec("_P")


def float_vectorize(f: Callable[_P, float]) -> Callable[_P, npt.NDArray[np.float64]]:
    return np.vectorize(f, otypes=[float])


@float_vectorize
def _calculate_carbon_dioxide_density(
    absolute_temperature: npt.NDArray[np.float64],
    pressure: npt.NDArray[np.float64],
    force_vapor: bool | Literal["auto"] = "auto",
    raise_error: bool = True,
) -> float: ...
```

**New code:**

```python
def _calculate_carbon_dioxide_density_scalar(
    absolute_temperature: float,
    pressure: float,
    force_vapor: bool | Literal["auto"] = "auto",
    raise_error: bool = True,
) -> float: ...


_calculate_carbon_dioxide_density = np.vectorize(
    _calculate_carbon_dioxide_density_scalar, otypes=[float]
)
```

The decorator was removed because it introduced unnecessary abstraction
(`ParamSpec`, `Callable` import) while doing nothing beyond a direct
`np.vectorize` call.

### `pyproject.toml`

- Removed `"sympy >= 1.13.3"` from `dependencies`
- Removed `"sympy"` from `allowedUntypedLibraries` (appeared twice)

---

## Verification

The existing test suite in `tests/span_wagner/test_co2_properties.py` exercises
`residual_helmholtz_energy()` indirectly through the CO₂ property calculation
functions (density, bulk modulus, viscosity, etc.). All tests pass identically
before and after this transition, confirming numerical equivalence of the
hardcoded expressions with the original sympy-generated ones.

The snapshot test data in `tests/data/snapshots/` provides regression coverage
against the published Span & Wagner reference values.
