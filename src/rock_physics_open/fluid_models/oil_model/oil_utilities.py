from typing import TypeAlias

import numpy as np
import numpy.typing as npt

ArrayLikeFloat: TypeAlias = npt.NDArray[np.float64] | float


def as_float_array(value: ArrayLikeFloat) -> npt.NDArray[np.float64]:
    return np.atleast_1d(np.asarray(value, dtype=np.float64))


def inputs_are_scalar(*values: ArrayLikeFloat) -> bool:
    return all(np.asarray(value).ndim == 0 for value in values)


def oil_density_to_api(rho0: ArrayLikeFloat) -> ArrayLikeFloat:
    """Convert oil density in g/cc to API gravity."""
    return 141.5 / rho0 - 131.5


def oil_api_to_density(rho0_api: ArrayLikeFloat) -> ArrayLikeFloat:
    """Convert API gravity to oil density in g/cc."""
    return 141.5 / (rho0_api + 131.5)


def oil_density_to_gcc(rho0: ArrayLikeFloat) -> ArrayLikeFloat:
    return rho0 / 1000.0


def oil_density_to_kg_m_3(rho0_gcc: ArrayLikeFloat) -> ArrayLikeFloat:
    return rho0_gcc * 1000.0
