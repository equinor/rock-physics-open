from typing import TypeAlias

import numpy as np
import numpy.typing as npt

ArrayLikeFloat: TypeAlias = npt.NDArray[np.float64] | float


def as_float_array(value: ArrayLikeFloat) -> npt.NDArray[np.float64]:
    """Ensure that an input will be cast to a numpy array with at least one
    dimension"""
    return np.atleast_1d(np.asarray(value, dtype=np.float64))


def inputs_are_scalar(*values: ArrayLikeFloat) -> bool:
    """Test if all inputs are scalar values"""
    return all(np.asarray(value).ndim == 0 for value in values)


def oil_density_to_api(rho0: ArrayLikeFloat) -> npt.NDArray[np.float64]:
    """Convert oil density in g/cc to API gravity."""
    return np.asarray(141.5 / np.asarray(rho0, dtype=np.float64) - 131.5)


def oil_api_to_density(rho0_api: ArrayLikeFloat) -> npt.NDArray[np.float64]:
    """Convert API gravity to oil density in g/cc."""
    return np.asarray(141.5 / (np.asarray(rho0_api, dtype=np.float64) + 131.5))


def oil_density_to_gcc(rho0: ArrayLikeFloat) -> npt.NDArray[np.float64]:
    """Convert oil density from kg/m³ to g/cc"""
    return np.asarray(np.asarray(rho0, dtype=np.float64) / 1000.0)


def oil_density_to_kg_m_3(rho0_gcc: ArrayLikeFloat) -> npt.NDArray[np.float64]:
    """Convert oil density from g/cc to kg/m³"""
    return np.asarray(np.asarray(rho0_gcc, dtype=np.float64) * 1000.0)
