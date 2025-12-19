from typing import Any, Protocol, TypeVar

import numpy as np
import numpy.typing as npt

_T = TypeVar("_T", bound=np.generic)

Array1D = np.ndarray[tuple[int], np.dtype[_T]]
Array2D = np.ndarray[tuple[int, int], np.dtype[_T]]
Array3D = np.ndarray[tuple[int, int, int], np.dtype[_T]]
Array4D = np.ndarray[tuple[int, int, int, int], np.dtype[_T]]


class OptCallable(Protocol):
    def __call__(
        self, x_data: npt.NDArray[np.float64], *args: Any, **kwargs: Any
    ) -> npt.NDArray[np.float64]: ...


class TMatrixCallable(Protocol):
    def __call__(
        self,
        k_min: Array1D[np.float64],
        mu_min: Array1D[np.float64],
        rho_min: Array1D[np.float64],
        k_fl: Array1D[np.float64],
        rho_fl: Array1D[np.float64],
        phi: Array1D[np.float64],
        perm: Array1D[np.float64],
        visco: Array1D[np.float64],
        alpha: Array2D[np.float64],
        v: Array2D[np.float64],
        tau: Any,
        frequency: float,
        angle: float,
        frac_inc_con: Array1D[np.float64],
        frac_inc_ani: Array1D[np.float64],
    ) -> tuple[
        Array1D[np.float64],
        Array1D[np.float64],
        Array1D[np.float64],
        Array1D[np.float64],
    ]: ...
