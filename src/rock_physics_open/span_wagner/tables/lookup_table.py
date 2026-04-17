import functools
from typing import Any, Callable

import numpy as np
import numpy.typing as npt
from scipy.interpolate import RegularGridInterpolator


@functools.lru_cache(maxsize=10)
def load_lookup_table_interpolator(
    filename: str,
) -> Callable[..., npt.NDArray[np.float64]]:
    data = np.load(filename)
    reg = RegularGridInterpolator(
        (data["x"], data["y"]), data["z_grid"], method="nearest", bounds_error=False
    )

    def _interp(
        _x: npt.NDArray[np.float64],
        _y: npt.NDArray[np.float64],
    ) -> npt.NDArray[Any]:
        _x, _y = np.asarray(_x), np.asarray(_y)
        pts = np.full((max(_x.size, _y.size), 2), fill_value=np.nan)
        pts[:, 0] = _x
        pts[:, 1] = _y
        res = reg(pts)
        if _x.ndim == 0 and _y.ndim == 0:
            return res[0]
        return res

    return _interp
