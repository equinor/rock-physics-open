"""
Simple conversions required for the material models.
"""

import numpy as np
import numpy.typing as npt


def celsius_to_kelvin(
    temperature: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Convert temperature from Celsius to kelvin
    """
    return temperature + 273.15
