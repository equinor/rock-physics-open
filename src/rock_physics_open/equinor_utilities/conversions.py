"""Simple conversions required for the material models.

.. deprecated::
    Import from :mod:`rock_physics_open.equinor_utilities.units` instead.
"""

from typing_extensions import deprecated

from rock_physics_open.equinor_utilities.units import FloatOrArray


@deprecated(
    "Import celsius_to_kelvin from rock_physics_open.equinor_utilities.units instead."
)
def celsius_to_kelvin(temperature_c: FloatOrArray) -> FloatOrArray:
    """Convert temperature from Celsius to Kelvin."""
    from rock_physics_open.equinor_utilities.units import (
        celsius_to_kelvin as _celsius_to_kelvin,
    )

    return _celsius_to_kelvin(temperature_c)


__all__ = ["celsius_to_kelvin"]
