"""
Simple conversions required for the material models.
"""


def celsius_to_kelvin(temperature: float) -> float:
    """
    Convert temperature from Celsius to kelvin
    """
    return temperature + 273.15
