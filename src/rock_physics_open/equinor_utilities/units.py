"""
Unit conversion functions for rock physics calculations.

Provides conversions between common units used in rock physics:
- Temperature: Celsius / Kelvin
- Pressure: Pascal / MegaPascal / Torr
- Density: kg/m³ / g/cc / API gravity
- Velocity: m/s / km/s
- Viscosity: Pa·s / centipoise
"""

from typing import Any, TypeVar

import numpy.typing as npt

FloatOrArray = TypeVar("FloatOrArray", float, npt.NDArray[Any])


def pa_to_torr(pressure_pa: FloatOrArray) -> FloatOrArray:
    """Convert pressure from Pascal to Torr."""
    return pressure_pa / 133.3224


def torr_to_pa(pressure_torr: FloatOrArray) -> FloatOrArray:
    """Convert pressure from Torr to Pascal."""
    return pressure_torr * 133.3224


def kg_m3_to_g_cc(density_kg_m3: FloatOrArray) -> FloatOrArray:
    """Convert density from kilograms per cubic meter to grams per cubic centimeter."""
    return density_kg_m3 / 1000.0


def g_cc_to_kg_m3(density_g_cc: FloatOrArray) -> FloatOrArray:
    """Convert density from grams per cubic centimeter to kilograms per cubic meter."""
    return density_g_cc * 1000.0


def pa_to_mpa(pressure_pa: FloatOrArray) -> FloatOrArray:
    """Convert pressure from Pascal to MegaPascal."""
    return pressure_pa / 1e6


def mpa_to_pa(pressure_mpa: FloatOrArray) -> FloatOrArray:
    """Convert pressure from MegaPascal to Pascal."""
    return pressure_mpa * 1e6


def pa_to_psi(pressure_pa: FloatOrArray) -> FloatOrArray:
    """Convert pressure from Pascal to pounds per square inch."""
    return pressure_pa / 6894.757


def psi_to_pa(pressure_psi: FloatOrArray) -> FloatOrArray:
    """Convert pressure from pounds per square inch to Pascal."""
    return pressure_psi * 6894.757


def celsius_to_kelvin(temperature_c: FloatOrArray) -> FloatOrArray:
    """Convert temperature from Celsius to Kelvin."""
    return temperature_c + 273.15


def kelvin_to_celsius(temperature_k: FloatOrArray) -> FloatOrArray:
    """Convert temperature from Kelvin to Celsius."""
    return temperature_k - 273.15


def celsius_to_fahrenheit(temperature_c: FloatOrArray) -> FloatOrArray:
    """Convert temperature from Celsius to Fahrenheit."""
    return temperature_c * 9.0 / 5.0 + 32.0


def fahrenheit_to_celsius(temperature_f: FloatOrArray) -> FloatOrArray:
    """Convert temperature from Fahrenheit to Celsius."""
    return (temperature_f - 32.0) * 5.0 / 9.0


def km_per_s_to_m_per_s(velocity_km_per_s: FloatOrArray) -> FloatOrArray:
    """Convert velocity from kilometers per second to meters per second."""
    return velocity_km_per_s * 1000.0


def m_per_s_to_km_per_s(velocity_m_per_s: FloatOrArray) -> FloatOrArray:
    """Convert velocity from meters per second to kilometers per second."""
    return velocity_m_per_s / 1000.0


def cp_to_pa_s(viscosity_cp: FloatOrArray) -> FloatOrArray:
    """Convert viscosity from centipoise to Pascal-seconds."""
    return viscosity_cp / 1000.0


def pa_s_to_cp(viscosity_pa_s: FloatOrArray) -> FloatOrArray:
    """Convert viscosity from Pascal-seconds to centipoise."""
    return viscosity_pa_s * 1000.0


def api_to_g_cc(api: FloatOrArray) -> FloatOrArray:
    """Convert API gravity to density in g/cc."""
    return 141.5 / (api + 131.5)


def api_to_kg_m3(api: FloatOrArray) -> FloatOrArray:
    """Convert API gravity to density in kg/m³."""
    return g_cc_to_kg_m3(api_to_g_cc(api))


def g_cc_to_api(density_g_cc: FloatOrArray) -> FloatOrArray:
    """Convert density in g/cc to API gravity."""
    return 141.5 / density_g_cc - 131.5


def kg_m3_to_api(density_kg_m3: FloatOrArray) -> FloatOrArray:
    """Convert density in kg/m³ to API gravity."""
    return g_cc_to_api(kg_m3_to_g_cc(density_kg_m3))
