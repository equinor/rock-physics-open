import warnings

import numpy as np
from syrupy.assertion import SnapshotAssertion

from rock_physics_open.equinor_utilities.units import celsius_to_kelvin
from rock_physics_open.span_wagner import co2_properties
from rock_physics_open.span_wagner.co2_properties import (
    CO2_CRITICAL_DENSITY,
    CO2_CRITICAL_PRESSURE,
    CO2_CRITICAL_TEMPERATURE,
    CO2_TRIPLE_PRESSURE,
    CO2_TRIPLE_TEMPERATURE,
    carbon_dioxide_density,
    melting_pressure,
    saturated_liquid_density,
    saturated_vapor_density,
    sublimation_pressure,
    vapor_pressure,
)

temp = 100.0 * np.linspace(0.8, 1.2, 101)
pres = 23.0e6 * np.linspace(0.8, 1.2, 101)
c1 = 0.1 * np.ones(101)
c2 = 0.1 * np.ones(101)
c3 = 0.1 * np.ones(101)
c4 = 0.1 * np.ones(101)
c5 = 0.1 * np.ones(101)
c6 = 0.1 * np.ones(101)
c7 = 0.1 * np.ones(101)
n2 = 0.1 * np.ones(101)
co2 = 0.05 * np.ones(101)
h2s = 0.05 * np.ones(101)
sgc7 = 0.81 * np.ones(101)
mwc7 = 161 * np.ones(101)
gr = 1.0 * np.linspace(0.7, 1.05, 101)


def test_co2_properties(snapshot: SnapshotAssertion):
    temp_sw = np.linspace(-56, 28, 101)
    assert snapshot == co2_properties(temp_sw, pres)


def test_co2_properties_above_critical_temperature_no_warning():
    """Temperatures above CO2 critical temperature should not produce RuntimeWarning."""
    temp_above_critical = np.linspace(32, 100, 50)  # All above critical (~31 °C)
    pres_high = 20.0e6 * np.ones(50)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        vel, den, k = co2_properties(temp_above_critical, pres_high)
    assert np.all(np.isfinite(vel))
    assert np.all(np.isfinite(den))
    assert np.all(np.isfinite(k))


def test_co2_properties_vapor_phase(snapshot: SnapshotAssertion):
    """Snapshot test for vapor phase: T between triple and critical, P below vapor pressure."""
    temp_vapor = np.linspace(-40, 20, 50)
    pres_low = 0.5e6 * np.ones(50)
    assert snapshot == co2_properties(temp_vapor, pres_low)


def test_co2_properties_below_triple_point(snapshot: SnapshotAssertion):
    """Snapshot test for below triple point temperature (gas phase, sublimation region)."""
    temp_low = np.linspace(-80, -60, 20)
    pres_low = 5000.0 * np.ones(20)  # 5 kPa = 0.005 MPa
    assert snapshot == co2_properties(temp_low, pres_low)


def test_co2_properties_high_temperature(snapshot: SnapshotAssertion):
    """Snapshot test for high temperatures well above critical point."""
    temp_high = np.linspace(100, 500, 50)
    pres_high = 50.0e6 * np.ones(50)
    assert snapshot == co2_properties(temp_high, pres_high)


def test_co2_properties_near_critical_supercritical(snapshot: SnapshotAssertion):
    """Snapshot test near the critical point on the supercritical side."""
    temp_near = np.linspace(32, 40, 20)
    pres_near = np.linspace(8.0e6, 15.0e6, 20)
    assert snapshot == co2_properties(temp_near, pres_near)


def test_co2_properties_wide_pressure_range(snapshot: SnapshotAssertion):
    """Snapshot test across a wide range of pressures at constant above-critical temperature."""
    n = 30
    temp_const = 50.0 * np.ones(n)
    pres_range = np.geomspace(1.0e6, 100.0e6, n)
    assert snapshot == co2_properties(temp_const, pres_range)


def test_co2_properties_mixed_phase_array(snapshot: SnapshotAssertion):
    """Snapshot test spanning multiple phase regions: below triple, vapor, liquid, supercritical."""
    temp_mixed = np.array([-80.0, -50.0, -10.0, 0.0, 15.0, 50.0, 200.0])
    pres_mixed = np.array([0.005e6, 0.5e6, 0.5e6, 10.0e6, 10.0e6, 10.0e6, 50.0e6])
    assert snapshot == co2_properties(temp_mixed, pres_mixed)


def test_vapor_pressure_at_triple_and_critical():
    """Vapor pressure should match known values at the triple and critical points."""
    vp_critical = vapor_pressure(np.array([CO2_CRITICAL_TEMPERATURE - 0.001]))
    np.testing.assert_allclose(vp_critical[0], CO2_CRITICAL_PRESSURE, rtol=1e-3)

    vp_triple = vapor_pressure(np.array([CO2_TRIPLE_TEMPERATURE + 0.001]))
    np.testing.assert_allclose(vp_triple[0], CO2_TRIPLE_PRESSURE, rtol=1e-3)


def test_sublimation_pressure_at_triple():
    """Sublimation pressure at triple point temperature should equal triple pressure."""
    sp = sublimation_pressure(np.array([CO2_TRIPLE_TEMPERATURE - 0.001]))
    np.testing.assert_allclose(sp[0], CO2_TRIPLE_PRESSURE, rtol=1e-3)


def test_melting_pressure_at_triple():
    """Melting pressure at triple point temperature should equal triple pressure."""
    mp = melting_pressure(np.array([CO2_TRIPLE_TEMPERATURE]))
    np.testing.assert_allclose(mp[0], CO2_TRIPLE_PRESSURE, rtol=1e-6)


def test_saturated_densities_at_critical():
    """At critical temperature, saturated liquid and vapor densities should approach critical density."""
    t_near_critical = np.array([CO2_CRITICAL_TEMPERATURE - 0.01])
    liq_den = saturated_liquid_density(t_near_critical)
    vap_den = saturated_vapor_density(t_near_critical)
    np.testing.assert_allclose(liq_den[0], CO2_CRITICAL_DENSITY, rtol=0.1)
    np.testing.assert_allclose(vap_den[0], CO2_CRITICAL_DENSITY, rtol=0.1)


def test_saturated_densities_ordering():
    """Saturated liquid density should always exceed saturated vapor density."""
    temps = np.linspace(CO2_TRIPLE_TEMPERATURE + 1, CO2_CRITICAL_TEMPERATURE - 1, 50)
    liq = saturated_liquid_density(temps)
    vap = saturated_vapor_density(temps)
    assert np.all(liq > vap)


def test_carbon_dioxide_density_force_vapor():
    """Test force_vapor parameter produces vapor-phase (lower) densities."""
    temps_k = celsius_to_kelvin(np.array([0.0, 10.0, 20.0]))
    vp = vapor_pressure(temps_k)
    den_liquid = carbon_dioxide_density(temps_k, vp, force_vapor=False)
    den_vapor = carbon_dioxide_density(temps_k, vp, force_vapor=True)
    assert np.all(den_liquid > den_vapor)


def test_carbon_dioxide_density_scalar_input():
    """Single-element array inputs should work and return a finite result."""
    temp_k = celsius_to_kelvin(np.array([20.0]))
    pres_mpa = np.array([10.0])
    result = carbon_dioxide_density(temp_k, pres_mpa)
    assert np.all(np.isfinite(result))
    assert np.all(result > 0)
