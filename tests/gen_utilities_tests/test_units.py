import numpy as np

from rock_physics_open.equinor_utilities.units import (
    api_to_g_cc,
    api_to_kg_m3,
    celsius_to_fahrenheit,
    celsius_to_kelvin,
    cp_to_pa_s,
    fahrenheit_to_celsius,
    g_cc_to_api,
    g_cc_to_kg_m3,
    kelvin_to_celsius,
    kg_m3_to_api,
    kg_m3_to_g_cc,
    km_per_s_to_m_per_s,
    m_per_s_to_km_per_s,
    mpa_to_pa,
    pa_s_to_cp,
    pa_to_mpa,
    pa_to_psi,
    pa_to_torr,
    psi_to_pa,
    torr_to_pa,
)


def test_celsius_kelvin():
    """Verify Celsius <-> Kelvin conversions for known reference points."""
    # Freezing point of water
    np.testing.assert_allclose(celsius_to_kelvin(0.0), 273.15)
    # Boiling point of water
    np.testing.assert_allclose(celsius_to_kelvin(100.0), 373.15)
    # Absolute zero
    np.testing.assert_allclose(celsius_to_kelvin(-273.15), 0.0, atol=1e-10)
    np.testing.assert_allclose(kelvin_to_celsius(0.0), -273.15)
    # Array input
    result = celsius_to_kelvin(np.array([0.0, 100.0, -40.0]))
    np.testing.assert_allclose(result, [273.15, 373.15, 233.15])
    # Roundtrip preserves value
    np.testing.assert_allclose(kelvin_to_celsius(celsius_to_kelvin(25.0)), 25.0)


def test_celsius_fahrenheit():
    """Verify Celsius <-> Fahrenheit conversions for known reference points."""
    # Freezing point of water
    np.testing.assert_allclose(celsius_to_fahrenheit(0.0), 32.0)
    # Boiling point of water
    np.testing.assert_allclose(celsius_to_fahrenheit(100.0), 212.0)
    # -40 is the crossover point where both scales are equal
    np.testing.assert_allclose(celsius_to_fahrenheit(-40.0), -40.0)
    # Inverse direction
    np.testing.assert_allclose(fahrenheit_to_celsius(32.0), 0.0)
    np.testing.assert_allclose(fahrenheit_to_celsius(212.0), 100.0)
    # Array input
    result = celsius_to_fahrenheit(np.array([0.0, 100.0, -40.0]))
    np.testing.assert_allclose(result, [32.0, 212.0, -40.0])
    # Roundtrip preserves value
    np.testing.assert_allclose(fahrenheit_to_celsius(celsius_to_fahrenheit(37.0)), 37.0)


def test_pa_mpa():
    """Verify Pascal <-> MegaPascal conversions."""
    # 1 MPa = 1e6 Pa
    np.testing.assert_allclose(pa_to_mpa(1e6), 1.0)
    np.testing.assert_allclose(mpa_to_pa(1.0), 1e6)
    # Array input
    np.testing.assert_allclose(pa_to_mpa(np.array([1e6, 2e6, 5e6])), [1.0, 2.0, 5.0])
    # Roundtrip preserves value
    np.testing.assert_allclose(mpa_to_pa(pa_to_mpa(5e7)), 5e7)


def test_pa_torr():
    """Verify Pascal <-> Torr conversions."""
    # 1 Torr = 133.3224 Pa
    np.testing.assert_allclose(pa_to_torr(133.3224), 1.0)
    np.testing.assert_allclose(torr_to_pa(1.0), 133.3224)
    # Roundtrip at 1 atm
    np.testing.assert_allclose(torr_to_pa(pa_to_torr(101325.0)), 101325.0)


def test_pa_psi():
    """Verify Pascal <-> PSI conversions."""
    # 1 psi = 6894.757 Pa
    np.testing.assert_allclose(pa_to_psi(6894.757), 1.0)
    np.testing.assert_allclose(psi_to_pa(1.0), 6894.757)
    # 1 atm ≈ 14.696 psi
    np.testing.assert_allclose(pa_to_psi(101325.0), 14.6959, rtol=1e-4)
    # Roundtrip preserves value
    np.testing.assert_allclose(psi_to_pa(pa_to_psi(101325.0)), 101325.0)


def test_kg_m3_g_cc():
    """Verify kg/m³ <-> g/cc density conversions."""
    # Scalar conversions
    np.testing.assert_allclose(kg_m3_to_g_cc(850.0), 0.85)
    np.testing.assert_allclose(kg_m3_to_g_cc(1000.0), 1.0)
    np.testing.assert_allclose(g_cc_to_kg_m3(0.85), 850.0)
    np.testing.assert_allclose(g_cc_to_kg_m3(1.0), 1000.0)
    # Array input in both directions
    np.testing.assert_allclose(
        kg_m3_to_g_cc(np.array([800.0, 850.0, 900.0, 1000.0])),
        [0.8, 0.85, 0.9, 1.0],
    )
    np.testing.assert_allclose(
        g_cc_to_kg_m3(np.array([0.8, 0.85, 0.9, 1.0])),
        [800.0, 850.0, 900.0, 1000.0],
    )
    # Roundtrip preserves value
    original = np.array([750.0, 800.0, 850.0, 900.0, 1000.0])
    np.testing.assert_allclose(g_cc_to_kg_m3(kg_m3_to_g_cc(original)), original)


def test_g_cc_api():
    """Verify g/cc <-> API gravity conversions using known oil industry references."""
    # Water at 1.0 g/cc = API 10
    np.testing.assert_allclose(g_cc_to_api(1.0), 10.0)
    np.testing.assert_allclose(api_to_g_cc(10.0), 1.0)
    # Light and heavy oil against the API formula
    np.testing.assert_allclose(g_cc_to_api(0.8), 141.5 / 0.8 - 131.5)
    np.testing.assert_allclose(g_cc_to_api(0.95), 141.5 / 0.95 - 131.5)
    # Array input in both directions
    densities = np.array([0.8, 0.85, 0.9, 0.95, 1.0])
    np.testing.assert_allclose(g_cc_to_api(densities), 141.5 / densities - 131.5)
    apis = np.array([10.0, 20.0, 30.0, 40.0])
    np.testing.assert_allclose(api_to_g_cc(apis), 141.5 / (apis + 131.5))
    # Roundtrip preserves value
    original = np.array([0.75, 0.80, 0.85, 0.90, 0.95, 1.0])
    np.testing.assert_allclose(api_to_g_cc(g_cc_to_api(original)), original)


def test_kg_m3_api():
    """Verify kg/m³ <-> API gravity conversions (combines kg/m³->g/cc->API)."""
    # Water at 1000 kg/m³ = API 10
    np.testing.assert_allclose(api_to_kg_m3(10.0), 1000.0)
    np.testing.assert_allclose(kg_m3_to_api(1000.0), 10.0)
    # Roundtrip preserves value
    np.testing.assert_allclose(kg_m3_to_api(api_to_kg_m3(30.0)), 30.0)


def test_velocity_km_s_m_s():
    """Verify km/s <-> m/s velocity conversions."""
    # 1 km/s = 1000 m/s
    np.testing.assert_allclose(km_per_s_to_m_per_s(1.0), 1000.0)
    np.testing.assert_allclose(m_per_s_to_km_per_s(1000.0), 1.0)
    # Array input with typical seismic velocities
    result = km_per_s_to_m_per_s(np.array([2.0, 3.5, 5.0]))
    np.testing.assert_allclose(result, [2000.0, 3500.0, 5000.0])
    # Roundtrip preserves value
    np.testing.assert_allclose(km_per_s_to_m_per_s(m_per_s_to_km_per_s(3500.0)), 3500.0)


def test_viscosity_cp_pa_s():
    """Verify centipoise <-> Pa*s viscosity conversions."""
    # 1 cP = 0.001 Pa*s
    np.testing.assert_allclose(cp_to_pa_s(1.0), 0.001)
    np.testing.assert_allclose(pa_s_to_cp(0.001), 1.0)
    # Roundtrip preserves value
    np.testing.assert_allclose(pa_s_to_cp(cp_to_pa_s(5.0)), 5.0)
