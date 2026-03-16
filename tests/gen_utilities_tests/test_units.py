import numpy as np

from rock_physics_open.equinor_utilities.units import (
    api_to_g_cc,
    api_to_kg_m3,
    celsius_to_kelvin,
    cp_to_pa_s,
    g_cc_to_api,
    g_cc_to_kg_m3,
    kg_m3_to_api,
    kg_m3_to_g_cc,
    km_per_s_to_m_per_s,
    m_per_s_to_km_per_s,
    mpa_to_pa,
    pa_s_to_cp,
    pa_to_mpa,
    pa_to_torr,
    torr_to_pa,
)


class TestCelsiusToKelvin:
    def test_zero_celsius(self) -> None:
        np.testing.assert_allclose(celsius_to_kelvin(0.0), 273.15)

    def test_boiling_point(self) -> None:
        np.testing.assert_allclose(celsius_to_kelvin(100.0), 373.15)

    def test_absolute_zero(self) -> None:
        np.testing.assert_allclose(celsius_to_kelvin(-273.15), 0.0, atol=1e-10)

    def test_array(self) -> None:
        result = celsius_to_kelvin(np.array([0.0, 100.0, -40.0]))
        np.testing.assert_allclose(result, [273.15, 373.15, 233.15])


class TestPressureConversions:
    def test_pa_to_mpa(self) -> None:
        np.testing.assert_allclose(pa_to_mpa(1e6), 1.0)

    def test_mpa_to_pa(self) -> None:
        np.testing.assert_allclose(mpa_to_pa(1.0), 1e6)

    def test_pa_mpa_roundtrip(self) -> None:
        np.testing.assert_allclose(mpa_to_pa(pa_to_mpa(5e7)), 5e7)

    def test_pa_to_torr(self) -> None:
        np.testing.assert_allclose(pa_to_torr(133.3224), 1.0)

    def test_torr_to_pa(self) -> None:
        np.testing.assert_allclose(torr_to_pa(1.0), 133.3224)

    def test_pa_torr_roundtrip(self) -> None:
        np.testing.assert_allclose(torr_to_pa(pa_to_torr(101325.0)), 101325.0)

    def test_array_pa_mpa(self) -> None:
        result = pa_to_mpa(np.array([1e6, 2e6, 5e6]))
        np.testing.assert_allclose(result, [1.0, 2.0, 5.0])


class TestGccToApi:
    def test_water_reference(self) -> None:
        """Water has API gravity of 10 at density 1.0 g/cc."""
        np.testing.assert_allclose(g_cc_to_api(1.0), 10.0)

    def test_light_oil(self) -> None:
        """Light oil at 0.8 g/cc should give ~45.4 API."""
        np.testing.assert_allclose(g_cc_to_api(0.8), 141.5 / 0.8 - 131.5)

    def test_heavy_oil(self) -> None:
        np.testing.assert_allclose(g_cc_to_api(0.95), 141.5 / 0.95 - 131.5)

    def test_array_input(self) -> None:
        densities = np.array([0.8, 0.85, 0.9, 0.95, 1.0])
        np.testing.assert_allclose(g_cc_to_api(densities), 141.5 / densities - 131.5)

    def test_returns_ndarray(self) -> None:
        assert isinstance(g_cc_to_api(np.array([0.85])), np.ndarray)


class TestApiToGcc:
    def test_water_reference(self) -> None:
        """API 10 corresponds to water density 1.0 g/cc."""
        np.testing.assert_allclose(api_to_g_cc(10.0), 1.0)

    def test_light_oil(self) -> None:
        api = 141.5 / 0.8 - 131.5
        np.testing.assert_allclose(api_to_g_cc(api), 0.8)

    def test_heavy_oil(self) -> None:
        api = 141.5 / 0.95 - 131.5
        np.testing.assert_allclose(api_to_g_cc(api), 0.95)

    def test_array_input(self) -> None:
        apis = np.array([10.0, 20.0, 30.0, 40.0])
        np.testing.assert_allclose(api_to_g_cc(apis), 141.5 / (apis + 131.5))

    def test_returns_ndarray(self) -> None:
        assert isinstance(api_to_g_cc(np.array([30.0])), np.ndarray)


class TestRoundTripApiGcc:
    def test_density_roundtrip(self) -> None:
        original = np.array([0.75, 0.80, 0.85, 0.90, 0.95, 1.0])
        np.testing.assert_allclose(api_to_g_cc(g_cc_to_api(original)), original)

    def test_api_roundtrip(self) -> None:
        original = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        np.testing.assert_allclose(g_cc_to_api(api_to_g_cc(original)), original)

    def test_scalar_roundtrip(self) -> None:
        np.testing.assert_allclose(api_to_g_cc(g_cc_to_api(0.87)), 0.87)


class TestApiToKgM3:
    def test_water_reference(self) -> None:
        np.testing.assert_allclose(api_to_kg_m3(10.0), 1000.0)

    def test_kg_m3_to_api_water(self) -> None:
        np.testing.assert_allclose(kg_m3_to_api(1000.0), 10.0)

    def test_roundtrip(self) -> None:
        np.testing.assert_allclose(kg_m3_to_api(api_to_kg_m3(30.0)), 30.0)


class TestKgM3ToGcc:
    def test_typical_oil(self) -> None:
        np.testing.assert_allclose(kg_m3_to_g_cc(850.0), 0.85)

    def test_water(self) -> None:
        np.testing.assert_allclose(kg_m3_to_g_cc(1000.0), 1.0)

    def test_zero(self) -> None:
        np.testing.assert_allclose(kg_m3_to_g_cc(0.0), 0.0)

    def test_array_input(self) -> None:
        np.testing.assert_allclose(
            kg_m3_to_g_cc(np.array([800.0, 850.0, 900.0, 1000.0])),
            [0.8, 0.85, 0.9, 1.0],
        )

    def test_returns_ndarray(self) -> None:
        assert isinstance(kg_m3_to_g_cc(np.array([850.0])), np.ndarray)


class TestGccToKgM3:
    def test_typical_oil(self) -> None:
        np.testing.assert_allclose(g_cc_to_kg_m3(0.85), 850.0)

    def test_water(self) -> None:
        np.testing.assert_allclose(g_cc_to_kg_m3(1.0), 1000.0)

    def test_zero(self) -> None:
        np.testing.assert_allclose(g_cc_to_kg_m3(0.0), 0.0)

    def test_array_input(self) -> None:
        np.testing.assert_allclose(
            g_cc_to_kg_m3(np.array([0.8, 0.85, 0.9, 1.0])),
            [800.0, 850.0, 900.0, 1000.0],
        )

    def test_returns_ndarray(self) -> None:
        assert isinstance(g_cc_to_kg_m3(np.array([0.85])), np.ndarray)


class TestRoundTripGccKgM3:
    def test_gcc_roundtrip(self) -> None:
        original = np.array([750.0, 800.0, 850.0, 900.0, 1000.0])
        np.testing.assert_allclose(g_cc_to_kg_m3(kg_m3_to_g_cc(original)), original)

    def test_kg_m3_roundtrip(self) -> None:
        original = np.array([0.75, 0.80, 0.85, 0.90, 1.0])
        np.testing.assert_allclose(kg_m3_to_g_cc(g_cc_to_kg_m3(original)), original)


class TestVelocityConversions:
    def test_km_s_to_m_s(self) -> None:
        np.testing.assert_allclose(km_per_s_to_m_per_s(1.0), 1000.0)

    def test_m_s_to_km_s(self) -> None:
        np.testing.assert_allclose(m_per_s_to_km_per_s(1000.0), 1.0)

    def test_velocity_roundtrip(self) -> None:
        np.testing.assert_allclose(
            km_per_s_to_m_per_s(m_per_s_to_km_per_s(3500.0)), 3500.0
        )

    def test_array_velocity(self) -> None:
        result = km_per_s_to_m_per_s(np.array([2.0, 3.5, 5.0]))
        np.testing.assert_allclose(result, [2000.0, 3500.0, 5000.0])


class TestViscosityConversions:
    def test_cp_to_pa_s(self) -> None:
        np.testing.assert_allclose(cp_to_pa_s(1.0), 0.001)

    def test_pa_s_to_cp(self) -> None:
        np.testing.assert_allclose(pa_s_to_cp(0.001), 1.0)

    def test_viscosity_roundtrip(self) -> None:
        np.testing.assert_allclose(pa_s_to_cp(cp_to_pa_s(5.0)), 5.0)
