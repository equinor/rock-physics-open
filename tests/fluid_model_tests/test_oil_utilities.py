import numpy as np

from rock_physics_open.fluid_models.oil_model.oil_utilities import (
    as_float_array,
    inputs_are_scalar,
    oil_api_to_density,
    oil_density_to_api,
    oil_density_to_gcc,
    oil_density_to_kg_m_3,
)


class TestAsFloatArray:
    def test_scalar_input(self):
        result = as_float_array(3.14)
        assert result.dtype == np.float64
        assert result.ndim >= 1
        np.testing.assert_array_equal(result, [3.14])

    def test_integer_input(self):
        result = as_float_array(5)
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, [5.0])

    def test_float_array_input(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = as_float_array(arr)
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_single_element_array(self):
        arr = np.array([42.0], dtype=np.float64)
        result = as_float_array(arr)
        assert result.dtype == np.float64
        assert result.ndim >= 1
        np.testing.assert_array_equal(result, [42.0])

    def test_already_float64(self):
        arr = np.array([1.5, 2.5], dtype=np.float64)
        result = as_float_array(arr)
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, [1.5, 2.5])


class TestInputsAreScalar:
    def test_all_scalars(self):
        assert inputs_are_scalar(1.0, 2.0, 3.0) is True

    def test_mixed_scalar_and_array(self):
        assert inputs_are_scalar(1.0, np.array([2.0, 3.0])) is False

    def test_all_arrays(self):
        assert inputs_are_scalar(np.array([1.0]), np.array([2.0])) is False

    def test_single_scalar(self):
        assert inputs_are_scalar(42.0) is True

    def test_zero_dim_array(self):
        assert inputs_are_scalar(np.float64(1.0)) is True


class TestOilDensityToApi:
    def test_water_reference(self):
        """Water has API gravity of 10 at density 1.0 g/cc."""
        result = oil_density_to_api(1.0)
        np.testing.assert_allclose(result, 10.0)

    def test_light_oil(self):
        """Light oil at 0.8 g/cc should give ~45.4 API."""
        result = oil_density_to_api(0.8)
        expected = 141.5 / 0.8 - 131.5
        np.testing.assert_allclose(result, expected)

    def test_heavy_oil(self):
        """Heavy oil at 0.95 g/cc."""
        result = oil_density_to_api(0.95)
        expected = 141.5 / 0.95 - 131.5
        np.testing.assert_allclose(result, expected)

    def test_array_input(self):
        densities = np.array([0.8, 0.85, 0.9, 0.95, 1.0])
        result = oil_density_to_api(densities)
        expected = 141.5 / densities - 131.5
        np.testing.assert_allclose(result, expected)

    def test_returns_ndarray(self):
        result = oil_density_to_api(0.85)
        assert isinstance(result, np.ndarray)


class TestOilApiToDensity:
    def test_water_reference(self):
        """API 10 corresponds to water density 1.0 g/cc."""
        result = oil_api_to_density(10.0)
        np.testing.assert_allclose(result, 1.0)

    def test_light_oil(self):
        """API ~45.4 corresponds to 0.8 g/cc."""
        api = 141.5 / 0.8 - 131.5
        result = oil_api_to_density(api)
        np.testing.assert_allclose(result, 0.8)

    def test_heavy_oil(self):
        """API gravity for heavy oil at 0.95 g/cc."""
        api = 141.5 / 0.95 - 131.5
        result = oil_api_to_density(api)
        np.testing.assert_allclose(result, 0.95)

    def test_array_input(self):
        apis = np.array([10.0, 20.0, 30.0, 40.0])
        result = oil_api_to_density(apis)
        expected = 141.5 / (apis + 131.5)
        np.testing.assert_allclose(result, expected)

    def test_returns_ndarray(self):
        result = oil_api_to_density(30.0)
        assert isinstance(result, np.ndarray)


class TestRoundTripApiDensity:
    def test_density_roundtrip(self):
        """Converting density -> API -> density should return the original."""
        original = np.array([0.75, 0.80, 0.85, 0.90, 0.95, 1.0])
        api = oil_density_to_api(original)
        recovered = oil_api_to_density(api)
        np.testing.assert_allclose(recovered, original)

    def test_api_roundtrip(self):
        """Converting API -> density -> API should return the original."""
        original = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        density = oil_api_to_density(original)
        recovered = oil_density_to_api(density)
        np.testing.assert_allclose(recovered, original)

    def test_scalar_roundtrip(self):
        density = 0.87
        np.testing.assert_allclose(
            oil_api_to_density(oil_density_to_api(density)), density
        )


class TestOilDensityToGcc:
    def test_typical_oil(self):
        """850 kg/m³ = 0.85 g/cc."""
        result = oil_density_to_gcc(850.0)
        np.testing.assert_allclose(result, 0.85)

    def test_water(self):
        """1000 kg/m³ = 1.0 g/cc."""
        result = oil_density_to_gcc(1000.0)
        np.testing.assert_allclose(result, 1.0)

    def test_zero(self):
        result = oil_density_to_gcc(0.0)
        np.testing.assert_allclose(result, 0.0)

    def test_array_input(self):
        densities_kg = np.array([800.0, 850.0, 900.0, 1000.0])
        result = oil_density_to_gcc(densities_kg)
        expected = np.array([0.8, 0.85, 0.9, 1.0])
        np.testing.assert_allclose(result, expected)

    def test_returns_ndarray(self):
        result = oil_density_to_gcc(850.0)
        assert isinstance(result, np.ndarray)


class TestOilDensityToKgM3:
    def test_typical_oil(self):
        """0.85 g/cc = 850 kg/m³."""
        result = oil_density_to_kg_m_3(0.85)
        np.testing.assert_allclose(result, 850.0)

    def test_water(self):
        """1.0 g/cc = 1000 kg/m³."""
        result = oil_density_to_kg_m_3(1.0)
        np.testing.assert_allclose(result, 1000.0)

    def test_zero(self):
        result = oil_density_to_kg_m_3(0.0)
        np.testing.assert_allclose(result, 0.0)

    def test_array_input(self):
        densities_gcc = np.array([0.8, 0.85, 0.9, 1.0])
        result = oil_density_to_kg_m_3(densities_gcc)
        expected = np.array([800.0, 850.0, 900.0, 1000.0])
        np.testing.assert_allclose(result, expected)

    def test_returns_ndarray(self):
        result = oil_density_to_kg_m_3(0.85)
        assert isinstance(result, np.ndarray)


class TestRoundTripGccKgM3:
    def test_gcc_roundtrip(self):
        """Converting kg/m³ -> g/cc -> kg/m³ should return the original."""
        original = np.array([750.0, 800.0, 850.0, 900.0, 1000.0])
        gcc = oil_density_to_gcc(original)
        recovered = oil_density_to_kg_m_3(gcc)
        np.testing.assert_allclose(recovered, original)

    def test_kg_m3_roundtrip(self):
        """Converting g/cc -> kg/m³ -> g/cc should return the original."""
        original = np.array([0.75, 0.80, 0.85, 0.90, 1.0])
        kg_m3 = oil_density_to_kg_m_3(original)
        recovered = oil_density_to_gcc(kg_m3)
        np.testing.assert_allclose(recovered, original)
