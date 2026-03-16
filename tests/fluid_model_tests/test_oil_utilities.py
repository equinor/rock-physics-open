import numpy as np

from rock_physics_open.fluid_models.oil_model.oil_utilities import (
    as_float_array,
    inputs_are_scalar,
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
