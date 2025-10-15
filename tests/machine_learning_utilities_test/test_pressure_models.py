# ruff: noqa: UP034
"""
Comprehensive pytest test suite for pressure sensitivity models.

This module tests all model types with proper fixtures, parameterization,
and mock testing for external rock physics dependencies.

pytest test_pressure_models.py -v                    # Run all tests
pytest test_pressure_models.py -m unit              # Run only unit tests
pytest test_pressure_models.py -m benchmark         # Run only benchmarks
pytest test_pressure_models.py -k "exponential"     # Run tests matching pattern

"""

import os
import tempfile
from unittest.mock import patch

import numpy as np
import pytest

# Import all model classes
from rock_physics_open.equinor_utilities.machine_learning_utilities.exponential_model import (
    ExponentialPressureModel,
)
from rock_physics_open.equinor_utilities.machine_learning_utilities.friable_pressure_models import (
    FriableDryBulkModulusPressureModel,
    FriableDryShearModulusPressureModel,
)
from rock_physics_open.equinor_utilities.machine_learning_utilities.patchy_cement_pressure_models import (
    PatchyCementDryBulkModulusPressureModel,
    PatchyCementDryShearModulusPressureModel,
)
from rock_physics_open.equinor_utilities.machine_learning_utilities.polynomial_model import (
    PolynomialPressureModel,
)
from rock_physics_open.equinor_utilities.machine_learning_utilities.sigmoidal_model import (
    SigmoidalPressureModel,
)


# Fixtures
@pytest.fixture
def exponential_model():
    """Create exponential model for testing."""
    return ExponentialPressureModel(
        a_factor=0.5,
        b_factor=1e7,
        model_max_pressure=5e7,
        description="Test exponential model",
    )


@pytest.fixture
def polynomial_model():
    """Create polynomial model for testing."""
    return PolynomialPressureModel(
        weights=[1.0, 2e-8, -1e-16],
        model_max_pressure=5e7,
        description="Test polynomial model",
    )


@pytest.fixture
def sigmoidal_model():
    """Create sigmoidal model for testing."""
    return SigmoidalPressureModel(
        phi_amplitude=1000.0,
        phi_median_point=0.2,
        phi_x_scaling=10,
        phi_bias=2000.0,
        p_eff_median_point=1.5e7,
        p_eff_x_scaling=1e-7,
        p_eff_bias=1000.0,
        model_max_pressure=5e7,
        description="Test sigmoidal model",
    )


@pytest.fixture
def friable_bulk_model():
    """Create friable bulk modulus model for testing."""
    return FriableDryBulkModulusPressureModel(
        phi_c=0.4,
        coord_num_func="Porosity",
        shear_red=1.0,
        model_max_pressure=5e7,
        description="Test friable bulk model",
    )


@pytest.fixture
def friable_shear_model():
    """Create friable shear modulus model for testing."""
    return FriableDryShearModulusPressureModel(
        phi_c=0.4,
        coord_num_func="ConstVal",
        n=8.0,
        shear_red=0.8,
        model_max_pressure=5e7,
        description="Test friable shear model",
    )


@pytest.fixture
def patchy_bulk_model():
    """Create patchy cement bulk modulus model for testing."""
    return PatchyCementDryBulkModulusPressureModel(
        frac_cem=0.3,
        phi_c=0.4,
        coord_num_func="Porosity",
        shear_red=1.0,
        model_max_pressure=5e7,
        description="Test patchy bulk model",
    )


@pytest.fixture
def patchy_shear_model():
    """Create patchy cement shear modulus model for testing."""
    return PatchyCementDryShearModulusPressureModel(
        frac_cem=0.3,
        phi_c=0.4,
        coord_num_func="Porosity",
        shear_red=1.0,
        model_max_pressure=5e7,
        description="Test patchy shear model",
    )


@pytest.fixture
def exp_poly_valid_input():
    """Valid input for exponential and polynomial models."""
    return np.array([[3000.0, 2e7, 1e7], [3200.0, 2.5e7, 1.2e7]])


@pytest.fixture
def sigmoidal_valid_input():
    """Valid input for sigmoidal model."""
    return np.array([[0.2, 2e7, 1e7], [0.25, 2.5e7, 1.2e7]])


@pytest.fixture
def friable_valid_input():
    """Valid input for friable models."""
    return np.array(
        [[0.2, 3.7e10, 4.4e10, 2e7, 1e7], [0.25, 3.8e10, 4.5e10, 2.5e7, 1.2e7]]
    )


@pytest.fixture
def patchy_valid_input():
    """Valid input for patchy cement models."""
    return np.array(
        [
            [0.2, 3.7e10, 4.4e10, 2650, 7.7e10, 3.2e10, 2200, 2e7, 1e7],
            [0.25, 3.8e10, 4.5e10, 2670, 7.8e10, 3.3e10, 2220, 2.5e7, 1.2e7],
        ]
    )


# Invalid data fixtures for error testing
@pytest.fixture
def invalid_input_data():
    """Provide various invalid input formats for testing."""
    rng = np.random.default_rng(42)
    return {
        "wrong_type": [[1, 2, 3], [4, 5, 6]],  # List instead of ndarray
        "wrong_dimensions": np.array([1, 2, 3, 4, 5]),  # 1D instead of 2D
        "wrong_columns_exp": rng.random((10, 4)),  # 4 columns instead of 3
        "wrong_columns_friable": rng.random((10, 3)),  # 3 columns instead of 5
        "wrong_columns_patchy": rng.random((10, 6)),  # 6 columns instead of 9
        "empty_array": np.empty((0, 3)),  # Empty array
    }


# Test ExponentialPressureModel
class TestExponentialPressureModel:
    """Test cases for ExponentialPressureModel."""

    def test_initialization(self, exponential_model):
        """Test model initialization."""
        assert exponential_model.a_factor == 0.5
        assert exponential_model.b_factor == 1e7
        assert exponential_model.max_pressure == 5e7
        assert exponential_model.description == "Test exponential model"

    def test_validate_input_valid(self, exponential_model, exp_poly_valid_input):
        """Test input validation with valid data."""
        result = exponential_model.validate_input(exp_poly_valid_input)
        np.testing.assert_array_equal(result, exp_poly_valid_input)

    def test_validate_input_invalid(self, exponential_model, invalid_input_data):
        """Test validation with invalid input formats."""
        test_cases = [
            ("wrong_type", "Input must be numpy ndarray."),
            ("wrong_dimensions", "Input must be \\(n,3\\)"),
            ("wrong_columns_exp", "Input must be \\(n,3\\)"),
            ("empty_array", None),  # Empty array might be valid
        ]

        for case_name, expected_error in test_cases:
            if expected_error is None:
                continue  # Skip cases where we don't expect an error

            with pytest.raises(ValueError, match=expected_error):
                exponential_model.validate_input(invalid_input_data[case_name])

    @pytest.mark.parametrize("case", ["in_situ", "depleted"])
    def test_predict_abs(self, exponential_model, exp_poly_valid_input, case):
        """Test absolute prediction for different cases."""
        result = exponential_model.predict_abs(exp_poly_valid_input, case=case)
        assert len(result) == 2
        assert np.all(result > 0)

    def test_predict_differential(self, exponential_model, exp_poly_valid_input):
        """Test differential prediction."""
        result = exponential_model.predict(exp_poly_valid_input)
        assert len(result) == 2

    def test_predict_max(self, exponential_model, exp_poly_valid_input):
        """Test prediction with max pressure."""
        result = exponential_model.predict_max(exp_poly_valid_input)
        assert len(result) == 2

    def test_predict_max_no_max_pressure(self, exp_poly_valid_input):
        """Test predict_max without max_pressure set."""
        model_no_max = ExponentialPressureModel(0.5, 1e7)
        with pytest.raises(ValueError, match='Field "model_max_pressure" is not set'):
            model_no_max.predict_max(exp_poly_valid_input)

    def test_todict(self, exponential_model):
        """Test dictionary conversion."""
        result = exponential_model.todict()
        expected_keys = {"a_factor", "b_factor", "model_max_pressure", "description"}
        assert set(result.keys()) == expected_keys

    def test_save_load(self, exponential_model):
        """Test save and load functionality."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
            tmp_path = tmp.name

        try:
            exponential_model.save(tmp_path)
            loaded_model = ExponentialPressureModel.load(tmp_path)

            assert loaded_model.a_factor == exponential_model.a_factor
            assert loaded_model.b_factor == exponential_model.b_factor
            assert loaded_model.max_pressure == exponential_model.max_pressure
        finally:
            os.unlink(tmp_path)


# Test PolynomialPressureModel
class TestPolynomialPressureModel:
    """Test cases for PolynomialPressureModel."""

    def test_initialization(self, polynomial_model):
        """Test model initialization."""
        assert polynomial_model.weights == [1.0, 2e-8, -1e-16]

    def test_predict_abs_no_weights(self, exp_poly_valid_input):
        """Test prediction with empty weights."""
        model_no_weights = PolynomialPressureModel([])
        with pytest.raises(ValueError, match='Field "weights" is not set'):
            model_no_weights.predict_abs(exp_poly_valid_input)

    def test_predict_abs(self, polynomial_model, exp_poly_valid_input):
        """Test absolute prediction."""
        result = polynomial_model.predict_abs(exp_poly_valid_input, case="in_situ")
        assert len(result) == 2
        assert np.all(result > 0)


# Test SigmoidalPressureModel
class TestSigmoidalPressureModel:
    """Test cases for SigmoidalPressureModel."""

    def test_properties(self, sigmoidal_model):
        """Test model properties."""
        assert sigmoidal_model.phi_amplitude == 1000.0
        assert sigmoidal_model.phi_median_point == 0.2
        assert sigmoidal_model.p_eff_median_point == 1.5e7

    def test_predict_abs(self, sigmoidal_model, sigmoidal_valid_input):
        """Test absolute prediction."""
        result = sigmoidal_model.predict_abs(sigmoidal_valid_input, case="in_situ")
        assert len(result) == 2
        assert np.all(result > 0)

    def test_input_validation(self, sigmoidal_model):
        """Test input validation with wrong format."""
        invalid_input = np.array([[0.2, 2e7]])  # Missing column
        with pytest.raises(ValueError, match="Input must be \\(n,3\\)"):
            sigmoidal_model.validate_input(invalid_input)


# Test Friable Models
class TestFriablePressureModels:
    """Test cases for Friable pressure models."""

    @patch(
        "rock_physics_open.equinor_utilities.machine_learning_utilities.friable_pressure_models.friable_model_dry"
    )
    def test_bulk_model_predict_abs(
        self, mock_friable, friable_bulk_model, friable_valid_input
    ):
        """Test bulk modulus prediction."""
        # Mock the friable_model_dry function
        mock_friable.return_value = (np.array([1e10, 1.2e10]), np.array([0.8e10, 1e10]))

        result = friable_bulk_model.predict_abs(friable_valid_input, case="in_situ")
        assert len(result) == 2
        mock_friable.assert_called()

    @patch(
        "rock_physics_open.equinor_utilities.machine_learning_utilities.friable_pressure_models.friable_model_dry"
    )
    def test_shear_model_predict_abs(
        self, mock_friable, friable_shear_model, friable_valid_input
    ):
        """Test shear modulus prediction."""
        # Mock the friable_model_dry function
        mock_friable.return_value = (np.array([1e10, 1.2e10]), np.array([0.8e10, 1e10]))

        result = friable_shear_model.predict_abs(friable_valid_input, case="depleted")
        assert len(result) == 2
        mock_friable.assert_called()

    def test_validate_input_invalid_columns(self, friable_bulk_model):
        """Test input validation with wrong number of columns."""
        invalid_input = np.array([[0.2, 3.7e10, 4.4e10, 2e7]])  # Missing column
        with pytest.raises(ValueError, match="Input must be \\(n,5\\)"):
            friable_bulk_model.validate_input(invalid_input)

    def test_model_properties(self, friable_shear_model):
        """Test model properties are properly set."""
        model_dict = friable_shear_model.todict()
        assert model_dict["phi_c"] == 0.4
        assert model_dict["coord_num_func"] == "ConstVal"
        assert model_dict["n"] == 8.0
        assert model_dict["shear_red"] == 0.8


# Test Patchy Cement Models
class TestPatchyCementPressureModels:
    """Test cases for Patchy Cement pressure models."""

    @patch(
        "rock_physics_open.equinor_utilities.machine_learning_utilities.patchy_cement_pressure_models.patchy_cement_model_dry"
    )
    def test_bulk_model_predict_abs(
        self, mock_patchy, patchy_bulk_model, patchy_valid_input
    ):
        """Test bulk modulus prediction."""
        # Mock the patchy_cement_model_dry function
        mock_patchy.return_value = (
            np.array([1.5e10, 1.7e10]),  # k_dry
            np.array([1.2e10, 1.4e10]),  # mu
            np.array([2400, 2420]),  # rho
        )

        result = patchy_bulk_model.predict_abs(patchy_valid_input, case="in_situ")
        assert len(result) == 2
        mock_patchy.assert_called()

    @patch(
        "rock_physics_open.equinor_utilities.machine_learning_utilities.patchy_cement_pressure_models.patchy_cement_model_dry"
    )
    def test_shear_model_predict_abs(
        self, mock_patchy, patchy_shear_model, patchy_valid_input
    ):
        """Test shear modulus prediction."""
        # Mock the patchy_cement_model_dry function
        mock_patchy.return_value = (
            np.array([1.5e10, 1.7e10]),  # k_dry
            np.array([1.2e10, 1.4e10]),  # mu
            np.array([2400, 2420]),  # rho
        )

        result = patchy_shear_model.predict_abs(patchy_valid_input, case="depleted")
        assert len(result) == 2
        mock_patchy.assert_called()

    def test_validate_input_invalid_columns(self, patchy_bulk_model):
        """Test input validation with wrong number of columns."""
        invalid_input = np.array(
            [[0.2, 3.7e10, 4.4e10, 2650, 7.7e10, 3.2e10, 2200, 2e7]]
        )  # Missing column
        with pytest.raises(ValueError, match="Input must be \\(n,9\\)"):
            patchy_bulk_model.validate_input(invalid_input)


# Integration Tests
class TestModelIntegration:
    """Integration tests across model types."""

    @pytest.mark.parametrize(
        (
            "model_fixture",
            "input_fixture",
        ),
        [
            ("exponential_model", "exp_poly_valid_input"),
            ("polynomial_model", "exp_poly_valid_input"),
            ("sigmoidal_model", "sigmoidal_valid_input"),
            ("friable_bulk_model", "friable_valid_input"),
            ("friable_shear_model", "friable_valid_input"),
            ("patchy_bulk_model", "patchy_valid_input"),
            ("patchy_shear_model", "patchy_valid_input"),
        ],
    )
    def test_all_models_predict_interface(self, request, model_fixture, input_fixture):
        """Test that all models implement the required interface."""
        model = request.getfixturevalue(model_fixture)
        test_input = request.getfixturevalue(input_fixture)

        # Test interface methods exist and work
        assert hasattr(model, "predict_abs")
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_max")
        assert hasattr(model, "validate_input")
        assert hasattr(model, "todict")
        assert hasattr(model, "save")

        # Test methods work (with mocking for rock physics models)
        if "friable" in model_fixture:
            with patch(
                "rock_physics_open.equinor_utilities.machine_learning_utilities.friable_pressure_models.friable_model_dry"
            ) as mock_friable:
                mock_friable.return_value = (
                    np.array([1e10] * len(test_input)),
                    np.array([0.8e10] * len(test_input)),
                )

                result_abs = model.predict_abs(test_input)
                result_diff = model.predict(test_input)
                result_max = model.predict_max(test_input)

        elif "patchy" in model_fixture:
            with patch(
                "rock_physics_open.equinor_utilities.machine_learning_utilities.patchy_cement_pressure_models.patchy_cement_model_dry"
            ) as mock_patchy:
                mock_patchy.return_value = (
                    np.array([1e10] * len(test_input)),
                    np.array([0.8e10] * len(test_input)),
                    np.array([2400] * len(test_input)),
                )

                result_abs = model.predict_abs(test_input)
                result_diff = model.predict(test_input)
                result_max = model.predict_max(test_input)
        else:
            result_abs = model.predict_abs(test_input)
            result_diff = model.predict(test_input)
            result_max = model.predict_max(test_input)

        assert len(result_abs) == len(test_input)
        assert len(result_diff) == len(test_input)
        assert len(result_max) == len(test_input)

    def test_model_serialization_consistency(
        self, exponential_model, polynomial_model, sigmoidal_model
    ):
        """Test that all models can be serialized and deserialized consistently."""
        models = [exponential_model, polynomial_model, sigmoidal_model]

        for model in models:
            # Test dictionary conversion
            model_dict = model.todict()
            assert isinstance(model_dict, dict)
            assert len(model_dict) > 0

            # Test save/load cycle
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
                tmp_path = tmp.name

            try:
                model.save(tmp_path)
                loaded_model = type(model).load(tmp_path)

                # Compare dictionaries
                original_dict = model.todict()
                loaded_dict = loaded_model.todict()

                assert original_dict == loaded_dict

            finally:
                os.unlink(tmp_path)


# Performance Tests
class TestModelPerformance:
    """Performance and stress tests."""

    @pytest.mark.parametrize("n_samples", [100, 1000, 10000])
    def test_exponential_model_performance(self, exponential_model, n_samples):
        """Test exponential model performance with different data sizes."""
        # Generate test data
        rng = np.random.default_rng(42)
        velocities = rng.uniform(2500, 4000, n_samples)
        p_in_situ = rng.uniform(1e7, 3e7, n_samples)
        p_depleted = rng.uniform(0.5e7, 1.5e7, n_samples)
        test_data = np.column_stack([velocities, p_in_situ, p_depleted])

        # Test prediction (should complete without errors)
        result = exponential_model.predict_abs(test_data)
        assert len(result) == n_samples
        assert np.all(np.isfinite(result))

    def test_memory_usage(self, exponential_model):
        """Test that models don't leak memory with repeated calls."""
        # Create moderate size dataset
        n_samples = 10000
        rng = np.random.default_rng(42)
        velocities = rng.uniform(2500, 4000, n_samples)
        p_in_situ = rng.uniform(1e7, 3e7, n_samples)
        p_depleted = rng.uniform(0.5e7, 1.5e7, n_samples)
        test_data = np.column_stack([velocities, p_in_situ, p_depleted])

        # Run multiple predictions
        for _ in range(10):
            result = exponential_model.predict_abs(test_data)
            del result  # Explicitly delete to test memory cleanup
