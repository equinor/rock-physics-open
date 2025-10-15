"""
Test configuration for machine learning utilities.

This conftest.py provides fixtures and configuration specific to pressure models
located in <project root>/tests/machine_learning_utilities_test/
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock

# Import all model classes for fixtures - adjusted for test directory structure
from rock_physics_open.equinor_utilities.machine_learning_utilities import (
    ExponentialPressureModel,
    PolynomialPressureModel,
    SigmoidalPressureModel,
    FriableDryBulkModulusPressureModel,
    FriableDryShearModulusPressureModel,
    PatchyCementDryBulkModulusPressureModel,
    PatchyCementDryShearModulusPressureModel,
)


# Test-specific pytest configuration
def pytest_configure(config):
    """Configure markers specific to machine learning utilities tests."""
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "pressure_model: marks tests related to pressure models")
    config.addinivalue_line("markers", "slow: marks tests as slow running")
    config.addinivalue_line("markers", "benchmark: marks performance benchmark tests")


def pytest_collection_modifyitems(config, items):
    """Add automatic markers based on test location and name."""
    for item in items:
        # Mark all tests in this directory as pressure_model tests
        item.add_marker(pytest.mark.pressure_model)

        # Add unit marker for most tests, integration for specific ones
        if "integration" in item.name or "end_to_end" in item.name:
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)

        # Mark slow tests
        if "benchmark" in item.name or "performance" in item.name or "stress" in item.name:
            item.add_marker(pytest.mark.slow)
            item.add_marker(pytest.mark.benchmark)


# Model fixtures with session scope for performance
@pytest.fixture(scope="session")
def exponential_model():
    """Create exponential model for testing."""
    return ExponentialPressureModel(
        a_factor=0.8,
        b_factor=2e7,
        model_max_pressure=3e7,
        description="Test exponential model"
    )


@pytest.fixture(scope="session")
def polynomial_model():
    """Create polynomial model for testing."""
    return PolynomialPressureModel(
        weights=[1000, 1e-7, -5e-15],
        model_max_pressure=3e7,
        description="Test polynomial model"
    )


@pytest.fixture(scope="session")
def sigmoidal_model():
    """Create sigmoidal model for testing."""
    return SigmoidalPressureModel(
        phi_amplitude=1500.0,
        phi_median_point=0.20,
        phi_x_scaling=10.0,
        phi_bias=2500.0,
        p_eff_median_point=1.5e7,
        p_eff_x_scaling=1e-7,
        p_eff_bias=1000.0,
        model_max_pressure=3e7,
        description="Test sigmoidal model"
    )


@pytest.fixture(scope="session")
def friable_bulk_model():
    """Create friable bulk modulus model for testing."""
    return FriableDryBulkModulusPressureModel(
        phi_c=0.4,
        coord_num_func="Porosity",
        n=None,
        shear_red=1.0,
        model_max_pressure=3e7,
        description="Test friable bulk model"
    )


@pytest.fixture(scope="session")
def friable_shear_model():
    """Create friable shear modulus model for testing."""
    return FriableDryShearModulusPressureModel(
        phi_c=0.4,
        coord_num_func="Porosity",
        n=None,
        shear_red=1.0,
        model_max_pressure=3e7,
        description="Test friable shear model"
    )


@pytest.fixture(scope="session")
def patchy_bulk_model():
    """Create patchy cement bulk modulus model for testing."""
    return PatchyCementDryBulkModulusPressureModel(
        frac_cem=0.1,
        phi_c=0.4,
        coord_num_func="Porosity",
        n=None,
        shear_red=1.0,
        model_max_pressure=3e7,
        description="Test patchy cement bulk model"
    )


@pytest.fixture(scope="session")
def patchy_shear_model():
    """Create patchy cement shear modulus model for testing."""
    return PatchyCementDryShearModulusPressureModel(
        frac_cem=0.1,
        phi_c=0.4,
        coord_num_func="Porosity",
        n=None,
        shear_red=1.0,
        model_max_pressure=3e7,
        description="Test patchy cement shear model"
    )


# Test data fixtures with deterministic random data
@pytest.fixture(scope="session")
def exp_poly_test_data():
    """Generate test data for exponential and polynomial models."""
    np.random.seed(42)  # Deterministic for reproducible tests
    n_samples = 100
    velocities = np.random.uniform(2500, 4000, n_samples)
    p_in_situ = np.random.uniform(1e7, 3e7, n_samples)
    p_depleted = np.random.uniform(0.5e7, 1.5e7, n_samples)
    return np.column_stack([velocities, p_in_situ, p_depleted])


@pytest.fixture(scope="session")
def sigmoidal_test_data():
    """Generate test data for sigmoidal model."""
    np.random.seed(42)
    n_samples = 100
    porosity = np.random.uniform(0.1, 0.35, n_samples)
    p_in_situ = np.random.uniform(1e7, 3e7, n_samples)
    p_depleted = np.random.uniform(0.5e7, 1.5e7, n_samples)
    return np.column_stack([porosity, p_in_situ, p_depleted])


@pytest.fixture(scope="session")
def friable_test_data():
    """Generate test data for friable models."""
    np.random.seed(42)
    n_samples = 50
    phi = np.random.uniform(0.1, 0.35, n_samples)
    k_min = np.random.uniform(35e9, 40e9, n_samples)
    mu_min = np.random.uniform(40e9, 45e9, n_samples)
    p_in_situ = np.random.uniform(1e7, 3e7, n_samples)
    p_depleted = np.random.uniform(0.5e7, 1.5e7, n_samples)
    return np.column_stack([phi, k_min, mu_min, p_in_situ, p_depleted])


@pytest.fixture(scope="session")
def patchy_test_data():
    """Generate test data for patchy cement models."""
    np.random.seed(42)
    n_samples = 50
    phi = np.random.uniform(0.1, 0.35, n_samples)
    k_min = np.random.uniform(35e9, 40e9, n_samples)
    mu_min = np.random.uniform(40e9, 45e9, n_samples)
    rho_min = np.random.uniform(2600, 2700, n_samples)
    k_cem = np.random.uniform(25e9, 30e9, n_samples)
    mu_cem = np.random.uniform(15e9, 20e9, n_samples)
    rho_cem = np.random.uniform(2200, 2300, n_samples)
    p_in_situ = np.random.uniform(1e7, 3e7, n_samples)
    p_depleted = np.random.uniform(0.5e7, 1.5e7, n_samples)
    return np.column_stack([phi, k_min, mu_min, rho_min, k_cem, mu_cem, rho_cem, p_in_situ, p_depleted])


# Mock fixtures for external rock physics functions
@pytest.fixture
def mock_friable_function():
    """Mock the friable_model_dry function."""
    with patch('rock_physics_open.sandstone_models.friable_models.friable_model_dry') as mock:
        def side_effect(k_min, mu_min, phi, p_eff, *args, **kwargs):
            # Return realistic bulk and shear modulus values
            k_dry = k_min * (1 - phi) * (1 + p_eff / 1e8)
            mu_dry = mu_min * (1 - phi) * (1 + p_eff / 1e8) * 0.6
            return k_dry, mu_dry

        mock.side_effect = side_effect
        yield mock


@pytest.fixture
def mock_patchy_function():
    """Mock the patchy_cement_model_dry function."""
    with patch('rock_physics_open.sandstone_models.patchy_cement_model.patchy_cement_model_dry') as mock:
        def side_effect(k_min, mu_min, rho_min, k_cem, mu_cem, rho_cem, phi, p_eff, *args, **kwargs):
            # Return realistic bulk modulus, shear modulus, and density values
            k_dry = k_min * (1 - phi) * (1 + p_eff / 1e8)
            mu_dry = mu_min * (1 - phi) * (1 + p_eff / 1e8) * 0.6
            rho_dry = rho_min * (1 - phi) + rho_cem * phi * 0.1
            return k_dry, mu_dry, rho_dry

        mock.side_effect = side_effect
        yield mock


# Invalid data fixtures for error testing
@pytest.fixture
def invalid_input_data():
    """Provide various invalid input formats for testing."""
    return {
        "wrong_type": [[1, 2, 3], [4, 5, 6]],  # List instead of ndarray
        "wrong_dimensions": np.array([1, 2, 3, 4, 5]),  # 1D instead of 2D
        "wrong_columns_exp": np.random.random((10, 4)),  # 4 columns instead of 3
        "wrong_columns_friable": np.random.random((10, 3)),  # 3 columns instead of 5
        "wrong_columns_patchy": np.random.random((10, 6)),  # 6 columns instead of 9
        "empty_array": np.empty((0, 3)),  # Empty array
    }


# Cleanup and utility fixtures
@pytest.fixture
def temp_file():
    """Create temporary file for testing save/load operations."""
    fd, path = tempfile.mkstemp(suffix='.pkl')
    os.close(fd)  # Close file descriptor, keep path
    yield path
    try:
        os.unlink(path)
    except (OSError, FileNotFoundError):
        pass  # Ignore cleanup errors


@pytest.fixture
def temp_directory():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        original_cwd = os.getcwd()
        os.chdir(tmp_dir)
        yield tmp_dir
        os.chdir(original_cwd)


@pytest.fixture(autouse=True)
def suppress_warnings():
    """Suppress expected warnings during testing."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        yield


# Performance testing utilities
@pytest.fixture(scope="session")
def performance_data_sizes():
    """Standard data sizes for performance testing."""
    return [100, 1000, 5000, 10000]


@pytest.fixture
def memory_monitor():
    """Monitor memory usage during tests."""
    try:
        import psutil
        process = psutil.Process()

        class MemoryMonitor:
            def __init__(self):
                self.initial_memory = process.memory_info().rss

            def get_current_usage(self):
                return process.memory_info().rss

            def get_delta(self):
                return process.memory_info().rss - self.initial_memory

        return MemoryMonitor()
    except ImportError:
        # Return mock if psutil not available
        class MockMemoryMonitor:
            def get_current_usage(self):
                return 0
            def get_delta(self):
                return 0

        return MockMemoryMonitor()


# Parametrized fixtures for comprehensive testing
@pytest.fixture(params=[
    "exponential_model",
    "polynomial_model",
    "sigmoidal_model"
])
def simple_model(request):
    """Parametrized fixture for models with simple input format (n,3)."""
    return request.getfixturevalue(request.param)


@pytest.fixture(params=[
    "friable_bulk_model",
    "friable_shear_model"
])
def friable_model(request):
    """Parametrized fixture for friable models."""
    return request.getfixturevalue(request.param)


@pytest.fixture(params=[
    "patchy_bulk_model",
    "patchy_shear_model"
])
def patchy_model(request):
    """Parametrized fixture for patchy cement models."""
    return request.getfixturevalue(request.param)


@pytest.fixture(params=[
    "exponential_model",
    "polynomial_model",
    "sigmoidal_model",
    "friable_bulk_model",
    "friable_shear_model",
    "patchy_bulk_model",
    "patchy_shear_model"
])
def any_model(request):
    """Parametrized fixture for all model types."""
    return request.getfixturevalue(request.param)