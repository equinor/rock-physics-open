"""
Performance benchmark tests for pressure sensitivity models using pytest.

These tests measure execution time and memory usage of different model types.

pytest test_benchmarks.py --tb=short                # Run benchmarks with short traceback
"""

import pytest
import time
import numpy as np
import psutil
import os
from typing import Dict, List

from rock_physics_open.equinor_utilities.machine_learning_utilities.exponential_model import \
    ExponentialPressureModel
from rock_physics_open.equinor_utilities.machine_learning_utilities.polynomial_model import PolynomialPressureModel
from rock_physics_open.equinor_utilities.machine_learning_utilities.sigmoidal_model import SigmoidalPressureModel


@pytest.fixture(scope="session")
def benchmark_models():
    """Create models for benchmarking."""
    return {
        "Exponential": ExponentialPressureModel(0.5, 1e7),
        "Polynomial": PolynomialPressureModel([1.0, 2e-8, -1e-16]),
        "Sigmoidal": SigmoidalPressureModel(1000, 0.2, 10, 2000, 1.5e7, 1e-7, 1000)
    }


@pytest.fixture(scope="session", params=[1000, 10000, 100000])
def sample_sizes(request):
    """Sample sizes for benchmarking."""
    return request.param


def create_test_data(n_samples: int, model_type: str) -> np.ndarray:
    """Create test data for specified model type."""
    np.random.seed(42)  # For reproducible results

    if model_type in ["Exponential", "Polynomial"]:
        velocities = np.random.uniform(2500, 4000, n_samples)
        p_in_situ = np.random.uniform(1e7, 3e7, n_samples)
        p_depleted = np.random.uniform(0.5e7, 1.5e7, n_samples)
        return np.column_stack([velocities, p_in_situ, p_depleted])

    elif model_type == "Sigmoidal":
        porosities = np.random.uniform(0.1, 0.35, n_samples)
        p_in_situ = np.random.uniform(1e7, 3e7, n_samples)
        p_depleted = np.random.uniform(0.5e7, 1.5e7, n_samples)
        return np.column_stack([porosities, p_in_situ, p_depleted])

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def measure_performance(model, data: np.ndarray, model_name: str) -> Dict[str, float]:
    """Measure performance metrics for a model."""
    # Get initial memory usage
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024  # MB

    # Warm up
    for _ in range(3):
        _ = model.predict_abs(data[:10])

    # Benchmark predict_abs
    start_time = time.perf_counter()
    for _ in range(10):
        result = model.predict_abs(data)
    predict_abs_time = (time.perf_counter() - start_time) / 10

    # Benchmark predict (differential)
    start_time = time.perf_counter()
    for _ in range(10):
        result = model.predict(data)
    predict_diff_time = (time.perf_counter() - start_time) / 10

    # Get final memory usage
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    memory_usage = mem_after - mem_before

    return {
        'predict_abs_time_ms': predict_abs_time * 1000,
        'predict_diff_time_ms': predict_diff_time * 1000,
        'memory_usage_mb': max(0, memory_usage),  # Ensure non-negative
        'samples_per_sec': len(data) / predict_abs_time
    }


@pytest.mark.benchmark
@pytest.mark.parametrize("model_name", ["Exponential", "Polynomial", "Sigmoidal"])
class TestModelBenchmarks:
    """Benchmark tests for all model types."""

    def test_model_performance(self, benchmark_models, model_name, sample_sizes):
        """Test model performance across different sample sizes."""
        model = benchmark_models[model_name]
        data = create_test_data(sample_sizes, model_name)

        metrics = measure_performance(model, data, model_name)

        # Performance assertions (adjust thresholds as needed)
        assert metrics['predict_abs_time_ms'] < 10000, f"predict_abs too slow for {model_name}"
        assert metrics['predict_diff_time_ms'] < 20000, f"predict too slow for {model_name}"
        assert metrics['samples_per_sec'] > 100, f"Throughput too low for {model_name}"

        # Print results for monitoring
        print(f"\n{model_name} Model - {sample_sizes:,} samples:")
        print(f"  predict_abs: {metrics['predict_abs_time_ms']:.2f}ms")
        print(f"  predict_diff: {metrics['predict_diff_time_ms']:.2f}ms")
        print(f"  throughput: {metrics['samples_per_sec']:.0f} samples/sec")
        print(f"  memory: {metrics['memory_usage_mb']:.2f}MB")

    def test_memory_efficiency(self, benchmark_models, model_name):
        """Test that models don't consume excessive memory."""
        model = benchmark_models[model_name]

        # Create large dataset
        data = create_test_data(50000, model_name)

        # Monitor memory during prediction
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024

        result = model.predict_abs(data)

        mem_after = process.memory_info().rss / 1024 / 1024
        memory_increase = mem_after - mem_before

        # Memory usage should be reasonable (less than 100MB for 50k samples)
        assert memory_increase < 100, f"Excessive memory usage: {memory_increase:.2f}MB"

        # Clean up
        del result
        del data

    def test_numerical_stability(self, benchmark_models, model_name):
        """Test numerical stability with extreme values."""
        model = benchmark_models[model_name]

        if model_name in ["Exponential", "Polynomial"]:
            # Test with extreme velocities and pressures
            extreme_data = np.array([
                [100.0, 1e9, 1e5],  # Very high pressure, very low velocity
                [10000.0, 1e5, 1e9],  # Very low pressure, very high velocity
                [3000.0, 1e8, 1e8]  # Equal pressures
            ])
        else:  # Sigmoidal
            extreme_data = np.array([
                [0.01, 1e9, 1e5],  # Very low porosity
                [0.99, 1e5, 1e9],  # Very high porosity
                [0.5, 1e8, 1e8]  # Equal pressures
            ])

        # Should not raise exceptions or produce NaN/inf
        result = model.predict_abs(extreme_data)
        assert np.all(np.isfinite(result)), f"Non-finite results for {model_name}"

        result_diff = model.predict(extreme_data)
        assert np.all(np.isfinite(result_diff)), f"Non-finite differential results for {model_name}"


@pytest.mark.benchmark
class TestComparativePerformance:
    """Compare performance across different model types."""

    def test_model_comparison(self, benchmark_models):
        """Compare performance across all model types."""
        sample_size = 10000
        results = {}

        for model_name, model in benchmark_models.items():
            data = create_test_data(sample_size, model_name)
            metrics = measure_performance(model, data, model_name)
            results[model_name] = metrics

        # Print comparison
        print(f"\nPerformance Comparison ({sample_size:,} samples):")
        print("-" * 60)
        print(f"{'Model':<12} {'Abs(ms)':<10} {'Diff(ms)':<11} {'Samples/s':<12}")
        print("-" * 60)

        for model_name, metrics in results.items():
            print(f"{model_name:<12} "
                  f"{metrics['predict_abs_time_ms']:<10.2f} "
                  f"{metrics['predict_diff_time_ms']:<11.2f} "
                  f"{metrics['samples_per_sec']:<12.0f}")

        # Find fastest model
        fastest_abs = min(results.items(), key=lambda x: x[1]['predict_abs_time_ms'])
        fastest_diff = min(results.items(), key=lambda x: x[1]['predict_diff_time_ms'])

        print(f"\nFastest predict_abs: {fastest_abs[0]}")
        print(f"Fastest predict: {fastest_diff[0]}")


if __name__ == '__main__':
    # Run benchmarks with specific markers
    pytest.main([__file__, "-v", "-m", "benchmark", "--tb=short"])