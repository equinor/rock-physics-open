import numpy as np
from syrupy.assertion import SnapshotAssertion

from rock_physics_open.fluid_models.brine_model import (
    brine_properties,
)
from rock_physics_open.fluid_models.brine_model.brine_properties import (
    brine_density,
    brine_primary_velocity,
    water,
    water_density,
    water_primary_velocity,
)

pres = 23.0e6 * np.linspace(0.8, 1.2, 101)
temp = 100.0 * np.linspace(0.8, 1.2, 101)
sal = 35000.0 * np.ones(101)


def test_brine_properties(snapshot: SnapshotAssertion):
    assert snapshot == brine_properties(temp, pres, sal)


def test_water_properties(snapshot: SnapshotAssertion):
    assert snapshot == water(temp, pres)


def test_water_density(snapshot: SnapshotAssertion):
    assert snapshot == water_density(temp, pres)


def test_water_velocity(snapshot: SnapshotAssertion):
    assert snapshot == water_primary_velocity(temp, pres)


def test_brine_density(snapshot: SnapshotAssertion):
    assert snapshot == brine_density(temp, pres, sal)


def test_brine_velocity(snapshot: SnapshotAssertion):
    assert snapshot == brine_primary_velocity(temp, pres, sal)


def test_water_brine_properties_float():
    """Make sure that input object type is reflected in output type."""
    for results in [
        water(temp[0], pres[0]),
        water_density(temp[0], pres[0]),
        water_primary_velocity(temp[0], pres[0]),
        brine_properties(temp[0], pres[0], sal[0]),
        brine_density(temp[0], pres[0], sal[0]),
        brine_primary_velocity(temp[0], pres[0], sal[0]),
    ]:
        if hasattr(results, "__iter__"):
            assert all(isinstance(arg, float) for arg in results)
        else:
            assert isinstance(results, float)
