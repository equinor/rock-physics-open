import numpy as np
from syrupy.assertion import SnapshotAssertion

from rock_physics_open.fluid_models import gas_properties

temp = 100.0 * np.linspace(0.8, 1.2, 101)
pres = 23.0e6 * np.linspace(0.8, 1.2, 101)
gr = 1.0 * np.linspace(0.7, 1.05, 101)


def test_gas_properties(snapshot: SnapshotAssertion):
    assert snapshot == gas_properties(temp, pres, gr)


def test_gas_properties_float():
    """Make sure that input object type is reflected in output type."""
    args = gas_properties(temp[0], pres[0], gr[0])
    assert all(isinstance(arg, float) for arg in args)
