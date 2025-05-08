import os

import numpy as np

from rock_physics_open.equinor_utilities.snapshot_test_utilities import (
    INITIATE,
    compare_snapshots,
    get_snapshot_name,
    read_snapshot,
    store_snapshot,
)
from rock_physics_open.fluid_models import gas_properties

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


def test_gas_properties():
    args = gas_properties(temp, pres, gr, model="model")

    if not os.path.isfile(get_snapshot_name()) or INITIATE:
        store_snapshot(get_snapshot_name(), *args)
    else:
        assert compare_snapshots(args, read_snapshot(get_snapshot_name()))


def test_gas_properties_float():
    """
    Make sure that input object type is reflected in output type
    """
    args = gas_properties(temp[0], pres[0], gr[0], model="model")
    assert all(isinstance(arg, float) for arg in args)
