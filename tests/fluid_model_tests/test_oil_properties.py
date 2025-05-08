import os

import numpy as np

from rock_physics_open.equinor_utilities.snapshot_test_utilities import (
    INITIATE,
    compare_snapshots,
    get_snapshot_name,
    read_snapshot,
    store_snapshot,
)
from rock_physics_open.fluid_models import oil_properties
from rock_physics_open.fluid_models.oil_model.live_oil_density import live_oil_density
from rock_physics_open.fluid_models.oil_model.live_oil_velocity import live_oil_velocity

temp = 100.0 * np.linspace(0.8, 1.2, 101)
pres = 30.0e6 * np.linspace(0.8, 1.2, 101)
rho_0 = 850.0 * np.ones(101)
gor = 120.0 * np.ones(101)
gr = 0.7 * np.ones(101)
gor_co2 = 50.0 * np.ones(101)
gr_co2 = 1.52 * np.ones(101)
pres_mpa = 30.0 * np.linspace(0.8, 1.2, 101)
rho_0_gcc = 0.850 * np.ones(101)


def test_oil_prop():
    args = oil_properties(temp, pres, rho_0, gor, gr)

    if not os.path.isfile(get_snapshot_name()) or INITIATE:
        store_snapshot(get_snapshot_name(), *args)
    else:
        assert compare_snapshots(args, read_snapshot(get_snapshot_name()))


def test_live_oil_density():
    args = live_oil_density(temp, pres_mpa, rho_0_gcc, gor, gr)

    if not os.path.isfile(get_snapshot_name()) or INITIATE:
        store_snapshot(get_snapshot_name(), args)
    else:
        assert compare_snapshots(args, read_snapshot(get_snapshot_name()))


def test_live_oil_velocity():
    args = live_oil_velocity(temp, pres_mpa, rho_0_gcc, gor, gr)

    if not os.path.isfile(get_snapshot_name()) or INITIATE:
        store_snapshot(get_snapshot_name(), args)
    else:
        assert compare_snapshots(args, read_snapshot(get_snapshot_name()))


def test_oil_properties_float():
    """
    Make sure that input object type is reflected in output type
    """
    for results in [
        oil_properties(temp[0], pres[0], rho_0[0], gor[0], gr[0]),
        live_oil_density(temp[0], pres_mpa[0], rho_0_gcc[0], gor[0], gr[0]),
        live_oil_velocity(temp[0], pres_mpa[0], rho_0_gcc[0], gor[0], gr[0]),
    ]:
        if hasattr(results, "__iter__"):
            assert all(isinstance(arg, float) for arg in results)
        else:
            assert isinstance(results, float)
