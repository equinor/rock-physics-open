import os

import numpy as np

from rock_physics_open.equinor_utilities.snapshot_test_utilities import (
    INITIATE,
    compare_snapshots,
    get_snapshot_name,
    read_snapshot,
    store_snapshot,
)
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
pres_mpa = pres * 1.0e-6
temp = 100.0 * np.linspace(0.8, 1.2, 101)
sal = 35000.0 * np.ones(101)
nacl = 80.0 * np.ones(101)
cacl = 15.0 * np.ones(101)
kcl = 5.0 * np.ones(101)
gor_co2 = 50 * np.ones(101)


def test_brine_properties():
    args = brine_properties(temp, pres, sal, nacl, kcl, cacl)

    if not os.path.isfile(get_snapshot_name()) or INITIATE:
        store_snapshot(get_snapshot_name(), *args)
    else:
        assert compare_snapshots(args, read_snapshot(get_snapshot_name()))


def test_water_properties():
    args = water(temp, pres)

    if not os.path.isfile(get_snapshot_name()) or INITIATE:
        store_snapshot(get_snapshot_name(), *args)
    else:
        assert compare_snapshots(args, read_snapshot(get_snapshot_name()))


def test_water_density():
    args = water_density(temp, pres_mpa)

    if not os.path.isfile(get_snapshot_name()) or INITIATE:
        store_snapshot(get_snapshot_name(), args)
    else:
        assert compare_snapshots(args, read_snapshot(get_snapshot_name()))


def test_water_velocity():
    args = water_primary_velocity(temp, pres_mpa)

    if not os.path.isfile(get_snapshot_name()) or INITIATE:
        store_snapshot(get_snapshot_name(), args)
    else:
        assert compare_snapshots(args, read_snapshot(get_snapshot_name()))


def test_brine_density():
    args = brine_density(temp, pres_mpa, sal)

    if not os.path.isfile(get_snapshot_name()) or INITIATE:
        store_snapshot(get_snapshot_name(), args)
    else:
        assert compare_snapshots(args, read_snapshot(get_snapshot_name()))


def test_brine_velocity():
    args = brine_primary_velocity(temp, pres_mpa, sal)

    if not os.path.isfile(get_snapshot_name()) or INITIATE:
        store_snapshot(get_snapshot_name(), args)
    else:
        assert compare_snapshots(args, read_snapshot(get_snapshot_name()))


def test_water_brine_properties_float():
    """
    Make sure that input object type is reflected in output type
    """
    for results in [
        water(temp[0], pres[0]),
        water_density(temp[0], pres_mpa[0]),
        water_primary_velocity(temp[0], pres_mpa[0]),
        brine_properties(temp[0], pres[0], sal[0]),
        brine_density(temp[0], pres_mpa[0], sal[0]),
        brine_primary_velocity(temp[0], pres_mpa[0], sal[0]),
    ]:
        if hasattr(results, "__iter__"):
            assert all(isinstance(arg, float) for arg in results)
        else:
            assert isinstance(results, float)
