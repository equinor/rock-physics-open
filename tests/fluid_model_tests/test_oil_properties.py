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
from rock_physics_open.fluid_models.oil_model.han_batzle_oil_model import (
    han_batzle_live_oil_density,
    han_batzle_live_oil_velocity,
)
from rock_physics_open.fluid_models.oil_model.live_oil_density import live_oil_density
from rock_physics_open.fluid_models.oil_model.live_oil_velocity import live_oil_velocity
from rock_physics_open.fluid_models.oil_model.oil_bubble_point import (
    bp_standing,
    max_gor_standing,
)
from rock_physics_open.fluid_models.oil_model.oil_properties import live_oil

temp = 100.0 * np.linspace(0.8, 1.2, 101)
pres = 30.0e6 * np.linspace(0.8, 1.2, 101)
rho_0 = 850.0 * np.ones(101)
gor = 120.0 * np.ones(101)
gr = 0.7 * np.ones(101)


def test_oil_prop():
    args = oil_properties(temp, pres, rho_0, gor, gr)

    if not os.path.isfile(get_snapshot_name()) or INITIATE:
        _ = store_snapshot(get_snapshot_name(), *args)
    else:
        assert compare_snapshots(args, read_snapshot(get_snapshot_name()))


def test_oil_prop_bw():
    args = oil_properties(temp, pres, rho_0, gor, gr, model_version="BW")

    if not os.path.isfile(get_snapshot_name()) or INITIATE:
        _ = store_snapshot(get_snapshot_name(), *args)
    else:
        assert compare_snapshots(args, read_snapshot(get_snapshot_name()))


def test_live_oil_density():
    args = live_oil_density(temp, rho_0, gor, gr)

    if not os.path.isfile(get_snapshot_name()) or INITIATE:
        _ = store_snapshot(get_snapshot_name(), args)
    else:
        assert compare_snapshots(args, read_snapshot(get_snapshot_name()))


def test_live_oil_velocity():
    args = live_oil_velocity(temp, pres, rho_0, gor, gr)

    if not os.path.isfile(get_snapshot_name()) or INITIATE:
        _ = store_snapshot(get_snapshot_name(), args)
    else:
        assert compare_snapshots(args, read_snapshot(get_snapshot_name()))


def test_han_batzle_live_oil_density():
    args = han_batzle_live_oil_density(temp, pres, rho_0, gor, gr)

    if not os.path.isfile(get_snapshot_name()) or INITIATE:
        _ = store_snapshot(get_snapshot_name(), args)
    else:
        assert compare_snapshots(args, read_snapshot(get_snapshot_name()))


def test_han_batzle_live_oil_velocity():
    args = han_batzle_live_oil_velocity(temp, pres, rho_0, gor, gr)

    if not os.path.isfile(get_snapshot_name()) or INITIATE:
        _ = store_snapshot(get_snapshot_name(), args)
    else:
        assert compare_snapshots(args, read_snapshot(get_snapshot_name()))


def test_live_oil_han_batzle():
    args = live_oil(
        temperature=temp,
        pressure=pres,
        reference_density=rho_0,
        gas_oil_ratio=gor,
        gas_gravity=gr,
        model_version="HB",
    )

    if not os.path.isfile(get_snapshot_name()) or INITIATE:
        _ = store_snapshot(get_snapshot_name(), *args)
    else:
        assert compare_snapshots(args, read_snapshot(get_snapshot_name()))


def test_max_gor_standing():
    """Snapshot test for max_gor_standing (Standing 1962 inverse bubble-point)."""
    args = max_gor_standing(rho_0, pres, gr, temp)

    if not os.path.isfile(get_snapshot_name()) or INITIATE:
        _ = store_snapshot(get_snapshot_name(), args)
    else:
        assert compare_snapshots(args, read_snapshot(get_snapshot_name()))


def test_max_gor_standing_float():
    """Verify that max_gor_standing returns a float when all inputs are scalar, and that it is consistent with bp_standing (round-trip)."""
    result = max_gor_standing(rho_0[0], pres[0], gr[0], temp[0])
    assert isinstance(result, float)

    # Round-trip: bp_standing(max_gor_standing(p)) ≈ p  (within ~2% empirical tolerance)
    bp = bp_standing(rho_0[0], result, gr[0], temp[0])
    assert isinstance(bp, float)
    assert abs(bp - pres[0]) / pres[0] < 0.02


def test_oil_properties_float():
    """Make sure that input object type is reflected in output type."""
    for results in [
        oil_properties(temp[0], pres[0], rho_0[0], gor[0], gr[0]),
        oil_properties(temp[0], pres[0], rho_0[0], gor[0], gr[0], model_version="BW"),
        live_oil_density(temp[0], rho_0[0], gor[0], gr[0]),
        live_oil_velocity(temp[0], pres[0], rho_0[0], gor[0], gr[0]),
        han_batzle_live_oil_density(temp[0], pres[0], rho_0[0], gor[0], gr[0]),
        han_batzle_live_oil_velocity(temp[0], pres[0], rho_0[0], gor[0], gr[0]),
        live_oil(
            temperature=temp[0],
            pressure=pres[0],
            reference_density=rho_0[0],
            gas_oil_ratio=gor[0],
            gas_gravity=gr[0],
            model_version="HB",
        ),
        live_oil(
            temperature=temp[0],
            pressure=pres[0],
            reference_density=rho_0[0],
            gas_oil_ratio=gor[0],
            gas_gravity=gr[0],
            model_version="BW",
        ),
    ]:
        if hasattr(results, "__iter__"):
            assert all(isinstance(arg, float) for arg in results)
        else:
            assert isinstance(results, float)
