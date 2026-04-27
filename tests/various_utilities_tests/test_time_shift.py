import numpy as np
from numpy.random import default_rng
from syrupy.assertion import SnapshotAssertion

from rock_physics_open.equinor_utilities.various_utilities import (
    time_shift_pp,
    time_shift_ps,
)


def test_time_shift_ps(snapshot: SnapshotAssertion):
    rg = default_rng(672190547)
    tvd = np.linspace(1900.0, 2100.0, 11)
    vp_base = 3500.0 * (0.5 + 0.5 * rg.random(11))
    vp_mon = 3500.0 * (0.5 + 0.5 * rg.random(11))
    vs_base = 1700.0 * (0.5 + 0.5 * rg.random(11))
    vs_mon = 1700.0 * (0.5 + 0.5 * rg.random(11))
    multiplier = 1
    assert snapshot == time_shift_ps(tvd, vp_base, vp_mon, vs_base, vs_mon, multiplier)


def test_time_shift_pp(snapshot: SnapshotAssertion):
    rg = default_rng(3874629384)
    tvd = np.linspace(1900.0, 2100.0, 11)
    vp_base = 3500.0 * (0.5 + 0.5 * rg.random(11))
    vp_mon = 3500.0 * (0.5 + 0.5 * rg.random(11))
    multiplier = 1
    assert snapshot == time_shift_pp(tvd, vp_base, vp_mon, multiplier)
