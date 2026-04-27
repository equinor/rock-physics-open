import numpy as np
from numpy.random import default_rng
from syrupy.assertion import SnapshotAssertion

from rock_physics_open.equinor_utilities.various_utilities import pressure


def test_pressure(snapshot: SnapshotAssertion):
    rg = default_rng(987654321)
    rho = 2650 * (0.8 + 0.2 * rg.random(11))
    tvd_msl = np.linspace(2200, 2500, 11)
    water_depth = 240.0
    p_form = 20.0e6
    tvd_p_form = 2300.0
    n = 0.89
    assert snapshot == pressure(rho, tvd_msl, water_depth, p_form, tvd_p_form, n)


def test_pressure_inf_nan(snapshot: SnapshotAssertion):
    rg = default_rng(987654321)
    rho = 2650 * (0.8 + 0.2 * rg.random(11))
    rho[4] = np.nan
    rho[7] = np.inf
    tvd_msl = np.linspace(2200, 2500, 11)
    tvd_msl[2] = np.nan
    water_depth = 240.0
    p_form = 20.0e6
    tvd_p_form = 2300.0
    n = 0.89
    assert snapshot == pressure(rho, tvd_msl, water_depth, p_form, tvd_p_form, n)
