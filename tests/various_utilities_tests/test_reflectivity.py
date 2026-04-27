import numpy as np
import pytest
from numpy.random import default_rng

from rock_physics_open.equinor_utilities.various_utilities import reflectivity

rg = default_rng(5947037623874)
vp = 3500 * (1.0 + 0.2 * rg.random(11))
vs = 1200 * (1.0 + 0.4 * rg.random(11))
rho = 2650 * (1.0 + 0.02 * rg.random(11))
theta = 10.0
k = 2.0


def test_reflectivity_AR_ok_inputs():
    r_aki_rich, idx = reflectivity(vp, vs, rho, theta=theta, k=k, model="AkiRichards")
    r_aki_rich_ref = np.array(
        [
            0.006805,
            0.0028961,
            0.0073632,
            -0.0016639,
            -0.0053192,
            -0.0114692,
            0.0059247,
            -0.0300648,
            0.015451,
            0.0143082,
            0.0143082,
        ]
    )
    idx_ref = np.ones(11).astype(bool)
    np.testing.assert_almost_equal(r_aki_rich, r_aki_rich_ref)
    np.testing.assert_almost_equal(idx, idx_ref)


def test_reflectivity_SG_ok_inputs():
    r_sm_gi, idx = reflectivity(vp, vs, rho, theta=theta, k=k, model="SmithGidlow")
    r_sm_gi_ref = np.array(
        [
            0.0041067,
            0.0033311,
            0.012066,
            -0.0041207,
            -0.0024095,
            -0.0165878,
            0.0082089,
            -0.0389886,
            0.0207101,
            0.0145037,
            0.0145037,
        ]
    )
    idx_ref = np.ones(11).astype(bool)
    np.testing.assert_almost_equal(r_sm_gi, r_sm_gi_ref)
    np.testing.assert_almost_equal(idx, idx_ref)


def test_reflectivity_nan():
    vp[5] = np.nan
    with pytest.raises(ValueError, match="Missing or illegal values in input"):
        _ = reflectivity(
            vp_inp=vp,
            vs_inp=vs,
            rho_inp=rho,
            theta=theta,
            k=k,
            model="AkiRichards",
        )
