import numpy as np
import pytest
from numpy.random import default_rng
from syrupy.assertion import SnapshotAssertion

from rock_physics_open.equinor_utilities.various_utilities import reflectivity

rg = default_rng(5947037623874)
vp = 3500 * (1.0 + 0.2 * rg.random(11))
vs = 1200 * (1.0 + 0.4 * rg.random(11))
rho = 2650 * (1.0 + 0.02 * rg.random(11))
theta = 10.0
k = 2.0


def test_reflectivity_AR_ok_inputs(snapshot: SnapshotAssertion):
    assert snapshot == reflectivity(vp, vs, rho, theta=theta, k=k, model="AkiRichards")


def test_reflectivity_SG_ok_inputs(snapshot: SnapshotAssertion):
    assert snapshot == reflectivity(vp, vs, rho, theta=theta, k=k, model="SmithGidlow")


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
