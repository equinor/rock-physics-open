import numpy as np
from numpy.random import default_rng
from syrupy.assertion import SnapshotAssertion

from rock_physics_open.equinor_utilities.std_functions import aki_richards, smith_gidlow

rg = default_rng(12345)
vp = 3500 * (1.0 + 0.2 * rg.random(11))
vs = 1200 * (1.0 + 0.4 * rg.random(11))
rho = 2650 * (1.0 + 0.02 * rg.random(11))
theta = np.ones(11)
k = 1.9


def test_aki_richards(snapshot: SnapshotAssertion):
    assert snapshot == aki_richards(vp, vs, rho, theta, k=2.0)


def test_smith_gidlow(snapshot: SnapshotAssertion):
    assert snapshot == smith_gidlow(vp, vs, rho, theta, k=2.0)
