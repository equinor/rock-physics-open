import unittest

import numpy as np
from numpy.random import default_rng

from rock_physics_open.equinor_utilities.std_functions import aki_richards, smith_gidlow

rg = default_rng(12345)
vp = 3500 * (1.0 + 0.2 * rg.random(11))
vs = 1200 * (1.0 + 0.4 * rg.random(11))
rho = 2650 * (1.0 + 0.02 * rg.random(11))
theta = np.ones(11)
k = 1.9


class ReflectivityTestCase(unittest.TestCase):
    def test_aki_richards(self):
        r_aki_rich = aki_richards(vp, vs, rho, theta, k=2.0)
        r_aki_rich_ref = np.array(
            [
                0.0048749,
                0.0206719,
                -0.0025819,
                -0.015988,
                -0.0030281,
                0.0119779,
                -0.0165907,
                0.023673,
                0.0103467,
                -0.0293775,
                -0.0293775,
            ]
        )
        np.testing.assert_almost_equal(r_aki_rich, r_aki_rich_ref)

    def test_smith_gidlow(self):
        r_sm_gi = smith_gidlow(vp, vs, rho, theta, k=2.0)
        r_sm_gi_ref = np.array(
            [
                0.0053144,
                0.027062,
                -0.0066175,
                -0.0161272,
                -0.0033894,
                0.0152015,
                -0.0238758,
                0.0280036,
                0.0144885,
                -0.038749,
                -0.038749,
            ]
        )
        np.testing.assert_almost_equal(r_sm_gi, r_sm_gi_ref)


if __name__ == "__main__":
    unittest.main()
