import unittest

import numpy as np
from numpy.random import default_rng

from rock_physics_open.equinor_utilities.various_utilities import hs_average


class HSTestCase(unittest.TestCase):
    def test_hashin_shtrikman_average(self):
        rg = default_rng(42)
        f = rg.random(10)
        k1 = 36.8e9 * np.ones_like(f)
        mu1 = 44.0e9 * np.ones_like(f)
        rho1 = 2650.0 * np.ones_like(f)
        k2 = 71.2e9 * np.ones_like(f)
        mu2 = 32.0e9 * np.ones_like(f)
        rho2 = 2710.0 * np.ones_like(f)
        vp, vs = hs_average(k1, mu1, rho1, k2, mu2, rho2, f)[0:2]
        vp_ref = np.array(
            [
                6045.198023,
                6174.5154768,
                6025.1116175,
                6067.5931606,
                6401.1173579,
                6005.143869,
                6048.6629321,
                6042.0277693,
                6374.082277,
                6168.6768996,
            ]
        )
        vs_ref = np.array(
            [
                3920.4844507,
                3703.175796,
                3977.4848829,
                3869.6688652,
                3492.0335032,
                4057.8105084,
                3911.9312645,
                3928.5836574,
                3512.3142984,
                3710.4311108,
            ]
        )
        np.testing.assert_almost_equal(vp, vp_ref)
        np.testing.assert_almost_equal(vs, vs_ref)


if __name__ == "__main__":
    unittest.main()
