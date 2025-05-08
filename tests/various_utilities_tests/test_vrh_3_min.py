import unittest

import numpy as np
from numpy.random import default_rng

from rock_physics_open.equinor_utilities.various_utilities import min_3_voigt_reuss_hill


class Min3VRHTestCase(unittest.TestCase):
    def test_3_mineral_voigt_reuss_hill(self):
        rg = default_rng(260363)
        f1 = 0.33 * rg.random(11)
        f2 = 0.33 * rg.random(11)
        f3 = 0.33 * rg.random(11)
        vp1 = 6050.0 * np.ones(11)
        vs1 = 4090.0 * np.ones(11)
        rho1 = 2650.0 * np.ones(11)
        vp2 = 6640.0 * np.ones(11)
        vs2 = 3440.0 * np.ones(11)
        rho2 = 2710.0 * np.ones(11)
        vp3 = 7340.0 * np.ones(11)
        vs3 = 3960.0 * np.ones(11)
        rho3 = 2870.0 * np.ones(11)
        vp, vs, rho = min_3_voigt_reuss_hill(
            vp1, vs1, rho1, f1, vp2, vs2, rho2, f2, vp3, vs3, rho3, f3
        )[0:3]
        vp_ref = np.array(
            [
                6717.9175171,
                6800.3189303,
                6624.4446804,
                6750.0741773,
                6174.0931249,
                6464.0869351,
                6798.4129712,
                6495.8033624,
                6826.4121872,
                6573.038569,
                6595.6224302,
            ]
        )
        vs_ref = np.array(
            [
                3747.7259178,
                3734.6876154,
                3855.2166557,
                3729.0785307,
                4017.1020017,
                4031.9058341,
                3745.7727205,
                3777.5636891,
                3812.1757253,
                3790.0322411,
                3906.8306075,
            ]
        )
        rho_ref = np.array(
            [
                2761.6394369,
                2772.287933,
                2756.8150973,
                2764.3215062,
                2676.6615413,
                2736.3677518,
                2773.3473826,
                2728.3556655,
                2785.0461618,
                2742.6151621,
                2755.545227,
            ]
        )
        np.testing.assert_almost_equal(vp, vp_ref)
        np.testing.assert_almost_equal(vs, vs_ref)
        np.testing.assert_almost_equal(rho, rho_ref)


if __name__ == "__main__":
    unittest.main()
