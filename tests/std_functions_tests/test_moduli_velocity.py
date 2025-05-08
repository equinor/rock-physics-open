import unittest

import numpy as np

from rock_physics_open.equinor_utilities.std_functions import moduli, velocity


class ModuliVelocityTestCase(unittest.TestCase):
    def test_moduli(self):
        vp = np.linspace(0.8, 1.2, 11) * 3500.0
        vs = np.linspace(0.8, 1.2, 11) * 1200.0
        rho = np.ones(11) * 2650.0
        k, mu = moduli(vp, vs, rho)
        k_ref = np.array(
            [
                17.51968,
                19.3154472,
                21.1988128,
                23.1697768,
                25.2283392,
                27.3745,
                29.6082592,
                31.9296168,
                34.3385728,
                36.8351272,
                39.41928,
            ]
        )
        mu_ref = np.array(
            [
                2.44224,
                2.6925696,
                2.9551104,
                3.2298624,
                3.5168256,
                3.816,
                4.1273856,
                4.4509824,
                4.7867904,
                5.1348096,
                5.49504,
            ]
        )
        np.testing.assert_almost_equal(k / 1.0e9, k_ref)
        np.testing.assert_almost_equal(mu / 1.0e9, mu_ref)

    def test_velocity(self):
        rho = np.ones(11) * 2650.0
        k = (
            np.array(
                [
                    17.51968,
                    19.3154472,
                    21.1988128,
                    23.1697768,
                    25.2283392,
                    27.3745,
                    29.6082592,
                    31.9296168,
                    34.3385728,
                    36.8351272,
                    39.41928,
                ]
            )
            * 1.0e9
        )
        mu = (
            np.array(
                [
                    2.44224,
                    2.6925696,
                    2.9551104,
                    3.2298624,
                    3.5168256,
                    3.816,
                    4.1273856,
                    4.4509824,
                    4.7867904,
                    5.1348096,
                    5.49504,
                ]
            )
            * 1.0e9
        )
        vp, vs, ai, vp_vs = velocity(k, mu, rho)
        vp_ref = np.linspace(0.8, 1.2, 11) * 3500.0
        vs_ref = np.linspace(0.8, 1.2, 11) * 1200.0
        ai_ref = rho * vp
        vp_vs_ref = vp / vs
        np.testing.assert_almost_equal(vp, vp_ref)
        np.testing.assert_almost_equal(vs, vs_ref)
        np.testing.assert_almost_equal(ai, ai_ref)
        np.testing.assert_almost_equal(vp_vs, vp_vs_ref)


if __name__ == "__main__":
    unittest.main()
