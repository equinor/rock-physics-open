import unittest

import numpy as np

from rock_physics_open.equinor_utilities.std_functions import hertz_mindlin


class HertzMindlinTestCase(unittest.TestCase):
    def test_hertz_mindlin(self):
        k1 = np.ones(11) * 36.6e9
        mu1 = np.ones(11) * 44.0e9
        phi_c = np.ones(11) * 0.4
        p = np.ones(11) * 30e6
        shear_red = np.linspace(0.0, 1.0, 11)
        n = np.ones(11) * 7.5
        k_dry, mu_dry = hertz_mindlin(k1, mu1, phi_c, p, shear_red, n)
        k_dry_ref = np.array(
            [
                1.9720104,
                1.9720104,
                1.9720104,
                1.9720104,
                1.9720104,
                1.9720104,
                1.9720104,
                1.9720104,
                1.9720104,
                1.9720104,
                1.9720104,
            ]
        )
        mu_dry_ref = np.array(
            [
                1.1832062,
                1.354167,
                1.5251277,
                1.6960885,
                1.8670492,
                2.0380099,
                2.2089707,
                2.3799314,
                2.5508921,
                2.7218529,
                2.8928136,
            ]
        )
        np.testing.assert_almost_equal(k_dry / 1e9, k_dry_ref)
        np.testing.assert_almost_equal(mu_dry / 1e9, mu_dry_ref)


if __name__ == "__main__":
    unittest.main()
