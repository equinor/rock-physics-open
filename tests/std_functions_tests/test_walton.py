import unittest

import numpy as np

from rock_physics_open.equinor_utilities.std_functions import walton_smooth


class WaltonTestCase(unittest.TestCase):
    def test_walton_smooth(self):
        k = np.ones(11) * 36.8e9
        mu = np.ones(11) * 44.0e9
        phi = np.linspace(0.1, 0.36, 11)
        p_eff = 15.0e6
        k_dry, mu_dry = walton_smooth(k, mu, phi, p_eff)
        k_dry_ref = np.array(
            [
                3.6200601,
                3.4159233,
                3.2186296,
                3.0281776,
                2.8445654,
                2.6677899,
                2.4978473,
                2.3347321,
                2.1784371,
                2.0289527,
                1.8862663,
            ]
        )
        mu_dry_ref = np.array(
            [
                2.172036,
                2.049554,
                1.9311778,
                1.8169066,
                1.7067392,
                1.6006739,
                1.4987084,
                1.4008393,
                1.3070623,
                1.2173716,
                1.1317598,
            ]
        )
        np.testing.assert_almost_equal(k_dry / 1.0e9, k_dry_ref)
        np.testing.assert_almost_equal(mu_dry / 1.0e9, mu_dry_ref)

    def test_walton_smooth_n(self):
        k = np.ones(11) * 36.8e9
        mu = np.ones(11) * 44.0e9
        phi = np.linspace(0.1, 0.36, 11)
        p_eff = 15.0e6
        n = 8.5
        k_dry, mu_dry = walton_smooth(k, mu, phi, p_eff, coord=n)
        k_dry_ref = np.array(
            [
                2.2321254,
                2.1889267,
                2.1452973,
                2.1012197,
                2.0566748,
                2.0116422,
                1.9660997,
                1.9200235,
                1.8733876,
                1.8261639,
                1.7783213,
            ]
        )
        mu_dry_ref = np.array(
            [
                1.3392753,
                1.313356,
                1.2871784,
                1.2607318,
                1.2340049,
                1.2069853,
                1.1796598,
                1.1520141,
                1.1240326,
                1.0956983,
                1.0669928,
            ]
        )
        np.testing.assert_almost_equal(k_dry / 1.0e9, k_dry_ref)
        np.testing.assert_almost_equal(mu_dry / 1.0e9, mu_dry_ref)


if __name__ == "__main__":
    unittest.main()
