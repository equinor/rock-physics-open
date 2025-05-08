import unittest

import numpy as np

from rock_physics_open.equinor_utilities.std_functions import (
    multi_voigt_reuss_hill,
    reuss,
    voigt,
    voigt_reuss_hill,
)

k1 = np.ones(11) * 36.6e9
mu1 = np.ones(11) * 44.0e9
k2 = np.ones(11) * 71.0e9
mu2 = np.ones(11) * 32.0e9
f1 = np.linspace(0, 1, 11)


class VoigtReussHillTestCase(unittest.TestCase):
    def test_voigt(self):
        k_v, mu_v = voigt(k1, mu1, k2, mu2, f1)
        k_v_ref = np.array(
            [71.0, 67.56, 64.12, 60.68, 57.24, 53.8, 50.36, 46.92, 43.48, 40.04, 36.6]
        )
        mu_v_ref = np.array(
            [32.0, 33.2, 34.4, 35.6, 36.8, 38.0, 39.2, 40.4, 41.6, 42.8, 44.0]
        )
        np.testing.assert_almost_equal(k_v / 1.0e9, k_v_ref)
        np.testing.assert_almost_equal(mu_v / 1.0e9, mu_v_ref)

    def test_reuss(self):
        k_r, mu_r = reuss(k1, mu1, k2, mu2, f1)
        k_r_ref = np.array(
            [
                71.0,
                64.9000999,
                59.7654094,
                55.3836317,
                51.6004766,
                48.3011152,
                45.3983229,
                42.8246539,
                40.5271366,
                38.4635879,
                36.6,
            ]
        )
        mu_r_ref = np.array(
            [
                32.0,
                32.8971963,
                33.8461538,
                34.8514851,
                35.9183673,
                37.0526316,
                38.2608696,
                39.5505618,
                40.9302326,
                42.4096386,
                44.0,
            ]
        )
        np.testing.assert_almost_equal(k_r / 1.0e9, k_r_ref)
        np.testing.assert_almost_equal(mu_r / 1.0e9, mu_r_ref)

    def test_voigt_reuss_hill(self):
        k_vrh, mu_vrh = voigt_reuss_hill(k1, mu1, k2, mu2, f1)
        k_vrh_ref = np.array(
            [
                71.0,
                66.23005,
                61.9427047,
                58.0318159,
                54.4202383,
                51.0505576,
                47.8791614,
                44.872327,
                42.0035683,
                39.251794,
                36.6,
            ]
        )
        mu_vrh_ref = np.array(
            [
                32.0,
                33.0485981,
                34.1230769,
                35.2257426,
                36.3591837,
                37.5263158,
                38.7304348,
                39.9752809,
                41.2651163,
                42.6048193,
                44.0,
            ]
        )
        np.testing.assert_almost_equal(k_vrh / 1.0e9, k_vrh_ref)
        np.testing.assert_almost_equal(mu_vrh / 1.0e9, mu_vrh_ref)

    def test_multi_voigt_reuss_hill(self):
        k_m_vrh, mu_m_vrh = multi_voigt_reuss_hill(k1, mu1, f1, k2, mu2, 1.0 - f1)
        k_m_vrh_ref = np.array(
            [
                71.0,
                66.23005,
                61.9427047,
                58.0318159,
                54.4202383,
                51.0505576,
                47.8791614,
                44.872327,
                42.0035683,
                39.251794,
                36.6,
            ]
        )
        mu_m_vrh_ref = np.array(
            [
                32.0,
                33.0485981,
                34.1230769,
                35.2257426,
                36.3591837,
                37.5263158,
                38.7304348,
                39.9752809,
                41.2651163,
                42.6048193,
                44.0,
            ]
        )
        np.testing.assert_almost_equal(k_m_vrh / 1.0e9, k_m_vrh_ref)
        np.testing.assert_almost_equal(mu_m_vrh / 1.0e9, mu_m_vrh_ref)


if __name__ == "__main__":
    unittest.main()
