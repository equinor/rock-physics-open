import unittest

import numpy as np

from rock_physics_open.equinor_utilities.std_functions import (
    hashin_shtrikman,
    hashin_shtrikman_average,
    hashin_shtrikman_walpole,
    multi_hashin_shtrikman,
)


class HashinShtrikmanTest(unittest.TestCase):
    def setup_hs(self):
        k1 = np.ones(11) * 36.6e9
        mu1 = np.ones(11) * 44.0e9
        k2 = np.ones(11) * 71.0e9
        mu2 = np.ones(11) * 32.0e9
        f1 = np.linspace(0, 1, 11)
        return k1, mu1, k2, mu2, f1

    def test_hs(self):
        k1, mu1, k2, mu2, f1 = self.setup_hs()
        k_hs, mu_hs = hashin_shtrikman(k1, mu1, k2, mu2, f1)
        k_hs_ref = np.array(
            [
                71.0,
                66.481021,
                62.266414,
                58.32643,
                54.635074,
                51.169532,
                47.909697,
                44.837783,
                41.937995,
                39.196261,
                36.6,
            ]
        )
        mu_hs_ref = np.array(
            [
                32.0,
                33.0436742,
                34.1180058,
                35.2243656,
                36.3642075,
                37.5390749,
                38.7506074,
                40.0005484,
                41.290754,
                42.6232015,
                44.0,
            ]
        )
        np.testing.assert_almost_equal(k_hs / 1e9, k_hs_ref, decimal=6)
        np.testing.assert_almost_equal(mu_hs / 1e9, mu_hs_ref, decimal=6)

    def test_hs_ave(self):
        k1, mu1, k2, mu2, f1 = self.setup_hs()
        k_av, mu_av = hashin_shtrikman_average(k1, mu1, k2, mu2, f1)
        k_av_ref = np.array(
            [
                71.0,
                66.376654,
                62.094281,
                58.116259,
                54.411059,
                50.951386,
                47.71349,
                44.6766,
                41.822475,
                39.135024,
                36.6,
            ]
        )
        mu_av_ref = np.array(
            [
                32.0,
                33.038928,
                34.109311,
                35.2126,
                36.350338,
                37.524166,
                38.735829,
                39.987191,
                41.280235,
                42.617082,
                44.0,
            ]
        )
        np.testing.assert_almost_equal(k_av / 1e9, k_av_ref, decimal=6)
        np.testing.assert_almost_equal(mu_av / 1e9, mu_av_ref, decimal=6)

    def test_hsw(self):
        k1, mu1, k2, mu2, f1 = self.setup_hs()
        k_hsw, mu_hsw = hashin_shtrikman_walpole(k1, mu1, k2, mu2, f1, bound="lower")
        k_hsw_ref = np.array(
            [
                71.0,
                66.272288,
                61.922148,
                57.906087,
                54.187043,
                50.733241,
                47.517283,
                44.515417,
                41.706955,
                39.073787,
                36.6,
            ]
        )
        mu_hsw_ref = np.array(
            [
                32.0,
                33.024474,
                34.082798,
                35.176679,
                36.307938,
                37.478526,
                38.690529,
                39.946185,
                41.247896,
                42.598241,
                44.0,
            ]
        )
        np.testing.assert_almost_equal(k_hsw / 1e9, k_hsw_ref, decimal=6)
        np.testing.assert_almost_equal(mu_hsw / 1e9, mu_hsw_ref, decimal=6)

    def test_multi_hs(self):
        k1, mu1, k2, mu2, f1 = self.setup_hs()
        k_m_hs, mu_m_hs = multi_hashin_shtrikman(
            k1, mu1, f1, k2, mu2, 1.0 - f1, mode="lower"
        )
        k_m_ref = np.array(
            [
                71.0,
                66.272288,
                61.922148,
                57.906087,
                54.187043,
                50.733241,
                47.517283,
                44.515417,
                41.706955,
                39.073787,
                36.6,
            ]
        )
        mu_m_ref = np.array(
            [
                32.0,
                33.024474,
                34.082798,
                35.176679,
                36.307938,
                37.478526,
                38.690529,
                39.946185,
                41.247896,
                42.598241,
                44.0,
            ]
        )
        np.testing.assert_almost_equal(k_m_hs / 1e9, k_m_ref, decimal=6)
        np.testing.assert_almost_equal(mu_m_hs / 1e9, mu_m_ref, decimal=6)
