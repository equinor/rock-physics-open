import unittest

import numpy as np

from rock_physics_open.equinor_utilities.std_functions import (
    gassmann,
    gassmann2,
    gassmann_dry,
)


class GassmannTest(unittest.TestCase):
    def setup_gassmann(self):
        k_brine = np.array([2.8e9, 2.8e9, 2.8e9])
        phie = np.array([0.0, 0.35, 0.1])
        k_min = np.array([36.6e9, 36.6e9, 36.6e9])
        k_dry = np.array([36.6e9, 15.0e9, 28.0e9])
        k_oil = np.array([0.8e9, 0.8e9, 0.8e9])
        return k_brine, phie, k_min, k_dry, k_oil

    def test_gassmann(self):
        k_brine, phie, k_min, k_dry, k_oil = self.setup_gassmann()
        k_sat = gassmann(k_dry, phie, k_brine, k_min)
        ksat_ref = np.array([36.6, 17.6473742, 29.4012504])
        np.testing.assert_almost_equal(k_sat / 1.0e9, ksat_ref)

    def test_gassmann2(self):
        k_brine, phie, k_min, k_dry, k_oil = self.setup_gassmann()
        k_sat = gassmann(k_dry, phie, k_brine, k_min)
        k_sat2 = gassmann2(k_sat, k_brine, k_oil, phie, k_min)
        ksat2_ref = np.array([36.6, 15.7843355, 28.4290396])
        np.testing.assert_almost_equal(k_sat2 / 1.0e9, ksat2_ref)

    def test_gassmann_dry(self):
        k_brine, phie, k_min, k_dry, k_oil = self.setup_gassmann()
        k_sat = gassmann(k_dry, phie, k_brine, k_min)
        k_dry2 = gassmann_dry(k_sat, phie, k_brine, k_min)
        k_dry2_ref = np.array([36.6e9, 15.0e9, 28.0e9]) / 1e9
        np.testing.assert_almost_equal(k_dry2 / 1.0e9, k_dry2_ref)
