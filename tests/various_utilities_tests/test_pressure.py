import unittest

import numpy as np
from numpy.random import default_rng

from rock_physics_open.equinor_utilities.various_utilities import pressure


class PressureTestCase(unittest.TestCase):
    def test_pressure(self):
        rg = default_rng(987654321)
        rho = 2650 * (0.8 + 0.2 * rg.random(11))
        tvd_msl = np.linspace(2200, 2500, 11)
        water_depth = 240.0
        p_form = 20.0e6
        tvd_p_form = 2300.0
        n = 0.89
        p_eff, p_lith = pressure(rho, tvd_msl, water_depth, p_form, tvd_p_form, n)
        p_eff_ref = np.array(
            [
                26965839.7074737,
                27378582.1519758,
                27791324.5964779,
                28204067.0409801,
                28341647.8558141,
                28341647.8558141,
                28341647.8558141,
                28341647.8558141,
                28341647.8558141,
                28341647.8558141,
                28341647.8558141,
            ]
        )
        p_lith_ref = np.array(
            [
                44156462.2597103,
                44806963.6761548,
                45461677.0549971,
                46141647.8558141,
                46788318.3400473,
                47456762.5944036,
                48138689.8657396,
                48778246.1823415,
                49432568.3607947,
                50158507.0041808,
                50806835.7838177,
            ]
        )
        np.testing.assert_almost_equal(p_lith, p_lith_ref)
        np.testing.assert_almost_equal(p_eff, p_eff_ref)

    def test_pressure_inf_nan(self):
        rg = default_rng(987654321)
        rho = 2650 * (0.8 + 0.2 * rg.random(11))
        rho[4] = np.nan
        rho[7] = np.inf
        tvd_msl = np.linspace(2200, 2500, 11)
        tvd_msl[2] = np.nan
        water_depth = 240.0
        p_form = 20.0e6
        tvd_p_form = 2300.0
        n = 0.89
        p_eff, p_lith = pressure(rho, tvd_msl, water_depth, p_form, tvd_p_form, n)
        p_eff_ref = np.array(
            [
                27608794.7187852,
                28031378.3114197,
                np.nan,
                28876545.4966886,
                np.nan,
                29017406.6942335,
                29017406.6942335,
                np.nan,
                29017406.6942335,
                29017406.6942335,
                29017406.6942335,
            ]
        )
        p_lith_ref = np.array(
            [
                44156462.2597103,
                45457465.0925993,
                np.nan,
                46817406.6942335,
                np.nan,
                47485850.9485897,
                48849705.4912617,
                np.nan,
                49504027.6697149,
                50229966.313101,
                50878295.0927379,
            ]
        )
        np.testing.assert_almost_equal(p_lith, p_lith_ref)
        np.testing.assert_almost_equal(p_eff, p_eff_ref)


if __name__ == "__main__":
    unittest.main()
