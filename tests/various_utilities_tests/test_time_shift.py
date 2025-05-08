import unittest

import numpy as np
from numpy.random import default_rng

from rock_physics_open.equinor_utilities.various_utilities import (
    time_shift_pp,
    time_shift_ps,
)


class TimeShiftTestCase(unittest.TestCase):
    def test_time_shift_ps(self):
        rg = default_rng(672190547)
        tvd = np.linspace(1900.0, 2100.0, 11)
        vp_base = 3500.0 * (0.5 + 0.5 * rg.random(11))
        vp_mon = 3500.0 * (0.5 + 0.5 * rg.random(11))
        vs_base = 1700.0 * (0.5 + 0.5 * rg.random(11))
        vs_mon = 1700.0 * (0.5 + 0.5 * rg.random(11))
        multiplier = 1
        twt_ps_shift = time_shift_ps(tvd, vp_base, vp_mon, vs_base, vs_mon, multiplier)
        twt_ref = np.array(
            [
                -1.487565,
                -1.8224189,
                1.0883962,
                0.0848656,
                2.6961336,
                -2.1949398,
                -4.3835154,
                0.2291774,
                -6.2099461,
                -5.7735105,
                -9.4191945,
            ]
        )
        np.testing.assert_almost_equal(twt_ps_shift, twt_ref)

    def test_time_shift_pp(self):
        rg = default_rng(3874629384)
        tvd = np.linspace(1900.0, 2100.0, 11)
        vp_base = 3500.0 * (0.5 + 0.5 * rg.random(11))
        vp_mon = 3500.0 * (0.5 + 0.5 * rg.random(11))
        multiplier = 1
        owt_pp_shift, twt_pp_shift = time_shift_pp(tvd, vp_base, vp_mon, multiplier)
        owt_ref = np.array(
            [
                -2.389087,
                -2.6406689,
                -1.1021602,
                -4.4961946,
                -0.858652,
                -2.7415121,
                -5.375477,
                -3.7926991,
                -6.0184832,
                -2.2550444,
                -5.1210616,
            ]
        )
        twt_ref = 2.0 * owt_ref
        np.testing.assert_almost_equal(owt_pp_shift, owt_ref)
        np.testing.assert_almost_equal(twt_pp_shift, twt_ref)


if __name__ == "__main__":
    unittest.main()
