import unittest

import numpy as np

from rock_physics_open.equinor_utilities.std_functions import backus_average


class BackusAverageTest(unittest.TestCase):
    def test_back_ave(self):
        vp1i = np.ones(10) * 3500.0
        vp2i = np.ones(10) * 2800.0
        vs1i = np.ones(10) * 1500.0
        vs2i = np.ones(10) * 1100.0
        rho1i = np.ones(10) * 2560.0
        rho2i = np.ones(10) * 2580.0
        f1 = np.linspace(0.0, 1.0, 10)
        vpv, vsv, vph, vsh, rho = backus_average(
            vp1i, vs1i, rho1i, vp2i, vs2i, rho2i, f1
        )
        vpv_expected = np.array(
            [
                2800.0,
                2858.1426623,
                2919.9567089,
                2985.8476926,
                3056.2868556,
                3131.8254803,
                3213.1132666,
                3300.9221468,
                3396.1775453,
                3500.0,
            ]
        )
        vsv_expected = np.array(
            [
                1100.0,
                1129.5925026,
                1161.6526796,
                1196.5453111,
                1234.7154897,
                1276.7128915,
                1323.2256613,
                1375.128761,
                1433.5546976,
                1500.0,
            ]
        )
        vph_expected = np.array(
            [
                2800.0,
                2878.0196313,
                2955.40214,
                3032.3691991,
                3109.1518316,
                3185.9970573,
                3263.1757229,
                3340.9921274,
                3419.7962809,
                3500.0,
            ]
        )
        vsh_expected = np.array(
            [
                1100.0,
                1150.9815901,
                1199.8813574,
                1246.944452,
                1292.3717418,
                1336.3302536,
                1378.9606476,
                1420.3826955,
                1460.6993775,
                1500.0,
            ]
        )
        rho_expected = np.array(
            [
                2580.0,
                2577.7777778,
                2575.5555556,
                2573.3333333,
                2571.1111111,
                2568.8888889,
                2566.6666667,
                2564.4444444,
                2562.2222222,
                2560.0,
            ]
        )
        np.testing.assert_almost_equal(vpv, vpv_expected)
        np.testing.assert_almost_equal(vsv, vsv_expected)
        np.testing.assert_almost_equal(vph, vph_expected)
        np.testing.assert_almost_equal(vsh, vsh_expected)
        np.testing.assert_almost_equal(rho, rho_expected)
