import unittest

import numpy as np

from rock_physics_open.equinor_utilities.std_functions import dvorkin_contact_cement


class DvorkinNurTest(unittest.TestCase):
    def test_dvorkin_nur(self):
        frac_cem = np.linspace(0.01, 0.1, 10)
        por0_sst = 0.4 * np.ones(10)
        mu0_sst = 44e9 * np.ones(10)
        k0_sst = 36.8e9 * np.ones(10)
        mu0_cem = 32e9 * np.ones(10)
        k0_cem = 71e9 * np.ones(10)
        vs_red = 0.25 * np.ones(10)
        c = 9.0
        k, mu = dvorkin_contact_cement(
            frac_cem, por0_sst, mu0_sst, k0_sst, mu0_cem, k0_cem, vs_red, c
        )
        k_expected = np.array(
            [
                2.8039220,
                3.9241764,
                4.7745324,
                5.4851787,
                6.1065427,
                6.6644869,
                7.1743746,
                7.6462158,
                8.0869637,
                8.5016807,
            ]
        )
        mu_expected = np.array(
            [
                2.2131204,
                3.0849503,
                3.745499,
                4.2966804,
                4.7779701,
                5.2096136,
                5.6036376,
                5.9678782,
                6.3077783,
                6.6273009,
            ]
        )
        np.testing.assert_almost_equal(k / 1.0e9, k_expected)
        np.testing.assert_almost_equal(mu / 1.0e9, mu_expected)
