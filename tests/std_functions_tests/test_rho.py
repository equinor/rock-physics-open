import unittest

import numpy as np
import pytest

from rock_physics_open.equinor_utilities.std_functions import rho_b, rho_m


class RhoTest(unittest.TestCase):
    def test_rhob(self):
        phi = np.linspace(0.0, 0.25, 10)
        rho_fluid = 1000 * np.ones_like(phi)
        rho_matrix = 3000 * np.ones_like(phi)

        rhob = rho_b(phi, rho_fluid, rho_matrix)
        np.testing.assert_almost_equal(rhob, np.linspace(3000, 2500, 10))

    def test_rhom(self):
        phi = np.linspace(0.0, 0.25, 10)
        rho_cem = 2000 * np.ones_like(phi)
        rho_min = 3000 * np.ones_like(phi)
        frac_cem = 0.05 * np.ones_like(phi)

        rho_mat = rho_m(frac_cem, phi, rho_cem, rho_min)
        rho_expected = np.array(
            [
                2950.0,
                2948.5714286,
                2947.0588235,
                2945.4545455,
                2943.75,
                2941.9354839,
                2940.0,
                2937.9310345,
                2935.7142857,
                2933.3333333,
            ]
        )
        np.testing.assert_almost_equal(rho_mat, rho_expected)

    def test_phi_one(self):
        phi = np.array([1.0])
        rho_cem = 2000 * np.ones_like(phi)
        rho_min = 3000 * np.ones_like(phi)
        frac_cem = 0.05 * np.ones_like(phi)

        with pytest.warns(UserWarning, match="phi out of range in 1 sample"):
            r_m = rho_m(frac_cem, phi, rho_cem, rho_min)
            assert np.isnan(r_m[0])
