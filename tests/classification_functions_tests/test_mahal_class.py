import unittest

import numpy as np
from numpy.random import default_rng

from rock_physics_open.equinor_utilities.classification_functions import mahal_class


class MahalClassTestCase(unittest.TestCase):
    def test_mahal_class(self):
        rg = default_rng(234769238476)
        obs = rg.random((11, 2))
        class_mean = np.array(
            [
                [0.4450828, 0.5032985],
                [0.4959657, 0.4961214],
                [0.5020109, 0.5090125],
            ]
        )
        class_cov = np.array(
            [
                [
                    [0.06409206, 0.0695167, 0.09727411],
                    [0.00732941, -0.01057039, 0.00734017],
                ],
                [
                    [0.00732941, -0.01057039, 0.00734017],
                    [0.07791386, 0.08905777, 0.07493538],
                ],
            ]
        )
        class_id = np.array([1, 2, 3])
        mahal_class_arr, mahal_dist, mahal_pp = mahal_class(
            obs, class_mean, class_cov, class_id
        )
        mahal_class_ref = np.array([3, 3, 1, 2, 3, 1, 3, 3, 2, 3, 1])
        mahal_dist_ref = np.array(
            [
                1.7566752,
                0.9532443,
                0.8388402,
                1.6136804,
                0.3358948,
                1.5194717,
                1.2497144,
                0.6221147,
                1.0361249,
                0.2204843,
                1.2178069,
            ]
        )
        mahal_pp_ref = np.array(
            [
                0.3657373,
                0.3392176,
                0.3635076,
                0.3976099,
                0.3662607,
                0.3604428,
                0.3710431,
                0.3795974,
                0.3387392,
                0.3620651,
                0.3627325,
            ]
        )
        np.testing.assert_almost_equal(mahal_class_arr, mahal_class_ref)
        np.testing.assert_almost_equal(mahal_dist, mahal_dist_ref)
        np.testing.assert_almost_equal(mahal_pp, mahal_pp_ref)

    def test_mahal_class_thresh(self):
        rg = default_rng(234769238476)
        obs = rg.random((11, 2))
        class_mean = np.array(
            [
                [0.4450828, 0.5032985],
                [0.4959657, 0.4961214],
                [0.5020109, 0.5090125],
            ]
        )
        class_cov = np.array(
            [
                [
                    [0.06409206, 0.0695167, 0.09727411],
                    [0.00732941, -0.01057039, 0.00734017],
                ],
                [
                    [0.00732941, -0.01057039, 0.00734017],
                    [0.07791386, 0.08905777, 0.07493538],
                ],
            ]
        )
        class_id = np.array([1, 2, 3])
        mahal_class_arr, mahal_dist, mahal_pp = mahal_class(
            obs, class_mean, class_cov, class_id, thresh=1.5
        )
        mahal_class_ref = np.array([0, 3, 1, 0, 3, 0, 3, 3, 2, 3, 1])
        mahal_dist_ref = np.array(
            [
                1.7566752,
                0.9532443,
                0.8388402,
                1.6136804,
                0.3358948,
                1.5194717,
                1.2497144,
                0.6221147,
                1.0361249,
                0.2204843,
                1.2178069,
            ]
        )
        mahal_pp_ref = np.array(
            [
                0.3657373,
                0.3392176,
                0.3635076,
                0.3976099,
                0.3662607,
                0.3604428,
                0.3710431,
                0.3795974,
                0.3387392,
                0.3620651,
                0.3627325,
            ]
        )
        np.testing.assert_almost_equal(mahal_class_arr, mahal_class_ref)
        np.testing.assert_almost_equal(mahal_dist, mahal_dist_ref)
        np.testing.assert_almost_equal(mahal_pp, mahal_pp_ref)


if __name__ == "__main__":
    unittest.main()
