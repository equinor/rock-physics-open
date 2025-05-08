import unittest

import numpy as np
from numpy.random import default_rng

from rock_physics_open.equinor_utilities.classification_functions import norm_class


class NormClassTestCase(unittest.TestCase):
    def test_norm_class(self):
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
        prior_prob = np.array([0.28, 0.35, 0.37])
        class_id = np.array([1, 2, 3])
        norm_class_arr, norm_dist, norm_pp = norm_class(
            obs, class_mean, class_cov, prior_prob, class_id
        )
        norm_class_ref = np.array([3, 2, 3, 2, 2, 3, 3, 3, 2, 2, 3])
        norm_dist_ref = np.array(
            [
                0.5917953,
                1.0196095,
                1.0218156,
                0.6947731,
                1.3181127,
                0.709255,
                0.8452758,
                1.1590756,
                0.9835508,
                1.3819751,
                0.8343481,
            ]
        )
        norm_pp_ref = np.array(
            [
                -0.356827,
                -0.3511402,
                -0.3451817,
                -0.3821536,
                -0.3609478,
                -0.3536818,
                -0.3579545,
                -0.3618086,
                -0.3529041,
                -0.361059,
                -0.345492,
            ]
        )
        np.testing.assert_almost_equal(norm_class_arr, norm_class_ref)
        np.testing.assert_almost_equal(norm_dist, norm_dist_ref)
        np.testing.assert_almost_equal(norm_pp, norm_pp_ref)

    def test_norm_class_thresh(self):
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
        prior_prob = np.array([0.28, 0.35, 0.37])
        class_id = np.array([1, 2, 3])
        norm_class_arr, norm_dist, norm_pp = norm_class(
            obs, class_mean, class_cov, prior_prob, class_id, thresh=1.2
        )
        norm_class_ref = np.array([3, 2, 3, 2, 0, 3, 3, 3, 2, 0, 3])
        norm_dist_ref = np.array(
            [
                0.5917953,
                1.0196095,
                1.0218156,
                0.6947731,
                1.3181127,
                0.709255,
                0.8452758,
                1.1590756,
                0.9835508,
                1.3819751,
                0.8343481,
            ]
        )
        norm_pp_ref = np.array(
            [
                -0.356827,
                -0.3511402,
                -0.3451817,
                -0.3821536,
                -0.3609478,
                -0.3536818,
                -0.3579545,
                -0.3618086,
                -0.3529041,
                -0.361059,
                -0.345492,
            ]
        )
        np.testing.assert_equal(norm_class_arr, norm_class_ref)
        np.testing.assert_almost_equal(norm_dist, norm_dist_ref)
        np.testing.assert_almost_equal(norm_pp, norm_pp_ref)


if __name__ == "__main__":
    unittest.main()
