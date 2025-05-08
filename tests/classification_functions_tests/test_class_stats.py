import unittest

import numpy as np
from numpy.random import default_rng

from rock_physics_open.equinor_utilities.classification_functions import gen_class_stats


class ClassStatsTestCase(unittest.TestCase):
    def test_class_stats(self):
        rg = default_rng(234769238476)
        obs = rg.random((100, 2))
        inp_id = rg.integers(1, 4, 100)
        class_mean, class_cov, prior_prob, class_counts, class_id = gen_class_stats(
            obs, inp_id
        )
        class_mean_ref = np.array(
            [
                [0.4450828, 0.5032985],
                [0.4959657, 0.4961214],
                [0.5020109, 0.5090125],
            ]
        )
        class_cov_ref = np.array(
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
        prior_prob_ref = np.array([0.28, 0.35, 0.37])
        class_count_ref = np.array([28, 35, 37])
        class_id_ref = np.array([1, 2, 3])
        np.testing.assert_almost_equal(class_mean, class_mean_ref)
        np.testing.assert_almost_equal(class_cov, class_cov_ref)
        np.testing.assert_almost_equal(prior_prob, prior_prob_ref)
        np.testing.assert_equal(class_counts, class_count_ref)
        np.testing.assert_equal(class_id, class_id_ref)


if __name__ == "__main__":
    unittest.main()
