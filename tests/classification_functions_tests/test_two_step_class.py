import unittest

import numpy as np
from numpy.random import default_rng

from rock_physics_open.equinor_utilities.classification_functions import (
    gen_two_step_class_stats,
)


class TwoStepClassTestCase(unittest.TestCase):
    def test_mahal_class_thresh(self):
        rg = default_rng(238476)
        obs = rg.random((11, 2))
        inp_class_id = np.array([3, 3, 1, 2, 3, 1, 3, 3, 2, 3, 1])
        (
            mean_class_id,
            class_cov,
            prior_prob,
            class_counts,
            class_id,
            mahal_class_id,
        ) = gen_two_step_class_stats(obs, inp_class_id, thresh=1.5)
        mahal_class_ref = np.array([3, 1, 1, 3, 0, 1, 3, 1, 2, 3, 1])
        class_id_ref = np.array([1, 2, 3])
        class_count_ref = np.array([3, 2, 5])
        prior_prob_ref = np.array([0.3, 0.2, 0.5])
        class_cov_ref = np.array(
            [
                [
                    [0.05270089, 0.03950745, 0.17263269],
                    [-0.01814353, -0.02054904, 0.11536174],
                ],
                [
                    [-0.01814353, -0.02054904, 0.11536174],
                    [0.00712315, 0.01068818, 0.08373218],
                ],
            ]
        )
        class_mean_ref = np.array(
            [
                [0.7015159, 0.8764056],
                [0.3122596, 0.2412942],
                [0.3509461, 0.5514007],
            ]
        )
        np.testing.assert_equal(mahal_class_id, mahal_class_ref)
        np.testing.assert_equal(class_id, class_id_ref)
        np.testing.assert_equal(class_counts, class_count_ref)
        np.testing.assert_almost_equal(prior_prob, prior_prob_ref)
        np.testing.assert_almost_equal(class_cov, class_cov_ref)
        np.testing.assert_almost_equal(mean_class_id, class_mean_ref)


if __name__ == "__main__":
    unittest.main()
