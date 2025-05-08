import unittest

import numpy as np
from numpy.random import default_rng

from rock_physics_open.equinor_utilities.classification_functions import lin_class


class LinClassTestCase(unittest.TestCase):
    def test_lin_class(self):
        rg = default_rng(234769238476)
        obs = rg.random((11, 2))
        class_mean = np.array(
            [
                [0.4450828, 0.5032985],
                [0.4959657, 0.4961214],
                [0.5020109, 0.5090125],
            ]
        )
        class_id = np.array([1, 2, 3])
        lin_class_arr, lin_dist = lin_class(obs, class_mean, class_id)
        lin_class_ref = np.array([1, 3, 1, 2, 3, 1, 3, 3, 1, 3, 1])
        lin_dist_ref = np.array(
            [
                0.4926242,
                0.2636271,
                0.2291065,
                0.4904759,
                0.0985496,
                0.4212475,
                0.3646983,
                0.1860129,
                0.2648174,
                0.0640302,
                0.3429182,
            ]
        )
        np.testing.assert_almost_equal(lin_class_arr, lin_class_ref)
        np.testing.assert_almost_equal(lin_dist, lin_dist_ref)


if __name__ == "__main__":
    unittest.main()
