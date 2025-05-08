import unittest

import numpy as np
from numpy.random import default_rng

from rock_physics_open.equinor_utilities.classification_functions import poly_class


class PolyClassTestCase(unittest.TestCase):
    def test_poly_class(self):
        rg = default_rng(234769238476)
        obs = rg.random((11, 2))
        class_poly = np.array(
            [
                [
                    [0.0, 0.0],
                    [0.5, 0.0],
                    [0.5, 0.5],
                    [0.0, 0.5],
                    [0.0, 0.0],
                ],
                [
                    [0.5, 0.0],
                    [1.0, 0.0],
                    [1.0, 0.5],
                    [0.5, 0.5],
                    [0.5, 0.0],
                ],
                [
                    [0.0, 0.5],
                    [1.0, 0.5],
                    [1.0, 1.0],
                    [0.0, 0.5],
                    [0.0, 0.0],
                ],
            ]
        )
        class_id = np.array([1, 2, 3])
        poly_class_arr = poly_class(obs, class_poly, class_id)
        poly_class_ref = np.array([1, 3, 1, 2, 2, 1, 3, 2, 0, 2, 1])
        np.testing.assert_equal(poly_class_arr, poly_class_ref)


if __name__ == "__main__":
    unittest.main()
