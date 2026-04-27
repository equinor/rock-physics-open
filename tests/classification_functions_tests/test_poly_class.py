import numpy as np
from numpy.random import default_rng
from syrupy.assertion import SnapshotAssertion

from rock_physics_open.equinor_utilities.classification_functions import poly_class


def test_poly_class(snapshot: SnapshotAssertion):
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
    assert snapshot == poly_class(obs, class_poly, class_id)
