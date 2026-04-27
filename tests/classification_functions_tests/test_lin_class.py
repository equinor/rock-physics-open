import numpy as np
from numpy.random import default_rng
from syrupy.assertion import SnapshotAssertion

from rock_physics_open.equinor_utilities.classification_functions import lin_class


def test_lin_class(snapshot: SnapshotAssertion):
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
    assert snapshot == lin_class(obs, class_mean, class_id)
