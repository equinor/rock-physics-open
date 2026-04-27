import numpy as np
from numpy.random import default_rng
from syrupy.assertion import SnapshotAssertion

from rock_physics_open.equinor_utilities.classification_functions import norm_class


def test_norm_class(snapshot: SnapshotAssertion):
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
    assert snapshot == norm_class(obs, class_mean, class_cov, prior_prob, class_id)


def test_norm_class_thresh(snapshot: SnapshotAssertion):
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
    assert snapshot == norm_class(
        obs, class_mean, class_cov, prior_prob, class_id, thresh=1.2
    )
