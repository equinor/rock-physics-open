import numpy as np
from numpy.random import default_rng
from syrupy.assertion import SnapshotAssertion

from rock_physics_open.equinor_utilities.classification_functions import (
    gen_two_step_class_stats,
)


def test_mahal_class_thresh(snapshot: SnapshotAssertion):
    rg = default_rng(238476)
    obs = rg.random((11, 2))
    inp_class_id = np.array([3, 3, 1, 2, 3, 1, 3, 2, 2, 3, 1])
    assert snapshot == gen_two_step_class_stats(obs, inp_class_id, thresh=1.5)
