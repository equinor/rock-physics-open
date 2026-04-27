from numpy.random import default_rng
from syrupy.assertion import SnapshotAssertion

from rock_physics_open.equinor_utilities.classification_functions import gen_class_stats


def test_class_stats(snapshot: SnapshotAssertion):
    rg = default_rng(234769238476)
    obs = rg.random((100, 2))
    inp_id = rg.integers(1, 4, 100)
    assert snapshot == gen_class_stats(obs, inp_id)
