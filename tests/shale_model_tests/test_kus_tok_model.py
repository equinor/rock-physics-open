import os

import numpy as np

from rock_physics_open.equinor_utilities.snapshot_test_utilities import (
    INITIATE,
    compare_snapshots,
    get_snapshot_name,
    read_snapshot,
    store_snapshot,
)
from rock_physics_open.shale_models import kuster_toksoz_model

k1 = 36.8e9 * np.ones(20)
mu1 = 44.0e9 * np.ones(20)
rho1 = 2650 * np.ones(20)
k2 = 11.0e9 * np.ones(20)
mu2 = 3e9 * np.ones(20)
rho2 = 2680 * np.ones(20)
frac1 = np.linspace(0.0, 1.0, 20)
asp_2 = 0.85 * np.ones(20)


def test_kus_tok_model():
    args = kuster_toksoz_model(k1, mu1, rho1, k2, mu2, rho2, frac1, asp_2)

    if not os.path.isfile(get_snapshot_name()) or INITIATE:
        store_snapshot(get_snapshot_name(), *args)
    else:
        assert compare_snapshots(args, read_snapshot(get_snapshot_name()))
