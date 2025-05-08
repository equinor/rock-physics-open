import os

import numpy as np

from rock_physics_open.equinor_utilities.snapshot_test_utilities import (
    INITIATE,
    compare_snapshots,
    get_snapshot_name,
    read_snapshot,
    store_snapshot,
)
from rock_physics_open.shale_models import (
    multi_sca,
    self_consistent_approximation_model,
)

k1 = 36.8e9 * np.ones(20)
mu1 = 44.0e9 * np.ones(20)
rho1 = 2650 * np.ones(20)
k2 = 15.0e9 * np.ones(20)
mu2 = 7.5e9 * np.ones(20)
rho2 = 2680 * np.ones(20)
k3 = 71.0e9 * np.ones(20)
mu3 = 32.5e9 * np.ones(20)
rho3 = 2720 * np.ones(20)
frac1 = np.linspace(0.0, 1.0, 20)
frac_inc = np.linspace(0.0, 0.4, 20)
frac_inc_1 = np.linspace(0.3, 0.2, 20)
asp1 = 0.5 * np.ones(20)
asp2 = 0.85 * np.ones(20)
tol = 1.0e-6


def test_sca_model():
    args = self_consistent_approximation_model(
        k1, mu1, rho1, k2, mu2, rho2, frac1, asp1, asp2, tol
    )

    if not os.path.isfile(get_snapshot_name()) or INITIATE:
        store_snapshot(get_snapshot_name(), *args)
    else:
        assert compare_snapshots(args, read_snapshot(get_snapshot_name()))


def test_multi_sca_model():
    args = multi_sca(
        k1, mu1, rho1, asp1, frac1, k2, mu2, rho2, asp2, 1 - frac1, tol=tol
    )

    if not os.path.isfile(get_snapshot_name()) or INITIATE:
        store_snapshot(get_snapshot_name(), *args)
    else:
        assert compare_snapshots(args, read_snapshot(get_snapshot_name()))
