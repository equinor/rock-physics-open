import os

import numpy as np

from rock_physics_open.equinor_utilities.snapshot_test_utilities import (
    INITIATE,
    compare_snapshots,
    get_snapshot_name,
    read_snapshot,
    store_snapshot,
)
from rock_physics_open.equinor_utilities.std_functions import moduli
from rock_physics_open.equinor_utilities.various_utilities import (
    gassmann_dry_model,
    gassmann_model,
    gassmann_sub_model,
)
from rock_physics_open.sandstone_models import friable_model, friable_model_dry

k_min = np.ones(11) * 36.8e9
mu_min = np.ones(11) * 44.0e9
rho_min = np.ones(11) * 2650.0
por = np.linspace(0.15, 0.35, 11)
p_eff = np.ones(11) * 10.0e6
k_fl_orig = np.ones(11) * 0.8e9
rho_fl_orig = np.ones(11) * 850.0
k_fl_sub = np.ones(11) * 2.7e9
rho_fl_sub = np.ones(11) * 1005.0

rho_dry = rho_min * (1.0 - por)
k_dry, mu_dry = friable_model_dry(k_min, mu_min, por, p_eff, 0.4, "porosity", 8, 0.5)
vp_sat, vs_sat, rho_sat, _, _ = friable_model(
    k_min, mu_min, rho_min, k_fl_orig, rho_fl_orig, por, p_eff, 0.4, "porosity", 8, 0.5
)
k_sat, mu_sat = moduli(vp_sat, vs_sat, rho_sat)


def test_gassmann():
    args = gassmann_model(k_min, k_fl_orig, rho_fl_orig, k_dry, mu_dry, rho_dry, por)

    if not os.path.isfile(get_snapshot_name()) or INITIATE:
        store_snapshot(get_snapshot_name(), *args)
    else:
        assert compare_snapshots(args, read_snapshot(get_snapshot_name()))


def test_gassmann_dry():
    args = gassmann_dry_model(
        k_min, k_fl_orig, rho_fl_orig, k_sat, mu_sat, rho_sat, por
    )

    if not os.path.isfile(get_snapshot_name()) or INITIATE:
        store_snapshot(get_snapshot_name(), *args)
    else:
        assert compare_snapshots(args, read_snapshot(get_snapshot_name()))


def test_gassmann_sub():
    args = gassmann_sub_model(
        k_min, k_fl_orig, rho_fl_orig, k_fl_sub, rho_fl_sub, k_sat, mu_dry, rho_sat, por
    )

    if not os.path.isfile(get_snapshot_name()) or INITIATE:
        store_snapshot(get_snapshot_name(), *args)
    else:
        assert compare_snapshots(args, read_snapshot(get_snapshot_name()))
