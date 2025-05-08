import os

import numpy as np
import pandas as pd

from rock_physics_open.equinor_utilities.snapshot_test_utilities import (
    INITIATE,
    compare_snapshots,
    get_snapshot_name,
    read_snapshot,
    store_snapshot,
)
from rock_physics_open.sandstone_models import (
    constant_cement_model_optimisation,
    friable_model_optimisation,
    patchy_cement_model_optimisation,
)
from tests.config import TESTDATA_DIR

os.chdir(TESTDATA_DIR)
data_df = pd.read_csv("sandstone_optimisation.csv")
vp = data_df["VP"].to_numpy()
vs = data_df["VS"].to_numpy()
rhob = data_df["RHOB"].to_numpy()
phit = data_df["PHIT"].to_numpy()

k_min = 36.8e9 * np.ones_like(phit)
mu_min = 44.0e9 * np.ones_like(phit)
rho_min = 2650 * np.ones_like(phit)
k_cem = 36.8e9 * np.ones_like(phit)
mu_cem = 44.0e9 * np.ones_like(phit)
rho_cem = 2650 * np.ones_like(phit)
k_fl = 2.7e9 * np.ones_like(phit)
rho_fl = 1005 * np.ones_like(phit)
p_eff = 20.0e6 * np.ones_like(phit)
phi_c = 0.45


def test_friable_optimisation(data_dir):
    file_name = str(data_dir.joinpath("friable_model_optimisation.pkl"))

    args = friable_model_optimisation(
        k_min,
        mu_min,
        rho_min,
        k_fl,
        rho_fl,
        phit,
        p_eff,
        vp,
        vs,
        rhob,
        file_out_str=file_name,
    )

    if not os.path.isfile(get_snapshot_name()) or INITIATE:
        store_snapshot(get_snapshot_name(), *args)
    else:
        assert compare_snapshots(args, read_snapshot(get_snapshot_name()))


def test_constant_cement_optimisation(data_dir):
    file_name = str(data_dir.joinpath("constant_cement_model_optimisation.pkl"))

    args = constant_cement_model_optimisation(
        k_min,
        mu_min,
        rho_min,
        k_cem,
        mu_cem,
        rho_cem,
        k_fl,
        rho_fl,
        phit,
        vp,
        vs,
        rhob,
        file_out_str=file_name,
    )

    if not os.path.isfile(get_snapshot_name()) or INITIATE:
        store_snapshot(get_snapshot_name(), *args)
    else:
        assert compare_snapshots(args, read_snapshot(get_snapshot_name()))


def test_patchy_cement_optimisation(data_dir):
    file_name = str(data_dir.joinpath("patchy_cement_model_optimisation.pkl"))

    args = patchy_cement_model_optimisation(
        k_min,
        mu_min,
        rho_min,
        k_cem,
        mu_cem,
        rho_cem,
        k_fl,
        rho_fl,
        phit,
        p_eff,
        vp,
        vs,
        rhob,
        phi_c,
        file_out_str=file_name,
    )

    if not os.path.isfile(get_snapshot_name()) or INITIATE:
        store_snapshot(get_snapshot_name(), *args)
    else:
        assert compare_snapshots(args, read_snapshot(get_snapshot_name()))
