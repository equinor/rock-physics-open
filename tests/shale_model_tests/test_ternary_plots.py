import os

import pytest
from numpy.random import default_rng

from rock_physics_open.equinor_utilities.snapshot_test_utilities import (
    INITIATE,
    compare_snapshots,
    get_snapshot_name,
    read_snapshot,
    store_snapshot,
)
from rock_physics_open.ternary_plots import run_ternary

rg = default_rng(5947037623874)
quartz = rg.random(11)
carb = rg.random(11)
clay = rg.random(11)
kero = rg.random(11)
phi = rg.random(11)
misc = rg.random(11)
misc_log_type = "Vp"
well_name = "35_11_15"


@pytest.mark.use_graphics
def test_ternary():
    args = run_ternary(
        quartz,
        carb,
        clay,
        kero,
        phi,
        misc,
        misc_log_type,
        well_name,
        draw_figures=False,
    )

    if not os.path.isfile(get_snapshot_name()) or INITIATE:
        store_snapshot(get_snapshot_name(), *args)
    else:
        assert compare_snapshots(args, read_snapshot(get_snapshot_name()))
