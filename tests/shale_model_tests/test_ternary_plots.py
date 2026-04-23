from numpy.random import default_rng
from syrupy.assertion import SnapshotAssertion

from rock_physics_open.ternary_plots import run_ternary


def test_ternary(snapshot: SnapshotAssertion):
    rg = default_rng(5947037623874)
    quartz = rg.random(11)
    carb = rg.random(11)
    clay = rg.random(11)
    kero = rg.random(11)
    phi = rg.random(11)
    misc = rg.random(11)
    misc_log_type = "Vp"
    well_name = "35_11_15"

    assert snapshot == run_ternary(
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
