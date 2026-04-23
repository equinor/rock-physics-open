import numpy as np
from syrupy.assertion import SnapshotAssertion

from rock_physics_open.sandstone_models import friable_model, friable_model_dry
from rock_physics_open.sandstone_models.friable_models import CoordinateNumberFunction

k_min = 36.8e9 * np.ones(20)
mu_min = 44.0e9 * np.ones(20)
rho_min = 2650 * np.ones(20)
k_fl = 2.7e9 * np.ones(20)
rho_fl = 1005 * np.ones(20)
phi = np.linspace(0.0, 0.50, 20)
p_eff = 20.0e6 * np.ones(20)
phi_c = 0.45
coord_num_func: CoordinateNumberFunction = "PorBased"
n = 8.0
shear_red = 0.25


def test_friable_model(snapshot: SnapshotAssertion):
    assert snapshot == friable_model(
        k_min=k_min,
        mu_min=mu_min,
        rho_min=rho_min,
        k_fl=k_fl,
        rho_fl=rho_fl,
        phi=phi,
        p_eff=p_eff,
        phi_c=phi_c,
        coord_num_func=coord_num_func,
        n=n,
        shear_red=shear_red,
    )


def test_friable_model_dry(snapshot: SnapshotAssertion):
    assert snapshot == friable_model_dry(
        k_min=k_min,
        mu_min=mu_min,
        phi=phi,
        p_eff=p_eff,
        phi_c=phi_c,
        coord_num_func=coord_num_func,
        n=n,
        shear_red=shear_red,
    )
