import numpy as np
from syrupy.assertion import SnapshotAssertion

from rock_physics_open.sandstone_models import (
    constant_cement_model,
    constant_cement_model_dry,
)

k_min = 36.8e9 * np.ones(20)
mu_min = 44.0e9 * np.ones(20)
rho_min = 2650 * np.ones(20)
k_cem = 36.8e9 * np.ones(20)
mu_cem = 44.0e9 * np.ones(20)
rho_cem = 2650 * np.ones(20)
k_fl = 2.7e9 * np.ones(20)
rho_fl = 1005 * np.ones(20)
phi = np.linspace(0.0, 0.45, 20)
frac_cem = 0.05
phi_c = 0.45
n = 8.0
shear_red = 0.25


def test_constant_cement_model_dry(snapshot: SnapshotAssertion):
    assert snapshot == constant_cement_model_dry(
        k_min, mu_min, k_cem, mu_cem, phi, frac_cem, phi_c, n, shear_red
    )


def test_constant_cement_model(snapshot: SnapshotAssertion):
    assert snapshot == constant_cement_model(
        k_min=k_min,
        mu_min=mu_min,
        rho_min=rho_min,
        k_cem=k_cem,
        mu_cem=mu_cem,
        rho_cem=rho_cem,
        k_fl=k_fl,
        rho_fl=rho_fl,
        phi=phi,
        frac_cem=frac_cem,
        phi_c=phi_c,
        n=n,
        shear_red=shear_red,
    )


new_phi = np.linspace(0.0, 0.40, 20)
low_phi_c = 0.40
high_frac_cem = 0.10


def test_constant_cement_model_high_phi(snapshot: SnapshotAssertion):
    assert snapshot == (
        result := constant_cement_model(
            k_min=k_min,
            mu_min=mu_min,
            rho_min=rho_min,
            k_cem=k_cem,
            mu_cem=mu_cem,
            rho_cem=rho_cem,
            k_fl=k_fl,
            rho_fl=rho_fl,
            phi=new_phi,
            frac_cem=high_frac_cem,
            phi_c=low_phi_c,
            n=n,
            shear_red=shear_red,
        )
    )
    # assert that there are NaN values in the output at the correct indices
    expected_idx_nan = new_phi > low_phi_c - high_frac_cem
    assert np.all(expected_idx_nan == np.isnan(result[0]))


def test_constant_cement_model_high_phi_extrapolate(snapshot: SnapshotAssertion):
    # Set flag for extrapolation to True and assert that there are no NaN values in the output
    assert snapshot == (
        result := constant_cement_model(
            k_min=k_min,
            mu_min=mu_min,
            rho_min=rho_min,
            k_cem=k_cem,
            mu_cem=mu_cem,
            rho_cem=rho_cem,
            k_fl=k_fl,
            rho_fl=rho_fl,
            phi=new_phi,
            frac_cem=high_frac_cem,
            phi_c=low_phi_c,
            n=n,
            shear_red=shear_red,
            extrapolate_to_max_phi=True,
        )
    )
    assert not np.any(np.isnan(result[0]))
