import numpy as np
from syrupy.assertion import SnapshotAssertion

from rock_physics_open.equinor_utilities.std_functions import (
    multi_voigt_reuss_hill,
    reuss,
    voigt,
    voigt_reuss_hill,
)

k1 = np.ones(11) * 36.6e9
mu1 = np.ones(11) * 44.0e9
k2 = np.ones(11) * 71.0e9
mu2 = np.ones(11) * 32.0e9
f1 = np.linspace(0, 1, 11)


def test_voigt(snapshot: SnapshotAssertion):
    assert snapshot == voigt(k1, mu1, k2, mu2, f1)


def test_reuss(snapshot: SnapshotAssertion):
    assert snapshot == reuss(k1, mu1, k2, mu2, f1)


def test_voigt_reuss_hill(snapshot: SnapshotAssertion):
    assert snapshot == voigt_reuss_hill(k1, mu1, k2, mu2, f1)


def test_multi_voigt_reuss_hill(snapshot: SnapshotAssertion):
    assert snapshot == multi_voigt_reuss_hill(k1, mu1, f1, k2, mu2, 1.0 - f1)
