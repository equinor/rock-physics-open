import numpy as np
from syrupy.assertion import SnapshotAssertion

from rock_physics_open.equinor_utilities.std_functions import (
    hashin_shtrikman,
    hashin_shtrikman_average,
    hashin_shtrikman_walpole,
    multi_hashin_shtrikman,
)

k1 = np.ones(11) * 36.6e9
mu1 = np.ones(11) * 44.0e9
k2 = np.ones(11) * 71.0e9
mu2 = np.ones(11) * 32.0e9
f1 = np.linspace(0, 1, 11)


def test_hs(snapshot: SnapshotAssertion):
    assert snapshot == hashin_shtrikman(k1, mu1, k2, mu2, f1)


def test_hs_ave(snapshot: SnapshotAssertion):
    assert snapshot == hashin_shtrikman_average(k1, mu1, k2, mu2, f1)


def test_hsw(snapshot: SnapshotAssertion):
    assert snapshot == hashin_shtrikman_walpole(k1, mu1, k2, mu2, f1, bound="lower")


def test_multi_hs(snapshot: SnapshotAssertion):
    assert snapshot == multi_hashin_shtrikman(
        k1, mu1, f1, k2, mu2, 1.0 - f1, mode="lower"
    )
