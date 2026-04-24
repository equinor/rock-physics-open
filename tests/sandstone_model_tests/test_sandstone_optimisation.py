from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from syrupy.assertion import SnapshotAssertion

from rock_physics_open.sandstone_models import (
    constant_cement_model_optimisation,
    friable_model_optimisation,
    patchy_cement_model_optimisation,
)


@dataclass(frozen=True)
class SandstoneOptParams:
    vp: npt.NDArray[np.float64]
    vs: npt.NDArray[np.float64]
    rhob: npt.NDArray[np.float64]
    phit: npt.NDArray[np.float64]
    k_min: npt.NDArray[np.float64]
    mu_min: npt.NDArray[np.float64]
    rho_min: npt.NDArray[np.float64]
    k_cem: npt.NDArray[np.float64]
    mu_cem: npt.NDArray[np.float64]
    rho_cem: npt.NDArray[np.float64]
    k_fl: npt.NDArray[np.float64]
    rho_fl: npt.NDArray[np.float64]
    p_eff: npt.NDArray[np.float64]
    phi_c: float


@pytest.fixture
def params(testdata: Path) -> SandstoneOptParams:
    data_df = pd.read_csv(testdata / "sandstone_optimisation.csv")
    phit = data_df["PHIT"].to_numpy()
    return SandstoneOptParams(
        vp=data_df["VP"].to_numpy(),
        vs=data_df["VS"].to_numpy(),
        rhob=data_df["RHOB"].to_numpy(),
        phit=phit,
        k_min=36.8e9 * np.ones_like(phit),
        mu_min=44.0e9 * np.ones_like(phit),
        rho_min=2650 * np.ones_like(phit),
        k_cem=36.8e9 * np.ones_like(phit),
        mu_cem=44.0e9 * np.ones_like(phit),
        rho_cem=2650 * np.ones_like(phit),
        k_fl=2.7e9 * np.ones_like(phit),
        rho_fl=1005 * np.ones_like(phit),
        p_eff=20.0e6 * np.ones_like(phit),
        phi_c=0.45,
    )


def test_friable_optimisation(
    snapshot: SnapshotAssertion,
    testdata: Path,
    params: SandstoneOptParams,
):
    assert snapshot == friable_model_optimisation(
        k_min=params.k_min,
        mu_min=params.mu_min,
        rho_min=params.rho_min,
        k_fl=params.k_fl,
        rho_fl=params.rho_fl,
        por=params.phit,
        p_eff=params.p_eff,
        vp=params.vp,
        vs=params.vs,
        rhob=params.rhob,
        file_out_str=str(testdata / "friable_model_optimisation.pkl"),
    )


def test_constant_cement_optimisation(
    snapshot: SnapshotAssertion,
    testdata: Path,
    params: SandstoneOptParams,
):
    assert snapshot == constant_cement_model_optimisation(
        k_min=params.k_min,
        mu_min=params.mu_min,
        rho_min=params.rho_min,
        k_cem=params.k_cem,
        mu_cem=params.mu_cem,
        rho_cem=params.rho_cem,
        k_fl=params.k_fl,
        rho_fl=params.rho_fl,
        por=params.phit,
        vp=params.vp,
        vs=params.vs,
        rhob=params.rhob,
        file_out_str=str(testdata / "constant_cement_model_optimisation.pkl"),
    )


def test_patchy_cement_optimisation(
    snapshot: SnapshotAssertion,
    testdata: Path,
    params: SandstoneOptParams,
):
    assert snapshot == patchy_cement_model_optimisation(
        k_min=params.k_min,
        mu_min=params.mu_min,
        rho_min=params.rho_min,
        k_cem=params.k_cem,
        mu_cem=params.mu_cem,
        rho_cem=params.rho_cem,
        k_fl=params.k_fl,
        rho_fl=params.rho_fl,
        por=params.phit,
        p_eff=params.p_eff,
        vp=params.vp,
        vs=params.vs,
        rhob=params.rhob,
        phi_c=params.phi_c,
        file_out_str=str(testdata / "patchy_cement_model_optimisation.pkl"),
    )
