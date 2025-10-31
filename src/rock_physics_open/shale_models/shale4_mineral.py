from typing import Literal, cast

import numpy as np
import numpy.typing as npt

from rock_physics_open.equinor_utilities import std_functions
from rock_physics_open.equinor_utilities.gen_utilities import dim_check_vector

from .dem import dem_model
from .multi_sca import multi_sca


def shale_model_4_mineral_dem(
    k1: npt.NDArray[np.float64],
    mu1: npt.NDArray[np.float64],
    rho1: npt.NDArray[np.float64],
    k2: npt.NDArray[np.float64],
    mu2: npt.NDArray[np.float64],
    rho2: npt.NDArray[np.float64],
    k3: npt.NDArray[np.float64],
    mu3: npt.NDArray[np.float64],
    rho3: npt.NDArray[np.float64],
    k4: npt.NDArray[np.float64],
    mu4: npt.NDArray[np.float64],
    rho4: npt.NDArray[np.float64],
    k_fl: npt.NDArray[np.float64],
    rho_fl: npt.NDArray[np.float64],
    phi: npt.NDArray[np.float64],
    f1: npt.NDArray[np.float64],
    f2: npt.NDArray[np.float64],
    f3: npt.NDArray[np.float64],
    rhob_inp: npt.NDArray[np.float64],
    asp1: npt.NDArray[np.float64],
    asp2: npt.NDArray[np.float64],
    asp3: npt.NDArray[np.float64],
    asp4: npt.NDArray[np.float64],
    asp: npt.NDArray[np.float64],
    mod_type: Literal["SCA", "VRH"] = "SCA",
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """
    Simple shale model with mixture of matrix minerals in self-consistent approximation
    or Voigt-Reuss-Hill average. In the latter case the aspect ratios are ignored.
    Fluid filled porosity is included through a DEM inclusion model.

    All k, mu inputs have unit [Pa], all rho inputs have unit [kg/m^3], phi and all f and asp have unit [fraction].

    Parameters
    ----------
    k1, mu1, rho1 : np.ndarray
        Mineral 1 properties [Quartz and feldspar].
    k2, mu2, rho2 : np.ndarray
        Mineral 2 properties [Kerogen].
    k3, mu3, rho3 : np.ndarray
        Mineral 3 properties [Clay].
    k4, mu4, rho4 : np.ndarray
        Mineral 4 properties [Carbonates].
    k_fl, rho_fl : np.ndarray
        Fluid properties.
    phi : np.ndarray
        Phi parameter.
    f1 : np.ndarray
        Fraction of mineral 1.
    f2 : np.ndarray
        Fraction of mineral 2.
    f3 : np.ndarray
        Fraction of mineral 3.
    rhob_inp : np.ndarray
        Observed density [kg/m^3].
    asp1 : np.ndarray
        Aspect ratio mineral 1 inclusions.
    asp2 : np.ndarray
        Aspect ratio mineral 2 inclusions.
    asp3 : np.ndarray
        Aspect ratio mineral 3 inclusions.
    asp4 : np.ndarray
        Aspect ratio mineral 4 inclusions.
    asp : np.ndarray
        Porosity aspect ratio.
    mod_type : str
        One of 'SCA' or 'VRH'.

    Returns
    -------
    tuple
        k, mu, rhob, vp, vs, rho_factor : (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray).
        k - effective bulk modulus [Pa], mu - effective shear modulus [Pa], rhob - effective density [kg/m^3],
        vp - p-wave velocity [m/s], vs - shear wave velocity [m/s], rho_factor - ratio between modelled and observed
        density [ratio].
    """

    #   tol: DEM model calculation tolerance <= Set as a hardcoded value, not
    #   found to influence the results
    tol = 1.0e-6

    # Calculate effective mineral properties

    # Too harsh to raise exceptions - do a quiet normalisation instead,
    tot = f1 + f2 + f3
    idx = tot > 1.0
    f1[idx] /= tot[idx]
    f2[idx] /= tot[idx]
    f3[idx] /= tot[idx]
    f4 = 1.0 - f1 - f2 - f3

    if mod_type == "SCA":
        k_mat, mu_mat, rho_mat = multi_sca(
            k1,
            mu1,
            rho1,
            f1,
            asp1,
            k2,
            mu2,
            rho2,
            f2,
            asp2,
            k3,
            mu3,
            rho3,
            f3,
            asp3,
            k4,
            mu4,
            rho4,
            f4,
            asp4,
            tol=tol,
        )
    elif mod_type == "VRH":
        k_mat, mu_mat = std_functions.multi_voigt_reuss_hill(
            k1, mu1, f1, k2, mu2, f2, k3, mu3, f3, k4, mu4, f4
        )
        rho_mat = f1 * rho1 + f2 * rho2 + f3 * rho3 + f4 * rho4
    else:
        raise ValueError(
            f'{__file__}: unknown type: {mod_type}, should be one of "SCA", "VRH"'  # pyright: ignore[reportUnreachable] | Kept for backward compatibility
        )

    k_mat, mu_mat, rho_mat, k_fl, rho_fl, phi, asp = cast(
        list[npt.NDArray[np.float64]],
        dim_check_vector((k_mat, mu_mat, rho_mat, k_fl, rho_fl, phi, asp)),
    )
    k, mu, rhob = dem_model(
        k1=k_mat,
        mu1=mu_mat,
        rho1=rho_mat,
        k2=k_fl,
        mu2=np.zeros(len(k_fl)),
        rho2=rho_fl,
        frac2=phi,
        asp2=asp,
        tol=tol,
    )

    rho_factor = rhob / rhob_inp

    vp, vs, _, _ = std_functions.velocity(
        k=k,
        mu=mu,
        rhob=rhob,
    )

    return k, mu, rhob, vp, vs, rho_factor
