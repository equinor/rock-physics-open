import numpy as np
import numpy.typing as npt

from .dem import dem_model


def dem_model_dual_por(
    k1: npt.NDArray[np.float64],
    mu1: npt.NDArray[np.float64],
    rho1: npt.NDArray[np.float64],
    k2: npt.NDArray[np.float64],
    mu2: npt.NDArray[np.float64],
    rho2: npt.NDArray[np.float64],
    k3: npt.NDArray[np.float64],
    mu3: npt.NDArray[np.float64],
    rho3: npt.NDArray[np.float64],
    frac_inc: npt.NDArray[np.float64],
    frac_inc_1: npt.NDArray[np.float64],
    asp_1: npt.NDArray[np.float64],
    asp_2: npt.NDArray[np.float64],
    tol: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Differential effective media model with two sets of inclusions.

    Parameters
    ----------
    k1 : np.ndarray
        Matrix bulk modulus [Pa].
    mu1 : np.ndarray
        Matrix shear modulus [Pa].
    rho1 : np.ndarray
        Matrix bulk density [kg/m^3].
    k2 : np.ndarray
        Inclusion 1 bulk modulus [Pa].
    mu2 : np.ndarray
        Inclusion 1 shear modulus [Pa].
    rho2 : np.ndarray
        Inclusion 1 bulk density [kg/m^3]
    k3 : np.ndarray
        Inclusion 2 bulk modulus [Pa].
    mu3 : np.ndarray
        Inclusion 2 shear modulus [Pa].
    rho3 : np.ndarray
        Inclusion 2 bulk density [kg/m^3].
    frac_inc : np.ndarray
        Total fraction of inclusions [fraction].
    frac_inc_1 : np.ndarray
        Fraction of inclusions belonging to type 1 [fraction].
    asp_1 : np.ndarray
        Aspect ratio of inclusion 1 [fraction].
    asp_2 : np.ndarray
        Aspect ratio of inclusion 2 [fraction].
    tol : float
        Tolerance of accuracy [unitless].

    Returns
    -------
    tuple
        k_dem_dual, mu_dem_dual, rhob_dem_dual : np.ndarray
        k_dem_dual: bulk modulus [Pa], mu_dem_dual: shear modulus [Pa], rhob_dem_dual: bulk density [kg/m^3].
    """
    # Include the Type 1 inclusions into the matrix first, then run again with inclusions Type 2
    k_dem1, mu_dem1, rhob_dem1 = dem_model(
        k1=k1,
        mu1=mu1,
        rho1=rho1,
        k2=k2,
        mu2=mu2,
        rho2=rho2,
        frac2=frac_inc * frac_inc_1,
        asp2=asp_1,
        tol=tol,
    )
    k_dem_dual, mu_dem_dual, rhob_dem_dual = dem_model(
        k1=k_dem1,
        mu1=mu_dem1,
        rho1=rhob_dem1,
        k2=k3,
        mu2=mu3,
        rho2=rho3,
        frac2=frac_inc * (1.0 - frac_inc_1),
        asp2=asp_2,
        tol=tol,
    )

    return k_dem_dual, mu_dem_dual, rhob_dem_dual
