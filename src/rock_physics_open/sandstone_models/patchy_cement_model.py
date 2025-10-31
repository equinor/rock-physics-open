from typing import cast

import numpy as np
import numpy.typing as npt

from rock_physics_open.equinor_utilities import std_functions
from rock_physics_open.equinor_utilities.gen_utilities import (
    dim_check_vector,
    filter_input_log,
    filter_output,
)

from .constant_cement_models import constant_cement_model_dry
from .friable_models import CoordinateNumberFunction, friable_model_dry

FRAC_CEM_UP = 0.1
P_EFF_LOW = 20.0e6


def constant_cement_model_pcm(
    kmin: npt.NDArray[np.float64],
    mymin: npt.NDArray[np.float64],
    kcem: npt.NDArray[np.float64],
    mycem: npt.NDArray[np.float64],
    kzero: npt.NDArray[np.float64],
    myzero: npt.NDArray[np.float64],
    phi: npt.NDArray[np.float64] | float,
    cem_frac: npt.NDArray[np.float64] | float,
    phic: npt.NDArray[np.float64] | float,
    n: float,
    red_shear: float,
):
    """
    kdry, mydry = contantcementmodel_pcm(kmin, mymin, kcem, mycem, kzero, myzero, phi, cem_frac, phic, n, red_shear)
    """

    #   Contact cement model (Dvorkin-Nur)for given cem_frac
    kcc, mycc = std_functions.dvorkin_contact_cement(
        frac_cem=cem_frac,
        por0_sst=phic,
        mu0_sst=mymin,
        k0_sst=kmin,
        mu0_cem=mycem,
        k0_cem=kcem,
        vs_red=red_shear,
        c=n,
    )

    #   Fraction of zero-porosity end member
    f1 = 1 - phi / (phic - cem_frac)

    #   Interpolating using Hashin -Shtrikman lower bound = Constant cement model.
    #   Same mineral point as upper and lower bound in patchy cement model
    kdry, mydry = std_functions.hashin_shtrikman_walpole(
        k1=kzero,
        mu1=myzero,
        k2=kcc,
        mu2=mycc,
        f1=f1,
        bound="lower",
    )

    return kdry, mydry


def patchy_cement_model_weight(
    k_min: npt.NDArray[np.float64],
    mu_min: npt.NDArray[np.float64],
    rho_min: npt.NDArray[np.float64],
    k_cem: npt.NDArray[np.float64],
    mu_cem: npt.NDArray[np.float64],
    rho_cem: npt.NDArray[np.float64],
    k_fl: npt.NDArray[np.float64],
    rho_fl: npt.NDArray[np.float64],
    phi: npt.NDArray[np.float64],
    p_eff: npt.NDArray[np.float64],
    frac_cem: npt.NDArray[np.float64] | float,
    phi_c: float,
    coord_num_func: CoordinateNumberFunction,
    n: float,
    shear_red: npt.NDArray[np.float64] | float,
    weight_k: npt.NDArray[np.float64] | float,
    weight_mu: npt.NDArray[np.float64] | float,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """
    Patchy cement model for sands that are a combination of friable model and constant cement model. No fluid or
    pressure substitution. Input variables for weight of K and Mu determine the model's position between upper and
    lower bound.

    Parameters
    ----------
    k_min : np.ndarray
        Mineral bulk modulus [Pa].
    mu_min : np.ndarray
        Mineral shear modulus [Pa].
    rho_min : np.ndarray
        Mineral bulk density [kg/m^3].
    k_cem : np.ndarray
        Sandstone cement bulk modulus [Pa].
    mu_cem : np.ndarray
        Sandstone cement shear modulus [Pa].
    rho_cem : np.ndarray
        Cement bulk density [kg/m^3].
    k_fl : np.ndarray
        Fluid bulk modulus [Pa].
    rho_fl : np.ndarray
        Fluid bulk density [kg/m^3].
    phi : np.ndarray
        Total porosity [fraction].
    p_eff : np.ndarray
        Effective pressure [Pa].
    frac_cem : float
        Upper bound cement volume fraction [fraction].
    phi_c : float
        Critical porosity [fraction].
    coord_num_func : str
        Indication if coordination number should be calculated from porosity or kept constant.
    n : float
        Coordination number [unitless].
    shear_red : float
        Shear reduction factor for sandstone [fraction].
    weight_k : float
        Weight between friable and cemented model for bulk modulus.
    weight_mu : float
        Weight between friable and cemented model for shear modulus.

    Returns
    -------
    tuple
        k, mu, rhob, vp, vs : np.ndarray.
        vp :Saturated P-velocity [m/s] after fluid and pressure substitution,
        vs : Saturated S-velocity [m/s] after fluid and pressure substitution,
        rhob : Saturated density [kg/m3] after fluid and pressure substitution,
        k : Saturated rock bulk modulus [Pa],
        mu : Shear modulus [Pa].
    """

    k_zero, mu_zero = std_functions.hashin_shtrikman_walpole(
        k1=k_cem,
        mu1=mu_cem,
        k2=k_min,
        mu2=mu_min,
        f1=frac_cem,
        bound="lower",
    )

    # In this implementation of the patchy cement model the given cement fraction for the constant cement model defines
    # the upper bound, and the effective pressure for the friable model defines the lower bound

    k_fri, mu_fri = friable_model_dry(
        k_min=k_zero,
        mu_min=mu_zero,
        phi=phi,
        p_eff=p_eff,
        phi_c=phi_c,
        coord_num_func=coord_num_func,
        n=n,
        shear_red=shear_red,
    )

    k_up, mu_up = constant_cement_model_dry(
        k_min=k_min,
        mu_min=mu_min,
        k_cem=k_cem,
        mu_cem=mu_cem,
        phi=phi,
        frac_cem=frac_cem,
        phi_c=phi_c,
        n=n,
        shear_red=shear_red,
        return_k_zero=False,
        extrapolate_to_max_phi=True,
    )

    k_dry = k_fri + weight_k * (k_up - k_fri)
    mu = mu_fri + weight_mu * (mu_up - mu_fri)

    k = std_functions.gassmann(
        k_dry,
        por=phi,
        k_fl=k_fl,
        k_min=k_zero,
    )

    weight_rho = 0.5 * (weight_k + weight_mu)
    rhob = (
        phi * rho_fl
        + (1 - phi - frac_cem * weight_rho) * rho_min
        + frac_cem * weight_rho * rho_cem
    )

    vp, vs, ai, vpvs = std_functions.velocity(k=k, mu=mu, rhob=rhob)

    return vp, vs, rhob, ai, vpvs


def patchy_cement_model_cem_frac(
    k_min: npt.NDArray[np.float64],
    mu_min: npt.NDArray[np.float64],
    rho_min: npt.NDArray[np.float64],
    k_cem: npt.NDArray[np.float64],
    mu_cem: npt.NDArray[np.float64],
    rho_cem: npt.NDArray[np.float64],
    k_fl: npt.NDArray[np.float64],
    rho_fl: npt.NDArray[np.float64],
    phi: npt.NDArray[np.float64],
    p_eff: npt.NDArray[np.float64],
    frac_cem: float,
    phi_c: float,
    coord_num_func: CoordinateNumberFunction,
    n: float,
    shear_red: float,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """
    Patchy cement model for sands that are a combination of friable model and constant cement model. No fluid or
    pressure substitution. In this implementation of the patchy cement model the given cement fraction for the constant
    cement model defines the upper bound, and the effective pressure for the friable model defines the lower bound

    Parameters
    ----------
    k_min : np.ndarray
        Mineral bulk modulus [Pa].
    mu_min : np.ndarray
        Mineral shear modulus [Pa].
    rho_min : np.ndarray
        Mineral bulk density [kg/m^3].
    k_cem : np.ndarray
        Sandstone cement bulk modulus [Pa].
    mu_cem : np.ndarray
        Sandstone cement shear modulus [Pa].
    rho_cem : np.ndarray
        Cement bulk density [kg/m^3].
    k_fl : np.ndarray
        Fluid bulk modulus [Pa].
    rho_fl : np.ndarray
        Fluid bulk density [kg/m^3].
    phi : np.ndarray
        Total porosity [fraction].
    p_eff : np.ndarray
        Effective pressure [Pa].
    frac_cem : float
        Upper bound cement volume fraction [fraction].
    phi_c : float
        Critical porosity [fraction].
    coord_num_func : str
        Indication if coordination number should be calculated from porosity or kept constant, either "ConstVal" or
        "PoreBased" [default]
    n : float
        Coordination number [unitless].
    shear_red : float
        Shear reduction factor for sandstone [fraction].

    Returns
    -------
    tuple
        vp, vs, rhob, ai, vpvs : np.ndarray.
        vp :Saturated P-velocity [m/s] after fluid and pressure substitution,
        vs : Saturated S-velocity [m/s] after fluid and pressure substitution,
        rhob : Saturated density [kg/m3] after fluid and pressure substitution,
        ai : Saturated rock acoustic impedance [kg/m3 * m/s] after fluid and pressure substitution,
        vpvs : Saturated rock velocity ratio [ratio].
    """

    k_dry, mu, _ = patchy_cement_model_dry(
        k_min=k_min,
        mu_min=mu_min,
        rho_min=rho_min,
        k_cem=k_cem,
        mu_cem=mu_cem,
        rho_cem=rho_cem,
        phi=phi,
        p_eff=p_eff,
        frac_cem=frac_cem,
        phi_c=phi_c,
        coord_num_func=coord_num_func,
        n=n,
        shear_red=shear_red,
    )
    k_zero, _ = std_functions.hashin_shtrikman_walpole(
        k1=k_cem,
        mu1=mu_cem,
        k2=k_min,
        mu2=mu_min,
        f1=FRAC_CEM_UP,
        bound="lower",
    )

    k = std_functions.gassmann(
        k_dry=k_dry,
        por=phi,
        k_fl=k_fl,
        k_min=k_zero,
    )

    rhob = phi * rho_fl + (1 - phi - frac_cem) * rho_min + frac_cem * rho_cem

    vp, vs, ai, vpvs = std_functions.velocity(k=k, mu=mu, rhob=rhob)

    return vp, vs, rhob, ai, vpvs


def patchy_cement_model_dry(
    k_min: npt.NDArray[np.float64],
    mu_min: npt.NDArray[np.float64],
    rho_min: npt.NDArray[np.float64],
    k_cem: npt.NDArray[np.float64],
    mu_cem: npt.NDArray[np.float64],
    rho_cem: npt.NDArray[np.float64],
    phi: npt.NDArray[np.float64],
    p_eff: npt.NDArray[np.float64],
    frac_cem: float,
    phi_c: float,
    coord_num_func: CoordinateNumberFunction,
    n: float,
    shear_red: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Patchy cement model for sands that are a combination of friable model and constant cement model. No fluid or
    pressure substitution. In this implementation of the patchy cement model the given cement fraction for the constant
    cement model defines the upper bound, and the effective pressure for the friable model defines the lower bound

    Parameters
    ----------
    k_min : np.ndarray
        Mineral bulk modulus [Pa].
    mu_min : np.ndarray
        Mineral shear modulus [Pa].
    rho_min : np.ndarray
        Mineral bulk density [kg/m^3].
    k_cem : np.ndarray
        Sandstone cement bulk modulus [Pa].
    mu_cem : np.ndarray
        Sandstone cement shear modulus [Pa].
    rho_cem : np.ndarray
        Cement bulk density [kg/m^3].
    phi : np.ndarray
        Total porosity [fraction].
    p_eff : np.ndarray
        Effective pressure [Pa].
    frac_cem : float
        Upper bound cement volume fraction [fraction].
    phi_c : float
        Critical porosity [fraction].
    coord_num_func : str
        Indication if coordination number should be calculated from porosity or kept constant, either "ConstVal" or
        "PoreBased" [default]
    n : float
        Coordination number [unitless].
    shear_red : float
        Shear reduction factor for sandstone [fraction].

    Returns
    -------
    tuple
        k:dry, mu, rho_dry : np.ndarray
        k_dry: dry rock bulk modulus [Pa],
        mu : dry rock shear modulus [Pa],
        rho_dry : dry rock density [kg/m3].,
    """
    # There are cases which suffer from a lack of consistency check at this stage,
    # add dim_check_vector and filter input/output
    phi, k_min, mu_min, rho_min, k_cem, mu_cem, rho_cem, p_eff = cast(
        list[npt.NDArray[np.float64]],
        dim_check_vector((phi, k_min, mu_min, rho_min, k_cem, mu_cem, rho_cem, p_eff)),
    )
    (idx, (phi, k_min, mu_min, rho_min, k_cem, mu_cem, rho_cem, p_eff)) = cast(
        tuple[npt.NDArray[np.bool_], list[npt.NDArray[np.float64]]],
        filter_input_log((phi, k_min, mu_min, rho_min, k_cem, mu_cem, rho_cem, p_eff)),
    )

    k_zero, mu_zero = std_functions.hashin_shtrikman_walpole(
        k1=k_cem,
        mu1=mu_cem,
        k2=k_min,
        mu2=mu_min,
        f1=FRAC_CEM_UP,
        bound="lower",
    )

    k_low, mu_low = friable_model_dry(
        k_min=k_zero,
        mu_min=mu_zero,
        phi=phi,
        p_eff=P_EFF_LOW * np.ones_like(phi),
        phi_c=phi_c,
        coord_num_func=coord_num_func,
        n=n,
        shear_red=shear_red,
    )

    k_fri, mu_fri = friable_model_dry(
        k_min=k_zero,
        mu_min=mu_zero,
        phi=phi,
        p_eff=p_eff,
        phi_c=phi_c,
        coord_num_func=coord_num_func,
        n=n,
        shear_red=shear_red,
    )

    k_up, mu_up = constant_cement_model_dry(
        k_min=k_min,
        mu_min=mu_min,
        k_cem=k_cem,
        mu_cem=mu_cem,
        phi=phi,
        frac_cem=FRAC_CEM_UP,
        phi_c=phi_c,
        n=n,
        shear_red=shear_red,
        extrapolate_to_max_phi=True,
        return_k_zero=False,
    )

    # Special case for the constant cement model that represents the mean of the data
    k_cc, mu_cc = constant_cement_model_pcm(
        kmin=k_min,
        mymin=mu_min,
        kcem=k_cem,
        mycem=mu_cem,
        kzero=k_zero,
        myzero=mu_zero,
        phi=phi,
        cem_frac=frac_cem,
        phic=phi_c,
        n=n,
        red_shear=shear_red,
    )

    idwk = k_up == k_low
    idwmu = mu_up == mu_low

    weight_k = np.ones(k_zero.shape)
    weight_mu = np.ones(mu_zero.shape)

    weight_k[~idwk] = (k_cc[~idwk] - k_low[~idwk]) / (k_up[~idwk] - k_low[~idwk])
    weight_mu[~idwmu] = (mu_cc[~idwmu] - mu_low[~idwmu]) / (
        mu_up[~idwmu] - mu_low[~idwmu]
    )

    weight_mu = np.clip(weight_mu, 0.0, 1.0)
    weight_k = np.clip(weight_k, 0.0, 1.0)

    k_dry = k_fri + weight_k * (k_up - k_fri)
    mu = mu_fri + weight_mu * (mu_up - mu_fri)

    rho_dry = (1 - phi - frac_cem) * rho_min + frac_cem * rho_cem

    k_dry, mu, rho_dry = cast(
        list[npt.NDArray[np.float64]], filter_output(idx, (k_dry, mu, rho_dry))
    )

    return k_dry, mu, rho_dry
