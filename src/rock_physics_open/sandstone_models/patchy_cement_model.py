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
    Patchy cement model for sands that are a combination of friable model and constant cement model.

    No fluid or pressure substitution.
    Input variables for weight of K and Mu determine the model's position between upper and lower bound.

    Parameters
    ----------
    k_min
        Mineral bulk modulus [Pa].
    mu_min
        Mineral shear modulus [Pa].
    rho_min
        Mineral bulk density [kg/m^3].
    k_cem
        Sandstone cement bulk modulus [Pa].
    mu_cem
        Sandstone cement shear modulus [Pa].
    rho_cem
        Cement bulk density [kg/m^3].
    k_fl
        Fluid bulk modulus [Pa].
    rho_fl
        Fluid bulk density [kg/m^3].
    phi
        Total porosity [fraction].
    p_eff
        Effective pressure [Pa].
    frac_cem
        Upper bound cement volume fraction [fraction].
    phi_c
        Critical porosity [fraction].
    coord_num_func
        Indication if coordination number should be calculated from porosity or kept constant.
    n
        Coordination number [unitless].
    shear_red
        Shear reduction factor for sandstone [fraction].
    weight_k
        Weight between friable and cemented model for bulk modulus.
    weight_mu
        Weight between friable and cemented model for shear modulus.

    Returns
    -------
    k
        Saturated rock bulk modulus [Pa].
    mu
        Shear modulus [Pa].
    rhob
        Saturated density after fluid and pressure substitution [kg/m^3].
    vp
        Saturated P-velocity after fluid and pressure substitution [m/s].
    vs
        Saturated S-velocity after fluid and pressure substitution [m/s].
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

    k_up, mu_up, _ = constant_cement_model_dry(
        k_min=k_min,
        mu_min=mu_min,
        k_cem=k_cem,
        mu_cem=mu_cem,
        phi=phi,
        frac_cem=frac_cem,
        phi_c=phi_c,
        n=n,
        shear_red=shear_red,
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
    Patchy cement model for sands that are a combination of friable model and constant cement model.

    No fluid or pressure substitution. In this implementation of the patchy cement model the given cement fraction for the constant
    cement model defines the upper bound, and the effective pressure for the friable model defines the lower bound.

    Parameters
    ----------
    k_min
        Mineral bulk modulus [Pa].
    mu_min
        Mineral shear modulus [Pa].
    rho_min
        Mineral bulk density [kg/m^3].
    k_cem
        Sandstone cement bulk modulus [Pa].
    mu_cem
        Sandstone cement shear modulus [Pa].
    rho_cem
        Cement bulk density [kg/m^3].
    k_fl
        Fluid bulk modulus [Pa].
    rho_fl
        Fluid bulk density [kg/m^3].
    phi
        Total porosity [fraction].
    p_eff
        Effective pressure [Pa].
    frac_cem
        Upper bound cement volume fraction [fraction].
    phi_c
        Critical porosity [fraction].
    coord_num_func
        Indication if coordination number should be calculated from porosity or kept constant, either "ConstVal" or
        "PoreBased" [default]
    n
        Coordination number [unitless].
    shear_red
        Shear reduction factor for sandstone [fraction].

    Returns
    -------
    vp
        Saturated P-velocity after fluid and pressure substitution [m/s].
    vs
        Saturated S-velocity after fluid and pressure substitution [m/s].
    rhob
        Saturated density after fluid and pressure substitution [kg/m^3].
    ai
        Saturated rock acoustic impedance after fluid and pressure substitution [kg/m^3 x m/s].
    vpvs
        Saturated rock velocity ratio [ratio].
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
    Patchy cement model for sands that are a combination of friable model and constant cement model.

    No fluid or pressure substitution. In this implementation of the patchy cement model the given cement fraction for the constant
    cement model defines the upper bound, and the effective pressure for the friable model defines the lower bound.

    Parameters
    ----------
    k_min
        Mineral bulk modulus [Pa].
    mu_min
        Mineral shear modulus [Pa].
    rho_min
        Mineral bulk density [kg/m^3].
    k_cem
        Sandstone cement bulk modulus [Pa].
    mu_cem
        Sandstone cement shear modulus [Pa].
    rho_cem
        Cement bulk density [kg/m^3].
    phi
        Total porosity [fraction].
    p_eff
        Effective pressure [Pa].
    frac_cem
        Upper bound cement volume fraction [fraction].
    phi_c
        Critical porosity [fraction].
    coord_num_func
        Indication if coordination number should be calculated from porosity or kept constant, either "ConstVal" or
        "PoreBased" [default]
    n
        Coordination number [unitless].
    shear_red
        Shear reduction factor for sandstone [fraction].

    Returns
    -------
    k_dry
        Dry rock bulk modulus [Pa].
    mu
        Dry rock shear modulus [Pa].
    rho_dry
        Dry rock density [kg/m^3].
    """
    # There are cases which suffer from a lack of consistency check at this stage,
    # add dim_check_vector and filter input/output
    phi, k_min, mu_min, rho_min, k_cem, mu_cem, rho_cem, p_eff = dim_check_vector(
        (phi, k_min, mu_min, rho_min, k_cem, mu_cem, rho_cem, p_eff)
    )
    (idx, (phi, k_min, mu_min, rho_min, k_cem, mu_cem, rho_cem, p_eff)) = (
        filter_input_log((phi, k_min, mu_min, rho_min, k_cem, mu_cem, rho_cem, p_eff))
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

    k_up, mu_up, _ = constant_cement_model_dry(
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

    k_dry, mu, rho_dry = filter_output(idx, (k_dry, mu, rho_dry))

    return k_dry, mu, rho_dry
