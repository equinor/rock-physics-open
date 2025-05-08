import numpy as np

from rock_physics_open.equinor_utilities import std_functions

from .constant_cement_models import constant_cement_model_dry
from .friable_models import friable_model_dry

FRAC_CEM_UP = 0.1
P_EFF_LOW = 20.0e6


def constant_cement_model_pcm(
    kmin, mymin, kcem, mycem, kzero, myzero, phi, cem_frac, phic, n, red_shear
):
    """
    kdry, mydry = contantcementmodel_pcm(kmin, mymin, kcem, mycem, kzero, myzero, phi, cem_frac, phic, n, red_shear)
    """

    #   Contact cement model (Dvorkin-Nur)for given cem_frac
    kcc, mycc = std_functions.dvorkin_contact_cement(
        cem_frac, phic, mymin, kmin, mycem, kcem, red_shear, n
    )

    #   Fraction of zero-porosity end member
    f1 = 1 - phi / (phic - cem_frac)

    #   Interpolating using Hashin -Shtrikman lower bound = Constant cement model.
    #   Same mineral point as upper and lower bound in patchy cement model
    kdry, mydry = std_functions.hashin_shtrikman_walpole(
        kzero, myzero, kcc, mycc, f1, bound="lower"
    )

    return kdry, mydry


def patchy_cement_model_weight(
    k_min,
    mu_min,
    rho_min,
    k_cem,
    mu_cem,
    rho_cem,
    k_fl,
    rho_fl,
    phi,
    p_eff,
    frac_cem,
    phi_c,
    coord_num_func,
    n,
    shear_red,
    weight_k,
    weight_mu,
):
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
    shear_red : float
        Shear reduction factor for sandstone [fraction].
    phi_c : float
        Critical porosity [fraction].
    n : float
        Coordination number [unitless].
    coord_num_func : str
        Indication if coordination number should be calculated from porosity or kept constant.
    weight_k : float
        Weight between friable and cemented model for bulk modulus.
    weight_mu : float
        Weight between friable and cemented model for shear modulus.

    Returns
    -------
    tuple
        k, mu, rhob, vp, vs : (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray).
        vp :Saturated P-velocity [m/s] after fluid and pressure substitution,
        vs : Saturated S-velocity [m/s] after fluid and pressure substitution,
        rhob : Saturated density [kg/m3] after fluid and pressure substitution,
        k : Saturated rock bulk modulus [Pa],
        mu : Shear modulus [Pa].
    """

    k_zero, mu_zero = std_functions.hashin_shtrikman_walpole(
        k_cem, mu_cem, k_min, mu_min, frac_cem, bound="lower"
    )

    # In this implementation of the patchy cement model the given cement fraction for the constant cement model defines
    # the upper bound, and the effective pressure for the friable model defines the lower bound

    k_fri, mu_fri = friable_model_dry(
        k_zero, mu_zero, phi, p_eff, phi_c, coord_num_func, n, shear_red
    )[0:2]

    k_up, mu_up = constant_cement_model_dry(
        k_min,
        mu_min,
        k_cem,
        mu_cem,
        phi,
        frac_cem,
        phi_c,
        n,
        shear_red,
        extrapolate_to_max_phi=True,
    )[0:2]

    k_dry = k_fri + weight_k * (k_up - k_fri)
    mu = mu_fri + weight_mu * (mu_up - mu_fri)

    k = std_functions.gassmann(k_dry, phi, k_fl, k_zero)

    weight_rho = 0.5 * (weight_k + weight_mu)
    rhob = (
        phi * rho_fl
        + (1 - phi - frac_cem * weight_rho) * rho_min
        + frac_cem * weight_rho * rho_cem
    )

    vp, vs, ai, vpvs = std_functions.velocity(k, mu, rhob)

    return vp, vs, rhob, ai, vpvs


def patchy_cement_model_cem_frac(
    k_min,
    mu_min,
    rho_min,
    k_cem,
    mu_cem,
    rho_cem,
    k_fl,
    rho_fl,
    phi,
    p_eff,
    frac_cem,
    phi_c,
    coord_num_func,
    n,
    shear_red,
):
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
    shear_red : float
        Shear reduction factor for sandstone [fraction].
    phi_c : float
        Critical porosity [fraction].
    n : float
        Coordination number [unitless].
    coord_num_func : str
        Indication if coordination number should be calculated from porosity or kept constant, either "ConstVal" or
        "PoreBased" [default]

    Returns
    -------
    tuple
        vp, vs, rhob, ai, vpvs : (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray).
        vp :Saturated P-velocity [m/s] after fluid and pressure substitution,
        vs : Saturated S-velocity [m/s] after fluid and pressure substitution,
        rhob : Saturated density [kg/m3] after fluid and pressure substitution,
        ai : Saturated rock acoustic impedance [kg/m3 * m/s] after fluid and pressure substitution,
        vpvs : Saturated rock velocity ratio [ratio].
    """

    k_zero, mu_zero = std_functions.hashin_shtrikman_walpole(
        k_cem, mu_cem, k_min, mu_min, FRAC_CEM_UP, bound="lower"
    )

    k_low, mu_low = friable_model_dry(
        k_zero,
        mu_zero,
        phi,
        P_EFF_LOW * np.ones_like(phi),
        phi_c,
        coord_num_func,
        n,
        shear_red,
    )[0:2]

    k_fri, mu_fri = friable_model_dry(
        k_zero, mu_zero, phi, p_eff, phi_c, coord_num_func, n, shear_red
    )[0:2]

    k_up, mu_up = constant_cement_model_dry(
        k_min,
        mu_min,
        k_cem,
        mu_cem,
        phi,
        FRAC_CEM_UP,
        phi_c,
        n,
        shear_red,
        extrapolate_to_max_phi=True,
    )[0:2]

    # Special case for the constant cement model that represents the mean of the data
    k_cc, mu_cc = constant_cement_model_pcm(
        k_min,
        mu_min,
        k_cem,
        mu_cem,
        k_zero,
        mu_zero,
        phi,
        frac_cem,
        phi_c,
        n,
        shear_red,
    )[0:2]

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

    k = std_functions.gassmann(k_dry, phi, k_fl, k_zero)

    rhob = phi * rho_fl + (1 - phi - frac_cem) * rho_min + frac_cem * rho_cem

    vp, vs, ai, vpvs = std_functions.velocity(k, mu, rhob)

    return vp, vs, rhob, ai, vpvs
