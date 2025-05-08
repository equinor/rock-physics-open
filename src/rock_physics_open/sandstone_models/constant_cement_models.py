import numpy as np
from scipy.interpolate import interp1d

from rock_physics_open.equinor_utilities import gen_utilities, std_functions


def constant_cement_model(
    k_min,
    mu_min,
    rho_min,
    k_cem,
    mu_cem,
    rho_cem,
    k_fl,
    rho_fl,
    phi,
    frac_cem,
    phi_c,
    n,
    shear_red,
    extrapolate_to_max_phi=False,
):
    """
    Constant cement model is a sandstone model that combined a cemented and a friable sand, so that a constant
    proportion of the rock volume is made up of grain-bonding cement. Variation in porosity is due to grain sorting,
    i.e. a well-sorted sand will have high porosity, and a poorly sorted one will have low porosity. In the extreme
    end all porosity is removed and the effective mineral properties is returned.

    Mineral properties k_min, mu_min, rho_min are effective properties. For mixtures of minerals, effective
    properties are calculated by Hashin-Shtrikman or similar.

    Cement properties k_cem, mu_cem, rho_cem and cement fraction, the latter as a part of the whole volume.

    Fluid properties k_fl, rho_fl are in situ properties calculated by fluid models using reservoir properties.

    Critical porosity phi_c is input to Dvorkin-Nur function. Coordination number n is normally set to a fixed
    value (default 9) but it is possible to  override this. Porosity phi is used in Hashin-Shtrikman mixing
    (together with phi_c) and Gassmann saturation.

    Shear reduction parameter shear_red is used to account for tangential frictionless grain contacts

    All inputs are assumed to be vectors of the same length

    Parameters
    ----------
    k_min : np.ndarray
        Mineral bulk modulus [Pa].
    mu_min : np.ndarray
        Mineral shear modulus [Pa].
    rho_min : np.ndarray
        Mineral bulk density [kg/m^3].
    k_cem : np.ndarray
        Cement bulk modulus [Pa].
    mu_cem : np.ndarray
        Cement shear modulus [Pa].
    rho_cem : np.ndarray
        Cement bulk density [kg/m^3].
    k_fl : np.ndarray
        Fluid bulk modulus [Pa].
    rho_fl : np.ndarray
        Fluid bulk density [kg/m^3].
    phi : np.ndarray
        Porosity [fraction].
    frac_cem : np.ndarray or float
        Cement fraction [fraction].
    phi_c : float
        Critical porosity [fraction].
    n : float
        Coordination number [unitless].
    shear_red : float
        Shear reduction factor [fraction].
    extrapolate_to_max_phi : bool
        If True, the model will extrapolate to the maximum porosity value (phi_c - frac_cem) if the input porosity
        exceeds this value. If False, the model will return NaN for porosity values exceeding this limit.

    Returns
    -------
    tuple
        vp, vs, rho, ai, vpvs  : (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        vp [m/s] and vs [m/s], bulk density [kg/m^3], ai [m/s x kg/m^3], vpvs [ratio] of saturated rock.
    """
    k_zero, k_dry, mu = constant_cement_model_dry(
        k_min,
        mu_min,
        k_cem,
        mu_cem,
        phi,
        frac_cem,
        phi_c,
        n,
        shear_red,
        extrapolate_to_max_phi=extrapolate_to_max_phi,
        return_k_zero=True,
    )
    # Saturated rock incompressibility is calculated with Gassmann
    k = std_functions.gassmann(k_dry, phi, k_fl, k_zero)

    # Bulk density
    rho = phi * rho_fl + (1 - phi - frac_cem) * rho_min + frac_cem * rho_cem
    # Velocity
    vp, vs, ai, vpvs = std_functions.velocity(k, mu, rho)

    return vp, vs, rho, ai, vpvs


def constant_cement_model_dry(
    k_min,
    mu_min,
    k_cem,
    mu_cem,
    phi,
    frac_cem,
    phi_c,
    n,
    shear_red,
    extrapolate_to_max_phi=False,
    return_k_zero=False,
):
    """
    Dry rock version of the constant cement model. The method is identical to the constant cement model function,
    except that a saturation step is not performed at the end.

    Parameters
    ----------
    k_min : np.ndarray
        Mineral bulk modulus [Pa].
    mu_min : np.ndarray
        Mineral shear modulus [Pa].
    k_cem : np.ndarray
        Cement bulk modulus [Pa].
    mu_cem : np.ndarray
        Cement shear modulus [Pa].
    phi : np.ndarray
        Porosity [fraction].
    frac_cem : float or np.ndarray
        Cement fraction [fraction].
    phi_c : float
        Critical porosity [fraction].
    n : float
        Coordination number [unitless].
    shear_red : float
        Shear reduction factor [fraction].
    extrapolate_to_max_phi : bool
        If True, the model will extrapolate to the maximum porosity value (phi_c - frac_cem) if the input porosity
        exceeds this value. If False, the model will return NaN for porosity values exceeding this limit.
    return_k_zero : bool
        If True, the model will return the zero-porosity end member bulk modulus k_zero in addition to the dry rock
        bulk modulus k_dry and shear modulus mu.

    Returns
    -------
    tuple
        k_dry, mu : (np.ndarray, np.ndarray).
        Bulk modulus k [Pa] and shear modulus mu [Pa] of dry rock.
    """
    # First check if there are input values that are unphysical, i.e. negative values, separate between dry and
    # saturated rock properties. Use the filter_input_log function to identify these values
    idx, (k_min, mu_min, k_cem, mu_cem, phi) = gen_utilities.filter_input_log(
        (k_min, mu_min, k_cem, mu_cem, phi)
    )

    # Identify porosity values that exceed (phi_c - frac_cem). This is regardless of setting for extrapolate_to_max_phi
    idx_phi = np.where(phi > phi_c - frac_cem)[0]

    # At the zero-porosity point, all original porosity (critical porosity -
    # cement fraction) is filled with grains. The cement fraction surrounds the
    # original grains, so they will be fraction 1 according to the geometrical
    # interpretation of Hashin-Shtrikman
    k_zero, mu_zero = std_functions.hashin_shtrikman_walpole(
        k_cem, mu_cem, k_min, mu_min, frac_cem, bound="lower"
    )

    # Dry rock properties of high-porosity end member calculated with
    # Dvorkin-Nur equation. Cement is assumed to be evenly distributed on the
    # grains (scheme 2 in Dvorkin and Nur's original paper)

    k_cc, mu_cc = std_functions.dvorkin_contact_cement(
        frac_cem, phi_c, mu_min, k_min, mu_cem, k_cem, shear_red, n
    )

    # Hashin-Shtrikman lower bound describes the dry rock property mixing from
    # mineral properties to high-end porosity.

    # Fraction of zero-porosity end member
    f1 = 1 - phi / (phi_c - frac_cem)

    k_dry, mu = std_functions.hashin_shtrikman_walpole(
        k_zero, mu_zero, k_cc, mu_cc, f1, bound="lower"
    )

    # If extrapolate_to_max_phi is True, create extrapolation functions for k_dry and mu using scipy's interp1d
    if extrapolate_to_max_phi:
        k_dry_func = interp1d(phi, k_dry, fill_value="extrapolate")
        mu_func = interp1d(phi, mu, fill_value="extrapolate")
        k_dry[idx_phi] = k_dry_func(phi[idx_phi])
        mu[idx_phi] = mu_func(phi[idx_phi])
    else:
        k_dry[idx_phi] = np.nan
        mu[idx_phi] = np.nan

    k_dry, mu = gen_utilities.filter_output(idx, (k_dry, mu))
    if return_k_zero:
        return k_zero, k_dry, mu
    return k_dry, mu
