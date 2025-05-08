from rock_physics_open.equinor_utilities import gen_utilities, std_functions


def contact_cement_model(
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
):
    """
    Contact cement model is a sandstone model where all variation in porosity is explained by variation in grain contact
    cement. This is leads to the stiffest transition from critical porosity to the mineral point.

    Mineral properties k_min, muMin, rho_min are effective properties. For mixtures of minerals, effective properties
    are calculated by Hashin-Shtrikman or similar.

    Cement properties kCem, muCem, rhoCem and cement fraction, the latter as a part of the whole volume.

    Fluid properties k_fl, rho_fl are in situ properties calculated by fluid models using reservoir properties.

    Critical porosity phiC is input to Dvorkin-Nur function. Coordination number n is normally set to a fixed value
    (default 9) but it is possible to  override this. Porosity phi is used in Hashin-Shtrikman mixing (together with
    phiC) and Gassmann saturation.

    Shear reduction parameter shearRed is used to account for tangential frictionless grain contacts.

    All inputs are assumed to be vectors of the same length.


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
    frac_cem : float
        Cement fraction [fraction].
    phi_c : float
        Critical porosity [fraction].
    n : float
        Coordination number [unitless].
    shear_red : float
        Shear reduction factor [fraction].

    Returns
    -------
    tuple
        vp, vs, rho, ai, vpvs  : (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        vp [m/s] and vs [m/s], bulk density [kg/m^3], ai [m/s x kg/m^3], vpvs [ratio] of saturated rock.
    """

    # Identify porosity values that are above (phi_c - frac_cem), for which the model is not defined
    (
        idx_phi,
        (k_min, mu_min, rho_min, k_cem, mu_cem, rho_cem, k_fl, rho_fl, phi, _),
    ) = gen_utilities.filter_input_log(
        (
            k_min,
            mu_min,
            rho_min,
            k_cem,
            mu_cem,
            rho_cem,
            k_fl,
            rho_fl,
            phi,
            phi_c - frac_cem - phi,
        )
    )

    # Expand input parameters to arrays
    phi, frac_cem, phi_c, n, shear_red = gen_utilities.dim_check_vector(
        (phi, frac_cem, phi_c, n, shear_red)
    )

    # At the zero-porosity point, all original porosity (critical porosity) is
    # filled with cement. The cement surrounds the original grains, so they
    # will be fraction 1 according to the geometrical interpretation of
    # Hashin-Shtrikman
    k_zero, mu_zero = std_functions.hashin_shtrikman_walpole(
        k_cem, mu_cem, k_min, mu_min, phi_c, bound="lower"
    )

    # Dry rock properties of high-porosity end member calculated with
    # Dvorkin-Nur equation. Cement is assumed to be evenly distributed on the
    # grains (scheme 2 in Dvorkin and Nur's original paper)

    k_cc, mu_cc = std_functions.dvorkin_contact_cement(
        frac_cem, phi_c, mu_zero, k_zero, mu_cem, k_cem, shear_red, n
    )

    # Hashin-Shtrikman upper bound describes the dry rock property mixing from
    # mineral properties to high-end porosity.

    # Fraction of zero-porosity end member
    f1 = 1 - phi / (phi_c - frac_cem)

    k_dry, mu = std_functions.hashin_shtrikman_walpole(
        k_zero, mu_zero, k_cc, mu_cc, f1, bound="upper"
    )

    # Saturated rock incompressibility is calculated with Gassmann
    k = std_functions.gassmann(k_dry, phi, k_fl, k_zero)

    # Bulk density
    rho = phi * rho_fl + (1 - phi_c) * rho_min + (phi_c - phi) * rho_cem

    vp, vs, ai, vpvs = std_functions.velocity(k, mu, rho)

    # Restore original array length
    vp, vs, rho, ai, vpvs = gen_utilities.filter_output(
        idx_phi, (vp, vs, rho, ai, vpvs)
    )

    return vp, vs, rho, ai, vpvs
