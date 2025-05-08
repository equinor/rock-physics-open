from rock_physics_open.equinor_utilities import std_functions


def gassmann_sub_model(
    k_min,
    k_fl_orig,
    rho_fl_orig,
    k_fl_sub,
    rho_fl_sub,
    k_sat_orig,
    mu,
    rho_sat_orig,
    por,
):
    """
    Gassmann model to go from one saturated state to another.

    Parameters
    ----------
    k_min : np.ndarray
        Mineral bulk modulus [Pa].
    k_fl_orig : np.ndarray
        Original fluid  bulk modulus [Pa].
    rho_fl_orig : np.ndarray
        Original fluid density [lg/m^3].
    k_fl_sub : np.ndarray
        Sunstituted fluid  bulk modulus [Pa].
    rho_fl_sub : np.ndarray
        Substituted fluid density [lg/m^3].
    k_sat_orig : np.ndarray
        Saturated rock bulk modulus with original fluid  [Pa].
    mu : np.ndarray
        Rock shear modulus [Pa].
    rho_sat_orig : np.ndarray
        Saturated rock density with original fluid [kg/m^3].
    por : np.ndarray
        Porosity [fraction].

    Returns
    -------
    tuple
        vp_sat, vs_sat, rho_sat, ai_sat, vpvs_sat, k_sat, mu : (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray).
        vp_sat, vs_sat:  saturated velocities [m/s], rho_sat: saturated density [kg/m^3], ai_sat: saturated acoustic
        impedance [kg/m^3 x m/s], vpvs_sat: saturated velocity ratio [unitless], k_sat, mu: saturated bulk modulus and
        shear modulus (the latter unchanged from dry state) [Pa].
    """
    rho_sat_sub = rho_sat_orig + por * (rho_fl_sub - rho_fl_orig)
    k_sat_sub = std_functions.gassmann2(k_sat_orig, k_fl_orig, k_fl_sub, por, k_min)
    vp_sat_sub, vs_sat_sub, ai_sat_sub, vpvs_sat_sub = std_functions.velocity(
        k_sat_sub, mu, rho_sat_sub
    )

    return vp_sat_sub, vs_sat_sub, rho_sat_sub, ai_sat_sub, vpvs_sat_sub, k_sat_sub, mu
