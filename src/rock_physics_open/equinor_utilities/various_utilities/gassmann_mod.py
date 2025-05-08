from rock_physics_open.equinor_utilities import std_functions


def gassmann_model(k_min, k_fl, rho_fl, k_dry, mu, rho_dry, por):
    """
    Gassmann model to go from dry rock to saturated state.

    Parameters
    ----------
    k_min : np.ndarray
        Mineral bulk modulus [Pa].
    k_fl : np.ndarray
        Fluid  bulk modulus [Pa].
    rho_fl : np.ndarray
        Fluid density [lg/m^3].
    k_dry : np.ndarray
        Dry rock bulk modulus [Pa].
    mu : np.ndarray
        Dry rock shear modulus [Pa].
    rho_dry : np.ndarray
        Dry rock density [kg/m^3].
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
    rho_sat = rho_dry + por * rho_fl
    k_sat = std_functions.gassmann(k_dry, por, k_fl, k_min)
    vp_sat, vs_sat, ai_sat, vpvs_sat = std_functions.velocity(k_sat, mu, rho_sat)

    return vp_sat, vs_sat, rho_sat, ai_sat, vpvs_sat, k_sat, mu
