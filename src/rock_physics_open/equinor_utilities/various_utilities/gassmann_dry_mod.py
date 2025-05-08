from rock_physics_open.equinor_utilities import std_functions


def gassmann_dry_model(k_min, k_fl, rho_fl, k_sat, mu, rho_sat, por):
    """
    Gassmann model to go from saturated rock to dry state.

    Parameters
    ----------
    k_min : np.ndarray
        Mineral bulk modulus [Pa].
    k_fl : np.ndarray
        Fluid  bulk modulus [Pa].
    rho_fl : np.ndarray
        Fluid density [lg/m^3].
    k_sat : np.ndarray
        Saturated rock bulk modulus [Pa].
    mu : np.ndarray
        Saturated rock shear modulus [Pa].
    rho_sat : np.ndarray
        Saturated rock density [kg/m^3].
    por : np.ndarray
        Porosity [fraction].

    Returns
    -------
    tuple
        vp_dry, vs_dry, rho_dry, ai_dry, vpvs_dry, k_dry, mu : (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray).
        vp_dry, vs_dry:  dry velocities [m/s], rho_dry: dry density [kg/m^3], ai_dry: dry acoustic impedance
        [kg/m^3 x m/s], vpvs_dry: dry velocity ratio [unitless], k_dry, mu: dry bulk modulus and shear modulus (the
        latter unchanged from saturated state) [Pa].
    """
    rho_dry = rho_sat - por * rho_fl
    k_dry = std_functions.gassmann_dry(k_sat, por, k_fl, k_min)
    vp_dry, vs_dry, ai_dry, vpvs_dry = std_functions.velocity(k_dry, mu, rho_dry)

    return vp_dry, vs_dry, rho_dry, ai_dry, vpvs_dry, k_dry, mu
