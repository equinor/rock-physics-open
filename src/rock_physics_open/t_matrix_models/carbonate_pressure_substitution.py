import pandas as pd

from rock_physics_open.equinor_utilities.machine_learning_utilities import (
    run_regression,
)


def carbonate_pressure_model(
    rho_fluid,
    vp_in_situ,
    vs_in_situ,
    rho_in_situ,
    vp_fluid_sub,
    vs_fluid_sub,
    rho_fluid_sub,
    phi,
    pres_overburden,
    pres_formation,
    pres_form_depleted,
    vp_model,
    vs_model,
    model_path,
    b_add_fluid_sub=False,
):
    """
    Function for estimating relative changes in Vp and Vs as a function of formation pressure depletion for carbonates.
    The models that are used are based on laboratory plug measurements at close to in situ conditions,
    and with simulated formation pressure depletion. The training and validation data set is based on pre-salt
    carbonates from producing fields from the Brazilian continental shelf.

    Similar to other substitution routines, the estimated difference between in situ case and depletion case is added
    to observed logs.

    A fluid substitution is assumed to be run in advance of the pressure substitution, and the depletion effect is added
    to the substituted logs. In case there is no preceding fluid substitution, the in situ logs are presented for both
    the in situ and fluid substituted case.

    Parameters
    ----------
    rho_fluid : np.ndarray
        Fluid density [kg/m^3]
    vp_in_situ : np.ndarray
        In situ Vp [m/s]
    vs_in_situ : np.ndarray
        In situ Vs [m/s]
    rho_in_situ : np.ndarray
        In situ bulk density [kg/m^3]
    vp_fluid_sub : np.ndarray
        Fluid substituted Vp [m/s]
    vs_fluid_sub : np.ndarray
        Fluid substituted Vs [m/s]
    rho_fluid_sub : np.ndarray
        Fluid substituted bulk density [kg/m^3]
    phi : np.ndarray
        Porosity [fraction]
    pres_overburden : np.ndarray
        Overburden pressure [Pa]
    pres_formation : np.ndarray
        In situ formation pressure [Pa]
    pres_form_depleted : np.ndarray
        Depleted formation pressure [Pa]
    vp_model : Path
        Full name to neural network model for Vp
    vs_model : Path
        Full name to neural network model for Vs
    model_path : Path
        Path to model directory
    b_add_fluid_sub : bool
        Control whether effect of fluid substitution should be added to the final result

    Returns
    -------
    Tuple[np.ndarray]
        vp_pres_sub : pressure substituted p-velocity [m/s]
        vs_pres_sub : pressure substituted s-velocity [m/s]
        rho_pres_sub : fluid substituted density is returned as the pressure substitution does not change this [kg/m^3]
        ai_pres_sub : derived acoustic impedance [m/s x kg/m^3]
        vpvs_pres_sub : derived vp/vs-ratio [fraction]
    """

    # Plug measurements are reported or dry rock density, bulk density needs to be corrected for fluid effect
    rho_dry = rho_in_situ - rho_fluid * phi

    # Change unit to comply with laboratory units
    # Porosity in percent
    phi_percent = phi * 100.0
    # Pressure in MPa
    pres_overburden *= 1.0e-6
    pres_formation *= 1.0e-6
    pres_form_depleted *= 1.0e-6
    # Density in g/cm^3
    rho_dry *= 1.0e-3

    # Generate DataFrame with necessary inputs for in situ and depleted cases
    input_dict = {
        "PHIT": phi_percent,
        "VP": vp_in_situ,
        "VSX": vs_in_situ,
        "PEFF_in_situ": pres_overburden - pres_formation,
        "PEFF_depleted": pres_overburden - pres_form_depleted,
    }
    input_df = pd.DataFrame(input_dict)

    results_df = run_regression(
        input_df, str(vp_model), str(vs_model), model_dir=str(model_path)
    )
    results_df.columns = ["vp_delta", "vs_delta"]

    if b_add_fluid_sub:
        start_vp = vp_fluid_sub
        start_vs = vs_fluid_sub
        start_rho = rho_fluid_sub
    else:
        start_vp = vp_in_situ
        start_vs = vs_in_situ
        start_rho = rho_in_situ

    vp_pres_sub = (start_vp + results_df["vp_delta"]).to_numpy()
    vs_pres_sub = (start_vs + results_df["vs_delta"]).to_numpy()
    rho_pres_sub = start_rho
    ai_pres_sub = vp_pres_sub * rho_pres_sub
    vpvs_pres_sub = vp_pres_sub / vs_pres_sub

    return vp_pres_sub, vs_pres_sub, rho_pres_sub, ai_pres_sub, vpvs_pres_sub
