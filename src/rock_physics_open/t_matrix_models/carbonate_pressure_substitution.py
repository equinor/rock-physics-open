from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from rock_physics_open.equinor_utilities.machine_learning_utilities.run_regression import (
    run_regression,
)
from rock_physics_open.equinor_utilities.units import pa_to_mpa


def carbonate_pressure_model(
    vp_in_situ: npt.NDArray[np.float64],
    vs_in_situ: npt.NDArray[np.float64],
    rho_in_situ: npt.NDArray[np.float64],
    vp_fluid_sub: npt.NDArray[np.float64],
    vs_fluid_sub: npt.NDArray[np.float64],
    rho_fluid_sub: npt.NDArray[np.float64],
    phi: npt.NDArray[np.float64],
    pres_overburden: npt.NDArray[np.float64],
    pres_formation: npt.NDArray[np.float64],
    pres_form_depleted: npt.NDArray[np.float64],
    vp_model: Path,
    vs_model: Path,
    model_path: Path,
    b_add_fluid_sub: bool = False,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
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
    vp_in_situ
        In situ Vp [m/s]
    vs_in_situ
        In situ Vs [m/s]
    rho_in_situ
        In situ bulk density [kg/m^3]
    vp_fluid_sub
        Fluid substituted Vp [m/s]
    vs_fluid_sub
        Fluid substituted Vs [m/s]
    rho_fluid_sub
        Fluid substituted bulk density [kg/m^3]
    phi
        Porosity [fraction]
    pres_overburden
        Overburden pressure [Pa]
    pres_formation
        In situ formation pressure [Pa]
    pres_form_depleted
        Depleted formation pressure [Pa]
    vp_model
        Full name to neural network model for Vp
    vs_model
        Full name to neural network model for Vs
    model_path
        Path to model directory
    b_add_fluid_sub
        Control whether effect of fluid substitution should be added to the final result

    Returns
    -------
    vp_pres_sub
        Pressure substituted p-velocity [m/s].
    vs_pres_sub
        Pressure substituted s-velocity [m/s].
    rho_pres_sub
        Fluid substituted density (unchanged by pressure substitution) [kg/m^3].
    ai_pres_sub
        Derived acoustic impedance [m/s x kg/m^3].
    vpvs_pres_sub
        Derived Vp/Vs ratio [fraction].
    """
    # Change unit to comply with laboratory units
    # Porosity in percent
    phi_percent = phi * 100.0
    # Pressure in MPa
    pres_overburden = pa_to_mpa(pres_overburden)
    pres_formation = pa_to_mpa(pres_formation)
    pres_form_depleted = pa_to_mpa(pres_form_depleted)

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
        inp_df=input_df,
        first_model_file_name=str(vp_model),
        second_model_file_name=str(vs_model),
        model_dir=str(model_path),
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

    vp_pres_sub = start_vp + results_df["vp_delta"].to_numpy()
    vs_pres_sub = start_vs + results_df["vs_delta"].to_numpy()
    rho_pres_sub = start_rho
    ai_pres_sub = vp_pres_sub * rho_pres_sub
    vpvs_pres_sub = vp_pres_sub / vs_pres_sub

    return vp_pres_sub, vs_pres_sub, rho_pres_sub, ai_pres_sub, vpvs_pres_sub
