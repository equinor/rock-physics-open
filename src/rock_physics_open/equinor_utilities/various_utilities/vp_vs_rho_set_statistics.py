import os

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from .display_result_statistics import disp_result_stats


def vp_vs_rho_stats(
    vp_observed,
    vs_observed,
    rho_observed,
    vp_estimated,
    vs_estimated,
    rho_estimated,
    fname,
    estimated_set_names,
    well_names,
    file_mode="a",
    disp_results=True,
):
    """
    Utility to estimate statistics between vp-vs-rho sets - observed and estimated values. The results are displayed
    on screen (optional) and saved to a .csv file. If the file exists, the results will be appended.
    The statistics consists of RAE (relative absolute error) [fraction], RRMSE (relative root mean squared error)
    [fraction] and R^2 (coefficient of determination) [fraction].


    Parameters
    ----------
    vp_observed : np.ndarray or list
        Observed vp [m/s].
    vs_observed : np.ndarray or list
        Observed vs [m/s].
    rho_observed : np.ndarray or list
        Observed density [kg/m^3].
    vp_estimated : np.ndarray or list
        Estimated vp [m/s].
    vs_estimated : np.ndarray or list
        Estimated vs [m/s].
    rho_estimated : np.ndarray or list
        Estimated density [kg/m^3].
    fname : str
        File name for saved results.
    estimated_set_names : str or list
        Name of the estimated vp-vs-rho set(s).
    well_names : str or list
        Well name of the vp-vs-rho set(s).
    file_mode : str
        File open mode, default append.
    disp_results : bool
        Display results on screen.
    """
    if isinstance(estimated_set_names, str):
        estimated_set_names = [estimated_set_names]
    if isinstance(well_names, str):
        well_names = [well_names]

    _verify(
        vp_observed,
        vs_observed,
        rho_observed,
        vp_estimated,
        vs_estimated,
        rho_estimated,
        set_names=estimated_set_names,
        well_names=well_names,
        file_mode=file_mode,
    )

    est_frame_columns = [
        "Well",
        "Vp RMAE",
        "Vp RRMSE",
        "Vp R2",
        "Vs RMAE",
        "Vs RRMSE",
        "Vs R2",
        "Rho RMAE",
        "Rho RRMSE",
        "Rho R2",
    ]
    est_frame = pd.DataFrame(columns=est_frame_columns, index=estimated_set_names)
    est_frame.index.name = "Estimated set name"
    est_frame.iloc[:, 0] = well_names

    # If inputs are found to satisfy expectations in _verify, and they are numpy arrays, cast to lists, and run through
    if isinstance(vp_observed, np.ndarray):
        vp_observed = [vp_observed]
        vs_observed = [vs_observed]
        rho_observed = [rho_observed]
        vp_estimated = [vp_estimated]
        vs_estimated = [vs_estimated]
        rho_estimated = [rho_estimated]

    for i in range(len(vp_observed)):
        res = []
        for obs, est in zip(
            [vp_observed[i], vs_observed[i], rho_observed[i]],
            [vp_estimated[i], vs_estimated[i], rho_estimated[i]],
        ):
            res.append(np.mean(np.abs((est.flatten() - obs.flatten()) / obs.flatten())))
            res.append(
                np.sqrt(
                    np.mean(np.square((est.flatten() - obs.flatten()) / obs.flatten()))
                )
            )
            res.append(r2_score(obs.flatten(), est.flatten()))

        res_dict = dict(zip(est_frame_columns[1:], res))
        est_frame.iloc[i, 1:] = res_dict
        if disp_results:
            disp_result_stats(
                estimated_set_names[i], res, est_frame_columns[1:], values_only=True
            )
    # Test if the file already exists. If so, and the file_mode is set to 'a', drop the column headers when writing
    # the file
    if os.path.exists(fname) and file_mode == "a":
        est_frame.to_csv(fname, mode=file_mode, header=False)
    else:
        est_frame.to_csv(fname, mode=file_mode)


def _verify(*args, set_names=None, well_names=None, file_mode=None):
    """Verify that arguments are either numpy arrays or lists of numpy arrays.
    Raises
    ------
    ValueError
        If not np.ndarray or list of such.
    ValueError
        If an entry contains NaNs.
    ValueError
        If an entry contains Infs.
    ValueError
        If mismatch in argument lengths.
    ValueError
        If file mode not 'a' or 'w'.
    """
    # Verify that args are either numpy arrays or list of arrays
    for arg in args:
        if isinstance(arg, np.ndarray):
            arg = [arg]
        for this_arg in arg:
            if not isinstance(this_arg, np.ndarray):
                raise ValueError(f"{__file__}: input not numpy array: {type(arg)}")
            if np.any(np.isnan(this_arg)):
                raise ValueError(f"{__file__}: input contains NaNs")
            if np.any(np.isinf(this_arg)):
                raise ValueError(f"{__file__}: input contains Infs")
        if not len(arg) == len(set_names) == len(well_names):
            raise ValueError(f"{__file__}: mismatch in argument lengths")
    if not (file_mode == "a" or file_mode == "w"):
        raise ValueError(f'{__file__}: file_mode must be one of ["a", "w"]')
