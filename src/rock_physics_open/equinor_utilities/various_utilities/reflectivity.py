import numpy as np

from rock_physics_open.equinor_utilities import gen_utilities, std_functions


def reflectivity(vp_inp, vs_inp, rho_inp, theta=0.0, k=2.0, model="AkiRichards"):
    """
    Reflectivity model according to Aki and Richards or Smith and Gidlow for weak contrasts
    and angles less than critical angle.

    In this function it is not allowed to have any missing values in the input logs.
    Instead of interpolating here without the user knowing, raise an input value
    exception and leave it to the user to provide complete logs.

    Parameters
    ----------
    vp_inp : np.ndarray
        Compressional wave velocity [m/s].
    vs_inp : np.ndarray
        Shear wave velocity [m/s].
    rho_inp : np.ndarray
        Bulk density [kg/m^3].
    theta : float
        Incidence angle [radians] (default value 0).
    k : float
        Background Vp/Vs ratio [ratio] (default value 2.0).
    model : str
        One of 'AkiRichards' (default) or 'SmithGidlow'.

    Returns
    -------
    tuple
        refl_coef, idx_inp : np.ndarray.
        refl_coef: reflection coefficient [ratio],
        idx_inp: index to accepted part of the input arrays [bool].
    """

    vp, vs, rho, theta, k = gen_utilities.dim_check_vector(
        (vp_inp, vs_inp, rho_inp, theta, k)
    )

    idx_inp, (vp, vs, rho, theta, k) = gen_utilities.filter_input_log(
        [vp, vs, rho, theta, k], positive=True
    )

    if np.any(~idx_inp):
        # Only NaNs at the start or end? Find the first and last valid sample and check
        # if there are any invalid samples in between
        first_samp = np.where(idx_inp)[0][0]
        last_samp = np.where(idx_inp)[0][-1]
        if np.any(~idx_inp[first_samp : last_samp + 1]):
            # Find the culprit(s)
            idx_vp = gen_utilities.filter_input_log(
                [vp_inp[first_samp : last_samp + 1]], positive=True
            )[0]
            idx_vs = gen_utilities.filter_input_log(
                [vs_inp[first_samp : last_samp + 1]], positive=True
            )[0]
            idx_rho = gen_utilities.filter_input_log(
                [rho_inp[first_samp : last_samp + 1]], positive=True
            )[0]
            log_str = (
                int(np.any(~idx_vp)) * "Vp, "
                + int(np.any(~idx_vs)) * "Vs, "
                + int(np.any(~idx_rho)) * "Rho, "
            )
            pl_str = (
                (int(np.any(~idx_vp)) + int(np.any(~idx_vs)) + int(np.any(~idx_rho)))
                > 1
            ) * "s"
            raise ValueError(
                "{0:} reflectivity: Missing or illegal values in input log{1:}: {2:}interpolation of input log{1:} "
                "is needed\n".format(model, log_str, pl_str)
            )

    if model == "AkiRichards":
        refl_coef = std_functions.aki_richards(vp, vs, rho, theta, k)
    elif model == "SmithGidlow":
        refl_coef = std_functions.smith_gidlow(vp, vs, rho, theta, k)
    else:
        raise ValueError(
            f'{__file__}: unknown model: {model}, should be one of "AkiRichards", "SmithGidlow"'
        )

    return refl_coef, idx_inp
