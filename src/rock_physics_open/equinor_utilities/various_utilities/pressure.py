import numpy as np


def pressure(rho, tvd_msl, water_depth, p_form, tvd_p_form, n):
    """
    Function to estimate overburden pressure and vertical effective stress (lithostatic pressure)
    based on density.

    Parameters
    ----------
    rho : np.ndarray
        Density log [kg/m3].
    tvd_msl : np.ndarray
        Vertical depth log [m].
    water_depth : float
        Down to this point the difference between formation pressure and overburden pressure shall be zero [m].
    p_form : float
        Formation pressure [Pa].
    tvd_p_form : float
        Depth of formation pressure point [m].
    n: float
        Biot coefficient [unitless].

    Returns
    -------
    tuple
        p_eff, p_lith : np.ndarray.
        p_eff [Pa] - effective pressure,
        p_lith [Pa] - overburden pressure.
    """

    # Standard brine density with salinity 40000 ppm, 2 MPa and 4 deg. C
    rho_brine = 1.03e3
    # Gravity constant
    g = 9.80665
    # Input log length
    log_length = len(rho)

    # Allocate output logs
    p_eff = np.ones(log_length) * np.nan
    p_lith = np.ones(log_length) * np.nan

    # Does the depth log start at, above or below the sea bottom? We want it to
    # start at sea bottom, so we get rid of values above this point.
    # Also, as the tvd log normally is calculated by calling application, there can be
    # undefined values, flagged as negative or NaN
    idx = np.ones_like(rho, dtype=bool)
    idx_inf_nan = np.any(
        [np.isnan(rho), np.isinf(rho), np.isnan(tvd_msl), np.isinf(tvd_msl)], axis=0
    )
    idx[idx_inf_nan] = False
    idx[~idx_inf_nan] = np.logical_and(
        tvd_msl[~idx_inf_nan] >= water_depth, rho[~idx_inf_nan] > 0
    )

    # We need a starting point for pressure at water bottom - use a standard
    # density value
    p_wb = g * rho_brine * water_depth

    # p_form is a single value. No functionality is added at present to handle p_form in log version
    # Find which depth in the log that matches the calibration point best
    calib_point = np.argmin(abs(tvd_msl[idx] - tvd_p_form))

    # Overburden pressure
    dz = np.diff(tvd_msl[idx])
    # Append one sample to match the density log length
    dz = np.append(dz, dz[-1])

    # Find a starting point for the litho pressure at the start of the depth log
    # Density at water bottom is assumed to be for a sand with 40% porosity. Take the average
    # of this and the first observation of density log
    ave_dens = 0.5 * ((2650 * 0.6 + rho_brine * 0.4) + rho[0])
    p_lith_start = ave_dens * g * (tvd_msl[0] - water_depth)

    # Estimate the overburden pressure as the gravity of the cumulative bulk
    # density plus the calculated starting point and the fluid pressure at sea bottom
    p_lith[idx] = np.cumsum(g * dz * rho[idx]) + p_lith_start + p_wb

    # Find the effective/differential pressure at the calibration point
    p_eff_calib = p_lith[idx][calib_point] - n * p_form
    # In the absence of any better alternative - make a linear interpolation
    # from zero at water bottom to the calculated effective pressure at the
    # calibration point
    p_eff[idx] = np.interp(tvd_msl[idx], [water_depth, tvd_p_form], [0, p_eff_calib])

    p_lith[idx] = np.interp(tvd_msl[idx], tvd_msl[idx], p_lith[idx])

    return p_eff, p_lith
