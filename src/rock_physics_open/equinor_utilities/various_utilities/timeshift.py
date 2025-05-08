import numpy as np


def time_shift_pp(tvd, vp_base, vp_mon, multiplier):
    """
    Cumulative time shift calculation for 4D case. According to Equinor standard
    the time shift is negative for an increase in velocity from base to monitor
    survey.

    Parameters
    ----------
    tvd : np.ndarray
        True vertical depth along wellbore [m].
    vp_base : np.ndarray
        Initial Vp [m/s].
    vp_mon : np.ndarray
        vp at time of monitor survey [m/s].
    multiplier : int
        Time shift multiplier.

    Returns
    -------
    tuple
        owt_pp_shift, twt_pp_shift : np.ndarray.
        owt_pp_shift: one way time shift [ms],
        twt_pp_shift: two way time shift [ms].

    Notes
    -----
    Original function by Sascha Bussat, Equinor
    Ported to Python by Harald Flesche, Equinor 2016.

    """
    dx = np.diff(tvd)
    dx = np.append(dx, dx[-1])

    time_base = np.cumsum(dx / vp_base)
    time_monitor = np.cumsum(dx / vp_mon)

    owt_pp_shift = (time_monitor - time_base) * 1000 * multiplier
    twt_pp_shift = 2 * owt_pp_shift

    return owt_pp_shift, twt_pp_shift


def time_shift_ps(tvd, vp_base, vp_mon, vs_base, vs_mon, multiplier):
    """
    Cumulative time shift calculation for 4D case. According to Equinor standard
    the time shift is negative for an increase in velocity from base to monitor
    survey.

    Parameters
    ----------
    tvd : np.ndarray
        True vertical depth along wellbore [m].
    vp_base : np.ndarray
        Initial vp [m/s].
    vp_mon : np.ndarray
        vs at time of monitor survey [m/s].
    vs_base : np.ndarray
        Initial vp [m/s].
    vs_mon : np.ndarray
        vs at time of monitor survey [m/s].
    multiplier : int
        Time shift multiplier.

    Returns
    -------
    np.ndarray
        twt_ps_shift: two way time shift [ms]

    Notes
    -----
    Original function by Sascha Bussat, Equinor
    Ported to Python by Harald Flesche, Equinor 2016.

    """
    dx = np.diff(tvd)
    dx = np.append(dx, dx[-1])

    time_base_vp = np.cumsum(dx / vp_base)
    time_monitor_vp = np.cumsum(dx / vp_mon)

    time_base_vs = np.cumsum(dx / vs_base)
    time_monitor_vs = np.cumsum(dx / vs_mon)

    return (
        ((time_monitor_vp - time_base_vp) + (time_monitor_vs - time_base_vs))
        * 1000
        * multiplier
    )
