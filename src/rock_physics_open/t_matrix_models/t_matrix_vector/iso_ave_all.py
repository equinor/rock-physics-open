import numpy as np

from .iso_av import iso_av_vec


def iso_av_all_vec(x, case_iso):
    """Returns an multi dimensional matrix with isotropic elements. (nx6x6x(numbers of inclusions) matrix).
    Not used in the present implementation - direct call to iso_av_vec from main function instead


    Parameters
    ----------
    x : np.ndarray
        An nx6x6x(numbers of inclusions) matrix.
    case_iso : int
        Control parameter.

    Returns
    -------
    np.ndarray,
        Isotropic values.

    Examples
    --------
    caseIso : control parameter
    if caseIso = 0 then all the pore types are isotropic,
    if caseIso = 1 then there is a mix of isotropic and anisotropic pores,
    if caseIso = 2 then all the pore types are anisotropic.

    Notes
    -----
    HFLE 01.11.2020: Anisotropic inclusions are set to have the same aspect ratio as the isotropic ones.
    """
    if not (x.ndim == 4 and x.shape[1] == x.shape[2]):
        raise ValueError(f"{__name__}: mismatch in inputs variables dimension/shape")

    no_inclusions = x.shape[3]
    x_iso = np.zeros(x.shape)

    # if caseIso = 0 then all the pore types are isotropic
    if case_iso == 0:
        for j in range(no_inclusions):
            x_iso[:, :, j] = iso_av_vec(x[:, :, j])
    # if caseIso = 1 then all the pore types are isotropic
    elif case_iso == 1:
        for j in range(no_inclusions):
            x_iso[:, :, j] = iso_av_vec(x[:, :, j])

        x_iso[:, :, (no_inclusions // 2) : no_inclusions] = x[
            :, :, (no_inclusions // 2) : no_inclusions
        ]
    elif case_iso == 2:
        x_iso = x

    return x_iso
