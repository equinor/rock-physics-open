from typing import Literal, cast

import numpy as np
import numpy.typing as npt

from rock_physics_open.equinor_utilities import gen_utilities, std_functions

CoordinateNumberFunction = Literal["PorBased", "ConstVal"]


def friable_model(
    k_min: npt.NDArray[np.float64],
    mu_min: npt.NDArray[np.float64],
    rho_min: npt.NDArray[np.float64],
    k_fl: npt.NDArray[np.float64],
    rho_fl: npt.NDArray[np.float64],
    phi: npt.NDArray[np.float64],
    p_eff: npt.NDArray[np.float64],
    phi_c: float,
    coord_num_func: CoordinateNumberFunction,
    n: float,
    shear_red: float,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """
    Friable (non-cemented) sandstone model. All porosity variation is due to sorting, i.e. porosity filled with
    smaller grains. This model is pressure sensitive, but without asymptotic behaviour, which could be expected.

    The same model is also used for shale, with different set of parameters.

    Mineral properties k_min, mu_min, rho_min are effective properties. For
    mixtures of minerals, effective properties are calculated by
    Hashin-Shtrikman or similar.

    Fluid properties k_fl, rho_fl are in situ properties calculated by a fluid model using reservoir properties.

    Effective pressure p_eff is normally calculated as differential
    pressure, i.e. overburden pressure minus pore pressure.

    Critical porosity phi_c is input to Hertz-Mindlin function. Coordination
    number n is normally calculated from critical porosity, but it is
    possible to  override this through coord_num_func and parameter n.
    Porosity phi is used in Hashin-Shtrikman mixing (together with phi_c)
    and Gassmann saturation. Porosity values above the critical porosity are undefined and NaN is returned.

    Shear reduction parameter shearRed is used to account for tangential
    frictionless grain contacts.

    All inputs are assumed to be vectors of the same length. Single parameters
    are expanded to column vectors matching other inputs.

    Parameters
    ----------
    k_min : np.ndarray
        Mineral bulk modulus [Pa].
    mu_min : np.ndarray
        Mineral shear modulus [Pa].
    rho_min : np.ndarray
        Mineral bulk density [kg/m^3].
    k_fl : np.ndarray
        Fluid bulk modulus [Pa].
    rho_fl : np.ndarray
        Fluid bulk density [kg/m^3].
    phi : np.ndarray
        Porosity [fraction].
    p_eff : np.ndarray
        Effective pressure [Pa].
    phi_c : float
        Critical porosity [fraction].
    coord_num_func : str
        Indication if coordination number should be calculated from porosity or kept constant.
    n : float
        Coordination number [unitless].
    shear_red : float
        Shear reduction factor [fraction].

    Returns
    -------
    tuple
        vp, vs, rho, ai, vpvs  : np.ndarray
        vp [m/s] and vs [m/s], bulk density [kg/m^3], ai [m/s x kg/m^3], vpvs [ratio] of saturated rock.
    """
    k_dry, mu = friable_model_dry(
        k_min=k_min,
        mu_min=mu_min,
        phi=phi,
        p_eff=p_eff,
        phi_c=phi_c,
        coord_num_func=coord_num_func,
        n=n,
        shear_red=shear_red,
    )

    # Saturated rock incompressibility is calculated with Gassmann
    k = std_functions.gassmann(
        k_dry=k_dry,
        por=phi,
        k_fl=k_fl,
        k_min=k_min,
    )

    # Bulk density
    rhob = std_functions.rho_b(
        phi=phi,
        rho_f=rho_fl,
        rho_mat=rho_min,
    )

    vp, vs, ai, vpvs = std_functions.velocity(
        k=k,
        mu=mu,
        rhob=rhob,
    )

    return vp, vs, rhob, ai, vpvs


def friable_model_dry(
    k_min: npt.NDArray[np.float64],
    mu_min: npt.NDArray[np.float64],
    phi: npt.NDArray[np.float64],
    p_eff: npt.NDArray[np.float64],
    phi_c: float,
    coord_num_func: CoordinateNumberFunction,
    n: float | None,
    shear_red: npt.NDArray[np.float64] | float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Dry rock version of friable sandstone model.

    Mineral properties k_min, mu_min, are effective properties. For
    mixtures of minerals, effective properties are calculated by
    Hashin-Shtrikman or similar.

    Effective pressure p_eff is normally calculated as differential
    pressure, i.e. overburden pressure minus pore pressure.

    Critical porosity phi_c is input to Hertz-Mindlin function. Coordination
    number n is normally calculated from critical porosity, but it is
    possible to  override this through coordNumFunc and parameter n.
    Porosity phi is used in Hashin-Shtrikman mixing (together with phi_c).
    Porosity values above the critical porosity are undefined and NaN is returned.

    Shear reduction parameter shearRed is used to account for tangential
    frictionless grain contacts.

    All inputs are assumed to be vectors of the same length. Single parameters
    are expanded to column vectors matching other inputs.

    Parameters
    ----------
    k_min : np.ndarray
        Mineral bulk modulus [Pa].
    mu_min : np.ndarray
        Mineral shear modulus [Pa].
    phi : np.ndarray
        Porosity [fraction].
    p_eff : np.ndarray
        Effective pressure [Pa].
    phi_c : float
        Critical porosity [fraction].
    coord_num_func : str
        Indication if coordination number should be calculated from porosity or kept constant.
    n : float | None
        Coordination number [unitless].
    shear_red : float
        Shear reduction factor [fraction].

    Returns
    -------
    tuple
        k, mu : np.ndarray.
        Bulk modulus k [Pa], shear modulus mu [Pa] of dry rock.
    """
    # Expand floats to arrays, check for equal length
    phi, phi_c_, shear_red_, k_min, mu_min, p_eff, n_ = cast(
        list[npt.NDArray[np.float64]],
        gen_utilities.dim_check_vector(
            (phi, phi_c, shear_red, k_min, mu_min, p_eff, n)
        ),
    )
    # Valid porosity values are less or equal to the critical porosity
    # Use filter_input_log to remove values that do not comply with this
    (
        idx_phi,
        (phi, phi_c_, shear_red_, k_min, mu_min, p_eff, _),
    ) = cast(
        tuple[npt.NDArray[np.bool_], list[npt.NDArray[np.float64]]],
        gen_utilities.filter_input_log(
            (phi, phi_c_, shear_red_, k_min, mu_min, p_eff, phi_c - phi)
        ),
    )

    # Dry rock properties of high-porosity end member calculated with
    # Hertz-Mindlin equation

    # Override coordination number calculation based on porosity
    if coord_num_func == "ConstVal":
        k_hm, mu_hm = std_functions.hertz_mindlin(
            k=k_min,
            mu=mu_min,
            phi_c=phi_c_,
            p=p_eff,
            shear_red=shear_red,
            coord=n_,
        )
    else:
        # Porosity based coordination number
        k_hm, mu_hm = std_functions.hertz_mindlin(
            k=k_min,
            mu=mu_min,
            phi_c=phi_c_,
            p=p_eff,
            shear_red=shear_red,
        )

    # Hashin-Shtrikman lower bound describes the dry rock property mixing from
    # mineral properties to high-end porosity.

    # Fraction of solid
    f1 = 1 - phi / phi_c

    k_dry, mu = std_functions.hashin_shtrikman_walpole(
        k_min, mu_min, k_hm, mu_hm, f1, bound="lower"
    )

    k_dry, mu = cast(
        list[npt.NDArray[np.float64]], gen_utilities.filter_output(idx_phi, (k_dry, mu))
    )

    return k_dry, mu
