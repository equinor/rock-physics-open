import os
import pickle

# from scipy.optimize import minimize, Bounds
import sys

import numpy as np
from scipy.optimize import curve_fit


def curve_fit_wrapper(x_init, opt_func, x_data, y_data, *args, **opt_kwargs):
    """Use in tests with scipy.optimize.minimize instead of curve_fit.

    Parameters
    ----------
    x_init : np.ndarray
        Initial guess for parameters.
    opt_func : callable
        Function to optimize.
    x_data : np.ndarray
        Input data to opt_func.
    y_data : np.ndarray
        Results that the optimisation should match.
    args :
        args opt_func.
    opt_kwargs :
        kwargs opt_func.

    Returns
    -------
    np.ndarray
        res values.
    """
    y_pred = opt_func(x_data, *x_init, *args, **opt_kwargs)
    return np.sum(np.sqrt((y_data - y_pred) ** 2))


def gen_opt_routine(
    opt_function, x_data_orig, y_data, x_init, low_bound, high_bound, **opt_kwargs
):
    """
    This function is a lean method for running optimisation with the given opt_function in curve_fit. Predicted values,
    residuals to the observed values and optimal parameters are returned.

    Parameters
    ----------
    opt_function : callable
        function to optimise
    x_data_orig : np.ndarray
        input data to the function - independent variables
    y_data : np.ndarray
        results that the optimisation should match - dependent variables
    x_init : np.ndarray
        initial guess for parameters
    low_bound : np.ndarray
        parameter low bound
    high_bound : np.ndarray
        parameter high bound
    opt_kwargs : dict
        optional meta-parameters to the optimisation function

    Returns
    -------
    tuple
        y_pred, y_res, opt_params : (np.ndarray, np.ndarray, np.ndarray).
        y_pred : predicted values,
        y_res : residual values,
        opt_params : optimal model parameters.
    """
    try:
        opt_params, param_cov = curve_fit(
            opt_function,
            x_data_orig,
            y_data.flatten("F"),
            x_init,
            bounds=(low_bound, high_bound),
            method="trf",
            loss="soft_l1",
            **opt_kwargs,
        )

    except ValueError:
        raise ValueError(
            "gen_opt_routine: failed in optimisation step: {}".format(
                str(sys.exc_info())
            )
        )
    else:
        y_pred = np.reshape(
            opt_function(x_data_orig, *opt_params), y_data.shape, order="F"
        )
        y_res = y_pred - y_data

        # Alternative implementation, not shown to improve results
        # alt_opt_params = minimize(curve_fit_wrapper, x_init, args=(opt_function, x_data_orig, y_data.flatten('F')),
        #                          bounds=Bounds(low_bound, high_bound), method='SLSQP', options={'maxiter': 10000})
        # y_pred_1 = np.reshape(opt_function(x_data_orig, *alt_opt_params['x'], **opt_kwargs), y_data.shape, order='F')
        # y_res_1 = y_pred_1 - y_data
        # return y_pred_1, y_res_1, alt_opt_params['x']

        return y_pred, y_res, opt_params


def gen_mod_routine(opt_function, xdata_orig, ydata_shape, opt_params):
    """Predict modelled values based on an earlier optimisation run for optimal model parameters.

    Parameters
    ----------
    opt_function : callable
        Function to optimise.
    xdata_orig : np.ndarray
        Input data to the function - independent variables.
    ydata_shape : (int, int)
        Shape of y_data.
    opt_params : np.ndarray
        Optimal model parameters.

    Returns
    -------
    np.ndarray
        Predicted values.
    """
    # Estimation of values
    return np.reshape(opt_function(xdata_orig, *opt_params), ydata_shape, order="F")


def gen_sub_routine(opt_function, xdata_orig, xdata_new, ydata, opt_params):
    """General substitution function based on a calibrated/optimised model and with two sets of input parameters.
    The substituted values are calculated as the original observations plus the difference of the two modelling
    steps.

    Parameters
    ----------
    opt_function : callable
        Function to optimise.
    xdata_orig : np.ndarray
        Input data to the function step 1 - independent variables.
    xdata_new : np.ndarray
        Input data to the function step 2 - independent variables.
    ydata : np.ndarray
        Original observed values step 1.
    opt_params : np.ndarray
        Set of optimal parameters to model.

    Returns
    -------
    tuple
        y_final, y_pred, y_res : (np.ndarray, np.ndarray, np.ndarray).
        Original observed data + difference in estimation between steps 0 and 1, y_pred - modelled data,
        y_res - residuals, opt_params - best parameter setting.
    """
    # Estimation of initial values
    y_pred = np.reshape(opt_function(xdata_orig, *opt_params), ydata.shape, order="F")
    # Estimation step for substituted fluid properties
    y_subst = np.reshape(opt_function(xdata_new, *opt_params), ydata.shape, "F")

    y_res = y_pred - ydata
    y_diff = y_subst - y_pred
    y_final = ydata + y_diff

    return y_final, y_pred, y_res


def save_opt_params(
    opt_type: str,
    opt_params: np.ndarray,
    file_name: str = "opt_params.pkl",
    well_name: str = "Unknown well",
):
    """
    Utility to save optimal parameters as a pickle file in a more readable format so that the optimisation method can be
    recognised.

    Parameters
    ----------
    opt_type : str
        String defining optimisation type.
    opt_params : np.ndarray
        Numpy array with parameters from optimisation.
    file_name : str, optional
        File to save results to, by default 'opt_params.pkl'.
    well_name : str, optional
        Name of the well which is used in optimisation, by default 'Unknown well'.

    Raises
    ------
    ValueError
        If unknown optimisation opt_type.
    """
    # Save the optimal parameters with info
    if opt_type == "min":  # optimisation with mineral input from well
        opt_param_dict = {
            "well_name": well_name,
            "opt_ver": opt_type,
            "f_ani": opt_params[0],
            "f_con": opt_params[1],
            "alpha_opt": opt_params[2:4],
            "v_opt": opt_params[4],
            "opt_vec": opt_params,
        }
    elif opt_type == "exp":
        opt_param_dict = {
            "well_name": well_name,
            "opt_ver": opt_type,
            "f_ani": opt_params[0],
            "f_con": opt_params[1],
            "alpha_opt": opt_params[2:4],
            "v_opt": opt_params[4],
            "k_carb": opt_params[5],
            "mu_carb": opt_params[6],
            "rho_carb": opt_params[7],
            "k_sh": opt_params[8],
            "mu_sh": opt_params[9],
            "rho_sh": opt_params[10],
            "opt_vec": opt_params,
        }
    elif opt_type == "pat_cem":
        opt_param_dict = {
            "well_name": well_name,
            "opt_ver": opt_type,
            "weight_k": opt_params[0],
            "weight_mu": opt_params[1],
            "shear_red": opt_params[2],
            "frac_cem": opt_params[3],
            "opt_vec": opt_params,
        }
    elif opt_type == "const_cem":
        opt_param_dict = {
            "well_name": well_name,
            "opt_ver": opt_type,
            "phi_c": opt_params[0],
            "shear_red": opt_params[1],
            "frac_cem": opt_params[2],
            "opt_vec": opt_params,
        }
    elif opt_type == "friable":
        opt_param_dict = {
            "well_name": well_name,
            "opt_ver": opt_type,
            "phi_c": opt_params[0],
            "shear_red": opt_params[1],
            "opt_vec": opt_params,
        }
    else:
        raise ValueError(
            "save_opt_params: unknown optimisation opt_type: {}".format(opt_type)
        )

    with open(file_name, "wb") as file_out:
        pickle.dump(opt_param_dict, file_out)


def opt_param_info():
    """Hard coded dictionaries returned.
    Returns
    -------
    tuple
        parameter_translation_dict, value_translation_dict, type_translation_dict.
    """
    parameter_translation_dict = {
        "opt_ver": "Optimisation version",
        "no_incl_sets": "Number of inclusion sets",
        "ang_sym": "Angle of symmetry plane [°]",
        "f_ani": "Fraction of anisotropic inclusions",
        "f_con": "Fraction of connected inclusions",
        "alpha_opt": "Optimal aspect ratios for inclusion sets",
        "v_opt": "Ratio of volume for inclusion sets",
        "k_carb": "Matrix (carbonate) bulk modulus [Pa]",
        "mu_carb": "Matrix (carbonate) shear modulus [Pa]",
        "rho_carb": "Matrix (carbonate) density [kg/m^3]",
        "k_sh": "Mud/shale bulk modulus [Pa]",
        "mu_sh": "Mud/shale shear modulus [Pa]",
        "rho_sh": "Mud/shale density [kg/m^3]",
        "k_sst": "Sst bulk modulus [Pa]",
        "mu_sst": "Sst shear modulus [Pa]",
        "rho_sst": "Sst density [kg/m^3]",
        "frac_cem": "Cement fraction [fraction]",
        "phi_c": "Critical porosity [fraction]",
        "shear_red": "Reduction in tangential friction [fraction]",
        "weight_k": "Bulk modulus weight for constant cement model",
        "weight_mu": "Shear modulus weight for constant cement model",
    }
    value_translation_dict = {
        "ang_sym": 90.0,
        "k_carb": 95.0e9,
        "mu_carb": 45.0e9,
        "rho_carb": 2950.0,
        "k_sh": 35.0e9,
        "mu_sh": 20.0e9,
        "rho_sh": 2750.0,
        "k_sst": 45.0e9,
        "mu_sst": 50.0e9,
        "rho_sst": 2750.0,
    }
    type_translation_dict = {
        "min": "PETEC (Mineral input) optimisation",
        "exp": "Exploration type optimisation",
        "pat_cem": "Patchy cement model",
        "const_cem": "Constant cement model",
        "friable": "Friable sand model",
    }
    return parameter_translation_dict, value_translation_dict, type_translation_dict


def load_opt_params(file_name: str):
    """Utility to load parameter file from optimisation run.

    Parameters
    ----------
    file_name : str
        Input file name including path.

    Returns
    -------
    tuple
        opt_type: model type, no_sets: number of inclusion sets, opt_param: with all parameters for model.
    """
    with open(file_name, "rb") as fin:
        param_dict = pickle.load(fin)
        opt_type = param_dict["opt_ver"]
        opt_param = param_dict["opt_vec"]
        opt_dict = param_dict

        return opt_type, opt_param, opt_dict


def opt_param_to_ascii(
    in_file, display_results=True, out_file=None, well_name="Unknown well", **kwargs
):
    """Utility to convert stored optimised parameters to ascii and display results or save to file.

    Parameters
    ----------
    in_file : str
        File name for stored optimised parameters.
    display_results : bool
        Display results on screen, default True.
    out_file : str or None
        Optional store optimised parameters in ascii file.
    well_name : str
        Optional name of the well that is used in optimisation.
    """
    with open(in_file, "rb") as f_in:
        param_dict = pickle.load(f_in)
        if well_name.lower() == "unknown well":
            well_name = param_dict.pop("well_name", "Unknown Well")

        (
            parameter_translation_dict,
            value_translation_dict,
            type_translation_dict,
        ) = opt_param_info()

        item = []
        value = []
        disp_string = ""
        for opt_key, opt_value in param_dict.items():
            if opt_key in parameter_translation_dict:
                if opt_key in value_translation_dict:
                    opt_value = opt_value * value_translation_dict[opt_key]
                    opt_str = f" {opt_value:.4f}"
                elif opt_key == "opt_ver":
                    opt_str = type_translation_dict[opt_value]
                elif opt_key == "v_opt":
                    opt_value = np.append(opt_value, 1.0 - np.sum(opt_value))
                    opt_str = f" {opt_value:}"
                else:
                    if isinstance(opt_value, float):
                        opt_str = f" {opt_value:.4f}"
                    else:
                        opt_str = f" {opt_value:}"
                item.append(f"{parameter_translation_dict[opt_key]}: ")
                value.append(opt_str)
                disp_string += f"{parameter_translation_dict[opt_key]}: {opt_str}\n"
        info_array = np.stack((item, value), axis=1)

        if display_results:
            from tkinter import END, Entry, Tk

            class Table:
                def __init__(self, tk_root, no_rows, no_cols, info):
                    # code for creating table
                    str_len = np.vectorize(len)
                    text_justify = ["right", "left"]
                    for i in range(no_rows):
                        for j in range(no_cols):
                            just = text_justify[0] if j == 0 else text_justify[1]
                            max_len = np.max(str_len(info[:, j]))
                            self.e = Entry(
                                root,
                                width=max_len + 2,
                                fg="black",
                                font=("Consolas", 11, "normal"),
                                justify=just,
                            )
                            self.e.grid(row=i, column=j)
                            self.e.insert(END, info[i][j])

            root = Tk(**kwargs)
            if well_name.lower() == "unknown well":
                root.title("T Matrix Optimised Parameters")
            else:
                root.title(well_name)
            if sys.platform.startswith("win"):
                ico_file = os.path.join(os.path.dirname(__file__), "Equinor_logo.ico")
                root.iconbitmap(ico_file)
            Table(root, info_array.shape[0], info_array.shape[1], info_array)
            root.attributes("-topmost", True)
            root.mainloop()

        if out_file is not None:
            with open(out_file, "w") as f_out:
                f_out.write(disp_string)

        return
