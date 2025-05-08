import os
from re import match

import numpy as np
import pandas as pd

from .dummy_vars import generate_dummy_vars
from .import_ml_models import import_model


def _read_models(*model_files, model_dir=None):
    # Find the directory of the model files, change working directory, return to original directory at end of function
    orig_dir = os.getcwd()
    if model_dir is None:
        model_dir, _ = os.path.split(model_files[0])
    os.chdir(model_dir)
    # Allocate lists and read model
    reg_model, scaler, ohe, label_var, label_unit, feat_var, cat_var = (
        [] for _ in range(7)
    )
    answer_lists = [reg_model, scaler, ohe, label_var, label_unit, feat_var, cat_var]
    for mod_name in model_files:
        answer = import_model(mod_name)
        for ans, lst in zip(answer, answer_lists):
            lst.append(ans)

    # Need to modify names
    col_names, col_units = ([] for _ in range(2))
    for i in range(len(label_var)):
        col_names.append(label_var[i] + "_" + model_files[i].replace(label_var[i], ""))
        col_units.append(label_unit[i])

    os.chdir(orig_dir)

    return (
        reg_model,
        scaler,
        ohe,
        label_var,
        label_unit,
        feat_var,
        cat_var,
        col_names,
        col_units,
    )


def _perform_regression(
    inp_frame, col_names, feat_var, cat_var, ohe, scaler, reg_model
):
    depth = inp_frame.index.to_numpy()

    res_frame = pd.DataFrame(index=depth, columns=col_names)

    for j, model_name in enumerate(col_names):
        tmp_frame = inp_frame.copy()

        # Limit to columns used in estimation before dropping NaNs
        num_var = [i for i in feat_var[j] if not bool(match(r"x\d", i))]
        no_num_var = len(num_var)
        if cat_var[j]:
            num_var.append(cat_var[j])
        tmp_frame = tmp_frame[num_var]
        idx_na_n = tmp_frame.isna().any(axis=1)

        if cat_var[j]:
            dum_features, _, dum_var_names = generate_dummy_vars(
                tmp_frame.loc[~idx_na_n], cat_var[j], ohe=ohe[j]
            )
            # Add dummy features to data frame
            kept_dum_var = []
            for i, name in enumerate(dum_var_names):
                if name in feat_var[j]:
                    tmp_frame.loc[~idx_na_n, name] = dum_features[:, i]
                    kept_dum_var.append(name)
            tmp_frame.drop(columns=[cat_var[j]], inplace=True)

            # Need to assure that we have the correct sequence of features
            tmp_frame = tmp_frame.reindex(columns=feat_var[j])

            new_features = np.zeros((np.sum(~idx_na_n), tmp_frame.shape[1]))
            # Make scaling optional
            if scaler[j] is not None:
                new_features[:, :no_num_var] = scaler[j].transform(
                    tmp_frame.to_numpy()[~idx_na_n, :no_num_var]
                )
            else:
                new_features[:, :no_num_var] = tmp_frame.to_numpy()[
                    ~idx_na_n, :no_num_var
                ]
            new_features[:, no_num_var:] = tmp_frame.loc[
                ~idx_na_n, kept_dum_var
            ].to_numpy()
        else:
            # Much simpler if there are no dummy variables
            # Need to assure that we have the correct sequence of features
            tmp_frame = tmp_frame.reindex(columns=feat_var[j])
            # Make scaling optional
            if scaler[j] is not None:
                new_features = scaler[j].transform(tmp_frame.to_numpy()[~idx_na_n, :])
            else:
                new_features = tmp_frame.to_numpy()[~idx_na_n, :]

        new_var = np.ones(depth.shape[0]) * np.nan
        new_var[~idx_na_n] = reg_model[j].predict(new_features).flatten()
        res_frame[col_names[j]] = new_var

    return res_frame


def run_regression(inp_df, vp_model_file_name, vs_model_file_name, model_dir=None):
    """
    Estimate Vp and Vs by neural network regression with multiple inputs.

    Parameters
    ----------
    inp_df : pd.DataFrame
        Input logs required for the regression.
    vp_model_file_name : str
        Full file name for vp model.
    vs_model_file_name : str
        Full file name for vs model.
    model_dir : str
        Directory.

    Returns
    -------
    vp, vs : pd.DataFrame
        Estimated vp and vs as series in Pandas DataFrame.
    """

    (
        regression_model,
        scaler_obj,
        ohe_obj,
        label_var,
        label_var_unit,
        feature_var,
        category_var,
        column_names,
        column_units,
    ) = _read_models(vp_model_file_name, vs_model_file_name, model_dir=model_dir)
    return _perform_regression(
        inp_df,
        column_names,
        feature_var,
        category_var,
        ohe_obj,
        scaler_obj,
        regression_model,
    )
