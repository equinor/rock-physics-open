import os
from collections.abc import Sequence
from pathlib import Path
from re import match
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from .dummy_vars import generate_dummy_vars
from .import_ml_models import AnyModel, import_model


def _read_models(
    *model_files: str, model_dir: str | Path | None = None
) -> tuple[
    list[AnyModel],
    list[RobustScaler],
    list[OneHotEncoder | None],
    list[str],
    list[str],
    list[list[str]],
    list[Any],
    list[str],
    list[str],
]:
    # Find the directory of the model files, change working directory, return to original directory at end of function
    orig_dir = os.getcwd()
    if model_dir is None:
        model_dir, _ = os.path.split(model_files[0])
    os.chdir(model_dir)
    # Allocate lists and read model
    reg_models: list[AnyModel] = []
    scalers: list[RobustScaler] = []
    ohes: list[OneHotEncoder | None] = []
    label_vars: list[str] = []
    label_units: list[str] = []
    feat_vars: list[list[str]] = []
    cat_vars: list[Any] = []
    col_names: list[str] = []
    col_units: list[str] = []

    for mod_name in model_files:
        models, scaler, ohe, label_var, label_unit, feat_var, cat_var = import_model(
            model_file_name=mod_name
        )
        reg_models.append(models)
        scalers.append(scaler)
        ohes.append(ohe)
        label_vars.append(label_var)
        # Need to modify names
        col_names.append(label_var + "_" + mod_name.replace(label_var, ""))
        col_units.append(label_unit)
        label_units.append(label_unit)
        feat_vars.append(feat_var)
        cat_vars.append(cat_var)

        # Need to modify names
        col_names, col_units = ([] for _ in range(2))
        for i in range(len(label_vars)):
            col_names.append(
                label_vars[i] + "_" + model_files[i].replace(label_vars[i], "")
            )
            col_units.append(label_units[i])

    os.chdir(orig_dir)

    return (
        reg_models,
        scalers,
        ohes,
        label_vars,
        label_units,
        feat_vars,
        cat_vars,
        col_names,
        col_units,
    )


def _perform_regression(
    inp_frame: pd.DataFrame,
    col_names: list[str],
    feat_var: list[list[str]],
    cat_var: str | list[str],
    ohe: Sequence[OneHotEncoder | None],
    scaler: Sequence[RobustScaler | None],
    reg_model: Sequence[AnyModel],
) -> pd.DataFrame:
    depth = inp_frame.index.to_numpy()

    res_frame = pd.DataFrame(index=depth, columns=col_names)

    for j, _ in enumerate(col_names):
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
                inp_frame=tmp_frame.loc[~idx_na_n],
                class_var=cat_var[j],
                ohe=ohe[j],
            )
            # Add dummy features to data frame
            kept_dum_var: list[str] = []
            for i, name in enumerate(dum_var_names):
                if name in feat_var[j]:
                    tmp_frame.loc[~idx_na_n, name] = dum_features[:, i]
                    kept_dum_var.append(name)
            tmp_frame.drop(columns=[cat_var[j]], inplace=True)

            # Need to assure that we have the correct sequence of features
            tmp_frame = tmp_frame.reindex(columns=feat_var[j])

            new_features = np.zeros((np.sum(~idx_na_n), tmp_frame.shape[1]))
            # Make scaling optional
            scaler_ = scaler[j]
            if scaler_ is not None:
                new_features = (
                    cast(  # Casting since scikit-learn is not yet fully typed.
                        npt.NDArray[np.float64],
                        scaler_.transform(tmp_frame.to_numpy()[~idx_na_n, :]),
                    )
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
            scaler_ = scaler[j]
            if scaler_ is not None:
                new_features = (
                    cast(  # Casting since scikit-learn is not yet fully typed.
                        npt.NDArray[np.float64],
                        scaler_.transform(tmp_frame.to_numpy()[~idx_na_n, :]),
                    )
                )
            else:
                new_features = tmp_frame.to_numpy()[~idx_na_n, :]

        new_var = np.ones(depth.shape[0]) * np.nan
        new_var[~idx_na_n] = reg_model[j].predict(new_features).flatten()
        res_frame[col_names[j]] = new_var

    return res_frame


def run_regression(
    inp_df: pd.DataFrame,
    first_model_file_name: str,
    second_model_file_name: str,
    model_dir: str | None = None,
) -> pd.DataFrame:
    """
    Estimate Vp and Vs by neural network regression with multiple inputs.

    Parameters
    ----------
    inp_df : pd.DataFrame
        Input logs required for the regression.
    first_model_file_name : str
        Full file name for vp model.
    second_model_file_name : str
        Full file name for vs model.
    model_dir : str | None
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
        _,
        _,
        feature_var,
        category_var,
        column_names,
        _,
    ) = _read_models(first_model_file_name, second_model_file_name, model_dir=model_dir)
    return _perform_regression(
        inp_frame=inp_df,
        col_names=column_names,
        feat_var=feature_var,
        cat_var=category_var,
        ohe=ohe_obj,
        scaler=scaler_obj,
        reg_model=regression_model,
    )
