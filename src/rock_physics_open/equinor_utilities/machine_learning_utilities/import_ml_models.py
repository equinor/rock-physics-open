import warnings
from pathlib import Path
from pickle import load

from sklearn.preprocessing import OneHotEncoder, RobustScaler

from .exponential_model import ExponentialPressureModel
from .polynomial_model import PolynomialPressureModel
from .sigmoidal_model import SigmoidalPressureModel

AnyModel = ExponentialPressureModel | PolynomialPressureModel | SigmoidalPressureModel


def import_model(
    model_file_name: str,
) -> tuple[
    AnyModel,
    RobustScaler,
    OneHotEncoder | None,
    str,
    str,
    list[str],
    str | list[str],
]:
    """
    Utility to import a pickled dict containing information needed to run a classification or regression based on a calibrated model.

    Parameters
    ----------
    model_file_name
        Full name including path for model file.

    Returns
    -------
    models
        Regression or classification model (e.g. from sklearn or keras).
    scaler
        Preprocessing Robust Scaler.
    ohe
        One-hot encoder for categorical variables, if any.
    label_var
        Name(s) of label variable(s).
    label_unit
        Unit(s) of label variable(s).
    feature_var
        Names of feature variables.
    cat_var
        Categorical variables encoded with the one-hot-encoder.

    Raises
    ------
    ValueError
        If model type is unknown.
    """
    with Path(model_file_name).open("rb") as fin, warnings.catch_warnings():
        # 11.04.2021 HFLE: There is an issue that is not connected to the local function, in that a warning is issued
        # when the model is loaded, claiming that it is of an older version. This is debugged in detail, and the model
        # IS of the correct version, so the error arise elsewhere. To avoid confusion, the warning is suppressed here

        warnings.simplefilter("ignore", category=UserWarning)
        mod_dict = load(fin)

    if mod_dict["model_type"] == "Sigmoid":
        models = SigmoidalPressureModel.load(mod_dict["nn_mod"])
    elif mod_dict["model_type"] == "Exponential":
        models = ExponentialPressureModel.load(mod_dict["nn_mod"])
    elif mod_dict["model_type"] == "Polynomial":
        models = PolynomialPressureModel.load(mod_dict["nn_mod"])
    else:
        raise ValueError("unknown model type {}".format(mod_dict["model_type"]))

    ohe: OneHotEncoder | None = None
    cat_var: str | list[str] = []
    try:
        if mod_dict["ohe"]:
            with Path(mod_dict["ohe"]).open("rb") as f:
                ohe_dict = load(f)
                ohe = ohe_dict["ohe"]
                cat_var = ohe_dict["cat_var"]
    except (FileExistsError, FileNotFoundError):
        pass

    return (
        models,
        mod_dict["scaler"],
        ohe,
        mod_dict["label_var"],
        mod_dict["label_units"],
        mod_dict["feature_var"],
        cat_var,
    )
