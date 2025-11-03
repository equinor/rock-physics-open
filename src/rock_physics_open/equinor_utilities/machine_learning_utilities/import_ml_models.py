import pickle
import warnings
from typing import Any, Literal, Protocol, TypedDict, cast

import numpy as np
import numpy.typing as npt
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from .base_pressure_model import BasePressureModel
from .exponential_model import ExponentialPressureModel
from .polynomial_model import PolynomialPressureModel
from .sigmoidal_model import SigmoidalPressureModel


class OheDict(TypedDict):
    ohe: OneHotEncoder
    cat_var: str | list[str]


class ModelDict(TypedDict):
    model_type: Literal[
        "Sigmoid",
        "Exponential",
        "Polynomial",
        "GridSearchCV",
        "ExtraTrees",
        "Keras_nn",
    ]
    model: Any
    nn_mod: str
    scaler: RobustScaler
    label_var: str
    label_units: str
    feature_var: list[str]
    ohe: str | None


def _load_ohe(ohe_file_name: str) -> tuple[OneHotEncoder, str | list[str]]:
    with (
        warnings.catch_warnings(record=True, category=InconsistentVersionWarning) as w,
        open(ohe_file_name, "rb") as f,
    ):
        ohe_dict: OheDict = pickle.load(f)

    ohe = ohe_dict["ohe"]
    if w:
        old_categories = np.array(  # pyright: ignore[reportUnknownVariableType]
            ohe.categories_,  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        ).reshape(-1, 1)
        ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit(
            old_categories
        )
    return ohe, ohe_dict["cat_var"]


class SupportsPredict(Protocol):
    def predict(self, *args: npt.NDArray[np.float64]) -> Any: ...


def import_model(
    model_file_name: str,
) -> tuple[
    BasePressureModel | SupportsPredict,
    RobustScaler,
    OneHotEncoder | None,
    str,
    str,
    list[str],
    str | list[str],
]:
    """
    Utility to import a pickled dict containing information needed to run a classification or regression based on
    a calibrated model.

    Parameters
    ----------
    model_file_name : str
        Full name including path for model file.

    Returns
    -------
    models, scaler, ohe, label_var, label_units, feature_var, cat_var : Any
        models: various regression or classification models from e.g. sklearn or tensorflow keras, scaler: preprocessing
        Robust Scaler, label_var: name(s) of label variable(s), label_unit: unit(s) of label variable(s), cat_var:
        categorical variables that should be encoded with one-hot-encoder.
    """

    with open(model_file_name, "rb") as fin, warnings.catch_warnings():
        # 11.04.2021 HFLE: There is an issue that is not connected to the local function, in that a warning is issued
        # when the model is loaded, claiming that it is of an older version. This is debugged in detail, and the model
        # IS of the correct version, so the error arise elsewhere. To avoid confusion, the warning is suppressed here

        warnings.simplefilter("ignore", category=UserWarning)
        mod_dict: ModelDict = pickle.load(fin)

    model_type = mod_dict["model_type"]

    if model_type == "Sigmoid":
        models = SigmoidalPressureModel.load(mod_dict["nn_mod"])
    elif model_type == "Exponential":
        models = ExponentialPressureModel.load(mod_dict["nn_mod"])
    elif model_type == "Polynomial":
        models = PolynomialPressureModel.load(mod_dict["nn_mod"])
    elif model_type in ["GridSearchCV", "ExtraTrees"]:
        models = cast(SupportsPredict, mod_dict["model"])
    elif model_type == "Keras_nn":
        try:
            from keras.models import load_model

            models = cast(SupportsPredict, load_model(mod_dict["nn_mod"]))
        except ImportError as e:
            raise ImportError(
                "To use Keras models, please install `rock-physics-open` using the [tensorflow] extra."
            ) from e
    else:
        raise ValueError("unknown model type {}".format(model_type))

    ohe: OneHotEncoder | None = None
    cat_var: str | list[str] = []
    try:
        if mod_dict["ohe"]:
            ohe, cat_var = _load_ohe(mod_dict["ohe"])
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
