import pickle
from abc import ABC, abstractmethod
from typing import Any, Self

import numpy as np


class BasePressureModel(ABC):
    """
    Abstract base class for pressure sensitivity models.

    All pressure models follow the convention:
    - predict(): returns differential change (depleted - in_situ)
    - predict_abs(): returns absolute values for specified case
    - predict_max(): uses model_max_pressure instead of depleted pressure

    Input validation is delegated to concrete implementations since
    each model has different column requirements.
    """

    def __init__(self, model_max_pressure: float | None = None, description: str = ""):
        """
        Initialize base pressure model.

        Parameters
        ----------
        model_max_pressure : float | None
            Maximum pressure for predict_max method. Required for predict_max to work.
        description : str
            Human-readable description of the model instance.
        """
        self._model_max_pressure: float | None = model_max_pressure
        self._description: str = description

    @property
    def max_pressure(self) -> float | None:
        """Maximum pressure setting for predict_max method."""
        return self._model_max_pressure

    @property
    def description(self) -> str:
        """Model description."""
        return self._description

    def predict(self, inp_arr: np.ndarray) -> np.ndarray:
        """
        Predict differential change: result(depleted) - result(in_situ).

        Parameters
        ----------
        inp_arr : np.ndarray
            Input array with pressure columns and other model-specific parameters.

        Returns
        -------
        np.ndarray
            Differential change values.
        """
        arr = self.validate_input(inp_arr)
        return self.predict_abs(arr, case="depleted") - self.predict_abs(
            arr, case="in_situ"
        )

    def predict_max(self, inp_arr: np.ndarray) -> np.ndarray:
        """
        Predict using model_max_pressure instead of depleted pressure.

        Parameters
        ----------
        inp_arr : np.ndarray
            Input array where last column (depleted pressure) will be replaced.

        Returns
        -------
        np.ndarray
            Values at model_max_pressure minus values at in_situ pressure.

        Raises
        ------
        ValueError
            If model_max_pressure is not set.
        """
        if self._model_max_pressure is None:
            raise ValueError('Field "model_max_pressure" is not set')

        arr = self.validate_input(inp_arr).copy()
        # Replace last column (assumed to be depleted pressure) with max pressure
        arr[:, -1] = self._model_max_pressure
        return self.predict_abs(arr, case="depleted") - self.predict_abs(
            arr, case="in_situ"
        )

    @abstractmethod
    def validate_input(self, inp_arr: np.ndarray) -> np.ndarray:
        """
        Validate input array format for this specific model.

        Parameters
        ----------
        inp_arr : np.ndarray
            Input array to validate.

        Returns
        -------
        np.ndarray
            Validated input array.

        Raises
        ------
        ValueError
            If input format is invalid for this model.
        """

    @abstractmethod
    def predict_abs(self, inp_arr: np.ndarray, case: str = "in_situ") -> np.ndarray:
        """
        Predict absolute values for specified pressure case.

        Parameters
        ----------
        inp_arr : np.ndarray
            Validated input array.
        case : str
            Either "in_situ" or "depleted" to specify which pressure to use.

        Returns
        -------
        np.ndarray
            Absolute predicted values.
        """

    @abstractmethod
    def todict(self) -> dict[str, Any]:
        """
        Convert model to dictionary for serialization.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all model parameters.
        """

    def save(self, file: str | bytes) -> None:
        """
        Save model to pickle file.

        Parameters
        ----------
        file : str | bytes
            File path for saving.
        """
        with open(file, "wb") as f_out:
            pickle.dump(self.todict(), f_out)

    @classmethod
    @abstractmethod
    def load(cls, file: str | bytes) -> Self:
        """
        Load model from pickle file.

        Parameters
        ----------
        file : str | bytes
            File path for loading.

        Returns
        -------
        BasePressureModel
            Loaded model instance.
        """
