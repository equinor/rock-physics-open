import pickle
from typing import Any, Self, final

import numpy as np
from typing_extensions import override

from .base_pressure_model import BasePressureModel


@final
class ExponentialPressureModel(BasePressureModel):
    """
    Exponential pressure sensitivity model for velocity prediction.

    Uses exponential decay function: v = v0 * (1 - a*exp(-p/b)) / (1 - a*exp(-p0/b))
    where v0 is reference velocity, p is pressure, a and b are model parameters.

    Input format (n,3): [velocity, p_eff_in_situ, p_eff_depleted]
    """

    def __init__(
        self,
        a_factor: float,
        b_factor: float,
        model_max_pressure: float | None = None,
        description: str = "",
    ):
        """
        Initialize exponential pressure model.

        Parameters
        ----------
        a_factor : float
            Exponential amplitude parameter [unitless].
        b_factor : float
            Exponential decay parameter [Pa].
        model_max_pressure : float | None
            Maximum pressure for predict_max method [Pa].
        description : str
            Model description.
        """
        super().__init__(model_max_pressure, description)
        self._a_factor = a_factor
        self._b_factor = b_factor

    @property
    def a_factor(self) -> float:
        """Exponential amplitude factor."""
        return self._a_factor

    @property
    def b_factor(self) -> float:
        """Exponential decay factor."""
        return self._b_factor

    @override
    def validate_input(self, inp_arr: np.ndarray) -> np.ndarray:
        """
        Validate input for exponential model.

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
            If input format is invalid.
        """
        if not isinstance(inp_arr, np.ndarray):  # pyright: ignore[reportUnnecessaryIsInstance] | Kept for backward compatibility
            raise ValueError("Input must be numpy ndarray.")  # pyright: ignore[reportUnreachable] | Kept for backward compatibility
        if inp_arr.ndim != 2 or inp_arr.shape[1] != 3:
            raise ValueError(
                "Input must be (n,3): [velocity, p_eff_in_situ, p_eff_depleted]"
            )
        return inp_arr

    @override
    def predict_abs(self, inp_arr: np.ndarray, case: str = "in_situ") -> np.ndarray:
        """
        Calculate absolute velocity for specified pressure case.

        Parameters
        ----------
        inp_arr : np.ndarray
            Validated input array (n,3).
        case : str
            Pressure case: "in_situ" or "depleted".

        Returns
        -------
        np.ndarray
            Velocity values [m/s].
        """
        arr = self.validate_input(inp_arr)

        vel = arr[:, 0]
        p_in_situ = arr[:, 1]
        p_depleted = arr[:, 2]

        p_eff = p_in_situ if case == "in_situ" else p_depleted

        return (
            vel
            * (1.0 - self._a_factor * np.exp(-p_eff / self._b_factor))
            / (1.0 - self._a_factor * np.exp(-p_in_situ / self._b_factor))
        )

    @override
    def todict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "a_factor": self._a_factor,
            "b_factor": self._b_factor,
            "model_max_pressure": self._model_max_pressure,
            "description": self._description,
        }

    @override
    @classmethod
    def load(cls, file: str | bytes) -> Self:
        """Load exponential model from pickle file."""
        with open(file, "rb") as f_in:
            d = pickle.load(f_in)

        return cls(
            a_factor=d["a_factor"],
            b_factor=d["b_factor"],
            model_max_pressure=d["model_max_pressure"],
            description=d["description"],
        )
