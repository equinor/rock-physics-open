import pickle
from typing import Any, Self, final

import numpy as np
from typing_extensions import override

from .base_pressure_model import BasePressureModel


@final
class PolynomialPressureModel(BasePressureModel):
    """
    Polynomial pressure sensitivity model for velocity prediction.

    Uses polynomial function: v = v0 * P(p_depl) / P(p_in_situ)
    where P(p) is a polynomial function with specified coefficients.

    Input format (n,3): [velocity, p_eff_in_situ, p_eff_depleted]
    """

    def __init__(
        self,
        weights: list[float],
        model_max_pressure: float | None = None,
        description: str = "",
    ):
        """
        Initialize polynomial pressure model.

        Parameters
        ----------
        weights : list[float]
            Polynomial coefficients [unitless]. First element is constant term,
            second is linear coefficient, etc.
        model_max_pressure : float | None
            Maximum pressure for predict_max method [Pa].
        description : str
            Model description.
        """
        super().__init__(model_max_pressure, description)
        self._weights = weights

    @property
    def weights(self) -> list[float]:
        """Polynomial coefficients."""
        return self._weights

    @override
    def validate_input(self, inp_arr: np.ndarray) -> np.ndarray:
        """
        Validate input for polynomial model.

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

        # Validate weights are set
        if not self._weights:
            raise ValueError('Field "weights" is not set.')

        vel = arr[:, 0]
        p_in_situ = arr[:, 1]
        p_depleted = arr[:, 2]

        # Create polynomial from weights
        polynomial_expr = np.polynomial.Polynomial(self._weights)

        # Select pressure based on case
        p_eff = p_in_situ if case == "in_situ" else p_depleted

        # Calculate velocity using polynomial pressure correction
        return vel * polynomial_expr(p_eff) / polynomial_expr(p_in_situ)

    @override
    def todict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "weights": self._weights,
            "model_max_pressure": self._model_max_pressure,
            "description": self._description,
        }

    @override
    @classmethod
    def load(cls, file: str | bytes) -> Self:
        """Load polynomial model from pickle file."""
        with open(file, "rb") as f_in:
            d = pickle.load(f_in)

        return cls(
            weights=d["weights"],
            model_max_pressure=d["model_max_pressure"],
            description=d["description"],
        )
