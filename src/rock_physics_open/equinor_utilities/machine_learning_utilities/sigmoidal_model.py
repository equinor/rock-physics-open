import pickle
from pathlib import Path
from typing import Any, Self, final

import numpy as np
from typing_extensions import override

from .base_pressure_model import BasePressureModel


@final
class SigmoidalPressureModel(BasePressureModel):
    """
    Sigmoidal pressure sensitivity model for velocity prediction.

    Uses nested sigmoid functions: velocity amplitude varies with porosity,
    and velocity varies sigmoidally with effective pressure using the amplitude.

    Input format (n,3): [porosity, p_eff_in_situ, p_eff_depleted]

    The model applies two sigmoid transformations:
    1. Porosity -> velocity amplitude using phi_model parameters
    2. Effective pressure -> velocity using p_eff_model parameters and amplitude

    Parameters
    ----------
    phi_amplitude
        Amplitude parameter for porosity sigmoid [m/s].
    phi_median_point
        Median point for porosity sigmoid [fraction].
    phi_x_scaling
        X-scaling parameter for porosity sigmoid [unitless].
    phi_bias
        Bias parameter for porosity sigmoid [m/s].
    p_eff_median_point
        Median point for pressure sigmoid [Pa].
    p_eff_x_scaling
        X-scaling parameter for pressure sigmoid [1/Pa].
    p_eff_bias
        Bias parameter for pressure sigmoid [m/s].
    model_max_pressure
        Maximum pressure for predict_max method [Pa].
    description
        Model description.
    """

    def __init__(
        self,
        phi_amplitude: float,
        phi_median_point: float,
        phi_x_scaling: float,
        phi_bias: float,
        p_eff_median_point: float,
        p_eff_x_scaling: float,
        p_eff_bias: float,
        model_max_pressure: float | None = None,
        description: str = "",
    ):
        super().__init__(model_max_pressure, description)
        # Porosity model parameters
        self._phi_amplitude = phi_amplitude
        self._phi_median_point = phi_median_point
        self._phi_x_scaling = phi_x_scaling
        self._phi_bias = phi_bias
        # Pressure model parameters
        self._p_eff_median_point = p_eff_median_point
        self._p_eff_x_scaling = p_eff_x_scaling
        self._p_eff_bias = p_eff_bias

    @property
    def phi_amplitude(self) -> float:
        """Porosity sigmoid amplitude."""
        return self._phi_amplitude

    @property
    def phi_median_point(self) -> float:
        """Porosity sigmoid median point."""
        return self._phi_median_point

    @property
    def phi_x_scaling(self) -> float:
        """Porosity sigmoid x-scaling."""
        return self._phi_x_scaling

    @property
    def phi_bias(self) -> float:
        """Porosity sigmoid bias."""
        return self._phi_bias

    @property
    def p_eff_median_point(self) -> float:
        """Pressure sigmoid median point."""
        return self._p_eff_median_point

    @property
    def p_eff_x_scaling(self) -> float:
        """Pressure sigmoid x-scaling."""
        return self._p_eff_x_scaling

    @property
    def p_eff_bias(self) -> float:
        """Pressure sigmoid bias."""
        return self._p_eff_bias

    @override
    def validate_input(self, inp_arr: np.ndarray) -> np.ndarray:
        """
        Validate input for sigmoidal model.

        Parameters
        ----------
        inp_arr
            Input array to validate.

        Returns
        -------
        Validated input array.

        Raises
        ------
        ValueError
            If input format is invalid.
        """
        if inp_arr.ndim != 2 or inp_arr.shape[1] != 3:
            raise ValueError(
                "Input must be (n,3): [porosity, p_eff_in_situ, p_eff_depleted]"
            )
        return inp_arr

    def _sigmoid_phi(self, phi: np.ndarray) -> np.ndarray:
        """
        Calculate velocity amplitude from porosity using sigmoid function.

        Parameters
        ----------
        phi
            Porosity values [fraction].

        Returns
        -------
        Velocity amplitude values [m/s].
        """
        return (
            self._phi_amplitude
            / (1 + np.exp(-self._phi_x_scaling * (phi - self._phi_median_point)))
            + self._phi_bias
        )

    def _sigmoid_p_eff(self, p_eff: np.ndarray, amplitude: np.ndarray) -> np.ndarray:
        """
        Calculate velocity from effective pressure using sigmoid function with amplitude.

        Parameters
        ----------
        p_eff
            Effective pressure values [Pa].
        amplitude
            Velocity amplitude values [m/s].

        Returns
        -------
        Velocity values [m/s].
        """
        return (
            amplitude
            / (1 + np.exp(-self._p_eff_x_scaling * (p_eff - self._p_eff_median_point)))
            + self._p_eff_bias
        )

    @override
    def predict_abs(self, inp_arr: np.ndarray, case: str = "in_situ") -> np.ndarray:
        """
        Calculate absolute velocity for specified pressure case.

        Parameters
        ----------
        inp_arr
            Validated input array (n,3).
        case
            Pressure case: "in_situ" or "depleted".

        Returns
        -------
        Velocity values [m/s].
        """
        arr = self.validate_input(inp_arr)

        phi = arr[:, 0]
        p_in_situ = arr[:, 1]
        p_depleted = arr[:, 2]

        # Calculate velocity amplitude from porosity
        velocity_amplitude = self._sigmoid_phi(phi)

        # Select pressure based on case
        p_eff = p_in_situ if case == "in_situ" else p_depleted

        # Calculate velocity from effective pressure and amplitude
        return self._sigmoid_p_eff(p_eff, velocity_amplitude)

    @override
    def todict(self) -> dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "phi_amplitude": self._phi_amplitude,
            "phi_median_point": self._phi_median_point,
            "phi_x_scaling": self._phi_x_scaling,
            "phi_bias": self._phi_bias,
            "p_eff_median_point": self._p_eff_median_point,
            "p_eff_x_scaling": self._p_eff_x_scaling,
            "p_eff_bias": self._p_eff_bias,
            "model_max_pressure": self._model_max_pressure,
            "description": self._description,
        }

    @override
    @classmethod
    def load(cls, file: str | Path) -> Self:
        """Load sigmoidal model from pickle file."""
        with Path(file).open("rb") as f_in:
            d = pickle.load(f_in)

        return cls(
            phi_amplitude=d["phi_amplitude"],
            phi_median_point=d["phi_median_point"],
            phi_x_scaling=d["phi_x_scaling"],
            phi_bias=d["phi_bias"],
            p_eff_median_point=d["p_eff_median_point"],
            p_eff_x_scaling=d["p_eff_x_scaling"],
            p_eff_bias=d["p_eff_bias"],
            model_max_pressure=d["model_max_pressure"],
            description=d["description"],
        )
