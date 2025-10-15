from __future__ import annotations

import pickle
from typing import Any, Self

import numpy as np

from rock_physics_open.sandstone_models.patchy_cement_model import (
    patchy_cement_model_dry,
)

from .base_pressure_model import BasePressureModel


class PatchyCementDryPressureModel(BasePressureModel):
    """
    Pressure sensitivity model using patchy cement dry rock properties.

    This model combines friable and cemented sandstone behavior, representing
    partially cemented sands with heterogeneous cement distribution.

    Input format (n,9): [phi, k_min, mu_min, rho_min, k_cem, mu_cem, rho_cem, p_eff_in_situ, p_eff_depleted]

    Where:
    - phi: porosity [fraction]
    - k_min: mineral bulk modulus [Pa]
    - mu_min: mineral shear modulus [Pa]
    - rho_min: mineral density [kg/m³]
    - k_cem: cement bulk modulus [Pa]
    - mu_cem: cement shear modulus [Pa]
    - rho_cem: cement density [kg/m³]
    - p_eff_in_situ: effective pressure in-situ [Pa]
    - p_eff_depleted: effective pressure depleted [Pa]
    """

    def __init__(
        self,
        frac_cem: float,
        phi_c: float,
        coord_num_func: str = "Porosity",
        n: float | None = None,
        shear_red: float = 1.0,
        model_max_pressure: float | None = None,
        description: str = "",
    ):
        """
        Initialize patchy cement dry pressure model.

        Parameters
        ----------
        frac_cem : float
            Cement volume fraction [fraction] defining upper bound behavior.
        phi_c : float
            Critical porosity [fraction].
        coord_num_func : str
            Coordination number method: "Porosity" or "ConstVal".
        n : float | None
            Coordination number [unitless]. Used if coord_num_func="ConstVal".
        shear_red : float
            Shear reduction factor [fraction].
        model_max_pressure : float | None
            Maximum pressure for predict_max method [Pa].
        description : str
            Model description.
        """
        super().__init__(model_max_pressure, description)
        self._frac_cem = frac_cem
        self._phi_c = phi_c
        self._coord_num_func = coord_num_func
        self._n = n
        self._shear_red = shear_red

    def validate_input(self, inp_arr: np.ndarray) -> np.ndarray:
        """
        Validate input array for patchy cement model.

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
            If input is not a 2D numpy array with 9 columns.
        """
        if not isinstance(inp_arr, np.ndarray):
            raise ValueError("Input must be numpy ndarray.")
        if inp_arr.ndim != 2 or inp_arr.shape[1] != 9:
            raise ValueError(
                "Input must be (n,9): [phi, k_min, mu_min, rho_min, k_cem, mu_cem, rho_cem, p_eff_in_situ, p_eff_depleted]"
            )
        return inp_arr

    def _compute_moduli(
        self, inp_arr: np.ndarray, case: str = "in_situ"
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate absolute bulk modulus for specified pressure case.

        Parameters
        ----------
        inp_arr : np.ndarray
            Validated input array (n,9).
        case : str
            Pressure case: 'in_situ' or 'depleted'.

        Returns
        -------
        np.ndarray
            Bulk modulus values [Pa].
        """
        arr = self.validate_input(inp_arr)

        # Parse input columns
        phi = arr[:, 0]
        k_min = arr[:, 1]
        mu_min = arr[:, 2]
        rho_min = arr[:, 3]
        k_cem = arr[:, 4]
        mu_cem = arr[:, 5]
        rho_cem = arr[:, 6]
        p_in_situ = arr[:, 7]
        p_depleted = arr[:, 8]

        # Select pressure based on case
        p_eff = p_in_situ if case == "in_situ" else p_depleted

        # Calculate dry bulk modulus using patchy cement model
        k_dry, mu, _ = patchy_cement_model_dry(
            k_min,
            mu_min,
            rho_min,
            k_cem,
            mu_cem,
            rho_cem,
            phi,
            p_eff,
            self._frac_cem,
            self._phi_c,
            self._coord_num_func,
            self._n,
            self._shear_red,
        )

        return k_dry, mu

    def todict(self) -> dict[str, Any]:
        """
        Convert model to dictionary for serialization.

        Returns
        -------
        dict[str, Any]
            Dictionary containing all model parameters.
        """
        return {
            "frac_cem": self._frac_cem,
            "phi_c": self._phi_c,
            "coord_num_func": self._coord_num_func,
            "n": self._n,
            "shear_red": self._shear_red,
            "model_max_pressure": self._model_max_pressure,
            "description": self._description,
        }

    @classmethod
    def load(cls, file: str | bytes) -> Self:
        """
        Load patchy cement model from pickle file.

        Parameters
        ----------
        file : str | bytes
            File path for loading.

        Returns
        -------
        PatchyCementDryPressureModel
            Loaded model instance.
        """
        with open(file, "rb") as f_in:
            d = pickle.load(f_in)

        return cls(
            frac_cem=d["frac_cem"],
            phi_c=d["phi_c"],
            coord_num_func=d["coord_num_func"],
            n=d["n"],
            shear_red=d["shear_red"],
            model_max_pressure=d["model_max_pressure"],
            description=d["description"],
        )


class PatchyCementDryShearModulusPressureModel(PatchyCementDryPressureModel):
    """
    Pressure sensitivity model using patchy cement dry rock properties.

    This model combines friable and cemented sandstone behavior, representing
    partially cemented sands with heterogeneous cement distribution.

    Input format (n,9): [phi, k_min, mu_min, rho_min, k_cem, mu_cem, rho_cem, p_eff_in_situ, p_eff_depleted]

    Where:
    - phi: porosity [fraction]
    - k_min: mineral bulk modulus [Pa]
    - mu_min: mineral shear modulus [Pa]
    - rho_min: mineral density [kg/m³]
    - k_cem: cement bulk modulus [Pa]
    - mu_cem: cement shear modulus [Pa]
    - rho_cem: cement density [kg/m³]
    - p_eff_in_situ: effective pressure in-situ [Pa]
    - p_eff_depleted: effective pressure depleted [Pa]
    """

    def predict_abs(self, inp_arr: np.ndarray, case: str = "in_situ") -> np.ndarray:
        """
        Calculate absolute shear modulus for specified pressure case.

        Parameters
        ----------
        inp_arr : np.ndarray
            Validated input array (n,9).
        case : str
            Pressure case: "in_situ" or "depleted".

        Returns
        -------
        np.ndarray
            Shear modulus values [Pa].
        """
        k_dry, mu = self._compute_moduli(inp_arr, case)
        return mu


class PatchyCementDryBulkModulusPressureModel(PatchyCementDryPressureModel):
    """
    Pressure sensitivity model using patchy cement dry rock properties.

    This model combines friable and cemented sandstone behavior, representing
    partially cemented sands with heterogeneous cement distribution.

    Input format (n,9): [phi, k_min, mu_min, rho_min, k_cem, mu_cem, rho_cem, p_eff_in_situ, p_eff_depleted]

    Where:
    - phi: porosity [fraction]
    - k_min: mineral bulk modulus [Pa]
    - mu_min: mineral shear modulus [Pa]
    - rho_min: mineral density [kg/m³]
    - k_cem: cement bulk modulus [Pa]
    - mu_cem: cement shear modulus [Pa]
    - rho_cem: cement density [kg/m³]
    - p_eff_in_situ: effective pressure in-situ [Pa]
    - p_eff_depleted: effective pressure depleted [Pa]
    """

    def predict_abs(self, inp_arr: np.ndarray, case: str = "in_situ") -> np.ndarray:
        """
        Calculate absolute bulk modulus for specified pressure case.

        Parameters
        ----------
        inp_arr : np.ndarray
            Validated input array (n,9).
        case : str
            Pressure case: "in_situ" or "depleted".

        Returns
        -------
        np.ndarray
            Bulk modulus values [Pa].
        """
        k_dry, mu = self._compute_moduli(inp_arr, case)
        return k_dry
