from __future__ import annotations

import pickle
from typing import Any, Self

import numpy as np

from rock_physics_open.sandstone_models.friable_models import friable_model_dry

from .base_pressure_model import BasePressureModel


class FriableDryPressureModel(BasePressureModel):
    """
    Pressure sensitivity model using friable dry rock bulk modulus.

    This model calculates pressure-dependent bulk modulus using the friable
    sandstone model, which represents unconsolidated sands where porosity
    variation is due to grain sorting.

    Input format (n,5): [phi, k_min, mu_min, p_eff_in_situ, p_eff_depleted]

    Where:
    - phi: porosity [fraction]
    - k_min: mineral bulk modulus per sample [Pa]
    - mu_min: mineral shear modulus per sample [Pa]
    - p_eff_in_situ: effective pressure in-situ [Pa]
    - p_eff_depleted: effective pressure depleted [Pa]

    The model returns bulk and shear modulus changes due to pressure variations.
    """

    def __init__(
        self,
        phi_c: float,
        coord_num_func: str = "Porosity",
        n: float | None = None,
        shear_red: float = 1.0,
        model_max_pressure: float | None = None,
        description: str = "",
    ):
        """
        Initialize friable dry pressure model.

        Parameters
        ----------
        phi_c : float
            Critical porosity [fraction]. Porosity above this value is undefined.
        coord_num_func : str
            Coordination number calculation method: "Porosity" or "ConstVal".
        n : float | None
            Coordination number [unitless]. Used only if coord_num_func="ConstVal".
        shear_red : float
            Shear reduction factor [fraction] for tangential grain contacts.
        model_max_pressure : float | None
            Maximum pressure for predict_max method [Pa].
        description : str
            Human-readable description of this model instance.
        """
        super().__init__(model_max_pressure, description)
        self._phi_c = phi_c
        self._coord_num_func = coord_num_func
        self._n = n
        self._shear_red = shear_red

    def validate_input(self, inp_arr: np.ndarray) -> np.ndarray:
        """
        Validate input array for friable model.

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
            If input is not a 2D numpy array with 5 columns.
        """
        if not isinstance(inp_arr, np.ndarray):
            raise ValueError("Input must be numpy ndarray.")
        if inp_arr.ndim != 2 or inp_arr.shape[1] != 5:
            raise ValueError(
                "Input must be (n,5): [phi, k_min, mu_min, p_eff_in_situ, p_eff_depleted]"
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
            Validated input array (n,5).
        case : str
            Pressure case: "in_situ" or "depleted".

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
        p_in_situ = arr[:, 3]
        p_depleted = arr[:, 4]

        # Select pressure based on case
        p_eff = p_in_situ if case == "in_situ" else p_depleted

        # Calculate dry bulk modulus using friable model
        k_dry, mu = friable_model_dry(
            k_min,
            mu_min,
            phi,
            p_eff,
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
        Load friable model from pickle file.

        Parameters
        ----------
        file : str, bytes
            File path for loading.

        Returns
        -------
        FriableDryPressureModel
            Loaded model instance.
        """
        with open(file, "rb") as f_in:
            d = pickle.load(f_in)

        return cls(
            phi_c=d["phi_c"],
            coord_num_func=d["coord_num_func"],
            n=d["n"],
            shear_red=d["shear_red"],
            model_max_pressure=d["model_max_pressure"],
            description=d["description"],
        )


class FriableDryShearModulusPressureModel(FriableDryPressureModel):
    """
    Pressure sensitivity model using friable dry rock bulk modulus.

    This model calculates pressure-dependent bulk modulus using the friable
    sandstone model, which represents unconsolidated sands where porosity
    variation is due to grain sorting.

    Input format (n,5): [phi, k_min, mu_min, p_eff_in_situ, p_eff_depleted]

    Where:
    - phi: porosity [fraction]
    - k_min: mineral bulk modulus per sample [Pa]
    - mu_min: mineral shear modulus per sample [Pa]
    - p_eff_in_situ: effective pressure in-situ [Pa]
    - p_eff_depleted: effective pressure depleted [Pa]

    The model returns shear modulus changes due to pressure variations.
    """

    def predict_abs(self, inp_arr: np.ndarray, case: str = "in_situ") -> np.ndarray:
        k_dry, mu = self._compute_moduli(inp_arr, case)
        return mu


class FriableDryBulkModulusPressureModel(FriableDryPressureModel):
    """
    Pressure sensitivity model using friable dry rock bulk modulus.

    This model calculates pressure-dependent bulk modulus using the friable
    sandstone model, which represents unconsolidated sands where porosity
    variation is due to grain sorting.

    Input format (n,5): [phi, k_min, mu_min, p_eff_in_situ, p_eff_depleted]

    Where:
    - phi: porosity [fraction]
    - k_min: mineral bulk modulus per sample [Pa]
    - mu_min: mineral shear modulus per sample [Pa]
    - p_eff_in_situ: effective pressure in-situ [Pa]
    - p_eff_depleted: effective pressure depleted [Pa]

    The model returns bulk modulus changes due to pressure variations.
    """

    def predict_abs(self, inp_arr: np.ndarray, case: str = "in_situ") -> np.ndarray:
        k_dry, mu = self._compute_moduli(inp_arr, case)
        return k_dry
