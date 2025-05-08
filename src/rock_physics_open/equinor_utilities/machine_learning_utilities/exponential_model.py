import pickle
from typing import Union

import numpy as np


def _verify_input(inp_arr):
    if isinstance(inp_arr, np.ndarray) and not (
        inp_arr.ndim == 2 and inp_arr.shape[1] == 3
    ):
        raise ValueError(
            "Input to predict method should be an nx3 numpy array with columns velocity, in situ "
            "pressure and depleted pressure"
        )


class CarbonateExponentialPressure:
    def __init__(
        self,
        a_factor: float = None,
        b_factor: float = None,
        model_max_pressure: float = None,
        description: str = "",
    ):
        self._a_factor = a_factor
        self._b_factor = b_factor
        self._model_max_pressure = model_max_pressure
        self._description = description

    def todict(self):
        return {
            "a_factor": self._a_factor,
            "b_factor": self._b_factor,
            "model_max_pressure": self._model_max_pressure,
            "description": self._description,
        }

    @property
    def a_factor(self) -> float:
        return self._a_factor

    @property
    def b_factor(self) -> float:
        return self._b_factor

    @property
    def max_pressure(self) -> float:
        return self._model_max_pressure

    @property
    def description(self) -> str:
        return self._description

    def predict(self, inp_arr: np.ndarray) -> Union[np.ndarray, None]:
        _verify_input(inp_arr)
        if not self._valid():
            return None
        vel = inp_arr[:, 0]
        eff_pres_in_situ = inp_arr[:, 1]
        eff_pres_depl = inp_arr[:, 2]
        # Return differential velocity to match alternative models
        return (
            vel
            * (1.0 - self._a_factor * np.exp(-eff_pres_depl / self._b_factor))
            / (1.0 - self._a_factor * np.exp(-eff_pres_in_situ / self._b_factor))
            - vel
        )

    def predict_max(self, inp_arr: np.ndarray) -> Union[np.ndarray, None]:
        _verify_input(inp_arr)
        if not self._valid():
            return None
        vel = inp_arr[:, 0]
        eff_pres_in_situ = inp_arr[:, 1]
        return (
            vel
            * (
                1.0
                - self._a_factor * np.exp(-self._model_max_pressure / self._b_factor)
            )
            / (1.0 - self._a_factor * np.exp(-eff_pres_in_situ / self.b_factor))
        )

    def predict_abs(self, inp_arr: np.ndarray) -> Union[np.ndarray, None]:
        _verify_input(inp_arr)
        if not self._valid():
            return None
        vel = inp_arr[:, 0]
        eff_pres_in_situ = inp_arr[:, 1]
        eff_pres_depl = inp_arr[:, 2]
        return (
            vel
            * (1.0 - self._a_factor * np.exp(-eff_pres_depl / self._b_factor))
            / (1.0 - self._a_factor * np.exp(-eff_pres_in_situ / self._b_factor))
        )

    def save(self, file):
        with open(file, "wb") as f_out:
            pickle.dump(self.todict(), f_out)

    @classmethod
    def load(cls, file):
        with open(file, "rb") as f_in:
            inp_pcl = pickle.load(f_in)
            return cls(
                a_factor=inp_pcl["a_factor"],
                b_factor=inp_pcl["b_factor"],
                model_max_pressure=inp_pcl["model_max_pressure"],
                description=inp_pcl["description"],
            )

    def _valid(self):
        if self.a_factor is None:
            raise ValueError('object field "a_factor" is not set')
        if self.b_factor is None:
            raise ValueError('object field "b_factor" is not set')
        if self.max_pressure is None:
            raise ValueError('object field "max_pressure" is not set')
        return True
