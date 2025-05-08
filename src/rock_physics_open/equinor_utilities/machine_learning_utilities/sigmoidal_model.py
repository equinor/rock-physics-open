import pickle
from io import BufferedIOBase
from typing import Union

import numpy as np


class Sigmoid:
    def __init__(
        self,
        amplitude: float = None,
        median_point: float = None,
        x_scaling: float = None,
        bias: float = None,
        description="",
    ):
        self._amplitude = amplitude
        self._median_point = median_point
        self._x_scaling = x_scaling
        self._bias = bias
        self._description = description

    def todict(self):
        return {
            "amplitude": self._amplitude,
            "median_point": self._median_point,
            "x_scaling": self._x_scaling,
            "bias": self._bias,
            "description": self._description,
        }

    @property
    def amplitude(self):
        return self._amplitude

    @property
    def median_point(self):
        return self._median_point

    @property
    def x_scaling(self):
        return self._x_scaling

    @property
    def bias(self):
        return self._bias

    @property
    def description(self):
        return self._description

    def predict(self, x):
        if isinstance(x, list):
            x = np.array(x)
        if not self._valid():
            return None
        return (
            self._amplitude / (1 + np.exp(-self._x_scaling * (x - self._median_point)))
            + self._bias
        )

    def predict_amp(self, x, amp):
        if isinstance(x, list):
            x = np.array(x)
        if isinstance(amp, list):
            amp = np.array(amp)
        if not self._valid(variant="amp"):
            return None
        return (
            amp / (1 + np.exp(-self._x_scaling * (x - self._median_point))) + self._bias
        )

    @classmethod
    def save(cls, file):
        with open(file, "wb") as f_out:
            pickle.dump(cls, f_out)

    @classmethod
    def load(cls, file):
        with open(file, "rb") as f_in:
            return pickle.load(f_in)

    def _valid(self, variant=None):
        if variant is None and self.amplitude is None:
            raise ValueError('object field "variant" is not set')
        if self.median_point is None:
            raise ValueError('object field "median_point" is not set')
        if self.x_scaling is None:
            raise ValueError('object field "x_scaling" is not set')
        if self.bias is None:
            raise ValueError('object field "bias" is not set')
        return True


class CarbonateSigmoidalPressure:
    def __init__(
        self,
        phi_model: Union[dict, Sigmoid] = None,
        p_eff_model: Union[dict, Sigmoid] = None,
    ):
        if isinstance(phi_model, Sigmoid):
            self._phi_model = phi_model
        elif isinstance(phi_model, dict):
            self._phi_model = Sigmoid(**phi_model)
        else:
            self._phi_model = None
        if isinstance(p_eff_model, Sigmoid):
            self._p_eff_model = p_eff_model
        elif isinstance(p_eff_model, dict):
            self._p_eff_model = Sigmoid(**p_eff_model)
        else:
            self._p_eff_model = None

    @property
    def phi_model(self):
        return self._phi_model

    @phi_model.setter
    def phi_model(self, phi_mod):
        if isinstance(phi_mod, Sigmoid):
            self._phi_model = phi_mod
        else:
            raise ValueError(
                f"{type(self)}: expected input Sigmoid object, received {type(phi_mod)}"
            )

    @property
    def p_eff_model(self):
        return self._p_eff_model

    @p_eff_model.setter
    def p_eff_model(self, p_eff_mod):
        if isinstance(p_eff_mod, Sigmoid):
            self._p_eff_model = p_eff_mod
        else:
            raise ValueError(
                f"{type(self)}: expected input Sigmoid object, received {type(p_eff_mod)}"
            )

    def predict(self, inp_arr: np.ndarray) -> np.ndarray:
        # Don't save any of the intermediate calculations, only return the difference between the effective pressure
        # cases. The method name is set to be the same as for other machine learning models
        self._validate_input(inp_arr)
        velocity_amp = self._phi_model.predict(inp_arr[:, 0].flatten())
        velocity_init = self._p_eff_model.predict_amp(
            inp_arr[:, 1].flatten(), velocity_amp
        )
        velocity_depl = self._p_eff_model.predict_amp(
            inp_arr[:, 2].flatten(), velocity_amp
        )
        return velocity_depl - velocity_init

    def predict_abs(self, inp_arr: np.ndarray, case: str = "in_situ") -> np.ndarray:
        # Method for access to absolute results, not just the difference
        self._validate_input(inp_arr)
        velocity_amp = self._phi_model.predict(inp_arr[:, 0].flatten())
        if case == "in_situ":
            return self._p_eff_model.predict_amp(inp_arr[:, 1].flatten(), velocity_amp)
        return self._p_eff_model.predict_amp(inp_arr[:, 2].flatten(), velocity_amp)

    def _validate_input(self, input_array):
        if not isinstance(input_array, np.ndarray):
            raise ValueError(
                f"{type(input_array)}: should be numpy array with shape n x 3"
            )
        if not ((input_array.ndim == 2) and (input_array.shape[1] == 3)):
            raise ValueError(
                f'{type(self)}: Input array should be of shape n x 3, with columns in sequence "PHIT", '
                f'"P_EFF_in_situ" and "P_EFF_depleted"'
            )

    def todict(self):
        return {
            "phi_model": self._phi_model.todict(),
            "p_eff_model": self._p_eff_model.todict(),
        }

    def save(self, file: Union[str, BufferedIOBase]):
        with open(file, "wb") as f_out:
            pickle.dump(self.todict(), f_out)

    @classmethod
    def load(cls, file: Union[str, BufferedIOBase]):
        with open(file, "rb") as f_in:
            load_dict = pickle.load(f_in)
            return cls(
                phi_model=load_dict["phi_model"], p_eff_model=load_dict["p_eff_model"]
            )
