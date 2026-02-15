import numpy as np
import numpy.typing as npt


def oil_density_to_api(
    rho0: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """_summary_

    Args:
        rho0 (npt.NDArray[np.float64]): oil density at standrd conditions [g/cc]

    Returns:
        npt.NDArray[np.float64]: oil gravity [API]
    """
    return 141.5 / rho0 - 131.5


def oil_api_to_density(
    rho0_api: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """_summary_

    Args:
        rho0 (npt.NDArray[np.float64]): oil density at standrd conditions [g/cc]

    Returns:
        npt.NDArray[np.float64]: oil gravity [API]
    """
    return 141.5 / (rho0_api + 131.5)


def oil_density_to_gcc(rho0: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return rho0 / 1000.0


def oil_density_to_kg_m_3(rho0_gcc: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return rho0_gcc * 1000.0
