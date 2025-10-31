import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from .gen_ternary_plot import _ternary_plot
from .shale_prop_ternary import _shale_prop_ternary
from .ternary_patches import _ternary_patches


def run_ternary(
    quartz: npt.NDArray[np.float64],
    carb: npt.NDArray[np.float64],
    clay: npt.NDArray[np.float64],
    kero: npt.NDArray[np.float64],
    phi: npt.NDArray[np.float64],
    misc: npt.NDArray[np.float64],
    misc_log_type: str,
    well_name: str,
    draw_figures: bool = True,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Combined call to three different ternary plots used to describe unconventionals (shale) models.

    Parameters
    ----------
    quartz : np.ndarray
        Quartz volume fraction [fraction].
    carb : np.ndarray
        Carbonate volume fraction [fraction].
    clay : np.ndarray
        Clay volume fraction [fraction].
    kero : np.ndarray
        Kerogen volume fraction [fraction].
    phi : np.ndarray
        Porosity [fraction].
    misc : np.ndarray
        Property used for colour coding [unknown].
    misc_log_type : str
        Plot annotation of log used for colour coding.
    well_name : str
        Plot heading with well name.
    draw_figures : bool
        Decide if figures are drawn or not, default is True.

    Returns
    -------
    tuple
        lith_class, hard : (np.ndarray, np.ndarray).
        lith_class: lithology class [int], hardness [float].
    """
    matplotlib.use("TkAgg")

    lith_class = _ternary_patches(
        quartz, carb, clay, kero, well_name, draw_figures=draw_figures
    )

    hard = _shale_prop_ternary(
        quartz=quartz,
        carb=carb,
        clay=clay,
        kero=kero,
        phit=phi,
        col_code=misc,
        name_col_code=misc_log_type,
        well_name=well_name,
        draw_figures=draw_figures,
    )

    _ternary_plot(
        data1=quartz,
        data2=carb,
        data3=clay,
        data4=kero,
        well_name=well_name,
        name_data1="Quartz",
        name_data2="Carbonate",
        name_data3="Clay",
        name_data4="Kerogen",
        draw_figures=draw_figures,
    )

    if draw_figures:
        plt.show()

    return lith_class, hard
