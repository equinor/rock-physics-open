from typing import cast

import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as plt
from matplotlib.colorbar import Colorbar
from matplotlib.text import Text

from .ternary_plot_utilities import (
    make_mesh,
    set_ternary_figure,
    ternary_coord_trans,
)


def ternary_plot(
    data1: npt.NDArray[np.float64],
    data2: npt.NDArray[np.float64],
    data3: npt.NDArray[np.float64],
    data4: npt.NDArray[np.float64],
    well_name: str,
    name_data1: str,
    name_data2: str,
    name_data3: str,
    name_data4: str,
    draw_figures: bool = True,
) -> None:
    """Plot three mineral phases in a ternary plot and use a fourth phase for colour coding."""
    _, ax = set_ternary_figure(
        delta_x=2,
        delta_y=2,
        title="Ternary Plot of Mineral Phases",
        well_name=well_name,
    )

    # Draw background
    vertices = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.1],
            [1.0, 0.0, 0.0],
        ]
    )
    vertices_xy = ternary_coord_trans(vertices)

    _ = plt.fill(vertices_xy[:, 0], vertices_xy[:, 1], color="w")
    _ = plt.plot(vertices_xy[:, 0], vertices_xy[:, 1], "k", linewidth=1)

    _, _ = make_mesh(ax)

    # Annotate the axes
    label_handles: list[Text] = []
    label_handles.append(
        ax.text(
            x=0.15,
            y=np.sqrt(3) / 4 + 0.05,
            s=name_data1,
            horizontalalignment="center",
            rotation=60,
        )
    )
    label_handles.append(
        ax.text(
            x=0.5,
            y=-0.075,
            s=name_data2,
            horizontalalignment="center",
        )
    )
    label_handles.append(
        ax.text(
            x=0.85,
            y=np.sqrt(3) / 4 + 0.05,
            s=name_data3,
            horizontalalignment="center",
            rotation=-60,
        )
    )
    plt.setp(label_handles, fontname="sans-serif", fontweight="bold", fontsize=11)

    # Plot data
    data_xy = ternary_coord_trans(data1, data2, data3)
    _ = plt.scatter(data_xy[:, 0], data_xy[:, 1], s=64, c=data4, zorder=4)

    hcb = cast(  # Casting due to incomplete type hints in matplotlib
        Colorbar,
        plt.colorbar(),
    )
    hcb.set_label(name_data4, fontname="sans-serif", fontweight="bold", fontsize=11)

    if draw_figures:
        plt.draw()
