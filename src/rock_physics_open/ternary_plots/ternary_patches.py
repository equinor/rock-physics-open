from typing import cast

import numpy as np
import numpy.typing as npt
from matplotlib import path as pa, pyplot as plt
from matplotlib.colorbar import Colorbar
from matplotlib.text import Text

from rock_physics_open.equinor_utilities.various_utilities.types import Array2D

from .ternary_plot_utilities import (
    make_mesh,
    set_ternary_figure,
    ternary_coord_trans,
)


def ternary_patches(
    quartz: npt.NDArray[np.float64],
    carb: npt.NDArray[np.float64],
    clay: npt.NDArray[np.float64],
    kero: npt.NDArray[np.float64],
    well_name: str,
    draw_figures: bool = True,
) -> npt.NDArray[np.float64]:
    """Class display of the shale.

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
    well_name : str
        Plot heading with well name.
    draw_figures : bool
        Decide if figures are drawn or not, default is True.

    Returns
    -------
    np.ndarray
        Lithology class [int].
    """
    _, ax = set_ternary_figure(
        delta_x=0,
        delta_y=0,
        title="Lithological Class Ternary Plot",
        well_name=well_name,
    )

    # Define patches 1 - 16
    names = [
        "Silica-\ndominated\nlithotype",
        "Clay-rich\nsiliceous mudstone",
        "Mixed\nsiliceous\nmudstone",
        "Carbonate-rich\nsiliceous mudstone",
        "Argilaceous/\nsiliceous\nmudstone",
        "Mixed\nmudstone",
        "Carbonate/\nsiliceous\nmudstone",
        "Silica-rich\nargillaceous mudstone",
        "Mixed\nargillaceous\nmudstone",
        "Argillaceous/\ncarbonate\nmudstone\n(marl)",
        "Mixed\ncarbonate\nmudstone",
        "Silica-rich\ncarbonate mudstone",
        "Clay-\ndominated\nlithotype",
        "Carbonate-rich\nargillaceous mudstone",
        "Clay-rich\ncarbonate mudstone",
        "Carbonate-\ndominated\nlithotype",
    ]
    text_ang = [0, 60, 0, 0, 0, 0, 0, 60, 0, 0, 0, 0, 0, -60, -60, 0]
    col = (
        1
        / 255
        * np.array(
            [
                [244, 237, 29],
                [249, 246, 142],
                [255, 230, 137],
                [249, 246, 137],
                [254, 220, 95],
                [222, 238, 197],
                [131, 209, 193],
                [214, 170, 40],
                [197, 166, 114],
                [169, 169, 134],
                [193, 215, 239],
                [150, 180, 218],
                [175, 116, 76],
                [169, 123, 44],
                [96, 149, 202],
                [114, 126, 188],
            ]
        )
    )

    vertices: list[Array2D[np.float64]] = [
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.8, 0.0, 0.2],
                [0.8, 0.2, 0.0],
            ]
        ),
        np.array(
            [
                [0.8, 0.0, 0.2],
                [0.5, 0.0, 0.5],
                [0.5, 0.1, 0.4],
                [0.8, 0.1, 0.1],
            ]
        ),
        np.array(
            [
                [0.8, 0.1, 0.1],
                [0.5, 0.1, 0.4],
                [0.5, 0.4, 0.1],
            ]
        ),
        np.array(
            [
                [0.8, 0.1, 0.1],
                [0.5, 0.4, 0.1],
                [0.5, 0.5, 0.0],
                [0.8, 0.2, 0.0],
            ]
        ),
        np.array(
            [
                [0.5, 0.0, 0.5],
                [0.3, 0.2, 0.5],
                [0.5, 0.2, 0.3],
            ]
        ),
        np.array(
            [
                [0.5, 0.2, 0.3],
                [0.3, 0.2, 0.5],
                [0.2, 0.3, 0.5],
                [0.2, 0.5, 0.3],
                [0.3, 0.5, 0.2],
                [0.5, 0.3, 0.2],
            ]
        ),
        np.array(
            [
                [0.5, 0.3, 0.2],
                [0.3, 0.5, 0.2],
                [0.5, 0.5, 0.0],
            ]
        ),
        np.array(
            [
                [0.5, 0.0, 0.5],
                [0.2, 0.0, 0.8],
                [0.1, 0.1, 0.8],
                [0.4, 0.1, 0.5],
            ]
        ),
        np.array(
            [
                [0.4, 0.1, 0.5],
                [0.1, 0.1, 0.8],
                [0.1, 0.4, 0.5],
            ]
        ),
        np.array(
            [
                [0.0, 0.5, 0.5],
                [0.2, 0.5, 0.3],
                [0.2, 0.3, 0.5],
            ]
        ),
        np.array(
            [
                [0.1, 0.5, 0.4],
                [0.1, 0.8, 0.1],
                [0.4, 0.5, 0.1],
            ]
        ),
        np.array(
            [
                [0.4, 0.5, 0.1],
                [0.1, 0.8, 0.1],
                [0.2, 0.8, 0.0],
                [0.5, 0.5, 0.0],
            ]
        ),
        np.array(
            [
                [0.2, 0.0, 0.8],
                [0.0, 0.0, 1.0],
                [0.0, 0.2, 0.8],
            ]
        ),
        np.array(
            [
                [0.0, 0.2, 0.8],
                [0.0, 0.5, 0.5],
                [0.1, 0.4, 0.5],
                [0.1, 0.1, 0.8],
            ]
        ),
        np.array(
            [
                [0.0, 0.5, 0.5],
                [0.0, 0.8, 0.2],
                [0.1, 0.8, 0.1],
                [0.1, 0.5, 0.4],
            ]
        ),
        np.array(
            [
                [0.0, 0.8, 0.2],
                [0.0, 1.0, 0.0],
                [0.2, 0.8, 0.0],
            ]
        ),
    ]

    vertices_xy = [ternary_coord_trans(v) for v in vertices]

    _ = [
        plt.fill(
            vertices_xy[i][:, 0],
            vertices_xy[i][:, 1],
            facecolor=col[i, :],
            edgecolor="k",
        )
        for i in range(len(vertices))
    ]
    _ = [
        plt.text(
            x=float(np.mean(vertices_xy[i][:, 0])),
            y=float(np.mean(vertices_xy[i][:, 1])),
            s=names[i],
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=8,
            rotation=text_ang[i],
        )
        for i in range(len(vertices))
    ]

    # Make coordinate mesh
    _, _ = make_mesh(ax)

    # Annotate the axes
    label_handles: list[Text] = []
    label_handles.append(
        plt.text(
            x=0.5,
            y=-0.075,
            s="Carbonate",
            horizontalalignment="center",
        )
    )
    label_handles.append(
        plt.text(
            x=0.15,
            y=np.sqrt(3) / 4 + 0.05,
            s="Quartz",
            horizontalalignment="center",
            rotation=60,
        )
    )
    label_handles.append(
        plt.text(
            x=0.85,
            y=np.sqrt(3) / 4 + 0.05,
            s="Clay",
            horizontalalignment="center",
            rotation=-60,
        )
    )
    plt.setp(label_handles, fontname="sans-serif", fontweight="bold", fontsize=11)

    # Plot data
    data_xy = ternary_coord_trans(quartz, carb, clay)
    _ = plt.scatter(x=data_xy[:, 0], y=data_xy[:, 1], s=64, c=kero, zorder=4)

    # Classification - start at 2001 to match predefined lithology classes in
    startclass = 2000
    lith_class = np.ones(quartz.shape) * np.nan
    for i in range(len(vertices_xy)):
        # class_path = pa.Path(vertices_xy[i], closed=True)
        class_path = pa.Path(vertices_xy[i])
        in_path = class_path.contains_points(data_xy)
        lith_class[in_path] = i + startclass

    hcb = cast(  # Casting due to incomplete typing in matplotlib
        Colorbar,
        plt.colorbar(),
    )
    hcb.set_label("Kerogen", fontname="sans-serif", fontweight="bold", fontsize=11)

    # plt.show() <= Show command must be issued from the controlling class method
    if draw_figures:
        plt.draw()
    return lith_class
