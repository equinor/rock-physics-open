import numpy as np
from matplotlib import pyplot as plt

from .ternary_plot_utilities import (
    _make_mesh,
    _set_ternary_figure,
    _ternary_coord_trans,
)


def _ternary_plot(
    data1,
    data2,
    data3,
    data4,
    well_name,
    name_data1,
    name_data2,
    name_data3,
    name_data4,
    draw_figures=True,
):
    """Plot three mineral phases in a ternary plot and use a fourth phase for colour coding."""
    fig, ax = _set_ternary_figure(2, 2, "Ternary Plot of Mineral Phases", well_name)

    # Draw background
    vertices = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.1],
            [1.0, 0.0, 0.0],
        ]
    )
    vertices_xy = _ternary_coord_trans(vertices)

    plt.fill(vertices_xy[:, 0], vertices_xy[:, 1], color="w")
    plt.plot(vertices_xy[:, 0], vertices_xy[:, 1], "k", linewidth=1)

    _, _ = _make_mesh(ax)

    # Annotate the axes
    label_handles = []
    label_handles.append(
        ax.text(
            0.15,
            np.sqrt(3) / 4 + 0.05,
            name_data1,
            horizontalalignment="center",
            rotation=60,
        )
    )
    label_handles.append(ax.text(0.5, -0.075, name_data2, horizontalalignment="center"))
    label_handles.append(
        ax.text(
            0.85,
            np.sqrt(3) / 4 + 0.05,
            name_data3,
            horizontalalignment="center",
            rotation=-60,
        )
    )
    plt.setp(label_handles, fontname="sans-serif", fontweight="bold", fontsize=11)

    # Plot data
    data_xy = _ternary_coord_trans(data1, data2, data3)
    plt.scatter(data_xy[:, 0], data_xy[:, 1], s=64, c=data4, zorder=4)

    hcb = plt.colorbar()
    hcb.set_label(name_data4, fontname="sans-serif", fontweight="bold", fontsize=11)

    if draw_figures:
        plt.draw()
