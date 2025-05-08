import numpy as np
from matplotlib import pyplot as plt

from .ternary_plot_utilities import _set_ternary_figure, _triangle_transform


def _shale_prop_ternary(
    quartz,
    carb,
    clay,
    kero,
    phit,
    col_code,
    name_col_code,
    well_name,
    draw_figures=True,
):
    """Calculate hardness.

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
    phit : np.ndarray
        Porosity [fraction].
    col_code : np.ndarray
        Property used for colour coding [unknown].
    name_col_code : str
        Plot annotation of log used for colour coding.
    well_name : str
        Plot heading with well name.

    Returns
    -------
    np.ndarray
        hardness [float].
    """
    KEROMAX = 0.25

    _, _ = _set_ternary_figure(
        1, 1, "Hardness vs. Organic Content Ternary Plot", well_name
    )

    # Define patches
    col = (
        1
        / 255
        * np.array(
            [
                [
                    208,
                    208,
                    208,
                ],
                [
                    198,
                    198,
                    198,
                ],
                [
                    189,
                    189,
                    189,
                ],
                [
                    179,
                    179,
                    179,
                ],
                [
                    169,
                    169,
                    169,
                ],
                [
                    223,
                    213,
                    203,
                ],
                [
                    214,
                    203,
                    193,
                ],
                [
                    204,
                    194,
                    183,
                ],
                [
                    195,
                    183,
                    174,
                ],
                [
                    187,
                    172,
                    164,
                ],
                [
                    239,
                    218,
                    197,
                ],
                [
                    229,
                    208,
                    188,
                ],
                [
                    221,
                    197,
                    178,
                ],
                [
                    213,
                    186,
                    169,
                ],
                [
                    205,
                    174,
                    159,
                ],
                [
                    254,
                    223,
                    192,
                ],
                [
                    246,
                    211,
                    182,
                ],
                [
                    239,
                    200,
                    173,
                ],
                [
                    231,
                    189,
                    164,
                ],
                [223, 177, 154],
            ]
        )
    )

    corner_points = np.array(
        [
            [0.00, 0.00],
            [0.25, 0.00],
            [0.25, 0.20],
            [0.00, 0.20],
        ]
    )

    start_x, start_y = np.meshgrid(
        np.arange(0, 1, 0.25), np.arange(0, 1, 0.2), indexing="ij"
    )
    start_x = start_x.flatten()
    start_y = start_y.flatten()
    vertices = []
    for i in range(len(start_x)):
        vertices.append(corner_points + [start_x[i], start_y[i]])

    vertices_xy = []
    for i in range(len(vertices)):
        vertices_xy.append(_triangle_transform(vertices[i]))

    h_p = []
    for i in range(len(vertices)):
        # h_p.append(plt.fill(vertices_xy[i][:,0],vertices_xy[i][:,1],facecolor=col[i,:],edgecolor='k',linewidth=0.5))
        h_p.append(
            plt.fill(vertices_xy[i][:, 0], vertices_xy[i][:, 1], color=col[i, :])
        )

    # Draw lines around patches
    x = np.array([[0], [1]])
    y = np.array([[0], [0]])
    for i in np.arange(0, 1, 0.2):
        xy = np.column_stack((x, y + i))
        t_xy = _triangle_transform(xy)
        plt.plot(t_xy[:, 0], t_xy[:, 1], "k", linewidth=1)
    x = np.array([[0], [0]])
    y = np.array([[0], [1]])
    for i in np.arange(0, 1.1, 0.25):
        xy = np.column_stack((x + i, y))
        t_xy = _triangle_transform(xy)
        plt.plot(t_xy[:, 0], t_xy[:, 1], "k", linewidth=1)

    # Annotate axes
    xi = np.linspace(0, 1, 5)
    yi = np.linspace(0, 0.25, 6)
    for j in range(len(xi)):
        plt.text(xi[j], -0.025, "%.2f" % (xi[j]))
    for j in range(len(yi)):
        plt.text(
            1 - 0.5 / 0.25 * yi[j] + 0.015,
            yi[j] * np.sqrt(3) / 2 / 0.25,
            "%.2f" % (yi[j]),
        )
    handles = []
    # Axes labels
    handles.append(
        plt.text(
            0.5,
            -0.07,
            "Hardness percentage\nClay+TOC+Phit <== ==> Quartz+Feldspar+Carbonate",
            horizontalalignment="center",
            verticalalignment="center",
        )
    )
    handles.append(
        plt.text(
            0.95,
            np.sqrt(3) / 4,
            "Organic percentage",
            horizontalalignment="center",
            verticalalignment="center",
            rotation=-62,
        )
    )
    plt.setp(handles, fontname="sans-serif", fontweight="bold", fontsize=11)

    # Region boundaries
    df = 0.1 * np.array([1, np.tan(np.pi / 6)])
    bb = np.array(
        [
            [1.00, 0.20],
            [1.00, 0.60],
        ]
    )
    bbt = _triangle_transform(bb)
    bet = bbt + df
    for j in range(bb.shape[0]):
        plt.plot(
            [
                bbt[j, 0],
                bet[j, 0],
            ],
            [
                bbt[j, 1],
                bet[j, 1],
            ],
            "k",
            linewidth=0.5,
        )
    df = 0.1 * np.array([-1, np.tan(np.pi / 6)])
    bb = np.array(
        [
            [0.00, 0.20],
            [0.00, 0.60],
        ]
    )
    bbt = _triangle_transform(bb)
    bet = bbt + df
    for j in range(bb.shape[0]):
        plt.plot(
            [
                bbt[j, 0],
                bet[j, 0],
            ],
            [
                bbt[j, 1],
                bet[j, 1],
            ],
            "k",
            linewidth=0.5,
        )

    # Annotate regions
    names = [
        "Organically lean\nclaystones",
        "Organically rich\nclaystones",
        "Organically dominated\nclaystones",
        "Organically dominated\nmudstones",
        "Organically rich\nmudstones",
        "Organically lean\nmudstones",
    ]
    text_ang = [60, 60, 60, -60, -60, -60]
    text_pos = np.array(
        [
            [0.0, 0.1],
            [0.0, 0.4],
            [0.0, 0.8],
            [1.0, 0.8],
            [1.0, 0.4],
            [1.0, 0.1],
        ]
    )
    text_pos = _triangle_transform(text_pos)
    text_pos = text_pos + np.array(
        [
            [-0.05, 0.0],
            [0.1, 0.05],
        ]
    ).reshape(2, 2).repeat(3, axis=0)
    for i in range(len(names)):
        plt.text(
            text_pos[i, 0],
            text_pos[i, 1],
            names[i],
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=10,
            rotation=text_ang[i],
        )

    # Calculate the hardness parameter
    soft = clay + kero + phit
    brit = quartz + carb

    hard = brit / (brit + soft)

    # Scale kerogen
    kero = np.minimum(kero, KEROMAX)
    kero_scal = kero / KEROMAX

    data_xy = _triangle_transform(np.column_stack((hard, kero_scal)))
    plt.scatter(data_xy[:, 0], data_xy[:, 1], s=64, c=col_code, zorder=4)

    hcb = plt.colorbar(pad=0.1)
    hcb.set_label(name_col_code, fontname="sans-serif", fontweight="bold", fontsize=11)

    # plt.show()<= Show command is issued in controlling class method
    if draw_figures:
        plt.draw()

    return hard
