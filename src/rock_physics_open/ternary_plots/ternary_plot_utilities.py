import ctypes
import os
import platform
import re

import numpy as np
from matplotlib import pyplot as plt

HORZRES = 8
VERTRES = 10

# The following code is only needed if the figure size is given in inches
# LOGPIXELSX = 88
# LOGPIXELSY = 90


def _set_ternary_figure(delta_x, delta_y, title, well_name):
    """Find screen sixe, and make a suitable figure size and position, shift window with
    number of steps given by deltaX and deltaY.
    """

    if platform.system() == "Windows":
        dc = ctypes.windll.user32.GetDC(0)
        pix_x = ctypes.windll.gdi32.GetDeviceCaps(dc, HORZRES)
        pix_y = ctypes.windll.gdi32.GetDeviceCaps(dc, VERTRES)
    elif platform.system() == "Linux":
        f = os.popen("xrandr | grep '*'")
        pix_x, pix_y = np.array(re.findall(r"\d+", f.read()), dtype=int)[0:2]
    elif platform.system() == "Darwin":
        f = os.popen("system_profiler SPDisplaysDataType | grep Resolution")
        pix_x, pix_y = np.array(re.findall(r"\d+", f.read()), dtype=int)[0:2]
    else:
        raise ValueError("Unrecognised operating system")

    # Add figure
    fig = plt.figure(title, facecolor=[0.9, 0.9, 0.9])
    # Set figure size and position
    mngr = plt.get_current_fig_manager()
    if platform.system() == "Windows":
        wid = int(pix_x * 0.4)
        hei = int(pix_y * 0.6)
        start_x = int(pix_x * (0.1 + delta_x / 5))
        start_y = int(pix_y * (0.1 + delta_y / 10))

        mngr.window.wm_geometry("{}x{}+{}+{}".format(wid, hei, start_x, start_y))
    else:
        wid = int(pix_x * 0.2)
        hei = int(pix_y * 0.6)
        start_x = int(pix_x * (0.1 + delta_x / 5))
        start_y = int(pix_y * (0.1 + delta_y / 10))
        mngr.window.wm_geometry("{}x{}+{}+{}".format(wid, hei, start_x, start_y))

    # Set default colour map
    plt.jet()

    ax = plt.axes(frameon=False, aspect="equal")
    ax.set_title(well_name)
    plt.ylim([-0.02, 1])
    plt.xlim([0, 1])
    ax.set_axis_off()

    return fig, ax


def _ternary_coord_trans(*args):
    """Routine to transform ternary coordinates to xy coordinates.
    Inputs can either be 3 separate coordinate arrays or a nX3 array
    The sum of the input coordinates should be one - the routine will normalise the inputs

    Returns
    -------
    np.ndarray
        Coordinates in nx2 numpy array.
    """
    trans_mat = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, 0.5 * np.sqrt(3)],
        ]
    )

    tern_coord = None
    if len(args) == 3:
        try:
            tern_coord = np.array(np.column_stack(args[:]))
        except ValueError:
            print(
                f"{__file__}: Could not combine inputs to _ternary_coord_trans routine"
            )
    elif len(args) == 1:
        tern_coord = np.array(args[0])
    else:
        raise ValueError(f"{__file__}: Unexpected input to TernaryCoordTrans routine")

    # Normalise inputs
    tot = tern_coord.sum(axis=1).reshape(tern_coord.shape[0], 1)
    tern_coord = tern_coord / tot

    return np.array(tern_coord @ trans_mat)


def _triangle_transform(xy1):
    """

    Parameters
    ----------
    xy1 : np.ndarray
        Input coordinates.

    Returns
    -------
    np.ndarray
        Transformed input.
    """
    xy = np.ones_like(xy1, dtype=float)
    xy[:, 0] = 0.5 + (xy1[:, 0] - 0.5) * (1 - xy1[:, 1])
    xy[:, 1] = xy1[:, 1] * np.sqrt(3) / 2

    return xy


def _make_mesh(ax):
    """Make coordinate mesh."""
    # Make coordinate mesh
    i = np.linspace(0, 1, 11).reshape(11, 1)
    text_handles = []
    line_handles = []
    # 1
    xy1 = _ternary_coord_trans(i, np.zeros(len(i)), 1 - i)
    xy2 = _ternary_coord_trans(i, 1 - i, np.zeros(len(i)))
    for j in range(len(i)):
        line_handles.append(
            ax.plot(
                [
                    xy1[j, 0],
                    xy2[j, 0],
                ],
                [
                    xy1[j, 1],
                    xy2[j, 1],
                ],
                ":k",
                linewidth=0.25,
            )
        )
    # 2
    xy1 = _ternary_coord_trans(np.zeros(len(i)), i, 1 - i)
    xy2 = _ternary_coord_trans(1 - i, i, np.zeros(len(i)))
    for j in range(len(i)):
        line_handles.append(
            ax.plot(
                [
                    xy1[j, 0],
                    xy2[j, 0],
                ],
                [
                    xy1[j, 1],
                    xy2[j, 1],
                ],
                ":k",
                linewidth=0.25,
            )
        )
    # 3
    xy1 = _ternary_coord_trans(1 - i, np.zeros(len(i)), i)
    xy2 = _ternary_coord_trans(np.zeros(len(i)), 1 - i, i)
    for j in range(len(i)):
        line_handles.append(
            ax.plot(
                [
                    xy1[j, 0],
                    xy2[j, 0],
                ],
                [
                    xy1[j, 1],
                    xy2[j, 1],
                ],
                ":k",
                linewidth=0.25,
            )
        )
    # Tick mark text
    xy1 = _ternary_coord_trans(1 - i, i, np.zeros(len(i)))
    for j in range(len(i)):
        text_handles.append(ax.text(xy1[j, 0], xy1[j, 1] - 0.025, "%.1f" % (i[j])))
    xy1 = _ternary_coord_trans(i, np.zeros(len(i)), 1 - i)
    for j in range(len(i)):
        text_handles.append(
            ax.text(xy1[j, 0] - 0.05, xy1[j, 1] + 0.025, "%.1f" % (i[j]))
        )
    xy1 = _ternary_coord_trans(np.zeros(len(i)), 1 - i, i)
    for j in range(len(i)):
        text_handles.append(ax.text(xy1[j, 0], xy1[j, 1] + 0.025, "%.1f" % (i[j])))
    plt.setp(text_handles, fontsize=10)

    return text_handles, line_handles
