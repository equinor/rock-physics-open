import os
import sys

import numpy as np


def disp_result_stats(title: str, arr: list | tuple, names_arr: list, **kwargs):
    """
    Display results utilizing tkinter.

    Parameters
    ----------
    title : Title.
    arr : items to display
    names_arr : array of names.
    """
    from tkinter import END, Entry, PhotoImage, Tk

    class Table:
        def __init__(self, tk_root, no_rows, no_cols, info):
            # code for creating table
            str_len = np.vectorize(len)
            text_justify = ["center", "left"]
            text_weight = ["bold", "normal"]
            for i in range(no_rows):
                weigh = text_weight[np.sign(i)]
                for j in range(no_cols):
                    just = text_justify[np.sign(i)]
                    max_len = np.max(str_len(info[:, j]))
                    self.e = Entry(
                        tk_root,
                        width=max_len + 2,
                        fg="black",
                        font=("Consolas", 12, weigh),
                        justify=just,
                    )
                    self.e.grid(row=i, column=j)
                    self.e.insert(END, info[i][j])

    values_only = kwargs.pop("values_only", False)
    root = Tk(**kwargs)
    root.title(title)
    if values_only:
        info_array = np.zeros((len(arr) + 1, 2)).astype(str)
        info_array[0, :] = ["Property", "Value"]
        for k in range(len(arr)):
            info_array[k + 1, 0] = f"{names_arr[k]}"
            info_array[k + 1, 1] = f"{arr[k]:.3g}"
    else:
        info_array = np.zeros((len(arr) + 1, 6)).astype(str)
        info_array[0, :] = ["Var", "Min", "Mean", "Max", "No. NaN", "No. of Inf"]
        for k in range(len(arr)):
            info_array[k + 1, 0] = f"{names_arr[k]}"
            info_array[k + 1, 1] = f"{np.nanmin(np.asarray(arr[k])):.3g}"
            info_array[k + 1, 2] = f"{np.nanmean(np.asarray(arr[k])):.3g}"
            info_array[k + 1, 3] = f"{np.nanmax(np.asarray(arr[k])):.3g}"
            info_array[k + 1, 4] = f"{np.sum(np.isnan(np.asarray(arr[k])))}"
            info_array[k + 1, 5] = f"{np.sum(np.isinf(np.asarray(arr[k])))}"

    Table(root, info_array.shape[0], info_array.shape[1], info_array)

    root.update_idletasks()
    window_height = root.winfo_height()
    window_width = root.winfo_width()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    x_coordinate = int((screen_width / 2) - (window_width / 2))
    y_coordinate = int((screen_height / 2) - (window_height / 2))

    root.geometry(
        "{}x{}+{}+{}".format(window_width, window_height, x_coordinate, y_coordinate)
    )

    if sys.platform.startswith("win"):
        root.iconbitmap(os.path.join(os.path.dirname(__file__), "Equinor_logo.ico"))
    else:
        logo = PhotoImage(
            file=os.path.join(os.path.dirname(__file__), "Equinor_logo.gif")
        )
        root.call("wm", "iconphoto", root._w, logo)

    root.mainloop()
