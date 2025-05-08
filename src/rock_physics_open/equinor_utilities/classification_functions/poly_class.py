import matplotlib.path as mplpath
import numpy as np


def poly_class(train_data, polygons, labels):
    """
    Points within the polygons are assigned to class labels. Point that do not
    fall within any polygon are set to NULL_CLASS.

    Parameters
    ----------

    train_data : np.ndarray
        Data points of two variables.
    polygons : np.ndarray
        Vertices of polygons in two-dimensional space.
    labels : np.ndarray
        Class label for each polygon.

    Returns
    -------
    np.ndarray
        Class id.
    """
    if len(labels) != len(polygons):
        raise ValueError("Number of labels are not matching number of polygons")

    # Create output variables
    idx_filtered = np.zeros(train_data.shape[0]).astype("bool")
    poly_class_id = np.zeros(train_data.shape[0]).astype("int")

    for i in range(len(polygons)):
        class_polygon = polygons[i]
        path = mplpath.Path(class_polygon)
        # Only points within the given polygon are used
        idx_poly = path.contains_points(train_data)
        poly_class_id[idx_poly] = labels[i]
        idx_filtered = np.logical_or(idx_filtered, idx_poly)
    # idx_filtered is no longer returned
    return poly_class_id
