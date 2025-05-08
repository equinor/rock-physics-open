import numpy as np
from sklearn.preprocessing import OneHotEncoder


def generate_dummy_vars(inp_frame, class_var, ohe=None):
    """
    From categorical variables generate a one-hot-encoder, i.e. each value in the categorical variable becomes a binary
    variable. See sklearn.preprocessing.OneHotEncoder.

    Parameters
    ----------
    inp_frame : pd.DataFrame
        Input data containing categorical variables.
    class_var : str
        Name of categorical variable.
    ohe : preprocessing.OneHotEncoder
        One-hot-encoder object.

    Returns
    -------
    dum_features, no_dummy_cols, dum_var_names : (np.ndarray, int, np.ndarray)
        dum_features: 2D array with transformed dummy variables, no_dummy_cols: number of columns in returned array,
        dum_var_names: automatically generated feature names.
    """
    from pandas.api.types import is_numeric_dtype

    if is_numeric_dtype(inp_frame[class_var]):
        # Make sure that the chosen indicator variable contains discrete values
        inp_frame = inp_frame.astype({class_var: "int32"})

    features_in = np.array(inp_frame[class_var]).reshape(-1, 1)

    if ohe is None:
        classes = features_in
        ohe = OneHotEncoder(categories="auto", sparse_output=False)
        ohe.fit(classes)

    dum_features = ohe.transform(features_in)
    no_dummy_cols = dum_features.shape[1]
    dum_var_names = ohe.get_feature_names_out()

    return dum_features, no_dummy_cols, dum_var_names
