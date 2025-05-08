def dict_value_to_float(input_dict):
    """
    Convert dictionary strings to floating point numbers. Each value can have multiple floats.

    Parameters
    ----------
    input_dict : dict
        Input dictionary.

    Returns
    -------
    dict
        Output dictionary.
    """

    for item in input_dict:
        if isinstance(input_dict[item], float):
            pass
        else:
            try:
                ff = float(input_dict[item])
                input_dict[item] = ff
            except ValueError:  # if a list or tuple is hidden in the string
                try:
                    ll = eval(input_dict[item])
                    ff = [float(i) for i in ll]
                    input_dict[item] = ff
                except ValueError:
                    raise ValueError(
                        "dict_value_to_float: not possible to convert value to float"
                    )

    return input_dict
