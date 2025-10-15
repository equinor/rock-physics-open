import inspect
import math
import re
from warnings import warn

import numpy as np
from pandas import DataFrame

from rock_physics_open.equinor_utilities.various_utilities import disp_result_stats

from .snapshots import get_snapshot_name


def compare_snapshots(
    test_results: np.ndarray | tuple | DataFrame,
    saved_results: tuple,
    name_arr=None,
    display_results: bool = False,
) -> bool:
    test_results = _validate_input(test_results, saved_results)

    if display_results:
        title = str(inspect.stack()[1].function)
        if not name_arr:
            name_arr = [f"arr_{i}" for i in range(len(test_results))]
        disp_result_stats(title, test_results, name_arr)

    r_tol = 0.01
    equal_nan = True
    no_difference_found = True

    for i, (test_item, saved_item) in enumerate(zip(test_results, saved_results)):
        try:
            if name_arr:
                err_msg = f"saved and generated result for {name_arr[i]} differ"
            else:
                err_msg = f"saved result for variable {i} differ"
            np.testing.assert_allclose(
                test_item,
                saved_item,
                rtol=r_tol,
                equal_nan=equal_nan,
                err_msg=err_msg,
            )
        except AssertionError as error:
            open_mode = "w" if no_difference_found else "a"
            no_difference_found = False
            warn(f"comparison test failed for item {i}: {error}")
            log_file = re.sub("npz", "log", get_snapshot_name(step=2))

            with open(log_file, open_mode) as file:
                file.write(
                    f"Test function: "
                    f"{get_snapshot_name(include_extension=False, include_snapshot_dir=False, include_filename=False)} \n"
                )
                if name_arr:
                    file.write(f"Test variable: {name_arr[i]} \n")
                else:
                    file.write(f"Test variable number: {i} \n")

                for line in str(error).splitlines():
                    mismatched_elements_index = (
                        line.replace(" ", "").lower().find("mismatchedelements")
                    )
                    if mismatched_elements_index != -1:
                        file.write(line + "\n")
                        continue

                    max_abs_diff_index = (
                        line.replace(" ", "").lower().find("maxabsolutedifference")
                    )
                    if max_abs_diff_index != -1:
                        file.write(line + "\n")
                        continue

                    max_rel_diff_index = (
                        line.replace(" ", "").lower().find("maxrelativedifference")
                    )
                    if max_rel_diff_index != -1:
                        file.write(line + "\n")
                        continue

                    reg_index = re.search(r"differ", line)

                    if reg_index:
                        if isinstance(test_item, np.ndarray):
                            differences, num_nans = _compare_ndarray(
                                saved_item, test_item, equal_nan, r_tol
                            )
                        elif isinstance(test_results, DataFrame):
                            differences, num_nans = _compare_df(
                                saved_item, test_item, equal_nan, r_tol
                            )
                        file.write("Number of NaN elements: " + str(num_nans) + "\n")
                        file.write("Index:\t\tSaved:\t\tGenerated:\n")

                        # Write test results and saved results differences to file
                        if len(differences) > 0:
                            tab = "\t"
                            for difference in differences:
                                file.write(
                                    f"{tab}[{difference[0]:4}]=> {difference[1]:.8g} != {difference[2]:.8g}\n"
                                )
                            file.write(f"{'_' * 40}\n")
    return no_difference_found


def _compare_ndarray(
    saved_array: np.ndarray, test_array: np.ndarray, eq_nan: bool, rel_tol: float
) -> (list, int):
    differ_indexes = np.where(saved_array != test_array)[0]
    differences = []
    num_nans = 0

    for index in differ_indexes:
        if eq_nan and (
            np.isnan(test_array[int(index)]) and np.isnan(saved_array[int(index)])
        ):
            num_nans += 1
        elif math.isclose(
            saved_array[int(index)], test_array[int(index)], rel_tol=rel_tol
        ):
            pass
        else:
            differences.append([index, saved_array[index], test_array[index]])
    return differences, num_nans


def _compare_df(saved_results, test_results, equal_nan, r_tol):
    return _compare_ndarray(
        saved_results.to_numpy().flatten(),
        test_results.to_numpy().flatten(),
        equal_nan,
        r_tol,
    )


def _validate_input(test_obj, saved_obj: tuple) -> tuple:
    # Check for compatibility of test results and stored data
    if isinstance(test_obj, (np.ndarray, DataFrame)):
        return_test_obj = (test_obj,)
    else:
        return_test_obj = test_obj
    if isinstance(return_test_obj, (tuple, list)):
        if len(saved_obj) != len(return_test_obj):
            raise ValueError(
                f"unable to compare snapshots, different number of saved: ({len(saved_obj)})"
                f"  and generated results ({len(test_obj)})"
            )
    else:
        raise ValueError(
            f"test_obj should be one of list, tuple, numpy array or pandas DataFrame, is {type(test_obj)}"
        )
    return return_test_obj
