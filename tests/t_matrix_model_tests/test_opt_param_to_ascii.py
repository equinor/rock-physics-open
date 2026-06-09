from pathlib import Path

from rock_physics_open.equinor_utilities.optimisation_utilities import (
    opt_param_to_ascii,
)


def test_opt_param_to_ascii_petec(testdata: Path, tmp_path: Path):
    tmatrix_parameters_petec = testdata / "petec_opt_param_test.pkl"
    opt_param_to_ascii(
        tmatrix_parameters_petec,
        display_results=False,
        out_file=tmp_path / "petec_opt_param.txt",
    )


def test_opt_param_to_ascii_exp(testdata: Path, tmp_path: Path):
    tmatrix_parameters_exp = testdata / "exp_opt_param_test.pkl"
    opt_param_to_ascii(
        tmatrix_parameters_exp,
        display_results=False,
        out_file=tmp_path / "exp_opt_param.txt",
    )
