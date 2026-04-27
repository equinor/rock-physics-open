from pathlib import Path

from rock_physics_open.equinor_utilities.optimisation_utilities import (
    opt_param_to_ascii,
)


def test_opt_param_to_ascii_display_petec(testdata: Path):
    in_file = testdata / "petec_opt_param_test.pkl"
    try:
        opt_param_to_ascii(in_file, display_results=False)
    except (ValueError, IOError):
        raise ValueError(f"Not possible to read input file {in_file}")
    out_file = testdata / "petec_opt_param.txt"
    try:
        opt_param_to_ascii(in_file, display_results=False, out_file=out_file)
    except (ValueError, IOError):
        raise ValueError(f"Not possible to write output file {out_file}")


def test_opt_param_to_ascii_display_exp(testdata: Path):
    in_file = testdata / "exp_opt_param_test.pkl"
    try:
        opt_param_to_ascii(in_file, display_results=False)
    except (ValueError, IOError):
        raise ValueError(f"Not possible to read input file {in_file}")
    out_file = testdata / "exp_opt_param.txt"
    try:
        opt_param_to_ascii(in_file, display_results=False, out_file=out_file)
    except (ValueError, IOError):
        raise ValueError(f"Not possible to write output file {out_file}")
