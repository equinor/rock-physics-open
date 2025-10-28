from pathlib import Path

import pytest

from rock_physics_open.equinor_utilities.optimisation_utilities import (
    opt_param_to_ascii,
)


def test_opt_param_to_ascii_display_petec(
    monkeypatch: pytest.MonkeyPatch, data_dir: Path
):
    monkeypatch.chdir(data_dir)
    in_file = data_dir.joinpath("petec_opt_param_test.pkl")
    try:
        opt_param_to_ascii(in_file, display_results=False)
    except (ValueError, IOError):
        raise ValueError(f"Not possible to read input file {in_file}")
    out_file = data_dir.joinpath("petec_opt_param.txt")
    try:
        opt_param_to_ascii(in_file, display_results=False, out_file=out_file)
    except (ValueError, IOError):
        raise ValueError(f"Not possible to write output file {out_file}")


def test_opt_param_to_ascii_display_exp(
    monkeypatch: pytest.MonkeyPatch, data_dir: Path
):
    monkeypatch.chdir(data_dir)
    in_file = data_dir.joinpath("exp_opt_param_test.pkl")
    try:
        opt_param_to_ascii(in_file, display_results=False)
    except (ValueError, IOError):
        raise ValueError(f"Not possible to read input file {in_file}")
    out_file = data_dir.joinpath("exp_opt_param.txt")
    try:
        opt_param_to_ascii(in_file, display_results=False, out_file=out_file)
    except (ValueError, IOError):
        raise ValueError(f"Not possible to write output file {out_file}")
