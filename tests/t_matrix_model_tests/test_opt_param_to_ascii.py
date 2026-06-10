from pathlib import Path

from syrupy.assertion import SnapshotAssertion

from rock_physics_open.equinor_utilities.optimisation_utilities import (
    opt_param_to_ascii,
)


def test_opt_param_to_ascii_petec(
    snapshot: SnapshotAssertion,
    tmatrix_parameters_petec: Path,
    tmp_path: Path,
):
    output_ascii_file = tmp_path / "petec_opt_param.txt"
    opt_param_to_ascii(
        tmatrix_parameters_petec,
        display_results=False,
        out_file=output_ascii_file,
    )
    assert snapshot == output_ascii_file.read_text().strip()


def test_opt_param_to_ascii_exp(
    snapshot: SnapshotAssertion,
    tmatrix_parameters_exp: Path,
    tmp_path: Path,
):
    output_ascii_file = tmp_path / "exp_opt_param.txt"
    opt_param_to_ascii(
        tmatrix_parameters_exp,
        display_results=False,
        out_file=output_ascii_file,
    )
    assert snapshot == output_ascii_file.read_text().strip()
