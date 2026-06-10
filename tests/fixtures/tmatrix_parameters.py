from pathlib import Path

import numpy as np
import pytest

from rock_physics_open.equinor_utilities.optimisation_utilities import (
    save_opt_params,
)


@pytest.fixture
def tmatrix_parameters_petec(tmp_path: Path) -> Path:
    """Save a PETEC (mineral-input) T-matrix optimization result and return its path."""
    path = tmp_path / "petec_opt_param.pkl"
    save_opt_params(
        opt_type="min",
        opt_params=np.array(
            [
                4.180014961850993e-12,
                0.9970481705955884,
                0.5005453826944681,
                0.05497137749304187,
                0.5000914817434468,
            ]
        ),
        file_name=str(path),
        well_name="unknown",
    )
    return path


@pytest.fixture
def tmatrix_parameters_exp(tmp_path: Path) -> Path:
    """Save an exploration-type T-matrix optimization result and return its path."""
    path = tmp_path / "exp_opt_param.pkl"
    save_opt_params(
        opt_type="exp",
        opt_params=np.array(
            [
                1.8160911526041597e-16,
                0.4198862974591682,
                0.6238244394081418,
                0.053580417501345845,
                0.6154599371967261,
                0.7449307161152314,
                0.6839728969027045,
                0.8983050847766154,
                0.30086923518283865,
                0.9999695439076245,
                0.9102511987120231,
            ]
        ),
        file_name=str(path),
        well_name="unknown",
    )
    return path
