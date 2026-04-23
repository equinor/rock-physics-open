import sys
from typing import cast

import numpy as np
import pytest
from syrupy.assertion import SnapshotAssertion
from syrupy.types import PropertyMatcher


def _format_numeric_snapshot_value(
    data: float | np.floating,
    significant_digits: int,
) -> str:
    """Format a single numeric value into scientific notation with desired precision."""
    if np.isnan(data):
        return "NaN"
    if data == 0:
        return "0"
    return f"{float(data):.{significant_digits - 1}e}"


def _snapshot_matcher(significant_digits: int) -> PropertyMatcher:
    """Build a recursive snapshot formatter using the requested numeric precision."""

    def _formatter(*, data: object, path: str) -> object:
        if isinstance(data, float | np.floating):
            return _format_numeric_snapshot_value(data, significant_digits)  # pyright: ignore[reportUnknownArgumentType]
        if isinstance(data, tuple):
            return tuple(_formatter(data=element, path=path) for element in data)  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
        if isinstance(data, list):
            return [_formatter(data=element, path=path) for element in data]  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
        if isinstance(data, dict):
            return {
                key: _formatter(data=value, path=path)  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
                for key, value in data.items()  # pyright: ignore[reportUnknownVariableType]
            }
        if isinstance(data, np.ndarray):
            return np.array2string(
                data,
                formatter={
                    "float_kind": lambda element: _format_numeric_snapshot_value(
                        element,
                        significant_digits,
                    )
                },
                threshold=sys.maxsize,  # Avoid summarization of large arrays
            )
        return data

    return cast(PropertyMatcher, _formatter)


@pytest.fixture
def snapshot(snapshot: SnapshotAssertion) -> SnapshotAssertion:
    """Compare with snapshot where numerical values are written with 9 significant digits."""
    return snapshot.with_defaults(matcher=_snapshot_matcher(significant_digits=9))
