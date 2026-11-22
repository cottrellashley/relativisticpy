r"""Labelled arrays: xarray's DataArrays.

A DataArray is data over its named dimensions: each dimension is the space of its axis,
named after it, and a dimension with coordinates is labelled by them, as the axes of a pandas
table are. Data from two DataArrays meets by label, never by position.
"""

from __future__ import annotations

import importlib
from typing import Any

from relativisticpy.integrations import Data, Example, python_object


def _temperatures() -> dict[str, Any]:
    xarray = importlib.import_module("xarray")  # where it is installed: it is no extra's
    readings = [[10.0, 12.0], [14.0, 20.0]]
    coordinates = {"day": ["mon", "tue"], "city": ["Oslo", "Rome"]}
    return {"temperature": xarray.DataArray(readings, coordinates, ("day", "city"))}


@python_object(
    "xarray",
    takes=["xarray.DataArray"],
    summary="a DataArray: data over its named dimensions, labelled by their coordinates",
    requires="xarray",
    examples=[
        Example(
            r"T_{dc} := \texttt{temperature}" "\n" r"\mathbb{E}_d[T_{dc}]",
            [12.0, 16.0],
            "temperature.mean('day'), in xarray",
            given=_temperatures,
        ),
        Example(
            r"T_{dc} := \texttt{temperature}" "\n" r"T_{d:tue\, c:Rome}",
            20.0,
            "temperature.sel(day='tue', city='Rome'), in xarray",
            given=_temperatures,
        ),
    ],
)
class DataArrays:
    def becomes(self, value: Any) -> Data:
        bases = []
        for dimension in value.dims:
            index = value.indexes.get(dimension)
            bases.append(None if _numbered(index) else tuple(str(x) for x in index))
        names = tuple(str(dimension) for dimension in value.dims)
        return Data(value.to_numpy(), tuple(bases), names)


def _numbered(index: Any) -> bool:
    """Whether a dimension has no coordinates, or the default numbering, 0, 1, 2, ..."""
    if index is None:
        return True
    return type(index).__name__ == "RangeIndex" and index.start == 0 and index.step == 1
