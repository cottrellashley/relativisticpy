r"""Tables: pandas's DataFrames and Series.

A DataFrame or a Series is labelled data. Its index labels the rows, the basis of the space
of their index, and the columns of a DataFrame the basis of another, each space named after
its axis where the axis has a name: the rules by which ``wb["P_{ta}"] = frame`` binds a
table. Data from two tables meets by label, never by position.
"""

from __future__ import annotations

from typing import Any

from relativisticpy.integrations import Data, Example, python_object


def _prices() -> dict[str, Any]:
    import pandas as pd

    dates = pd.Index(["2026-11-16", "2026-11-17", "2026-11-18"], name="date")
    prices = {"AAPL": [230.0, 232.0, 234.0], "NVDA": [140.0, 146.0, 143.0]}
    return {"prices": pd.DataFrame(prices, index=dates)}


def _aligned() -> dict[str, Any]:
    import pandas as pd

    weights = pd.Series([0.2, 0.8], index=pd.Index(["NVDA", "AAPL"], name="asset"))
    return {**_prices(), "weights": weights}


@python_object(
    "pandas",
    takes=["pandas.DataFrame", "pandas.Series"],
    summary="a DataFrame or Series: labelled data, its index and columns the bases of spaces "
    "named after its axes",
    requires="pandas",
    examples=[
        Example(
            r"P_{ta} := \texttt{prices}" "\n" r"\mathbb{E}_t[P_{ta}]",
            [232.0, 143.0],
            "prices.mean(), in pandas",
            given=_prices,
        ),
        Example(
            r"P_{ta} := \texttt{prices}" "\n" r"w^a := \texttt{weights}" "\n" r"P_{ta} w^a",
            [0.8 * 230.0 + 0.2 * 140.0, 0.8 * 232.0 + 0.2 * 146.0, 0.8 * 234.0 + 0.2 * 143.0],
            "prices @ weights.reindex(prices.columns): each asset weighed by its own name",
            given=_aligned,
        ),
    ],
)
class Tables:
    def becomes(self, value: Any) -> Data:
        axes = [value.index] if value.ndim == 1 else [value.index, value.columns]
        bases = tuple(None if _numbered(axis) else tuple(str(x) for x in axis) for axis in axes)
        names = tuple(str(axis.name) if axis.name is not None else None for axis in axes)
        return Data(value.to_numpy(), bases, names)


def _numbered(axis: Any) -> bool:
    """Whether a pandas axis is the default numbering, 0, 1, 2, ..., rather than labels."""
    return type(axis).__name__ == "RangeIndex" and axis.start == 0 and axis.step == 1
