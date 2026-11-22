r"""Arrays: NumPy's, and any other NumPy can read, through ``__array__``, the array API or
DLPack, as JAX's, CuPy's and PyTorch's.

An array is data, each axis over a space of its own size, and a NumPy number is a scalar. A
JAX array stays one, so that JAX computes with it; any other becomes a NumPy array.
"""

from __future__ import annotations

from typing import Any

from relativisticpy.errors import DataError
from relativisticpy.integrations import Data, Example, Scalar, python_object


def _signal() -> dict[str, Any]:
    import numpy as np

    return {"signal": np.array([1.0, 2.0, 3.0, 6.0])}


def _portfolio() -> dict[str, Any]:
    import numpy as np

    return {
        "weights": np.array([0.25, 0.75]),
        "covariance": np.array([[0.04, 0.01], [0.01, 0.09]]),
    }


@python_object(
    "arrays",
    takes=["numpy.ndarray", "numpy.generic", "__array__", "__array_namespace__", "__dlpack__"],
    summary="an array, or anything NumPy reads: data, each axis over a space of its size; "
    "a NumPy number is a scalar",
    requires="numpy",
    examples=[
        Example(
            r"x_t := \texttt{signal}" "\n" r"\mathbb{E}_t[x_t]",
            3.0,
            "numpy.mean([1, 2, 3, 6])",
            given=_signal,
        ),
        Example(
            r"w^a := \texttt{weights}"
            "\n"
            r"\Sigma_{ab} := \texttt{covariance}"
            "\n"
            r"w^a \Sigma_{ab} w^b",
            0.25**2 * 0.04 + 2 * 0.25 * 0.75 * 0.01 + 0.75**2 * 0.09,
            "w @ covariance @ w, in NumPy",
            given=_portfolio,
        ),
    ],
)
class Arrays:
    def becomes(self, value: Any) -> Data | Scalar:
        from relativisticpy.tensors import numerical

        np = numerical.numpy()
        if numerical.is_jax(value) or isinstance(value, np.ndarray | np.generic):
            array = value
        else:
            array = _read(np, value)
        if getattr(array, "dtype", np.dtype(float)).kind in "USa":
            raise DataError("the object is text: read numbers")
        if getattr(array, "shape", ()) == ():
            return Scalar(array if isinstance(array, np.generic) else np.asarray(array)[()])
        return Data(array)


def _read(np: Any, value: Any) -> Any:
    """``value`` as a NumPy array, by ``__array__`` or DLPack."""
    try:
        return np.asarray(value)
    except Exception as first:
        try:
            return np.from_dlpack(value)
        except Exception:
            raise DataError(
                f"NumPy cannot read the object: {type(first).__name__}: {first}",
                hint="copy it to a NumPy array in Python first, as numpy.asarray(x) does on a "
                "computer's own memory",
            ) from first
