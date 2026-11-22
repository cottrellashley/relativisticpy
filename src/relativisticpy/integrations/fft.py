r"""The discrete Fourier transform.

The transform of samples, :math:`X_k = \sum_{n=0}^{N-1} x_n e^{-2\pi i k n/N}`, and its
inverse, by their names, ``\operatorname{fft}(x_t)`` and ``\operatorname{ifft}(X_k)``:
numeric with the FFT of NumPy, or of JAX, for data, whose result is over the frequencies of
the index it transforms; exact with SymPy for numbers a program writes. The transform
written as its sum, on data, is computed as an FFT too, by a rewrite. Cooley and Tukey
(1965); Bracewell (2000), chapter 9.
"""

from __future__ import annotations

import cmath
from collections.abc import Callable, Sequence
from typing import Any

import sympy

from relativisticpy import display
from relativisticpy.errors import EvaluationError
from relativisticpy.integrations import Example, rewrite, transform
from relativisticpy.integrations import _transforms as transforms
from relativisticpy.tensors.spaces import Space

#: The space of samples each space of frequencies was made from, so that the inverse
#: transform gives back data over it.
_SAMPLES: dict[Space, Space] = {}


def _sequence(values: Any, transform: Callable[[list[Any]], list[Any]]) -> Any:
    """A transform of a sequence: of a list, a list; of a vector, a vector."""
    if isinstance(values, sympy.NDimArray):
        if values.rank() != 1:
            raise EvaluationError("the discrete Fourier transform is of a sequence, one index")
        return sympy.ImmutableDenseNDimArray(transform(list(values)))
    if not isinstance(values, list | tuple):
        raise EvaluationError(
            f"the discrete Fourier transform is of a sequence, not of {display.text(values)}",
            hint="transform a list, as in fft([1, 2, 0, -1]), or data over an index",
        )
    return transform(list(values))


def _dft(values: list[Any], sign: int = -1) -> list[Any]:
    """The transform of a sequence, exactly, with e^{-2πikn/N}, of any length: SymPy's own
    pads a sequence to a power of two, and has the other sign."""
    size = len(values)
    root = sympy.exp(sign * 2 * sympy.pi * sympy.I / size)
    return [
        sympy.expand_complex(sum(values[n] * root ** (k * n) for n in range(size)))
        for k in range(size)
    ]


def _inverse_dft(values: list[Any]) -> list[Any]:
    """The inverse transform, exactly, with e^{2πikn/N} and 1/N."""
    size = len(values)
    return [sympy.expand_complex(value / size) for value in _dft(values, 1)]


@transform(
    "fft",
    latex=r"\operatorname{fft}(x)",
    summary=r"the discrete Fourier transform; `X_k = \sum_n x_n e^{-2\pi i k n/N}`, by the fast "
    "algorithm: of data over an index, data over its frequencies",
    inverse="ifft",
    examples=[
        Example("fft([1, 2, 0, -1])", [2, 1 - 3 * sympy.I, 0, 1 + 3 * sympy.I],
                "numpy.fft.fft([1, 2, 0, -1])"),
        Example("N := 8\n"
                r"x_n := \cos\left(\frac{2\pi n}{N}\right), \quad n = 0, \ldots, N-1" "\n"
                r"\operatorname{fft}(x_n)",
                [0, 4, 0, 0, 0, 0, 0, 4],
                "numpy.fft.fft(numpy.cos(2 * numpy.pi * numpy.arange(8) / 8))"),
    ],
)  # fmt: skip
class FastFourierTransform:
    def exact(self, x: Any) -> Any:
        return _sequence(x, _dft)

    def numeric(self, xp: Any, x: Any) -> Any:
        return xp.fft.fft(x, axis=0)

    def space(self, over: Space) -> Space:
        """The frequencies of the samples a space holds, in cycles per step of its index, in
        the order of the transform's values, as ``numpy.fft.fftfreq`` gives them."""
        step = transforms.spacing(over.basis)
        basis = tuple(transforms.label(f) for f in self.labels(over.dim, step))
        space = Space(f"the frequencies of {over.name}", over.dim, basis, "complex")
        _SAMPLES[space] = over
        return space

    def labels(self, size: int, step: Any) -> list[Any]:
        return transforms.frequencies(size, step)

    def sum(self, x: Sequence[Any], k: int) -> complex:
        size = len(x)
        return complex(sum(x[n] * cmath.exp(-2j * cmath.pi * k * n / size) for n in range(size)))

    def energy(self, size: int) -> int:
        """Parseval's relation: the energy of the transform is N times the samples'."""
        return size


@transform(
    "ifft",
    latex=r"\operatorname{ifft}(X)",
    summary=r"the inverse discrete Fourier transform; `x_n = \frac{1}{N} \sum_k X_k "
    r"e^{2\pi i k n/N}`: of data over frequencies, data over the samples they are of",
    inverse="fft",
    examples=[
        Example("ifft([2, 1 - 3i, 0, 1 + 3i])", [1, 2, 0, -1],
                "numpy.fft.ifft([2, 1 - 3j, 0, 1 + 3j])"),
        Example("N := 8\n"
                r"x_n := n^2, \quad n = 0, \ldots, N-1" "\n"
                r"\operatorname{ifft}(\operatorname{fft}(x_n))",
                [0, 1, 4, 9, 16, 25, 36, 49],
                "numpy.fft.ifft(numpy.fft.fft(numpy.arange(8) ** 2))"),
    ],
)  # fmt: skip
class InverseFastFourierTransform:
    def exact(self, X: Any) -> Any:
        return _sequence(X, _inverse_dft)

    def numeric(self, xp: Any, X: Any) -> Any:
        return xp.fft.ifft(X, axis=0)

    def space(self, over: Space) -> Space:
        """The space of the samples the frequencies are of: the one they were made from,
        or else samples numbered from 0."""
        made = _SAMPLES.get(over)
        if made is not None:
            return made
        basis = tuple(transforms.label(v) for v in self.labels(over.dim, 1))
        return Space(f"the samples of {over.name}", over.dim, basis)

    def labels(self, size: int, step: Any) -> list[Any]:
        return list(range(size))

    def sum(self, X: Sequence[Any], n: int) -> complex:
        size = len(X)
        total = sum(X[k] * cmath.exp(2j * cmath.pi * k * n / size) for k in range(size))
        return complex(total) / size

    def energy(self, size: int) -> Any:
        """Parseval's relation: the energy of the inverse is that of the frequencies over N."""
        return sympy.Rational(1, size)


@rewrite(
    "dft_sum",
    computes="fft",
    latex=r"\sum_{n=0}^{N-1} x_n e^{-2\pi i k n/N}",
    summary="the discrete Fourier transform written as its sum; on data or a list, computed as "
    "an FFT",
    examples=[
        Example("N := 8\n"
                r"x_n := \cos\left(\frac{2\pi n}{N}\right) + \frac{n}{N}, \quad "
                r"n = 0, \ldots, N-1" "\n"
                r"X_k := \sum_{n=0}^{N-1} x_n e^{-2\pi i k n/N}, \quad k = 0, \ldots, N-1" "\n"
                "X_k",
                [3.5, 3.5 + 1.207106781186547j, -0.5 + 0.5j, -0.5 + 0.20710678118654763j, -0.5,
                 -0.5 - 0.20710678118654763j, -0.5 - 0.5j, 3.5 - 1.207106781186547j],
                "numpy.fft.fft(numpy.cos(2 * numpy.pi * n / 8) + n / 8), n = numpy.arange(8)",
                tolerance=1e-9),
        Example("x := (1, 2, 0, -1, 3, 0, 0, 2)\nN := 8\n"
                r"X_k := \sum_{n=0}^{N-1} x_n e^{-2\pi i k n/N}, \quad k = 0, \ldots, N-1" "\n"
                "X_k",
                [7, 1.5355339059327378 + 0.7071067811865476j, 4 - 1j,
                 -5.535533905932738 + 0.7071067811865476j, 1,
                 -5.535533905932738 - 0.7071067811865476j, 4 + 1j,
                 1.5355339059327378 - 0.7071067811865476j],
                "numpy.fft.fft([1, 2, 0, -1, 3, 0, 0, 2])", tolerance=1e-9),
    ],
)  # fmt: skip
class FourierSum:
    def matches(self, kernel: Any, n: Any, k: Any, size: int) -> Any:
        """The factor by which the sum is the FFT, where its kernel, in the positions n and
        k counted from 0, is a constant times e^{-2πikn/N}; ``None`` where it is not."""
        ratio = sympy.simplify(kernel / sympy.exp(-2 * sympy.pi * sympy.I * k * n / size))
        if ratio.has(n) or ratio.has(k):
            return None
        return ratio
