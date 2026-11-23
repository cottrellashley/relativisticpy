r"""Activation functions of neural networks.

Those that have one meaning wherever they are written, as books of machine learning write
them: the rectified linear unit and the softplus, `\operatorname{ReLU}(x)` and
`\operatorname{softplus}(x)` (design/notation.md, decision 26). Exactly with SymPy and,
for numbers, with NumPy or JAX. σ is not one of them: books also write it for a standard
deviation, and a program defines it, as `\sigma(a) := \frac{1}{1 + e^{-a}}`.
"""

from __future__ import annotations

from typing import Any

import sympy

from relativisticpy.errors import EvaluationError
from relativisticpy.integrations import Example, function


@function(
    "ReLU",
    latex=r"\operatorname{ReLU}(x)",
    summary="the rectified linear unit, max(0, x), of a real x",
    examples=[
        Example(r"\operatorname{ReLU}(-2) + \operatorname{ReLU}(3)", 3,
                "Goodfellow, Bengio and Courville (2016), section 6.3.1"),
        Example(r"\operatorname{ReLU}(-1.5)", 0, "numpy.maximum(0, -1.5)"),
    ],
    points=[(-0.7,), (0.3,), (1.6,)],
)  # fmt: skip
class RectifiedLinearUnit:
    def exact(self, x: Any) -> Any:
        if sympy.sympify(x).is_extended_real is False:
            raise _not_real()
        return sympy.Max(0, x)

    def numeric(self, xp: Any, x: Any) -> Any:
        if xp.iscomplexobj(x) and xp.any(xp.imag(x) != 0):
            raise _not_real()
        return xp.maximum(0, xp.real(x) if xp.iscomplexobj(x) else x)


@function(
    "softplus",
    latex=r"\operatorname{softplus}(x)",
    summary="the softplus, log(1 + e^x), a smooth ReLU",
    examples=[
        Example(r"\operatorname{softplus}(0)", sympy.log(2), "numpy.logaddexp(0, 0)"),
        Example(r"\operatorname{softplus}(1)", sympy.log(1 + sympy.E),
                "Goodfellow, Bengio and Courville (2016), section 3.10"),
    ],
    points=[(-0.7,), (0.3,), (1.6,)],
)  # fmt: skip
class Softplus:
    def exact(self, x: Any) -> Any:
        return sympy.log(1 + sympy.exp(x))

    def numeric(self, xp: Any, x: Any) -> Any:
        if xp.iscomplexobj(x):
            return xp.log1p(xp.exp(x))
        return xp.logaddexp(0, x)  # without overflow where x is large


def _not_real() -> EvaluationError:
    return EvaluationError(
        "ReLU is a function of a real number, and this is not one",
        hint="ReLU(x) is max(0, x), which compares x with 0, as only a real x can be",
    )
