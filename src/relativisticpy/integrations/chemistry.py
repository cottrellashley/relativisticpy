r"""The notation of chemistry.

The p of chemists is an operator, the cologarithm :math:`\mathrm{p}X = -\log_{10} X`,
written upright against what it takes, as in pH and :math:`\mathrm{p}K_\mathrm{a}`
(IUPAC Gold Book, "p"). An italic p stays a letter: a pressure, a momentum or a
probability.
"""

from __future__ import annotations

from typing import Any

import sympy

from relativisticpy.integrations import Example, function


@function(
    "colog",
    latex=[r"\mathrm{p}X", r"\mathrm{p}(X)"],
    summary=r"the p of chemists, the cologarithm `-\log_{10} X`, as in `\mathrm{p}K_\mathrm{a}`",
    examples=[
        Example(r"\mathrm{p}(10^{-7})", 7, "IUPAC Gold Book, pH: neutral water at 25 °C"),
        Example(
            "K_\\mathrm{a} := 1.8 \\times 10^{-5}\n\\mathrm{p}K_\\mathrm{a}",
            4.7447274948966935,
            "-math.log10(1.8e-5)",
        ),
        Example("colog(1000)", -3, "-math.log10(1000)"),
    ],
    points=[(1.8e-5,), (0.25,), (3.0,)],
)
class Cologarithm:
    def exact(self, X: Any) -> Any:
        return -sympy.log(X, 10)

    def numeric(self, xp: Any, X: Any) -> Any:
        return -xp.log10(X)
