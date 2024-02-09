import pytest
import sympy as sp
from relativisticpy.workbook.workbook import Workbook

# Do not import from sympy - only import from the SAME interface which Workbook points to implements everything.
from relativisticpy.symengine import (
    Symbol,
    Function,
    sin,
    tan,
    cos,
    integrate,
    solve,
    diff,
    expand,
    limit,
    simplify,
    dsolve,
    SymbolArray,
)

# Workbook.expr() => is used for one line solver function - like a "unit test" for workbook module.



@pytest.fixture
def Schwarzschild_Basis():
    x, y, z, t, r, theta, phi, G, M, c = sp.symbols("x y z t r theta phi G M c")

    basis = sp.MutableDenseNDimArray(
        [
            t,
            r,
            theta,
            phi,
        ]
    )

    return basis

@pytest.fixture
def Schwarzschild_Metric():
    x, y, z, t, r, theta, phi, G, M, c = sp.symbols("x y z t r theta phi G M c")

    metric = sp.MutableDenseNDimArray(
        [
            [-(1 - (2 * G * M) / (r)), 0, 0, 0],
            [0, 1 / (1 - (2 * G * M) / (r)), 0, 0],
            [0, 0, r**2, 0],
            [0, 0, 0, r**2 * sin(theta) ** 2],
        ]
    )

    inverse_metric = sp.MutableDenseNDimArray(
        [
            [-(1 / (1 - (2 * G * M) / (r))), 0, 0, 0],
            [0, (1 - (2 * G * M) / (r)), 0, 0],
            [0, 0, 1 / r**2, 0],
            [0, 0, 0, 1 / (r**2 * sin(theta) ** 2)],
        ]
    )

    return metric, inverse_metric


@pytest.fixture
def Schwarzschild_Connection():
    x, y, z, t, r, theta, phi, G, M, c = sp.symbols("x y z t r theta phi G M c")

    return sp.MutableDenseNDimArray([
        [
            [0, G * M / (r * (-2 * G * M + r)), 0, 0],
            [G * M / (r * (-2 * G * M + r)), 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        [
            [G * M * (-2 * G * M + r) / r**3, 0, 0, 0],
            [0, G * M / (r * (2 * G * M - r)), 0, 0],
            [0, 0, 2 * G * M - r, 0],
            [0, 0, 0, (4 * G * M - 2 * r) * sin(theta) ** 2 / 2],
        ],
        [
            [0, 0, 0, 0],
            [0, 0, 1 / r, 0],
            [0, 1 / r, 0, 0],
            [0, 0, 0, -sin(2 * theta) / 2],
        ],
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1 / r],
            [0, 0, 0, 1 / tan(theta)],
            [0, 1 / r, 1 / tan(theta), 0],
        ],
    ])


@pytest.fixture
def Schwarzschild_Ricci(): return sp.MutableDenseNDimArray([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

@pytest.fixture
def Schwarzschild_MetricScalar(): return sp.MutableDenseNDimArray(4)

@pytest.fixture
def Schwarzschild_Riemann():
    x, y, z, t, r, theta, phi, G, M, c = sp.symbols("x y z t r theta phi G M c")
    riemann = sp.MutableDenseNDimArray([
        [
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [
                [0, 2 * G * M / (r**2 * (-2 * G * M + r)), 0, 0],
                [2 * G * M / (r**2 * (2 * G * M - r)), 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            [[0, 0, -G * M / r, 0], [0, 0, 0, 0], [G * M / r, 0, 0, 0], [0, 0, 0, 0]],
            [
                [0, 0, 0, -G * M * sin(theta) ** 2 / r],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [G * M * sin(theta) ** 2 / r, 0, 0, 0],
            ],
        ],
        [
            [
                [0, 2 * G * M * (-2 * G * M + r) / r**4, 0, 0],
                [2 * G * M * (2 * G * M - r) / r**4, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 0, -G * M / r, 0], [0, G * M / r, 0, 0], [0, 0, 0, 0]],
            [
                [0, 0, 0, 0],
                [0, 0, 0, -G * M * sin(theta) ** 2 / r],
                [0, 0, 0, 0],
                [0, G * M * sin(theta) ** 2 / r, 0, 0],
            ],
        ],
        [
            [
                [0, 0, G * M * (2 * G * M - r) / r**4, 0],
                [0, 0, 0, 0],
                [G * M * (-2 * G * M + r) / r**4, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0],
                [0, 0, G * M / (r**2 * (-2 * G * M + r)), 0],
                [0, G * M / (r**2 * (2 * G * M - r)), 0, 0],
                [0, 0, 0, 0],
            ],
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 2 * G * M * sin(theta) ** 2 / r],
                [0, 0, -2 * G * M * sin(theta) ** 2 / r, 0],
            ],
        ],
        [
            [
                [0, 0, 0, G * M * (2 * G * M - r) / r**4],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [G * M * (-2 * G * M + r) / r**4, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0],
                [0, 0, 0, G * M / (r**2 * (-2 * G * M + r))],
                [0, 0, 0, 0],
                [0, G * M / (r**2 * (2 * G * M - r)), 0, 0],
            ],
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, -2 * G * M / r],
                [0, 0, 2 * G * M / r, 0],
            ],
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        ],
    ])
    return riemann
