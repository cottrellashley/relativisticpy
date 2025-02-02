import pytest
import sympy as smp
from relativisticpy.workbook.workbook import Workbook
from relativisticpy.diffgeom import (
    Ricci,
    Metric,
    RicciScalar,
    Riemann,
    KScalar,
    MetricScalar
)


def test_basic_tensor_component_generations_zeros():
    wb = Workbook()
    zeros = smp.MutableDenseNDimArray().zeros(2)

    res = wb.expr(
        """
                    Coordinates := [x, y]
                    T_{a} := [0, 0]
                    T_{a}
            """
    )
    assert smp.simplify(res.components) == zeros
    assert str(res.indices) == "_{a}"
    assert str(res.indices.basis) == '[x, y]'


def test_basic_tensor_component_generations_symbols():
    wb = Workbook()
    x, y, z, t, r, theta, phi, G, M, c = smp.symbols("x y z t r theta phi G M c")

    res = wb.expr(
        """
                    Coordinates := [t, r, theta]
                    T_{a}_{b} := [[1, 0, 0], [0, r**2, 0], [0, 0, r**2*sin(theta)**2]]
                    T_{a}_{b}
            """
    )
    assert smp.simplify(res.components) == smp.MutableDenseNDimArray(
        [[1, 0, 0], [0, r ** 2, 0], [0, 0, r ** 2 * smp.sin(theta) ** 2]])
    assert str(res.indices) == "_{a}_{b}"
    assert str(res.indices.basis) == '[t, r, theta]'


def test_covariant_derivative_metric_mapping(
):
    # This should do the following:
    # 1. D_{a}*g_{b}_{c} == Zero
    wb = Workbook()
    zeros = smp.MutableDenseNDimArray().zeros(4, 4, 4)

    res = wb.expr(
        """
                    Coordinates := [t, r, theta, phi] 
                    g_{mu}_{nu} := [[-(1 - (2 * G * M) / (r)), 0, 0, 0],[0, 1 / (1 - (2 * G * M) / (r)), 0, 0],[0, 0, r**2, 0],[0, 0, 0, r**2 * sin(theta) ** 2]]
                    T_{b}_{a}_{c} := D_{a}*g_{b}_{c}
                    T_{b}_{a}_{c}
            """
    )
    assert smp.simplify(res.components) == zeros
    assert str(res.indices) == "_{b}_{a}_{c}"
    assert str(res.indices.basis) == '[t, r, theta, phi]'


def test_tensor_multiplication_with_scalar(
):
    # This should do the following:
    # 1. D_{a}*g_{b}_{c} == Zero
    wb = Workbook()

    res = wb.expr(
        """
                        Coordinates := [t, r, theta, phi] 
                        g_{mu}_{nu} := [[-(1 - (2 * G * M) / (r)), 0, 0, 0],[0, 1 / (1 - (2 * G * M) / (r)), 0, 0],[0, 0, r**2, 0],[0, 0, 0, r**2 * sin(theta) ** 2]]
                        g_{a}_{b}*f(x)
            """
    )
    assert str(res.components) == '[[(2*G*M/r - 1)*f(x), 0, 0, 0], [0, f(x)/(-2*G*M/r + 1), 0, 0], [0, 0, r**2*f(x), 0], [0, 0, 0, r**2*f(x)*sin(theta)**2]]'
    assert str(res.indices) == "_{a}_{b}"
    assert str(res.indices.basis) == '[t, r, theta, phi]'


def test_vector_index_raise_lowering():
    # This should do the following:
    # 1. D_{a}*g_{b}_{c} == Zero
    wb = Workbook()

    res = wb.expr(
        """
                Coordinates := [t, r, theta, phi]
                g_{mu nu} := [[-(1 - (2 * G * M) / (c**2*r)), 0, 0, 0],[0, 1 / (1 - (2 * G * M) / (c**2*r)), 0, 0],[0, 0, r**2, 0],[0, 0, 0, r**2 * sin(theta) ** 2]]

                V_{a} := [1 + t , h(r, t), h(theta), h(r, t)]
                V_{a} - V^{b} * g_{a b}
            """
    )
    assert str(res.components) == '[0, 0, 0, 0]'
    assert str(res.indices) == "_{a}"
    assert str(res.indices.basis) == '[t, r, theta, phi]'
