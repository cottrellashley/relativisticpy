import pytest
import sympy as smp
from relativisticpy.workbook.workbook import Workbook
from relativisticpy.core import EinsteinArray, Metric
from relativisticpy.gr import (
    EinsteinTensor,
    Ricci,
    RicciScalar,
    Riemann,
    KScalar,
    MetricScalar,
    Connection
)

# TODO: <<< PUT THIS FUNCTION SOMEWHERE ELSE + SIMPLIFY IT AS ITs IMPLEMENTATION LOOKS HORIBLE >>>>>>
def equal(array1: smp.MutableDenseNDimArray, array2: smp.MutableDenseNDimArray):
    def find_non_zero_elements(arr, pos=None, results=None):
        if pos is None:
            pos = []
        if results is None:
            results = []

        if isinstance(
            arr, (smp.ImmutableDenseNDimArray, smp.MutableDenseNDimArray, list)
        ):
            for i, elem in enumerate(arr):
                find_non_zero_elements(elem, pos + [i], results)
        else:
            if arr != 0:
                results.append((pos, arr))

        return results

    if not isinstance(
        array1, (smp.ImmutableDenseNDimArray, smp.MutableDenseNDimArray, list)
    ):
        return False

    if not isinstance(
        array2, (smp.ImmutableDenseNDimArray, smp.MutableDenseNDimArray, list)
    ):
        return False

    if not array1.shape == array2.shape:
        return False

    list1 = find_non_zero_elements(array1)
    list2 = find_non_zero_elements(array2)

    if len(list1) != len(list2):
        return False

    if len(list1) == 0 and len(list2) == 0:
        return True

    list_res = []
    for i, items in enumerate(list1):
        if isinstance(
            items[1], (smp.Expr, smp.Symbol, smp.Function, smp.MutableDenseNDimArray)
        ):
            list_res.append(list2[i][0] == items[0] and list2[i][1].equals(items[1]))
        else:
            list_res.append(list2[i][0] == items[0] and list2[i][1] == items[1])
    return all(list_res)

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
    assert smp.simplify(res.components) == smp.MutableDenseNDimArray([[1, 0, 0], [0, r**2, 0], [0, 0, r**2*smp.sin(theta)**2]])
    assert str(res.indices) == "_{a}_{b}"


@pytest.mark.skip(reason="TDD =====> Implement TODO: User Can map Arbritary tensor to tensor expression <======== ")
def test_covariant_derivative_metric_mapping(
    Schwarzschild_Basis
):
    # This should do the following:
    # 1. D_{a}*g_{b}_{c} == Zero
    basis = Schwarzschild_Basis
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
    assert equal(res.basis, basis)


# Test Tensor multiplication with non-int, non-float, symbol scalar
    """
                Coordinates := [t, r, theta, phi] 
                g_{mu}_{nu} := [[-(1 - (2 * G * M) / (r)), 0, 0, 0],[0, 1 / (1 - (2 * G * M) / (r)), 0, 0],[0, 0, r**2, 0],[0, 0, 0, r**2 * sin(theta) ** 2]]
                g_{a}_{b}*f(x)
    """
