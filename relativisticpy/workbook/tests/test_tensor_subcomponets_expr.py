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
    Connection,
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

def test_metric_sub_vector_getter(Schwarzschild_Metric, Schwarzschild_Basis):
    metric, _ = Schwarzschild_Metric
    basis = Schwarzschild_Basis
    wb = Workbook()
    

    res = wb.expr(
        """
                Coordinates := [t, r, theta, phi] 
                g_{mu}_{nu} := [[-(1 - (2 * G * M) / (r)), 0, 0, 0],[0, 1 / (1 - (2 * G * M) / (r)), 0, 0],[0, 0, r**2, 0],[0, 0, 0, r**2 * sin(theta) ** 2]]
                g_{a:0}_{b:0}
                g_{a:1}_{b:1}
                g_{a:2}_{b:2}
                g_{a:3}_{b:3}
                g_{a:1}_{b:0}
                g_{a:0}_{b:1}
                g_{a:0}_{b}
                g_{a}_{b:2}
    """
    )
    assert res[0] == metric[0, 0]
    assert res[1] == metric[1, 1]
    assert res[2] == metric[2, 2]
    assert res[3] == metric[3, 3]
    assert res[4] == metric[1, 0]
    assert res[5] == metric[0, 1]
    assert res[6] == metric[0]
    assert res[7] == metric[:, 2]


def test_connection_sub_components(Schwarzschild_Connection, Schwarzschild_Basis):
    connection = Schwarzschild_Connection
    basis = Schwarzschild_Basis
    wb = Workbook()

    res = wb.expr(
        """
                Coordinates := [t, r, theta, phi] 
                g_{mu}_{nu} := [[-(1 - (2 * G * M) / (r)), 0, 0, 0],[0, 1 / (1 - (2 * G * M) / (r)), 0, 0],[0, 0, r**2, 0],[0, 0, 0, r**2 * sin(theta) ** 2]]
                C^{a:0}_{b:0}_{c:0}
                C^{a}_{b:0}_{c:0}
                C^{a:0}_{b}_{c:0}
                C^{a:0}_{b:0}_{c}
    """
    )
    assert res[0] == connection[0,0,0]
    assert res[1] == connection[:,0,0]
    assert res[2] == connection[0,:,0]
    assert res[3] == connection[0,0,:]

def test_ricci_sub_components(Schwarzschild_Ricci, Schwarzschild_Basis):
    ricci_components = Schwarzschild_Ricci

    wb = Workbook()

    res = wb.expr(
        """
                Coordinates := [t, r, theta, phi] 
                g_{mu}_{nu} := [[-(1 - (2 * G * M) / (r)), 0, 0, 0],[0, 1 / (1 - (2 * G * M) / (r)), 0, 0],[0, 0, r**2, 0],[0, 0, 0, r**2 * sin(theta) ** 2]]
                Ric_{a:0}_{b:0}
                Ric_{a:1}_{b}
    """
    )
    assert res[0] == ricci_components[0,0]
    assert res[1] == ricci_components[0]

def test_riemann_sub_components(Schwarzschild_Riemann, Schwarzschild_Basis):
    riemann_components = Schwarzschild_Riemann

    wb = Workbook()

    res = wb.expr(
        """
                Coordinates := [t, r, theta, phi] 
                g_{mu}_{nu} := [[-(1 - (2 * G * M) / (r)), 0, 0, 0],[0, 1 / (1 - (2 * G * M) / (r)), 0, 0],[0, 0, r**2, 0],[0, 0, 0, r**2 * sin(theta) ** 2]]
                R^{a:0}_{b:0}_{c:0}_{n:0}
                R^{a}_{b:0}_{c:0}_{n:0}
                R^{a:0}_{b:1}_{c:0}_{n}
                R^{a:1}_{b}_{c}_{n:3}
        """
    )
    assert res[0] == riemann_components[0, 0, 0, 0]
    assert res[1] == riemann_components[:, 0, 0, 0]
    assert res[2] == riemann_components[0, 1, 0, :]
    assert res[3] == riemann_components[1, :, :, 3]
