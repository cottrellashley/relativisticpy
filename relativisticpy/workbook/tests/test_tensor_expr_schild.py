import pytest
import sympy as smp
from relativisticpy.workbook.workbook import Workbook

# TODO: <<< PUT THIS FUNCTION SOMEWHERE ELSE + SIMPLIFY IT AS ITs IMPLEMENTATION LOOKS HORIBLE >>>>>>
def equal(array1: smp.MutableDenseNDimArray, array2: smp.MutableDenseNDimArray): 

    def find_non_zero_elements(arr, pos=None, results=None):
        if pos is None:
            pos = []
        if results is None:
            results = []

        if isinstance(arr, (smp.ImmutableDenseNDimArray, smp.MutableDenseNDimArray, list)):
            for i, elem in enumerate(arr):
                find_non_zero_elements(elem, pos + [i], results)
        else:
            if arr != 0:
                results.append((pos, arr))

        return results

    if not isinstance(array1, (smp.ImmutableDenseNDimArray, smp.MutableDenseNDimArray, list)):
        return False

    if not isinstance(array2, (smp.ImmutableDenseNDimArray, smp.MutableDenseNDimArray, list)):
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
        if isinstance(items[1], (smp.Expr, smp.Symbol, smp.Function, smp.MutableDenseNDimArray)):
            list_res.append(
                list2[i][0] == items[0] and list2[i][1].equals(items[1])
            )
        else:
            list_res.append(
                list2[i][0] == items[0] and list2[i][1] == items[1]
            )
    return all(list_res)

def test_workbook_new_line_components_definition(Schwarzschild_Metric):
    metric, inverse_metric = Schwarzschild_Metric
    wb = Workbook()

    res = wb.expr('''
                MetricSymbol := G 
                RicciSymbol := Ric 
                Coordinates := [t, r, theta, phi] 

                g_{mu}_{nu} := [
                                    [-(1 - (2 * G * M) / (r)), 0, 0, 0],
                                    [0, 1 / (1 - (2 * G * M) / (r)), 0, 0],
                                    [0, 0, r**2, 0],
                                    [0, 0, 0, r**2 * sin(theta) ** 2],
                                ]
                g_{mu}_{nu}
    ''')

    assert equal(res[0].components, metric)

def test_metric_init(Schwarzschild_Metric, Schwarzschild_Basis):
    metric, _ = Schwarzschild_Metric
    basis = Schwarzschild_Basis
    wb = Workbook()

    res = wb.expr('''
                MetricSymbol := G 
                RicciSymbol := Ric 
                Coordinates := [t, r, theta, phi] 
                g_{mu}_{nu} := [[-(1 - (2 * G * M) / (r)), 0, 0, 0],[0, 1 / (1 - (2 * G * M) / (r)), 0, 0],[0, 0, r**2, 0],[0, 0, 0, r**2 * sin(theta) ** 2]]
                g_{mu}_{nu}
    ''')
    assert equal(res[0].components, metric)
    assert str(res[0].indices) == '_{mu}_{nu}'
    assert equal(res[0].basis, basis)


def test_workbook_inverse_metric_components_match(Schwarzschild_Metric, Schwarzschild_Basis):
    _, inverse_metric = Schwarzschild_Metric
    basis = Schwarzschild_Basis
    wb = Workbook()

    res = wb.expr('''
                MetricSymbol := G 
                RicciSymbol := Ric 
                Coordinates := [t, r, theta, phi] 
                g_{mu}_{nu} := [[-(1 - (2 * G * M) / (r)), 0, 0, 0],[0, 1 / (1 - (2 * G * M) / (r)), 0, 0],[0, 0, r**2, 0],[0, 0, 0, r**2 * sin(theta) ** 2]]
                g^{a}^{b}
    ''')
    assert equal(res[0].components, inverse_metric)
    assert str(res[0].indices) == '^{a}^{b}'
    assert equal(res[0].basis, basis)


def test_ricci_generation(Schwarzschild_Ricci, Schwarzschild_Basis):
    ricci_components = Schwarzschild_Ricci
    basis = Schwarzschild_Basis

    wb = Workbook()

    res = wb.expr('''
                MetricSymbol := G 
                RicciSymbol := Ric
                Coordinates := [t, r, theta, phi] 
                g_{mu}_{nu} := [[-(1 - (2 * G * M) / (r)), 0, 0, 0],[0, 1 / (1 - (2 * G * M) / (r)), 0, 0],[0, 0, r**2, 0],[0, 0, 0, r**2 * sin(theta) ** 2]]
                Ric_{a}_{b}
    ''')
    assert equal(res[0].components, ricci_components)
    assert str(res[0].indices) == '_{a}_{b}'
    assert equal(res[0].basis, basis)

def test_ricci_inverse_generation(Schwarzschild_Riemann, Schwarzschild_Basis):
    riemann_components = Schwarzschild_Riemann
    basis = Schwarzschild_Basis

    wb = Workbook()

    res = wb.expr('''
                MetricSymbol := G 
                RiemannSymbol := R
                Coordinates := [t, r, theta, phi] 
                g_{mu}_{nu} := [[-(1 - (2 * G * M) / (r)), 0, 0, 0],[0, 1 / (1 - (2 * G * M) / (r)), 0, 0],[0, 0, r**2, 0],[0, 0, 0, r**2 * sin(theta) ** 2]]
                R^{a}_{b}_{c}_{d}
    ''')
    assert equal(res[0].components, riemann_components)
    assert str(res[0].indices) == '^{a}_{b}_{c}_{d}'
    assert equal(res[0].basis, basis)

