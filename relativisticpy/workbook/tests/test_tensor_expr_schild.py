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


def test_workbook_new_line_components_definition(Schwarzschild_Metric):
    metric, inverse_metric = Schwarzschild_Metric
    res = Workbook().expr(
        """
                Coordinates := [t, r, theta, phi] 

                g_{mu}_{nu} := [
                                    [-(1 - (2 * G * M) / (r)), 0, 0, 0],
                                    [0, 1 / (1 - (2 * G * M) / (r)), 0, 0],
                                    [0, 0, r**2, 0],
                                    [0, 0, 0, r**2 * sin(theta) ** 2]
                                ]
                g_{mu}_{nu}
        """
    )
    assert equal(res.components, metric)
    del res

def test_ricci_generation_from_riemann_contraction_caching_correctly(Schwarzschild_Ricci, Schwarzschild_Basis):
    ricci_components = Schwarzschild_Ricci
    basis = Schwarzschild_Basis

    wb = Workbook()
    wb.expr(
            """
                    Coordinates := [t, r, theta, phi] 
                    g_{mu}_{nu} := [[-(1 - (2 * G * M) / (r)), 0, 0, 0],[0, 1 / (1 - (2 * G * M) / (r)), 0, 0],[0, 0, r**2, 0],[0, 0, 0, r**2 * sin(theta) ** 2]]
                    R^{a}_{b}_{a}_{h}
                    R^{a}_{c}_{a}_{h}
            """
        )
    wb.expr(
        """
                R^{a}_{b}_{a}_{h}
        """
    )
    res = wb.expr(
        """
                R^{a}_{b}_{a}_{h}
        """
    )

    assert equal(smp.simplify(res.components), ricci_components)
    assert str(res.indices) == "_{b}_{h}"
    assert equal(res.basis, basis)
    del wb
    del res

def test_metric_init(Schwarzschild_Metric, Schwarzschild_Basis):
    metric, _ = Schwarzschild_Metric
    basis = Schwarzschild_Basis

    res = Workbook().expr(
        """
                Coordinates := [t, r, theta, phi] 
                g_{mu}_{nu} := [[-(1 - (2 * G * M) / (r)), 0, 0, 0],[0, 1 / (1 - (2 * G * M) / (r)), 0, 0],[0, 0, r**2, 0],[0, 0, 0, r**2 * sin(theta) ** 2]]
                g_{mu}_{nu}
    """
    )
    assert equal(res.components, metric)
    assert str(res.indices) == "_{mu}_{nu}"
    assert equal(res.basis, basis)
    del res


def test_workbook_inverse_metric_components_match(
    Schwarzschild_Metric, Schwarzschild_Basis
):
    _, inverse_metric = Schwarzschild_Metric
    basis = Schwarzschild_Basis

    res = Workbook().expr(
        """
                Coordinates := [t, r, theta, phi] 
                g_{mu}_{nu} := [[-(1 - (2 * G * M) / (r)), 0, 0, 0],[0, 1 / (1 - (2 * G * M) / (r)), 0, 0],[0, 0, r**2, 0],[0, 0, 0, r**2 * sin(theta) ** 2]]
                g^{a}^{b}
    """
    )
    assert equal(res.components, inverse_metric)
    assert str(res.indices) == "^{a}^{b}"
    assert equal(res.basis, basis)
    del res


def test_workbook_cron_delta_metric_result(Schwarzschild_Metric, Schwarzschild_Basis):
    cron_delta = smp.MutableDenseNDimArray(smp.diag(1, 1, 1, 1))
    basis = Schwarzschild_Basis

    res = Workbook().expr(
        """
                Coordinates := [t, r, theta, phi] 
                g_{mu}_{nu} := [[-(1 - (2 * G * M) / (r)), 0, 0, 0],[0, 1 / (1 - (2 * G * M) / (r)), 0, 0],[0, 0, r**2, 0],[0, 0, 0, r**2 * sin(theta) ** 2]]
                g_{a}_{b}*g^{b}^{c}
    """
    )
    assert equal(res.components, cron_delta)
    assert str(res.indices) == "_{a}^{c}"
    assert equal(res.basis, basis)
    del res


def test_workbook_metricScalar_result(Schwarzschild_MetricScalar, Schwarzschild_Basis):
    metric = Schwarzschild_MetricScalar
    basis = Schwarzschild_Basis

    res = Workbook().expr(
        """
                Coordinates := [t, r, theta, phi] 
                g_{mu}_{nu} := [[-(1 - (2 * G * M) / (r)), 0, 0, 0],[0, 1 / (1 - (2 * G * M) / (r)), 0, 0],[0, 0, r**2, 0],[0, 0, 0, r**2 * sin(theta) ** 2]]
                g
    """
    )

    assert res.components == 4
    assert str(res.indices) == ""
    assert equal(res.basis, basis)
    del res


def test_connection_generation(Schwarzschild_Connection, Schwarzschild_Basis):
    connection = Schwarzschild_Connection
    basis = Schwarzschild_Basis

    res = Workbook().expr(
        """
                Coordinates := [t, r, theta, phi] 
                g_{mu}_{nu} := [[-(1 - (2 * G * M) / (r)), 0, 0, 0],[0, 1 / (1 - (2 * G * M) / (r)), 0, 0],[0, 0, r**2, 0],[0, 0, 0, r**2 * sin(theta) ** 2]]
                C^{a}_{b}_{c}
    """
    )
    assert equal(res.components, connection)
    assert str(res.indices) == "^{a}_{b}_{c}"
    assert equal(res.basis, basis)
    del res


def test_ricci_generation(Schwarzschild_Ricci, Schwarzschild_Basis):
    ricci_components = Schwarzschild_Ricci
    basis = Schwarzschild_Basis

    res = Workbook().expr(
        """
                Coordinates := [t, r, theta, phi] 
                g_{mu}_{nu} := [[-(1 - (2 * G * M) / (r)), 0, 0, 0],[0, 1 / (1 - (2 * G * M) / (r)), 0, 0],[0, 0, r**2, 0],[0, 0, 0, r**2 * sin(theta) ** 2]]
                Ric_{a}_{b}
    """
    )
    assert equal(res.components, ricci_components)
    assert str(res.indices) == "_{a}_{b}"
    assert equal(res.basis, basis)
    del res


def test_riemann_generation(Schwarzschild_Riemann, Schwarzschild_Basis):
    riemann_components = Schwarzschild_Riemann
    basis = Schwarzschild_Basis

    res = Workbook().expr(
        """
                Coordinates := [t, r, theta, phi] 
                g_{mu}_{nu} := [[-(1 - (2 * G * M) / (r)), 0, 0, 0],[0, 1 / (1 - (2 * G * M) / (r)), 0, 0],[0, 0, r**2, 0],[0, 0, 0, r**2 * sin(theta) ** 2]]
                R^{a}_{b}_{c}_{h}
    """
    )
    assert equal(res.components, riemann_components)
    assert str(res.indices) == "^{a}_{b}_{c}_{h}"
    assert equal(res.basis, basis)
    del res

def test_ricci_generation_from_riemann_contraction(Schwarzschild_Ricci, Schwarzschild_Basis):
    ricci_components = Schwarzschild_Ricci
    basis = Schwarzschild_Basis

    res = Workbook().expr(
        """
                Coordinates := [t, r, theta, phi] 
                g_{mu}_{nu} := [[-(1 - (2 * G * M) / (r)), 0, 0, 0],[0, 1 / (1 - (2 * G * M) / (r)), 0, 0],[0, 0, r**2, 0],[0, 0, 0, r**2 * sin(theta) ** 2]]
                R^{a}_{b}_{a}_{h}
        """
    )
    assert equal(smp.simplify(res.components), ricci_components)
    assert str(res.indices) == "_{b}_{h}"
    assert equal(res.basis, basis)
    del res

def test_metric_multiplication(Schwarzschild_MetricScalar, Schwarzschild_Basis):
    metric = Schwarzschild_MetricScalar
    basis = Schwarzschild_Basis

    res = Workbook().expr(
        """
                Coordinates := [t, r, theta, phi] 
                g_{mu}_{nu} := [[-(1 - (2 * G * M) / (r)), 0, 0, 0],[0, 1 / (1 - (2 * G * M) / (r)), 0, 0],[0, 0, r**2, 0],[0, 0, 0, r**2 * sin(theta) ** 2]]
                g_{a}_{b}*g^{a}^{b}
    """
    )
    assert res == 4
    del res


def test_connection_formulal_equal_built_in_connection(Schwarzschild_Basis):
    basis = Schwarzschild_Basis
    zeros = smp.MutableDenseNDimArray().zeros(4, 4, 4)
    res = Workbook().expr(
        """
                    Coordinates := [t, r, theta, phi] 
                    g_{mu}_{nu} := [[-(1 - (2 * G * M) / (r)), 0, 0, 0],[0, 1 / (1 - (2 * G * M) / (r)), 0, 0],[0, 0, r**2, 0],[0, 0, 0, r**2 * sin(theta) ** 2]]
                    T^{a}_{c}_{f} := (1/2)*g^{a}^{b}*(d_{c}*g_{b}_{f} + d_{f}*g_{b}_{c} - d_{b}*g_{c}_{f}) - C^{a}_{c}_{f}
                    T^{a}_{c}_{f}
        """
    )

    assert smp.simplify(res.components) == zeros
    assert str(res.indices) == "^{a}_{c}_{f}"
    assert equal(res.basis, basis)
    del res


def test_riemann_formulal_comps_equal_built_in_riemann(Schwarzschild_Basis):
    basis = Schwarzschild_Basis
    zeros = smp.MutableDenseNDimArray().zeros(4, 4, 4, 4)
    res = Workbook().expr(
        """
                Coordinates := [t, r, theta, phi] 
                g_{mu}_{nu} := [[-(1 - (2 * G * M) / (r)), 0, 0, 0],[0, 1 / (1 - (2 * G * M) / (r)), 0, 0],[0, 0, r**2, 0],[0, 0, 0, r**2 * sin(theta) ** 2]]
                R^{a}_{m}_{b}_{n} - ( d_{b}*C^{a}_{n}_{m} + C^{a}_{b}_{l}*C^{l}_{n}_{m} - d_{n}*C^{a}_{b}_{m} - C^{a}_{n}_{l}*C^{l}_{b}_{m} )
    """
    )

    assert smp.simplify(res.components) == zeros
    assert str(res.indices) == "^{a}_{m}_{b}_{n}"  # _{b}^{a}_{n}_{m}
    assert equal(res.basis, basis)
    del res


def test_riemann_formulal_indices_equal_built_in_riemann(
    Schwarzschild_Riemann, Schwarzschild_Basis
):
    # This should do the following:
    # 1. Idetify that T is a Variable and of a Tensor Type => It is a tensor created by user and to be assigned by user
    # 2. Compute the RHS of the equation.
    # 3. Perform a check that the indices match - order does not matter but covariance of symbols does matter
    # 4. Re-arrange the resulting computed tensor into the index structure which the USER defined in the LHS of the equation
    # 5. Now there should be a tensor T stored in cache, with the computed components and indices defined by user.
    # In this case, the formula is the Riemann formulla and since usually the app returns the rusult in indices '_{b}^{a}_{n}_{m}'
    # Check that the Riemann components assigned to tensot T has been re-structure from: '_{b}^{a}_{n}_{m}' -> '^{a}_{m}_{b}_{n}'
    riemann_components = Schwarzschild_Riemann
    basis = Schwarzschild_Basis

    res = Workbook().expr(
            """
                    Coordinates := [t, r, theta, phi] 
                    g_{mu}_{nu} := [[-(1 - (2 * G * M) / (r)), 0, 0, 0],[0, 1 / (1 - (2 * G * M) / (r)), 0, 0],[0, 0, r**2, 0],[0, 0, 0, r**2 * sin(theta) ** 2]]
                    T^{a}_{m}_{b}_{n} := d_{b}*C^{a}_{n}_{m} + C^{a}_{b}_{l}*C^{l}_{n}_{m} - d_{n}*C^{a}_{b}_{m} - C^{a}_{n}_{l}*C^{l}_{b}_{m}
                    T^{a}_{m}_{b}_{n}
            """
    )
    assert equal(smp.simplify(res.components), riemann_components)
    assert str(res.indices) == "^{a}_{m}_{b}_{n}"
    assert equal(res.basis, basis)
    del res


def test_ricci_scalar(Schwarzschild_Basis):
    basis = Schwarzschild_Basis
    res = Workbook().expr(
        """
                    Coordinates := [t, r, theta, phi] 
                    g_{mu}_{nu} := [[-(1 - (2 * G * M) / (r)), 0, 0, 0],[0, 1 / (1 - (2 * G * M) / (r)), 0, 0],[0, 0, r**2, 0],[0, 0, 0, r**2 * sin(theta) ** 2]]
                    Ric
        """
    )

    assert res.components == 0
    assert str(res.indices) == ""
    assert type(res) == RicciScalar
    assert equal(res.basis, basis)
    del res


def test_einstein_tensor(Schwarzschild_Basis):
    basis = Schwarzschild_Basis
    zeros = smp.MutableDenseNDimArray().zeros(4, 4)
    res = Workbook().expr(
        """
                    Coordinates := [t, r, theta, phi] 
                    g_{mu}_{nu} := [[-(1 - (2 * G * M) / (r)), 0, 0, 0],[0, 1 / (1 - (2 * G * M) / (r)), 0, 0],[0, 0, r**2, 0],[0, 0, 0, r**2 * sin(theta) ** 2]]
                    G_{a}_{b}
        """
    )

    assert smp.simplify(res.components) == zeros
    assert type(res) == EinsteinTensor
    assert str(res.indices) == "_{a}_{b}"
    assert equal(res.basis, basis)
    del res

def test_einstein_tensor_computed_from_equation(Schwarzschild_Basis):
    basis = Schwarzschild_Basis
    zeros = smp.MutableDenseNDimArray().zeros(4, 4)
    res = Workbook().expr(
        """
                    Coordinates := [t, r, theta, phi] 
                    g_{mu}_{nu} := [[-(1 - (2 * G * M) / (r)), 0, 0, 0],[0, 1 / (1 - (2 * G * M) / (r)), 0, 0],[0, 0, r**2, 0],[0, 0, 0, r**2 * sin(theta) ** 2]]
                    Ric_{a}_{b} - Ric*g_{a}_{b}
                    Ric_{a}_{b} - g_{a}_{b}*Ric
                    Ric_{a}_{b} - Ric*g_{a}_{b}*Ric - G_{a}_{b}
        """
    )

    assert equal(smp.simplify(res[0].components), zeros)
    assert equal(smp.simplify(res[1].components), zeros)
    assert equal(smp.simplify(res[2].components), zeros)
    assert type(res[0]) == EinsteinArray
    assert type(res[1]) == EinsteinArray
    assert type(res[2]) == EinsteinArray
    assert str(res[0].indices) == "_{a}_{b}"
    assert str(res[1].indices) == "_{a}_{b}"
    assert str(res[2].indices) == "_{a}_{b}"
    assert equal(res[0].basis, basis)
    assert equal(res[1].basis, basis)
    assert equal(res[2].basis, basis)
    del res


@pytest.mark.skip(reason="TDD =====> Implement TODO: Covariant Derivative <======== ")
def test_covariant_derivative_metric_equals_zero(
    Schwarzschild_Basis
):
    # This should do the following:
    # 1. D_{a}*g_{b}_{c} == Zero
    basis = Schwarzschild_Basis
    zeros = smp.MutableDenseNDimArray().zeros(4, 4, 4)

    res = Workbook().expr(
            """
                    Coordinates := [t, r, theta, phi] 
                    g_{mu}_{nu} := [[-(1 - (2 * G * M) / (r)), 0, 0, 0],[0, 1 / (1 - (2 * G * M) / (r)), 0, 0],[0, 0, r**2, 0],[0, 0, 0, r**2 * sin(theta) ** 2]]
                    D_{a}*g_{b}_{c}
            """
    )
    assert equal(smp.simplify(res.components), zeros)
    assert str(res.indices) == "_{a}_{b}_{c}"
    assert equal(res.basis, basis)
    del res
