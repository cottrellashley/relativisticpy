import pytest
import sympy as smp

from relativisticpy.diffgeom.tensor import Tensor
from relativisticpy.workbook.workbook import Workbook
from relativisticpy.diffgeom import (
    Ricci,
    RicciScalar,
    Metric,
    Riemann,
    KScalar,
    MetricScalar,
    LeviCivitaConnection,
)
from relativisticpy.gr.einstein import EinsteinTensor


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


def test_workbook_new_line_components_definition():
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
    assert str(res.components) == '[[2*G*M/r - 1, 0, 0, 0], [0, 1/(-2*G*M/r + 1), 0, 0], [0, 0, r**2, 0], [0, 0, 0, r**2*sin(theta)**2]]'
    assert str(res.indices) == "_{mu}_{nu}"

def test_ricci_generation_from_riemann_contraction_caching_correctly():
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

    assert str(smp.simplify(res.components)) == "[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]"
    assert str(res.indices) == "_{b}_{h}"


def test_metric_init():
    res = Workbook().expr(
        """
                Coordinates := [t, r, theta, phi] 
                g_{mu}_{nu} := [[-(1 - (2 * G * M) / (r)), 0, 0, 0],[0, 1 / (1 - (2 * G * M) / (r)), 0, 0],[0, 0, r**2, 0],[0, 0, 0, r**2 * sin(theta) ** 2]]
                g_{mu}_{nu}
    """
    )
    assert str(res.components) == "[[2*G*M/r - 1, 0, 0, 0], [0, 1/(-2*G*M/r + 1), 0, 0], [0, 0, r**2, 0], [0, 0, 0, r**2*sin(theta)**2]]"
    assert str(res.indices) == "_{mu}_{nu}"


def test_workbook_inverse_metric_components_match(
):
    res = Workbook().expr(
        """
                Coordinates := [t, r, theta, phi] 
                g_{mu}_{nu} := [[-(1 - (2 * G * M) / (r)), 0, 0, 0],[0, 1 / (1 - (2 * G * M) / (r)), 0, 0],[0, 0, r**2, 0],[0, 0, 0, r**2 * sin(theta) ** 2]]
                g^{a}^{b}
    """
    )
    assert str(res.components) == "[[1/(2*G*M/r - 1), 0, 0, 0], [0, -2*G*M/r + 1, 0, 0], [0, 0, r**(-2), 0], [0, 0, 0, 1/(r**2*sin(theta)**2)]]"
    assert str(res.indices) == "^{a}^{b}"


def test_workbook_cron_delta_metric_result():
    cron_delta = smp.MutableDenseNDimArray(smp.diag(1, 1, 1, 1))

    res = Workbook().expr(
        """
                Coordinates := [t, r, theta, phi] 
                g_{mu}_{nu} := [[-(1 - (2 * G * M) / (r)), 0, 0, 0],[0, 1 / (1 - (2 * G * M) / (r)), 0, 0],[0, 0, r**2, 0],[0, 0, 0, r**2 * sin(theta) ** 2]]
                g_{a}_{b}*g^{b}^{c}
    """
    )
    assert equal(res.components, cron_delta)
    assert str(res.indices) == "_{a}^{c}"
    del res


def test_workbook_metricScalar_result():
    res = Workbook().expr(
        """
                Coordinates := [t, r, theta, phi] 
                g_{mu}_{nu} := [[-(1 - (2 * G * M) / (r)), 0, 0, 0],[0, 1 / (1 - (2 * G * M) / (r)), 0, 0],[0, 0, r**2, 0],[0, 0, 0, r**2 * sin(theta) ** 2]]
                g
    """
    )

    assert int(res.components) == 4
    assert str(res.indices) == ""


def test_LeviCivitaConnection_generation():
    res = Workbook().expr(
        """
                Coordinates := [t, r, theta, phi] 
                g_{mu}_{nu} := [[-(1 - (2 * G * M) / (r)), 0, 0, 0],[0, 1 / (1 - (2 * G * M) / (r)), 0, 0],[0, 0, r**2, 0],[0, 0, 0, r**2 * sin(theta) ** 2]]
                C^{a}_{b}_{c}
    """
    )
    assert str(res.components) == "[[[0, G*M/(r*(-2*G*M + r)), 0, 0], [G*M/(r*(-2*G*M + r)), 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[G*M*(-2*G*M + r)/r**3, 0, 0, 0], [0, G*M/(r*(2*G*M - r)), 0, 0], [0, 0, 2*G*M - r, 0], [0, 0, 0, (2*G*M - r)*sin(theta)**2]], [[0, 0, 0, 0], [0, 0, 1/r, 0], [0, 1/r, 0, 0], [0, 0, 0, -sin(2*theta)/2]], [[0, 0, 0, 0], [0, 0, 0, 1/r], [0, 0, 0, 1/tan(theta)], [0, 1/r, 1/tan(theta), 0]]]"
    assert str(res.indices) == "^{a}_{b}_{c}"


def test_ricci_generation():
    res = Workbook().expr(
        """
                Coordinates := [t, r, theta, phi] 
                g_{mu}_{nu} := [[-(1 - (2 * G * M) / (r)), 0, 0, 0],[0, 1 / (1 - (2 * G * M) / (r)), 0, 0],[0, 0, r**2, 0],[0, 0, 0, r**2 * sin(theta) ** 2]]
                Ric_{a}_{b}
    """
    )
    assert str(res.components) == "[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]"
    assert str(res.indices) == "_{a}_{b}"


def test_riemann_generation():
    res = Workbook().expr(
        """
                Coordinates := [t, r, theta, phi] 
                g_{mu}_{nu} := [[-(1 - (2 * G * M) / (r)), 0, 0, 0],[0, 1 / (1 - (2 * G * M) / (r)), 0, 0],[0, 0, r**2, 0],[0, 0, 0, r**2 * sin(theta) ** 2]]
                R^{a}_{b}_{c}_{h}
    """
    )
    assert str(res.components) == "[[[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[0, 2*G*M/(r**2*(-2*G*M + r)), 0, 0], [2*G*M/(r**2*(2*G*M - r)), 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, -G*M/r, 0], [0, 0, 0, 0], [G*M/r, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, -G*M*sin(theta)**2/r], [0, 0, 0, 0], [0, 0, 0, 0], [G*M*sin(theta)**2/r, 0, 0, 0]]], [[[0, 2*G*M*(-2*G*M + r)/r**4, 0, 0], [2*G*M*(2*G*M - r)/r**4, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, -G*M/r, 0], [0, G*M/r, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, 0, -G*M*sin(theta)**2/r], [0, 0, 0, 0], [0, G*M*sin(theta)**2/r, 0, 0]]], [[[0, 0, G*M*(2*G*M - r)/r**4, 0], [0, 0, 0, 0], [G*M*(-2*G*M + r)/r**4, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, G*M/(r**2*(-2*G*M + r)), 0], [0, G*M/(r**2*(2*G*M - r)), 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 2*G*M*sin(theta)**2/r], [0, 0, -2*G*M*sin(theta)**2/r, 0]]], [[[0, 0, 0, G*M*(2*G*M - r)/r**4], [0, 0, 0, 0], [0, 0, 0, 0], [G*M*(-2*G*M + r)/r**4, 0, 0, 0]], [[0, 0, 0, 0], [0, 0, 0, G*M/(r**2*(-2*G*M + r))], [0, 0, 0, 0], [0, G*M/(r**2*(2*G*M - r)), 0, 0]], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, -2*G*M/r], [0, 0, 2*G*M/r, 0]], [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]]"
    assert str(res.indices) == "^{a}_{b}_{c}_{h}"


def test_ricci_generation_from_riemann_contraction():
    res = Workbook().expr(
        """
                Coordinates := [t, r, theta, phi] 
                g_{mu}_{nu} := [[-(1 - (2 * G * M) / (r)), 0, 0, 0],[0, 1 / (1 - (2 * G * M) / (r)), 0, 0],[0, 0, r**2, 0],[0, 0, 0, r**2 * sin(theta) ** 2]]
                R^{a}_{b}_{a}_{h}
        """
    )
    assert str(smp.simplify(res.components)) == "[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]"
    assert str(res.indices) == "_{b}_{h}"


def test_metric_multiplication():
    res = Workbook().expr(
        """
                Coordinates := [t, r, theta, phi] 
                g_{mu}_{nu} := [[-(1 - (2 * G * M) / (r)), 0, 0, 0],[0, 1 / (1 - (2 * G * M) / (r)), 0, 0],[0, 0, r**2, 0],[0, 0, 0, r**2 * sin(theta) ** 2]]
                g_{a}_{b}*g^{a}^{b}
    """
    )
    assert int(res.components) == 4


def test_LeviCivitaConnection_formulal_equal_built_in_LeviCivitaConnection():
    zeros = smp.MutableDenseNDimArray().zeros(4, 4, 4)
    res = Workbook().expr(
        """
                    Coordinates := [t, r, theta, phi] 
                    g_{mu}_{nu} := [[-(1 - (2 * G * M) / (r)), 0, 0, 0],[0, 1 / (1 - (2 * G * M) / (r)), 0, 0],[0, 0, r**2, 0],[0, 0, 0, r**2 * sin(theta) ** 2]]
                    T^{a}_{c f} := (1/2)*g^{a}^{b}*(d_{c}*g_{b}_{f} + d_{f}*g_{b c} - d_{b}*g_{c f}) - C^{a}_{c f}
                    T^{a}_{c f}
        """
    )

    assert str(smp.simplify(res.components)) == str(zeros)
    assert str(res.indices) == "^{a}_{c}_{f}"


def test_riemann_formulal_comps_equal_built_in_riemann():
    zeros = smp.MutableDenseNDimArray().zeros(4, 4, 4, 4)
    res = Workbook().expr(
        """
                Coordinates := [t, r, theta, phi] 
                g_{mu}_{nu} := [[-(1 - (2 * G * M) / (r)), 0, 0, 0],[0, 1 / (1 - (2 * G * M) / (r)), 0, 0],[0, 0, r**2, 0],[0, 0, 0, r**2 * sin(theta) ** 2]]
                R^{a}_{m}_{b}_{n} - ( d_{b}*C^{a}_{n m} + C^{a}_{b l}*C^{l}_{n m} - d_{n}*C^{a}_{b m} - C^{a}_{n l}*C^{l}_{b m} )
    """
    )

    assert str(smp.simplify(res.components)) == str(zeros)
    assert str(res.indices) == "^{a}_{m}_{b}_{n}"  # _{b}^{a}_{n}_{m}


def test_riemann_formulal_indices_equal_built_in_riemann(
):
    # This should do the following:
    # 1. Idetify that T is a Variable and of a Tensor Type => It is a tensor created by user and to be assigned by user
    # 2. Compute the RHS of the equation.
    # 3. Perform a check that the indices match - order does not matter but covariance of symbols does matter
    # 4. Re-arrange the resulting computed tensor into the index structure which the USER defined in the LHS of the equation
    # 5. Now there should be a tensor T stored in cache, with the computed components and indices defined by user.
    # In this case, the formula is the Riemann formulla and since usually the app returns the rusult in indices '_{b}^{a}_{n}_{m}'
    # Check that the Riemann components assigned to tensot T has been re-structure from: '_{b}^{a}_{n}_{m}' -> '^{a}_{m}_{b}_{n}'

    res = Workbook().expr(
        """
                    Coordinates := [t, r, theta, phi] 
                    g_{mu}_{nu} := [[-(1 - (2 * G * M) / (r)), 0, 0, 0],[0, 1 / (1 - (2 * G * M) / (r)), 0, 0],[0, 0, r**2, 0],[0, 0, 0, r**2 * sin(theta) ** 2]]
                    T^{a}_{m}_{b}_{n} := d_{b}*C^{a}_{n}_{m} + C^{a}_{b}_{l}*C^{l}_{n}_{m} - d_{n}*C^{a}_{b}_{m} - C^{a}_{n}_{l}*C^{l}_{b}_{m}
                    T^{a}_{m b n}
            """
    )
    assert str(smp.simplify(res.components)) == "str(zeros)"
    assert str(res.indices) == "^{a}_{m}_{b}_{n}"


def test_ricci_scalar():
    res = Workbook().expr(
        """
                    Coordinates := [t, r, theta, phi] 
                    g_{mu}_{nu} := [[-(1 - (2 * G * M) / (r)), 0, 0, 0],[0, 1 / (1 - (2 * G * M) / (r)), 0, 0],[0, 0, r**2, 0],[0, 0, 0, r**2 * sin(theta) ** 2]]
                    Ric
        """
    )

    assert int(res.components) == 0
    assert str(res.indices) == ""
    assert type(res) == RicciScalar


def test_einstein_tensor():
    zeros = smp.MutableDenseNDimArray().zeros(4, 4)
    res = Workbook().expr(
        """
                    Coordinates := [t, r, theta, phi] 
                    g_{mu}_{nu} := [[-(1 - (2 * G * M) / (r)), 0, 0, 0],[0, 1 / (1 - (2 * G * M) / (r)), 0, 0],[0, 0, r**2, 0],[0, 0, 0, r**2 * sin(theta) ** 2]]
                    G_{a}_{b}
        """
    )

    assert str(res.components) == "[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]"
    assert type(res) == EinsteinTensor
    assert str(res.indices) == "_{a}_{b}"
    del res


def test_einstein_tensor_computed_from_equation():
    zeros = smp.MutableDenseNDimArray().zeros(4, 4)
    res = Workbook().exe(
        """
                    Coordinates := [t, r, theta, phi] 
                    g_{mu}_{nu} := [[-(1 - (2 * G * M) / (r)), 0, 0, 0],[0, 1 / (1 - (2 * G * M) / (r)), 0, 0],[0, 0, r**2, 0],[0, 0, 0, r**2 * sin(theta) ** 2]]
                    Ric_{a b} - Ric*g_{a b}
                    Ric_{a b} - g_{a b}*Ric
                    Ric_{a b} - Ric*g_{a b}*Ric - G_{a b}
        """
    )

    assert str(res[0].components) == "[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]"
    assert str(res[1].components) == "[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]"
    assert str(res[2].components) == "[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]"
    assert type(res[0]) == Tensor
    assert type(res[1]) == Tensor
    assert type(res[2]) == Tensor
    assert str(res[0].indices) == "_{a}_{b}"
    assert str(res[1].indices) == "_{a}_{b}"
    assert str(res[2].indices) == "_{a}_{b}"
    del res


def test_covariant_derivative_metric_equals_zero(
):
    # This should do the following:
    # 1. D_{a}*g_{b}_{c} == Zero
    zeros = smp.MutableDenseNDimArray().zeros(4, 4, 4)

    res = Workbook().exe(
        """
                    Coordinates := [t, r, theta, phi] 
                    g_{mu}_{nu} := [[-(1 - (2 * G * M) / (r)), 0, 0, 0],[0, 1 / (1 - (2 * G * M) / (r)), 0, 0],[0, 0, r**2, 0],[0, 0, 0, r**2 * sin(theta) ** 2]]
                    D_{a}*g_{b c}
            """
    )
    assert equal(smp.simplify(res.components), zeros)
    assert str(res.indices) == "_{a}_{b}_{c}"
    assert str(res.indices.basis) == "[t, r, theta, phi]"
