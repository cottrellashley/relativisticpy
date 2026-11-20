"""Component kernels: sparse Einstein summation, permutation, symmetrisation."""

import pytest
import sympy
from sympy import Array, diag, sin, symbols

from relativisticpy.tensors import arrays

t, r, theta, phi, M = symbols("t r theta phi M")
SCHWARZSCHILD = Array(diag(-(1 - 2 * M / r), 1 / (1 - 2 * M / r), r**2, r**2 * sin(theta) ** 2))
INVERSE = Array(sympy.Matrix(SCHWARZSCHILD.tolist()).inv())


def dense_einsum(operands, output, dim):
    """A slow, obviously correct reference: sum over every label assignment."""
    labels = sorted({label for _, ls in operands for label in ls}, key=str)
    summed = [label for label in labels if label not in output]
    result = {}
    for out in __import__("itertools").product(range(dim), repeat=len(output)):
        total = 0
        for inner in __import__("itertools").product(range(dim), repeat=len(summed)):
            assignment = dict(zip(output, out, strict=True)) | dict(zip(summed, inner, strict=True))
            term = 1
            for components, ls in operands:
                index = tuple(assignment[label] for label in ls)
                term *= components[index] if ls else components
            total += term
        result[out] = sympy.simplify(total)
    return result


class TestContract:
    def test_matrix_product(self):
        a = Array([[1, 2], [3, 4]])
        b = Array([[5, 6], [7, 8]])
        assert arrays.contract([(a, "ab"), (b, "bc")], "ac", 2) == Array([[19, 22], [43, 50]])

    def test_metric_times_inverse_is_the_identity(self):
        result = arrays.contract([(SCHWARZSCHILD, "ab"), (INVERSE, "bc")], "ac", 4)
        assert result.applyfunc(sympy.simplify) == Array(sympy.eye(4))

    def test_full_contraction_gives_a_scalar(self):
        result = arrays.contract([(SCHWARZSCHILD, "ab"), (INVERSE, "ab")], "", 4)
        assert sympy.simplify(result) == 4
        assert not isinstance(result, sympy.NDimArray)

    def test_trace_within_one_operand(self):
        a = Array([[1, 2], [3, 4]])
        assert arrays.contract([(a, "aa")], "", 2) == 5

    def test_partial_trace(self):
        a = Array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        assert arrays.contract([(a, "aab")], "b", 2) == Array([1 + 7, 2 + 8])

    def test_outer_product(self):
        u, v = Array([1, 2]), Array([3, 4])
        assert arrays.contract([(u, "a"), (v, "b")], "ab", 2) == Array([[3, 4], [6, 8]])

    def test_output_order_is_respected(self):
        a = Array([[1, 2], [3, 4]])
        assert arrays.contract([(a, "ab")], "ba", 2) == Array([[1, 3], [2, 4]])

    def test_scalar_operands(self):
        assert arrays.contract([(sympy.Integer(3), ""), (Array([1, 2]), "a")], "a", 2) == Array(
            [3, 6]
        )

    def test_zero_operand_gives_zeros(self):
        assert arrays.contract([(Array([0, 0]), "a"), (Array([1, 2]), "a")], "", 2) == 0

    def test_three_operands_match_a_dense_reference(self):
        x, y = symbols("x y")
        a = Array([[x, 1], [0, y]])
        b = Array([[1, x], [y, 0]])
        c = Array([x, y])
        operands = [(a, "ab"), (b, "bc"), (c, "c")]
        result = arrays.contract(operands, "a", 2)
        reference = dense_einsum(operands, "a", 2)
        assert [sympy.simplify(result[i] - reference[(i,)]) for i in range(2)] == [0, 0]

    def test_output_label_must_exist(self):
        with pytest.raises(ValueError, match="not in any operand"):
            arrays.contract([(Array([1, 2]), "a")], "b", 2)

    def test_rank_must_match_labels(self):
        with pytest.raises(ValueError, match="rank"):
            arrays.contract([(Array([1, 2]), "ab")], "a", 2)


class TestKernels:
    def test_permute(self):
        a = Array([[1, 2], [3, 4]])
        assert arrays.permute(a, [1, 0]) == Array([[1, 3], [2, 4]])
        assert arrays.permute(a, [0, 1]) is a

    def test_select(self):
        a = Array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        assert arrays.select(a, {0: 1}) == Array([[5, 6], [7, 8]])
        assert arrays.select(a, {0: 1, 2: 0}) == Array([5, 7])
        assert arrays.select(a, {0: 1, 1: 1, 2: 1}) == 8

    def test_partial_puts_the_new_axis_first(self):
        result = arrays.partial(SCHWARZSCHILD, (t, r, theta, phi))
        assert result.shape == (4, 4, 4)
        assert result[1, 2, 2] == 2 * r
        assert result[2, 3, 3] == 2 * r**2 * sin(theta) * sympy.cos(theta)

    def test_symmetrize(self):
        a = Array([[1, 2], [3, 4]])
        assert arrays.symmetrize(a, [0, 1], antisymmetric=False) == Array(
            [[1, sympy.Rational(5, 2)], [sympy.Rational(5, 2), 4]]
        )
        assert arrays.symmetrize(a, [0, 1], antisymmetric=True) == Array(
            [[0, -sympy.Rational(1, 2)], [sympy.Rational(1, 2), 0]]
        )

    def test_antisymmetrizing_three_axes(self):
        a = Array(
            [[[sympy.Symbol(f"a{i}{j}{k}") for k in range(2)] for j in range(2)] for i in range(2)]
        )
        result = arrays.symmetrize(a, [0, 1, 2], antisymmetric=True)
        assert result == arrays.zeros(2, 3)  # nothing is totally antisymmetric in two dimensions

    def test_apply_and_flatten(self):
        assert arrays.apply(Array([1, 2]), lambda c: c * 10) == Array([10, 20])
        assert arrays.apply(sympy.Integer(3), lambda c: c * 10) == 30
        assert arrays.flatten(Array([[1, 2], [3, 4]])) == [1, 2, 3, 4]
        assert arrays.flatten(sympy.Integer(7)) == [7]

    def test_conversions(self):
        assert arrays.to_array([[1, 2], [3, 4]]) == Array([[1, 2], [3, 4]])
        assert arrays.to_array(sympy.Matrix([[1, 2], [3, 4]])) == Array([[1, 2], [3, 4]])
        assert arrays.zeros(2, 0) == 0
        assert arrays.from_flat([1, 2, 3, 4], 2, 2) == Array([[1, 2], [3, 4]])
        assert arrays.rank(sympy.Integer(1)) == 0
