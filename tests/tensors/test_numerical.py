"""Numerical components: contraction with numpy.einsum, in slices when large, and with
sparse matrices."""

import numpy as np
import pytest
import scipy.sparse as sp
import sympy

from relativisticpy.errors import EvaluationError
from relativisticpy.tensors import arrays, numerical

rng = np.random.default_rng(3)


class TestContraction:
    def test_einstein_summation(self):
        x, y = rng.normal(size=(5, 3)), rng.normal(size=(3, 4))
        result = numerical.contract([(x, (0, 1)), (y, (1, 2))], (0, 2), (5, 3, 4))
        assert np.allclose(result, x @ y)

    def test_a_label_in_three_operands_is_one_sum(self):
        # E_t[x_{ta} y_{tb}] with the measure m^t: one contraction, not a product first.
        x, y, m = rng.normal(size=(6, 2)), rng.normal(size=(6, 3)), np.full(6, 1 / 6)
        result = numerical.contract([(x, (0, 1)), (y, (0, 2)), (m, (0,))], (1, 2), (6, 2, 3))
        assert np.allclose(result, x.T @ y / 6)

    def test_a_full_contraction_is_a_number(self):
        v = rng.normal(size=4)
        result = numerical.contract([(v, (0,)), (v, (0,))], (), (4,))
        assert np.ndim(result) == 0
        assert np.isclose(result, v @ v)

    def test_symbolic_numbers_meet_numerical_ones(self):
        half = sympy.ImmutableDenseNDimArray([sympy.Rational(1, 2)] * 2)
        result = arrays.contract([(half, (0,)), (np.array([2.0, 4.0]), (0,))], (), (2,))
        assert result == pytest.approx(3.0)

    def test_large_operands_are_contracted_in_slices(self, monkeypatch):
        monkeypatch.setattr(numerical, "CHUNK", 1_000)
        x = rng.normal(size=(5_000, 7))
        result = numerical.contract([(x, (0, 1)), (x, (0, 2))], (1, 2), (5_000, 7, 7))
        assert np.allclose(result, x.T @ x)

    def test_a_sparse_matrix_times_a_dense_operand(self):
        matrix = sp.random(50, 50, density=0.1, random_state=1, format="csr")
        v, w = rng.normal(size=50), rng.normal(size=(50, 3))
        assert np.allclose(numerical.contract([(matrix, (0, 1)), (v, (1,))], (0,)), matrix @ v)
        assert np.allclose(numerical.contract([(v, (0,)), (matrix, (0, 1))], (1,)), matrix.T @ v)
        assert np.allclose(
            numerical.contract([(matrix, (0, 1)), (w, (1, 2))], (2, 0)), (matrix @ w).T
        )

    def test_products_of_sparse_matrices_stay_sparse(self):
        a = sp.random(30, 30, density=0.1, random_state=2, format="csr")
        product = numerical.contract([(a, (0, 1)), (a, (1, 2))], (0, 2))
        assert sp.issparse(product)
        assert np.allclose(product.toarray(), (a @ a).toarray())
        v = rng.normal(size=30)
        chain = numerical.contract([(a, (0, 1)), (a, (1, 2)), (v, (2,))], (0,))
        assert np.allclose(chain, a @ (a @ v))
        assert np.isclose(numerical.contract([(v, (0,)), (a, (0, 1)), (v, (1,))], ()), v @ a @ v)
        assert np.isclose(numerical.contract([(a, (0, 0))], ()), a.diagonal().sum())

    def test_a_row_of_a_sparse_matrix(self):
        a = sp.random(500, 500, density=0.01, random_state=4, format="csr")
        assert np.allclose(numerical.select(a, {0: 3}), a.toarray()[3])
        assert np.allclose(numerical.select(a, {1: 7}), a.toarray()[:, 7])

    def test_a_symbol_cannot_meet_numbers(self):
        with pytest.raises(EvaluationError, match="x is a symbol"):
            numerical.number(sympy.Symbol("x"))


class TestComponents:
    def test_permute_select_and_symmetrize(self):
        x = rng.normal(size=(2, 3, 4))
        assert np.allclose(arrays.permute(x, (2, 0, 1)), np.transpose(x, (2, 0, 1)))
        assert np.allclose(arrays.select(x, {1: 2}), x[:, 2, :])
        square = rng.normal(size=(3, 3))
        assert np.allclose(
            arrays.symmetrize(square, (0, 1), antisymmetric=True), (square - square.T) / 2
        )

    def test_elementwise_functions_are_numpy_s(self):
        x = np.array([0.0, 1.0])
        assert np.allclose(numerical.apply(x, sympy.exp, "exp"), np.exp(x))
        # Without a NumPy version, the function is applied to each component.
        assert np.allclose(numerical.apply(x, lambda v: v**2 + 1, "custom"), x**2 + 1)

    def test_close_is_to_rounding_error(self):
        assert numerical.close(np.array([1.0, 2.0]), np.array([1.0, 2.0 + 1e-13]))
        assert not numerical.close(np.array([1.0, 2.0]), np.array([1.0, 2.1]))
        assert numerical.close(np.zeros(3), 0)

    def test_close_is_relative_to_the_numbers_compared(self):
        masses = np.array([9.109e-31, 1.673e-27])  # an electron and a proton, in kg
        assert not numerical.close(masses, 2 * masses)
        assert not numerical.close(masses, 0)
        # A residual is rounding error relative to the numbers it was computed from.
        assert not numerical.close(np.array([8.6e-12]), 0)
        with numerical.operands(1e4):
            assert numerical.close(np.array([8.6e-12]), 0)
            assert not numerical.close(np.array([8.6e-6]), 0)

    def test_what_is_numerical(self):
        assert numerical.is_numerical(np.zeros(2))
        assert numerical.is_numerical(np.float64(1.0))
        assert numerical.is_numerical(sp.eye(3, format="csr"))
        assert not numerical.is_numerical(sympy.Array([1, 2]))
        assert not numerical.is_numerical(2.0)
