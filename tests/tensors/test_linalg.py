"""Linear algebra on components, symbolic and numerical."""

import numpy as np
import pytest
import scipy.linalg
import scipy.sparse as sp
import sympy
from sympy import Array, Symbol, sqrt

from relativisticpy.errors import EvaluationError
from relativisticpy.tensors import linalg

rng = np.random.default_rng(5)
A = rng.normal(size=(4, 4))
SPD = A @ A.T + 4 * np.eye(4)  # symmetric positive definite, as a covariance


class TestSolving:
    def test_an_inverse(self):
        assert np.allclose(linalg.inverse(SPD) @ SPD, np.eye(4))
        assert linalg.inverse(Array([[2, 0], [0, 4]])) == Array(
            [[sympy.Rational(1, 2), 0], [0, sympy.Rational(1, 4)]]
        )

    def test_a_singular_tensor_has_no_inverse(self):
        with pytest.raises(EvaluationError, match="no inverse"):
            linalg.inverse(Array([[1, 2], [2, 4]]))

    @pytest.mark.parametrize("matrix", [SPD, A + 5 * np.eye(4)])
    def test_a_solve_by_cholesky_or_lu(self, matrix):
        b = rng.normal(size=(4, 2))
        assert np.allclose(matrix @ linalg.solve(matrix, b), b)
        assert np.allclose(matrix.T @ linalg.solve(matrix, b, transpose=True), b)

    def test_a_nearly_symmetric_matrix_is_solved_as_it_is(self):
        # NumPy's default tolerance calls it symmetric; Cholesky would read half of it.
        near = np.array([[1.0, 0.999], [0.999 + 9e-6, 1.0]])
        b = np.array([[1.0], [0.0]])
        assert np.allclose(linalg.solve(near, b), np.linalg.solve(near, b), rtol=1e-10, atol=0)
        upper = 1e-9 * np.array([[2.0, 1.0], [0.0, 2.0]])
        assert np.allclose(linalg.solve(upper, b), np.linalg.solve(upper, b), rtol=1e-10, atol=0)

    def test_a_symbolic_solve(self):
        x, y = Symbol("x"), Symbol("y")
        assert linalg.solve(Array([[x, 0], [0, y]]), Array([1, 1])) == Array([1 / x, 1 / y])

    def test_a_sparse_solve(self):
        matrix = sp.csr_matrix(SPD)
        b = rng.normal(size=4)
        assert np.allclose(linalg.solve(matrix, b).ravel(), np.linalg.solve(SPD, b))


class TestSpectra:
    def test_eigenvalues_in_increasing_order(self):
        values, vectors = linalg.eig(SPD, rows=1)
        assert np.allclose(values, np.linalg.eigvalsh(SPD))
        assert np.allclose(SPD @ vectors, vectors * values)

    def test_small_matrices_are_symmetric_only_if_they_are(self):
        small = 1e-9 * np.array([[1.0, 2.0], [3.0, 4.0]])
        values = linalg.eig(small, rows=1)[0] / 1e-9
        assert np.allclose(values, np.sort(np.linalg.eigvals(small).real) / 1e-9)
        rotation = 1e-9 * np.array([[0.0, -1.0], [1.0, 0.0]])  # its eigenvalues are ±i 1e-9
        values = np.sort_complex(linalg.eig(rotation, rows=1)[0] / 1e-9)
        assert np.allclose(values, [-1j, 1j])

    def test_relative_to_a_metric(self):
        metric = np.diag([1.0, 2.0, 3.0, 4.0])
        values, vectors = linalg.eig(SPD, rows=1, metric=metric)
        assert np.allclose(SPD @ vectors, metric @ vectors * values)

    def test_a_map_on_a_product_space(self):
        # Two spins: H^{ik}_{jl}, rows (i, k), columns (j, l)
        pauli = [
            np.array(m, dtype=complex)
            for m in ([[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]])
        ]
        h = sum(np.einsum("ij,kl->ikjl", p, p) for p in pauli)
        values, vectors = linalg.eig(h, rows=2)
        assert np.allclose(values, [-3, 1, 1, 1])
        assert vectors.shape == (2, 2, 4)

    def test_the_few_largest_or_smallest_of_a_sparse_matrix(self):
        n = 200
        path = sp.diags(
            [-np.ones(n - 1), 2 * np.ones(n), -np.ones(n - 1)], [-1, 0, 1], format="csr"
        )
        exact = np.sort(np.linalg.eigvalsh(path.toarray()))
        assert np.allclose(linalg.eig(path, rows=1, count=-2)[0], exact[:2])
        assert np.allclose(linalg.eig(path, rows=1, count=3)[0], exact[-3:])

    def test_symbolic_eigenvalues(self):
        epsilon, delta = sympy.symbols("epsilon Delta", positive=True)
        values, vectors = linalg.eig(Array([[epsilon, delta], [delta, -epsilon]]), rows=1)
        root = sqrt(delta**2 + epsilon**2)
        assert list(values) == [-root, root]
        assert vectors.shape == (2, 2)

    def test_singular_values(self):
        x = rng.normal(size=(6, 3))
        u, s, v = linalg.svd(x, rows=1)
        assert np.allclose((u * s) @ v.T, x)
        assert list(s) == sorted(s, reverse=True)


class TestOthers:
    def test_determinant_and_trace(self):
        assert np.isclose(linalg.det(SPD, 1), np.linalg.det(SPD))
        assert np.isclose(linalg.trace(SPD, 1), np.trace(SPD))
        x = Symbol("x")
        assert linalg.det(Array([[1, 0], [0, x**2]]), 1) == x**2

    @pytest.mark.parametrize(
        ("name", "reference"),
        [("exp", scipy.linalg.expm), ("sqrt", scipy.linalg.sqrtm), ("log", scipy.linalg.logm)],
    )
    def test_functions_of_a_matrix(self, name, reference):
        assert np.allclose(linalg.function(name, SPD, 1), reference(SPD))

    def test_the_exponential_of_a_symbolic_matrix(self):
        t = Symbol("t")
        rotation = linalg.function("exp", Array([[0, -t], [t, 0]]), 1)
        assert rotation == Array([[sympy.cos(t), -sympy.sin(t)], [sympy.sin(t), sympy.cos(t)]])

    def test_diagonal_and_elementwise(self):
        assert np.allclose(linalg.diag(np.array([1.0, 2.0])), np.diag([1.0, 2.0]))
        assert linalg.diag(Array([1, 2])) == Array([[1, 0], [0, 2]])
        x = rng.normal(size=(2, 3))
        assert np.allclose(linalg.hadamard(x, x), x * x)
        assert linalg.hadamard(Array([1, 2]), Array([3, 4])) == Array([3, 8])
