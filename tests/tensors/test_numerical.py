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

    def test_small_integers_do_not_overflow(self):
        pixels = rng.integers(0, 256, size=(40, 3)).astype(np.uint8)
        gram = numerical.contract([(pixels, (0, 1)), (pixels, (0, 2))], (1, 2))
        assert np.array_equal(gram, pixels.astype(np.int64).T @ pixels.astype(np.int64))
        mask = rng.integers(0, 2, size=(20, 3)).astype(bool)
        counts = numerical.contract([(mask, (0, 1)), (mask, (0, 2))], (1, 2))
        assert np.array_equal(counts, mask.astype(np.int64).T @ mask.astype(np.int64))

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

    def test_a_function_computes_with_its_own_numeric_implementation(self):
        x = np.array([0.0, 1.0])
        exp = numerical.apply(x, sympy.exp, numeric=lambda xp, array: xp.exp(array))
        assert np.allclose(exp, np.exp(x))
        # Without one, the function is translated, or applied to each component.
        assert np.allclose(numerical.apply(x, lambda v: v**2 + 1), x**2 + 1)

    def test_a_function_numpy_lacks_is_translated_once(self):
        import scipy.special

        calls = []

        def erf(value):
            calls.append(value)
            return sympy.erf(value)

        x = rng.normal(size=(50, 20))
        assert np.allclose(numerical.apply(x, erf), scipy.special.erf(x))
        assert len(calls) == 1  # into SciPy's erf, rather than once for every component
        assert np.allclose(numerical.apply(x, lambda value: sympy.S(2)), 2.0)

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


class TestValuesPastTheRealDomain:
    """Data has the value exact numbers have where an array leaves it undefined, nan, or
    gives the real part of a complex value (the review of the engines, 6)."""

    def test_a_function_is_continued_to_complex_numbers(self):
        found = numerical.continued(np, np.sqrt, np.array([4.0, -4.0]))
        assert np.allclose(found, [2, 2j])
        assert numerical.continued(np, np.sqrt, np.array([4.0, 9.0])).dtype.kind == "f"

    def test_a_real_part_alone_is_continued_below_a_number(self):
        from scipy.special import sici

        found = numerical.continued(np, lambda x: sici(x)[1], np.array([1.0, -1.0]), below=0)
        assert np.allclose(found, [0.33740392290096816, 0.33740392290096816 + np.pi * 1j])

    def test_what_numeric_leaves_undefined_is_computed_through_sympy(self):
        x = np.array([0.5, 2.0])
        found = numerical.apply(x, sympy.asin, numeric=lambda xp, array: xp.arcsin(array))
        assert np.allclose(found, [np.arcsin(0.5), complex(sympy.asin(2))])
        assert complex(sympy.asin(2)).imag < 0  # SymPy's side of the cut, not NumPy's

    def test_complex_data_numeric_cannot_take_is_computed_through_sympy(self):
        from scipy.special import ellipk

        z = np.array([0.5 + 1.0j, 2.0 - 3.0j])
        found = numerical.apply(z, sympy.elliptic_k, numeric=lambda xp, array: ellipk(array))
        assert np.allclose(found, [complex(sympy.elliptic_k(sympy.sympify(v)).n()) for v in z])
        with pytest.raises(TypeError):  # real data: the failure is numeric's own
            numerical.apply(np.array([0.5]), sympy.sin, numeric=lambda xp, array: len(xp))

    def test_a_number_missing_from_the_data_stays_missing(self):
        x = np.array([np.nan, 4.0, np.inf])
        found = numerical.apply(x, sympy.sqrt, numeric=lambda xp, array: xp.sqrt(array))
        assert np.isnan(found[0])
        assert found[1] == 2.0
        assert np.isinf(found[2])

    def test_the_arrays_of_jax_have_the_same_values(self):
        jnp = pytest.importorskip("jax.numpy")
        numerical.double_precision(pytest.importorskip("jax"))
        x = jnp.asarray([0.5, 2.0, -4.0])
        asin = numerical.apply(x, sympy.asin, numeric=lambda xp, array: xp.arcsin(array))
        assert np.allclose(np.asarray(asin)[:2], [np.arcsin(0.5), complex(sympy.asin(2))])
        assert numerical.is_jax(asin)
        assert np.allclose(np.asarray(numerical.continued(jnp, jnp.sqrt, x))[2], 2j)

    def test_what_jax_traces_is_as_numeric_computes_it(self):
        jax = pytest.importorskip("jax")
        numerical.double_precision(jax)

        def total(x):
            root = numerical.apply(
                x, sympy.sqrt, numeric=lambda xp, a: numerical.continued(xp, xp.sqrt, a)
            )
            return root.sum()

        gradient = jax.grad(total)(jax.numpy.asarray([4.0, 9.0]))
        assert np.allclose(np.asarray(gradient), [0.25, 1 / 6])
