r"""Index notation beyond spacetime: indices over spaces of any size, data bound from
Python, statistics as contractions with a measure, linear algebra, and the domains this
opens: portfolios, quantum systems, graphs and regressions."""

import re

import numpy as np
import pandas as pd
import pytest
import scipy.linalg
import scipy.sparse as sp
import sympy
from sklearn.covariance import LedoitWolf

from relativisticpy import CompileError, EvaluationError, IndexNotationError, Workbook
from relativisticpy.tensors import Space, Tensor

rng = np.random.default_rng(11)
RETURNS = 0.001 + rng.normal(0, 0.01, (300, 5))
MEAN = RETURNS.mean(axis=0)
COVARIANCE = np.cov(RETURNS, rowvar=False, bias=True)


@pytest.fixture
def market(wb):
    wb["R_{ta}"] = RETURNS
    wb.exe(r"\mu_a := \mathbb{E}_t[R_{ta}]")
    wb.exe(r"\Sigma_{ab} := \mathbb{E}_t[(R_{ta} - \mu_a)(R_{tb} - \mu_b)]")
    return wb


class TestSpaces:
    def test_declared_in_the_notation(self, wb):
        wb.exe(r"""
            \mathcal{A} := \mathbb{R}^{3}
            a, b \in \mathcal{A}
        """)
        wb["w^a"] = np.ones(3)
        assert wb["w"].spaces == (Space("A", 3),)
        with pytest.raises(CompileError, match="ranges over A, of dimension 3"):
            wb["v^b"] = np.ones(4)

    def test_a_dimension_is_a_whole_number(self, wb):
        for size in ("n", "2.5", "-2"):
            with pytest.raises(CompileError, match="the dimension of a space is a whole number"):
                wb.exe(rf"\mathcal{{A}} := \mathbb{{R}}^{{{size}}}")
        wb.exe(r"Integers := [n]")  # an integer, but of no known size
        with pytest.raises(CompileError, match="whole number, not n"):
            wb.exe(r"\mathcal{A} := \mathbb{R}^{n}")

    def test_an_index_ranges_over_one_space(self, wb):
        wb["R_{ta}"] = RETURNS
        message = "the index a ranges over the space of a, but is written here on an axis over"
        with pytest.raises(IndexNotationError, match=message):
            wb.expr(r"R_{ta} R_{at}")

    def test_spacetime_is_unchanged(self, wb):
        wb.exe(r"g_{\mu\nu} := Schwarzschild(M)")
        wb["w^a"] = np.ones(3)
        assert wb.expr(r"R_{\mu\nu} = 0")
        with pytest.raises(IndexNotationError, match="ranges over"):
            wb.expr(r"R_{\mu a} w^a")

    def test_binding_declares_nothing(self, wb):
        # t ranges over days where it is written on R, and is a coordinate elsewhere.
        wb["R_{ta}"] = RETURNS
        wb.exe(r"g_{\mu\nu} := Schwarzschild(M)")
        m, r = sympy.symbols("M r")
        assert sympy.simplify(wb.expr(r"\Gamma^{t}_{tr}") - m / (r * (r - 2 * m))) == 0
        # and the same label may range over another number of days in other data
        wb["X_{ta}"] = RETURNS[:20]
        assert wb["X"].spaces[0].dim == 20
        message = "the space of t, of dimension 300, but is written here on an axis over the"
        with pytest.raises(IndexNotationError, match=message):
            wb.expr(r"R_{ta} X_{tb}")

    def test_only_a_named_space_declares_its_members(self, wb):
        wb.exe(r"Coordinates := [t, x, y, z]" "\n" r"g_{\mu\nu} := diag(-1, 1, 1, 1)")
        with pytest.raises(CompileError, match="read but not computed"):
            wb.exe(r"x \in \mathbb{R}^{3}")  # in a paper: x is a coordinate, still
        assert wb.expr(r"\Gamma^{x}_{xx} = 0")
        # A coordinate's name on spacetime is a component, even where it is declared.
        wb.exe(r"\mathcal{T} := \mathbb{R}^{3}" "\n" r"t \in \mathcal{T}")
        assert wb.expr(r"\Gamma^{t}_{tx} = 0")

    def test_a_square_matrix_of_data_keeps_its_spaces_apart(self, wb):
        wb["R_{ta}"] = np.eye(3)  # three days of three assets
        assert wb["R"].spaces[0] != wb["R"].spaces[1]
        with pytest.raises(IndexNotationError, match="ranges over"):
            wb.expr(r"R_{ta} R_{at}")
        wb["S_{ab}"] = np.eye(3)  # a and b are alike: a covariance is over one space
        assert wb["S"].spaces[0] == wb["S"].spaces[1]

    def test_like_indices_of_other_tensors_share_a_space(self, wb):
        wb["S_{ab}"] = np.diag([1.0, 2.0])
        wb["v^b"] = np.array([1.0, 1.0])
        assert wb.expr(r"v^a S_{ab} v^b") == pytest.approx(3.0)

    def test_component_names_come_from_the_basis(self, wb):
        tickers = ["AAPL", "MSFT", "NVDA"]
        wb["w^a"] = pd.Series([0.5, 0.3, 0.2], index=pd.Index(tickers, name="assets"))
        assert wb.expr(r"w^{a:MSFT}") == pytest.approx(0.3)
        assert wb["w"].to_pandas().index.tolist() == tickers

    def test_pandas_data_is_aligned_by_label(self, wb):
        tickers = pd.Index(["AAPL", "MSFT", "NVDA"], name="assets")
        wb["w^a"] = pd.Series([0.5, 0.3, 0.2], index=tickers)
        reordered = pd.Index(["NVDA", "AAPL", "MSFT"], name="assets")
        wb["v_b"] = pd.Series([1.0, 2.0, 3.0], index=reordered)
        assert wb.expr(r"v_{b:NVDA}") == pytest.approx(1.0)
        assert wb.expr(r"w^a v_a") == pytest.approx(0.5 * 2 + 0.3 * 3 + 0.2 * 1)
        frame = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], columns=pd.Index(["x", "y"], name="pair"))
        wb["P_{tc}"] = frame
        wb["Q_{tc}"] = frame[["y", "x"]]
        assert np.allclose(wb.expr(r"P_{tc} - Q_{tc}").to_numpy(), 0)
        other = pd.Index(["AAPL", "MSFT", "GOOG"], name="assets")
        with pytest.raises(CompileError, match="not the elements of assets: GOOG is not one"):
            wb["x_a"] = pd.Series([1.0, 2.0, 3.0], index=other)

    def test_a_measure_is_taken_by_label(self, wb):
        days = ["d1", "d2", "d3"]
        weights = pd.Series([1.0, 0.0, 0.0], index=["d3", "d2", "d1"])
        wb.space("days", basis=days, labels="t", measure=weights)
        wb["R_{ta}"] = pd.DataFrame([[1.0], [2.0], [4.0]], index=pd.Index(days, name="days"))
        assert wb.expr(r"\mathbb{E}_t[R_{ta}]").to_numpy() == pytest.approx([4.0])
        with pytest.raises(CompileError, match="x is not one; d3 is missing"):
            wb.space("days", basis=days, measure=pd.Series([1.0, 1.0, 1.0], ["d1", "d2", "x"]))


class TestData:
    def test_numpy_pandas_lists_and_sympy(self, wb):
        frame = pd.DataFrame(RETURNS[:4, :2], columns=["x", "y"])
        wb["X_{ta}"] = frame
        assert wb["X"].to_pandas().columns.tolist() == ["x", "y"]
        wb["v^a"] = [1.0, 2.0]
        assert wb.expr(r"X_{ta} v^a").to_numpy() == pytest.approx(frame.to_numpy() @ [1.0, 2.0])
        wb["s^a"] = sympy.Array([sympy.Symbol("p"), 1])
        assert wb.expr(r"s^a s^b \delta_{ab}") == sympy.Symbol("p") ** 2 + 1

    def test_like_indices_of_one_tensor_share_a_space(self, wb):
        wb["w^a"] = np.ones(4)
        wb["S_{ab}"] = np.eye(4)
        assert wb.expr(r"w^a S_{ab} w^b") == pytest.approx(4.0)

    def test_y_t_indexes_the_tensor_y(self, wb):
        wb["y_t"] = np.arange(3.0)
        assert wb.expr("y_t y_t") == pytest.approx(5.0)

    def test_what_cannot_be_bound(self, wb):
        with pytest.raises(CompileError, match="has 1 indices but the data has 2 axes"):
            wb["w^a"] = np.eye(2)
        with pytest.raises(CompileError, match="cannot bind"):
            wb["2 + x"] = 1.0

    def test_data_is_copied_when_bound(self, wb):
        x = np.array([1.0, 2.0])
        wb["w^a"] = x
        assert wb.expr(r"w^a w^b \delta_{ab}") == pytest.approx(5.0)
        x[0] = 3.0  # changed in place, as an optimiser does, then bound again
        assert wb["w"].to_numpy().tolist() == [1.0, 2.0]
        wb["w^a"] = x
        assert wb.expr(r"w^a w^b \delta_{ab}") == pytest.approx(13.0)
        frame = pd.DataFrame({"x": [1.0, 2.0]})
        wb["X_{sc}"] = frame
        frame.iloc[0, 0] = 10.0
        assert wb["X"].to_numpy().ravel().tolist() == [1.0, 2.0]

    def test_integer_data_does_not_overflow(self, wb):
        pixels = rng.integers(0, 256, size=(40, 3)).astype(np.uint8)
        wb["X_{ta}"] = pixels
        exact = pixels.astype(np.int64).T @ pixels.astype(np.int64)
        assert np.array_equal(wb.expr(r"X_{ta} X_{tb}").to_numpy(), exact)

    def test_numbers_are_plain_python_numbers(self, wb):
        wb["w^a"] = np.array([3.0, 4.0])
        value = wb.expr(r"\|w\|")
        assert type(value) is float
        assert value == pytest.approx(5.0)


class TestStatistics:
    def test_means_and_covariances(self, market):
        assert market["mu"].to_numpy() == pytest.approx(MEAN)
        assert np.allclose(market["Sigma"].to_numpy(), COVARIANCE)
        assert np.allclose(
            market.expr(r"\operatorname{Cov}_t(R_{ta}, R_{tb})").to_numpy(), COVARIANCE
        )
        assert np.allclose(
            market.expr(r"\operatorname{Var}_t(R_{ta})").to_numpy(), RETURNS.var(axis=0)
        )

    def test_an_expectation_is_one_contraction_with_the_measure(self, market):
        listing = market.ir(r"\mathbb{E}_t[(R_{ta} - \mu_a)(R_{tb} - \mu_b)]")
        assert re.search(r"contract ab,ac,a->bc", listing)

    def test_the_expectation_of_a_sum_is_taken_term_by_term(self, market):
        written = r"\mathbb{E}_t[R_{ta} R_{tb} - 2 \mu_a \mu_b] + \mu_a \mu_b"
        kinds = [line.split()[-1] for line in market.ir(written).splitlines() if "Tensor" in line]
        assert max(kind.count("{") for kind in kinds) == 2  # nothing of days, assets and assets
        assert np.allclose(market.expr(written).to_numpy(), COVARIANCE)

    def test_a_declared_measure_weights_the_expectation(self, wb):
        weights = np.linspace(1, 2, 300)
        wb.space("days", 300, labels="t", measure=weights)
        wb["R_{ta}"] = RETURNS
        expected = np.average(RETURNS, axis=0, weights=weights)
        assert np.allclose(wb.expr(r"\mathbb{E}_t[R_{ta}]").to_numpy(), expected)

    def test_ledoit_wolf_from_scikit_learn(self, wb):
        wb["R_{ta}"] = RETURNS
        shrunk = wb.expr(r"\operatorname{LedoitWolf}_t(R_{ta})").to_numpy()
        assert np.allclose(shrunk, LedoitWolf().fit(RETURNS).covariance_)

    def test_ledoit_wolf_does_not_ignore_a_measure(self, wb):
        wb.space("days", 300, labels="t", measure=np.linspace(1, 2, 300))
        wb["R_{ta}"] = RETURNS
        with pytest.raises(CompileError, match="weighted by a measure"):
            wb.expr(r"\operatorname{LedoitWolf}_t(R_{ta})")
        wb.space("days", 300, labels="t", measure=np.ones(300))  # every day alike
        shrunk = wb.expr(r"\operatorname{LedoitWolf}_t(R_{ta})").to_numpy()
        assert np.allclose(shrunk, LedoitWolf().fit(RETURNS).covariance_)

    def test_a_statistic_needs_its_index(self, market):
        with pytest.raises(IndexNotationError, match="needs the index t"):
            market.expr(r"\mathbb{E}_t[\mu_a]")


class TestLinearAlgebra:
    def test_an_inverse_times_a_vector_is_a_solve(self, market):
        expression = r"(\Sigma^{-1})^{ab} \mu_b"
        assert "solve" in market.ir(expression)
        assert np.allclose(market.expr(expression).to_numpy(), np.linalg.solve(COVARIANCE, MEAN))

    def test_the_inverse_of_a_map_is_a_map(self, wb):
        a = rng.normal(size=(3, 3)) + 3 * np.eye(3)
        metric = np.array([[2.0, 0.3, 0.0], [0.3, 1.0, 0.2], [0.0, 0.2, 1.5]])
        v = rng.normal(size=3)
        wb.exe(r"\mathcal{A} := \mathbb{R}^{3}" "\n" r"a, b, c \in \mathcal{A}")
        wb["A^a_b"] = a
        wb["G_{ab}"] = metric
        wb["v^a"] = v
        wb.exe(r"g_{ab} := G_{ab}")  # raising and lowering would change the components
        assert np.allclose(wb.expr(r"(A^{-1})^{a}{}_{b} A^{b}{}_{c}").to_numpy(), np.eye(3))
        assert np.allclose(wb.expr(r"(A^{-1})^{a}{}_{b} v^{b}").to_numpy(), np.linalg.solve(a, v))
        assert str(wb.expr(r"\operatorname{inv}(A)").indices) == "^{b}_{a}"

    def test_the_inverse_goes_between_two_spaces_the_other_way(self, wb):
        m, v = rng.normal(size=(3, 3)) + 3 * np.eye(3), rng.normal(size=3)
        wb.exe(r"""
            \mathcal{A} := \mathbb{R}^{3}
            \mathcal{B} := \mathbb{R}^{3}
            a, c \in \mathcal{A}
            b, d \in \mathcal{B}
        """)
        wb["T_{ab}"] = m
        wb["v_a"] = v
        assert [space.name for space in wb.expr(r"(T^{-1})^{ba}").spaces] == ["B", "A"]
        assert np.allclose(wb.expr(r"(T^{-1})^{ba} T_{ad}").to_numpy(), np.eye(3))
        assert np.allclose(wb.expr(r"(T^{-1})^{ba} v_a").to_numpy(), np.linalg.solve(m, v))
        with pytest.raises(IndexNotationError, match="the index a ranges over A, but"):
            wb.expr(r"(T^{-1})^{ab}")

    def test_an_inverse_is_a_solve_only_where_the_product_is_one(self, market):
        assert market.expr(r"(\Sigma^{-1})^{ab} \Sigma_{ab}") == pytest.approx(5.0)
        assert market.expr(r"(\Sigma^{-1})^{ab} \Sigma_{ba}") == pytest.approx(5.0)
        for written in [
            r"(\Sigma^{-1})^{ab} \mu_b \mu_b",
            r"\mu_b \mu_b (\Sigma^{-1})^{ab}",
            r"(\Sigma^{-1})^{ab} \mu_a \mu_a",
        ]:
            with pytest.raises(IndexNotationError, match="appears 3 times"):
                market.expr(written)

    def test_what_an_inverse_multiplies_is_contracted_first(self, wb):
        x, y = rng.normal(size=(50, 3)), rng.normal(size=50)
        wb["X_{tk}"] = x
        wb["y_t"] = y
        wb.exe(r"G_{kl} := X_{tk} X_{tl}")
        written = r"(G^{-1})^{kl} X_{tl} y_t"
        solve = next(line for line in wb.ir(written).splitlines() if "call solve" in line)
        assert solve.endswith("Tensor^{k}")  # the solve of a vector, not of a matrix of rows
        assert np.allclose(wb.expr(written).to_numpy(), np.linalg.lstsq(x, y, rcond=None)[0])

    def test_eigenvalues_eigenvectors_and_the_spectrum(self, market):
        market.exe(r"\lambda_k, v^a{}_k := \operatorname{eig}(\Sigma^a{}_b)")
        values, vectors = market["lambda"].to_numpy(), market["v"].to_numpy()
        assert np.allclose(COVARIANCE @ vectors, vectors * values)
        assert np.allclose(market.expr(r"\operatorname{spec}(\Sigma)").to_numpy(), values)
        # The eigenvectors' second index ranges over a space of their own.
        with pytest.raises(IndexNotationError, match="ranges over"):
            market.expr(r"v^a{}_k \mu_k")

    def test_a_few_eigenvalues_of_any_map(self, wb):
        upper = np.triu(rng.normal(size=(8, 8)), 1) + np.diag(np.arange(8.0))  # 0, 1, ..., 7
        wb["A^a_b"] = upper
        assert np.allclose(wb.expr(r"\operatorname{spec}(A^a{}_b, 2)").to_numpy(), [6.0, 7.0])
        with pytest.raises(CompileError, match="whole number"):
            wb.expr(r"\operatorname{spec}(A^a{}_b, 0)")
        with pytest.raises(CompileError, match="has 8 eigenvalues, not 9"):
            wb.expr(r"\operatorname{spec}(A^a{}_b, 9)")
        wb["L^i_j"] = sp.diags([-np.ones(39), 2 * np.ones(40), -np.ones(39)], [-1, 0, 1])
        spectrum = wb.expr(r"\operatorname{spec}(L^i{}_j)")  # every one, as for a dense map
        assert spectrum.to_numpy().shape == (40,)
        assert spectrum.spaces[0].dim == 40

    def test_the_index_of_a_spectrum(self, wb):
        form = np.array([[2.0, 0.5], [0.5, 1.0]])
        wb["S_{ab}"] = form
        wb.exe(r"s_k := \operatorname{spec}(S)")
        assert np.allclose(wb["s"].to_numpy(), np.linalg.eigvalsh(form))
        for written in (r"s_k 1^k", r"1^k s_k", r"1^k \operatorname{spec}(S^a{}_b)_k"):
            assert wb.expr(written) == pytest.approx(3.0)
        wb["x^i"] = np.array([1.0, 2.0, 3.0])
        assert np.allclose(wb.expr(r"1^j \operatorname{diag}(x)_{ij}").to_numpy(), [1, 2, 3])
        wb.exe(r"e_ab := \operatorname{diag}(x)")
        assert np.allclose(wb["e"].to_numpy(), np.diag([1.0, 2.0, 3.0]))

    def test_singular_values(self, market):
        market.exe(r"U_{tk}, s_k, V_{ak} := \operatorname{svd}(R_{ta})")
        u, s, v = (market[name].to_numpy() for name in "UsV")
        assert np.allclose((u * s) @ v.T, RETURNS)

    def test_determinant_trace_norm_diagonal_and_elementwise(self, market):
        assert market.expr(r"\det(\Sigma^a{}_b)") == pytest.approx(np.linalg.det(COVARIANCE))
        assert market.expr(r"\operatorname{tr}(\Sigma^a{}_b)") == pytest.approx(
            np.trace(COVARIANCE)
        )
        assert market.expr(r"\|\mu\|") == pytest.approx(np.linalg.norm(MEAN))
        market.exe(r"D_{ab} := \operatorname{diag}(\mu)")
        assert np.allclose(market["D"].to_numpy(), np.diag(MEAN))
        assert np.allclose(market.expr(r"R_{ta} \odot R_{ta}").to_numpy(), RETURNS**2)

    def test_the_elementwise_product_of_lists(self, wb):
        assert wb.expr(r"[1, 2] \odot [3, 4]") == sympy.Array([3, 8])
        assert wb.expr(r"[[1, 2], [3, 4]] \odot [[5, 6], [7, 8]]") == sympy.Array(
            [[5, 12], [21, 32]]  # not the matrix product, [[19, 22], [43, 50]]
        )
        assert wb.expr(r"2 \odot [3, 4]") == sympy.Array([6, 8])
        with pytest.raises(EvaluationError, match="of one shape"):
            wb.expr(r"[1, 2] \odot [3, 4, 5]")

    def test_a_function_of_a_map(self, wb):
        hamiltonian = np.array([[1.0, 0.5], [0.5, -1.0]])
        wb["H^i_j"] = hamiltonian
        wb["t"] = 0.7
        wb.exe(r"U := \exp(-i H t)")
        assert np.allclose(wb["U"].to_numpy(), scipy.linalg.expm(-0.7j * hamiltonian))

    def test_the_square_root_of_a_matrix(self, wb):
        covariance = np.array([[2.0, -0.5], [-0.5, 1.0]])
        wb["\\Sigma_{ab}"] = covariance
        for written in (r"\sqrt{\Sigma}", r"\sqrt(\Sigma)"):
            assert np.allclose(wb.expr(written).to_numpy(), scipy.linalg.sqrtm(covariance))
        wb["W_{ab}"] = np.array([[4.0, 1.0], [1.0, 9.0]])  # with indices: of each component
        assert np.allclose(wb.expr(r"\sqrt{W_{ab}}").to_numpy(), [[2.0, 1.0], [1.0, 3.0]])

    def test_the_logarithm_of_each_component(self, wb):
        prices = 100 * np.exp(rng.normal(0, 0.01, (6, 2)))
        wb["P_{ta}"] = prices
        for written in (r"\log(P_{ta})", r"\log P_{ta}", r"\ln(P_{ta})"):
            assert np.allclose(wb.expr(written).to_numpy(), np.log(prices))
        assert np.allclose(wb.expr(r"\log(P_{ta}, 10)").to_numpy(), np.log10(prices))

    def test_numerical_equations_hold_to_rounding_error(self, market):
        assert market.expr(r"\Sigma_{ab} = \operatorname{Cov}_t(R_{ta}, R_{tb})")
        assert not market.expr(r"\Sigma_{ab} = 2 \Sigma_{ab}")

    def test_rounding_error_is_relative_to_the_numbers(self, wb):
        wb["m_a"] = np.array([9.109e-31, 1.673e-27])  # an electron and a proton, in kg
        assert not wb.expr(r"m_{a} = 2 m_{a}")
        assert not wb.expr(r"m_{a} = 0")
        wb["P_{ta}"] = 100 + 10 * rng.normal(size=(250, 4))  # prices, in dollars
        wb.exe(r"C_{ab} := \operatorname{Cov}_t(P_{ta}, P_{tb})")
        terms = [
            r"\mathbb{E}_t[P_{ta} P_{tb}] 1^a 1^b",
            r"\mathbb{E}_t[P_{ta}] 1^a \mathbb{E}_t[P_{tb}] 1^b",
            r"C_{ab} 1^a 1^b",
        ]
        residual = " - ".join(terms)  # rounding error in numbers near 10^5, which is zero
        assert wb.expr(f"{residual} = 0") is sympy.true
        assert wb.expr(f"{residual} = 10^{{-6}}") is not sympy.true

    def test_a_zero_tensor_has_the_shape_of_its_spaces(self, wb):
        wb.space("days", 4, labels="t")
        wb.space("assets", 2, labels="a")
        wb["R_{ta}"] = np.ones((4, 2))
        wb.exe(r"Z_{ta} := 0")
        assert wb["Z"].shape == (4, 2)
        assert np.allclose(wb.expr(r"Z_{ta} + R_{ta}").to_numpy(), 1.0)
        assert wb.expr(r"R_{ta} - R_{ta} = Z_{ta}")

    def test_an_equation_of_data_need_not_be_square(self, wb):
        x, y = sympy.symbols("x y")
        wb["X_{ta}"] = sympy.Array([[x, 0], [0, y], [1, 0], [0, 1]])
        wb["Y_{ta}"] = sympy.Array([[x, 0], [0, y], [1, 0], [0, 2]])
        assert wb.expr(r"X_{ta} = Y_{ta}").equations() == [sympy.Eq(1, 2, evaluate=False)]
        assert len(wb.expr(r"X_{ta} = 0").equations()) == 4


class TestPortfolios:
    def test_markowitz_tangency(self, market):
        market.exe(r"m^a := \frac{(\Sigma^{-1})^{ab} \mu_b}{1_c (\Sigma^{-1})^{cd} \mu_d}")
        x = np.linalg.solve(COVARIANCE, MEAN)
        assert np.allclose(market["m"].to_numpy(), x / x.sum())

    def test_risk_is_a_metric(self, market):
        market["w^a"] = np.full(5, 0.2)
        market.exe(r"g_{ab} := \Sigma_{ab}")
        w = np.full(5, 0.2)
        assert market.expr(r"w^a w_a") == pytest.approx(w @ COVARIANCE @ w)
        assert np.allclose(market.expr(r"x_a := w_a" "\n" "x_a").to_numpy(), COVARIANCE @ w)
        assert np.allclose(market.expr(r"\mu^a").to_numpy(), np.linalg.solve(COVARIANCE, MEAN))
        with pytest.raises(IndexNotationError, match="once up and once down"):
            market.expr(r"w_{a} w_{a}")

    def test_the_gradient_of_the_sharpe_ratio(self, market):
        market["w^a"] = np.full(5, 0.2)
        gradient = market.grad(r"\frac{\mu_a w^a}{\sqrt{w^a \Sigma_{ab} w^b}}", "w").to_numpy()
        w = np.full(5, 0.2)
        variance = w @ COVARIANCE @ w
        expected = MEAN / np.sqrt(variance) - (MEAN @ w) * (COVARIANCE @ w) / variance**1.5
        assert np.allclose(gradient, expected)


class TestQuantum:
    def test_two_spins(self, wb):
        wb.exe(r"\mathcal{H} := \mathbb{C}^{2}" "\n" r"i, j, k, l \in \mathcal{H}")
        wb["X^i_j"] = np.array([[0, 1], [1, 0]], dtype=complex)
        wb["Y^i_j"] = np.array([[0, -1j], [1j, 0]])
        wb["Z^i_j"] = np.array([[1, 0], [0, -1]], dtype=complex)
        wb.exe(r"H^{ik}{}_{jl} := X^i{}_j X^k{}_l + Y^i{}_j Y^k{}_l + Z^i{}_j Z^k{}_l")
        spectrum = wb.expr(r"\operatorname{spec}(H^{ik}{}_{jl})").to_numpy()
        assert np.allclose(spectrum, [-3, 1, 1, 1])

    def test_complex_components_are_conjugated(self, wb):
        wb.exe(r"\mathcal{H} := \mathbb{C}^{2}" "\n" r"i, j \in \mathcal{H}")
        wb["\\psi^i"] = np.array([1.0, 1.0j]) / np.sqrt(2)
        assert wb.expr(r"\|\psi\|") == pytest.approx(1.0)
        wb.exe(r"\phi^{i} := [1, i]")
        assert wb.expr(r"\|\phi\|") == sympy.sqrt(2)
        z = rng.normal(size=(200, 3)) + 1j * rng.normal(size=(200, 3))
        wb["Z_{ta}"] = z
        variance = wb.expr(r"\operatorname{Var}_t(Z_{ta})").to_numpy()
        assert np.isrealobj(variance)
        assert np.allclose(variance, z.var(axis=0))
        covariance = wb.expr(r"\operatorname{Cov}_t(Z_{ta}, Z_{tb})").to_numpy()
        assert np.allclose(covariance, np.cov(z, rowvar=False, bias=True))
        wb.exe(r"U_{tk}, s_k, V_{ak} := \operatorname{svd}(Z_{ta})")
        assert np.allclose(wb.expr(r"U_{tk} \operatorname{diag}(s)_{kl} V_{al}").to_numpy(), z)

    def test_a_symbolic_two_level_system(self, wb):
        wb.exe(r"""
            \mathcal{H} := \mathbb{C}^{2}
            i, j \in \mathcal{H}
            H^{i}{}_{j} := [[\epsilon, \Delta], [\Delta, -\epsilon]]
        """)
        epsilon, delta = sympy.symbols("epsilon Delta")
        root = sympy.sqrt(delta**2 + epsilon**2)
        assert list(wb.expr(r"\operatorname{spec}(H^i{}_j)").components) == [-root, root]


class TestGraphs:
    def test_the_fiedler_vector_cuts_the_graph(self, wb):
        adjacency = np.zeros((6, 6))
        for i, j in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)]:
            adjacency[i, j] = adjacency[j, i] = 1
        wb["A_{ij}"] = adjacency
        wb.exe(r"k_i := A_{ij} 1^j")
        wb.exe(r"L_{ij} := \operatorname{diag}(k)_{ij} - A_{ij}")
        wb.exe(r"\lambda_k, v^i{}_k := \operatorname{eig}(L^i{}_j)")
        fiedler = wb["v"].to_numpy()[:, 1]
        assert np.sign(fiedler[:3]).tolist() == [np.sign(fiedler[0])] * 3
        assert np.sign(fiedler[3:]).tolist() == [-np.sign(fiedler[0])] * 3

    def test_a_large_sparse_graph(self, wb):
        n = 1_000
        ring = sp.diags([np.ones(n - 1), np.ones(n - 1)], [1, -1], format="lil")
        ring[0, n - 1] = ring[n - 1, 0] = 1
        wb["L^i_j"] = sp.csr_matrix(sp.diags(np.asarray(ring.sum(1)).ravel()) - ring)
        smallest = np.sort(wb.expr(r"\operatorname{spec}(L^i{}_j, -3)").to_numpy())
        gap = 2 - 2 * np.cos(2 * np.pi / n)
        assert np.allclose(smallest, [0, gap, gap], atol=1e-8)


class TestSparse:
    def test_sparse_data_is_never_densified(self, wb):
        n = 200_000  # dense, 300 GB
        path = sp.diags([-np.ones(n - 1), 2 * np.ones(n), -np.ones(n - 1)], [-1, 0, 1])
        wb["L^i_j"] = path.tocsr()
        assert sp.issparse(wb.expr(r"L^i{}_j L^j{}_k").components)
        assert wb.expr(r"L^{i=0}{}_{j} 1^j") == pytest.approx(1.0)
        assert wb.expr(r"\operatorname{tr}(L^i{}_j)") == pytest.approx(2.0 * n)

    def test_sums_and_elementwise_products(self, wb):
        a = sp.random(5, 5, density=0.5, random_state=1, format="csr")
        b = rng.normal(size=(5, 5))
        wb["A_{ij}"] = a
        wb["B_{ij}"] = b
        assert np.allclose(wb.expr(r"A_{ij} \odot B_{ij}").to_numpy(), a.toarray() * b)
        wb.exe(r"C_{ij} := A_{ij} + B_{ij}")
        assert type(wb["C"].components) is np.ndarray  # not np.matrix
        assert np.allclose(wb.expr(r"C_{i=0 j}").to_numpy(), (a.toarray() + b)[0])
        variance = wb.expr(r"\operatorname{Var}_i(A_{ij})").to_numpy()
        assert np.allclose(variance, a.toarray().var(axis=0))

    def test_gradients_and_jax_with_a_sparse_matrix(self, wb):
        path = sp.csr_matrix(np.diag(2 * np.ones(5)) - np.diag(np.ones(4), 1))
        v = rng.normal(size=5)
        wb["L^i_j"] = path
        wb["v^i"] = v
        gradient = wb.grad(r"v^i L_{ij} v^j", "v").to_numpy()
        assert np.allclose(gradient, (path + path.T) @ v)
        with Workbook(backend="jax") as jax_wb:
            jax_wb["L^i_j"] = path
            jax_wb["v^i"] = v
            assert np.allclose(jax_wb.expr(r"L^i{}_j v^j").to_numpy(), path @ v)


class TestRegression:
    def test_the_normal_equations(self, wb):
        x = rng.normal(size=(200, 4))
        y = x @ np.array([1.0, -2.0, 0.5, 3.0]) + rng.normal(0, 0.1, 200)
        wb["X_{tk}"] = x
        wb["y_t"] = y
        wb.exe(r"G_{kl} := X_{tk} X_{tl}")
        wb.exe(r"\beta^k := (G^{-1})^{kl} X_{tl} y_t")
        assert np.allclose(wb["beta"].to_numpy(), np.linalg.lstsq(x, y, rcond=None)[0])


class TestJax:
    def test_the_jax_backend(self):
        with Workbook(backend="jax") as wb:
            wb["S_{ab}"] = COVARIANCE
            wb["w^a"] = np.full(5, 0.2)
            assert wb.expr(r"w^a S_{ab} w^b") == pytest.approx(0.04 * COVARIANCE.sum())
            assert type(wb.expr(r"S_{ab} w^b").components).__module__.startswith("jax")

    def test_spectra_with_jax(self):
        form, metric = np.array([[2.0, 0.5], [0.5, 1.0]]), np.diag([4.0, 1.0])
        with Workbook(backend="jax") as wb:
            wb.exe(r"\mathcal{A} := \mathbb{R}^{2}" "\n" r"a, b \in \mathcal{A}")
            wb["A^a_b"] = np.array([[1.0, 2.0], [0.0, 3.0]])
            wb["S_{ab}"] = form
            wb["G_{ab}"] = metric
            wb.exe(r"g_{ab} := G_{ab}")
            assert np.allclose(wb.expr(r"\operatorname{spec}(A^a{}_b)").to_numpy(), [1.0, 3.0])
            expected = scipy.linalg.eigh(form, metric, eigvals_only=True)
            assert np.allclose(wb.expr(r"\operatorname{spec}(S)").to_numpy(), expected)
            assert np.allclose(wb.expr(r"\operatorname{spec}(S, 1)").to_numpy(), expected[1:])

    def test_gradients_through_a_solve(self, market):
        gradient = market.grad(r"(\Sigma^{-1})^{ab} \mu_a \mu_b", "mu").to_numpy()
        assert np.allclose(gradient, 2 * np.linalg.solve(COVARIANCE, MEAN))

    def test_a_gradient_with_one_array_bound_to_two_names(self, wb):
        w = np.array([1.0, 2.0, 3.0])
        wb["w^a"] = w
        wb["b_a"] = w
        assert np.allclose(wb.grad(r"w^a b_a", "w").to_numpy(), w)

    def test_a_gradient_is_of_a_scalar(self, market):
        with pytest.raises(ValueError, match="scalar"):
            market.grad(r"\Sigma_{ab} \mu^b", "mu")


def test_results_are_tensors_with_their_spaces(market):
    sigma = market["Sigma"]
    assert isinstance(sigma, Tensor)
    assert sigma.numerical
    assert [space.name for space in sigma.spaces] == ["the space of a", "the space of a"]
    assert market["R"].latex() == r"R_{t a} \in \mathbb{R}^{300 \times 5}"
    first = f"{MEAN[0]:.6g}"  # six significant figures
    assert market.expr(r"\mu_a").latex().startswith(rf"\mu_{{a}} = \left[\begin{{matrix}}{first}")
