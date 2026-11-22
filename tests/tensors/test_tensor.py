"""The Tensor value: index notation in Python."""

import numpy as np
import pytest
import sympy
from sympy import Array, symbols

from relativisticpy import display
from relativisticpy.errors import EvaluationError, IndexNotationError
from relativisticpy.tensors import Index, Indices, Space, Tensor, contract

t, x = symbols("t x")
ETA = Tensor([[-1, 0], [0, 1]], "_{a b}", (t, x), name="eta")


class TestIndices:
    def test_str_is_the_text_the_language_reads(self):  # API-19
        assert str(Indices([Index("a", True), Index("b"), Index("c")])) == "^{a}_{bc}"
        assert str(Indices([Index("mu"), Index("nu")])) == r"_{\mu\nu}"
        assert str(Index("mu", True)) == r"^{\mu}"

    def test_latex_groups_neighbours(self):
        indices = Indices([Index("mu", True), Index("nu"), Index("rho")])
        assert indices.latex() == r"^{\mu}{}_{\nu\rho}"
        assert indices.latex(["t", "t", "r"]) == "^{t}{}_{tr}"
        assert Indices([Index("mu"), Index("a")]).latex() == r"_{\mu a}"

    def test_unicode(self):
        assert Indices("^{a}_{bc}").unicode() == "^a_{bc}"
        assert Indices(r"_{\mu\nu}").unicode() == "_{μν}"
        assert Indices("^{a}_{b}").unicode() == "^a_{b}"  # ^a_b would read as ^{a_b}

    def test_flip(self):
        assert Index("a").flip() == Index("a", True)

    def test_parsed_from_text(self):
        assert Tensor([1, 2], "^a").indices == Indices([Index("a", True)])
        assert str(Tensor(Array([[1, 0], [0, 1]]), r"^{\mu}_{\nu}").indices) == r"^{\mu}_{\nu}"

    def test_indices_read_text_as_a_tensor_does(self):  # API-32
        assert Indices("^{mu}") == Indices([Index("mu", True)])
        assert Indices("^{mu}").names == ("mu",)
        with pytest.raises(TypeError, match="Index values"):
            Indices(["^", "{"])

    def test_bad_index_text(self):
        with pytest.raises(ValueError, match="cannot read indices"):
            Tensor([1, 2], "+")
        with pytest.raises(ValueError, match="index labels only"):
            Tensor([1, 2], "_{a=0}")


class TestTensor:
    def test_shape(self):
        assert (ETA.rank, ETA.dim, ETA.shape, ETA.size, ETA.ndim) == (2, 2, (2, 2), 4, 2)
        assert ETA[0, 0] == -1
        assert len(ETA) == 2  # API-25: as NumPy, the first axis

    def test_len_and_iteration_agree(self):  # API-25
        gamma = Tensor(Array.zeros(4, 4, 4), "^{a}_{bc}")
        assert len(gamma) == len(list(gamma)) == 4
        numerical = Tensor(np.zeros((3, 3)), "_{ab}", spaces=(Space("s", 3), Space("s", 3)))
        assert len(numerical) == len(list(numerical)) == 3
        assert len(Tensor(5, "")) == 1

    def test_repr_shows_the_name_and_each_component(self):  # API-5, API-6, DOC-30
        assert repr(ETA) == (
            "η_{ab} in (t, x): 2 of 4 components non-zero\n  η_{tt} = −1\n  η_{xx} = 1"
        )
        assert str(ETA) == repr(ETA)

    def test_the_components_come_first(self):  # API-6
        with pytest.raises(TypeError, match=r"Did you mean Tensor\(\[-2, 3\], '_\{a\}'\)"):
            Tensor("_{a}", [-2, 3])

    def test_a_label_appears_once(self):  # API-32
        with pytest.raises(IndexNotationError, match="index a appears twice"):
            Tensor([[-1, 0], [0, 1]], "_{aa}", (t, x))

    def test_shape_must_fit_the_indices(self):  # API-32
        with pytest.raises(ValueError, match="2 axes, but there are 3 indices"):
            Tensor([[1, 2], [3, 4]], "_{a b c}")
        with pytest.raises(ValueError, match=r"must be square.*unless their spaces are given"):
            Tensor([[1, 2, 3]] * 5, "_{ta}")
        with pytest.raises(ValueError, match="coordinates"):
            Tensor([1, 2, 3], "_a", (t, x))

    def test_contraction(self):
        v = Tensor([2, 3], "^{b}", (t, x))
        lowered = ETA * v
        assert lowered == Tensor([-2, 3], "_{a}")
        assert lowered.coordinates == (t, x)

    def test_full_contraction_is_a_scalar(self):
        v = Tensor([2, 3], "^{a}", (t, x))
        assert contract(ETA, v, v.relabel("^{b}")) == -4 + 9

    def test_addition_aligns_by_name(self):
        a = Tensor([[1, 2], [3, 4]], "_{a b}")
        b = Tensor([[1, 2], [3, 4]], "_{b a}")
        assert (a + b).components == Array([[2, 5], [5, 8]])
        assert (a - b).components == Array([[0, -1], [1, 0]])

    def test_equality_aligns_by_name(self):
        a = Tensor([[1, 2], [3, 4]], "_{a b}")
        assert a == Tensor([[1, 3], [2, 4]], "_{b a}")
        assert a == a.T  # the same tensor, its axes in the other order
        assert a.T.to_numpy().tolist() == [[1, 3], [2, 4]]

    def test_sum_with_mismatched_indices(self):
        with pytest.raises(IndexNotationError, match="must agree"):
            Tensor([1, 2], "_a") + Tensor([1, 2], "_b")

    def test_scalars(self):
        assert (ETA * 2).components == Array([[-2, 0], [0, 2]])
        assert (2 * ETA).components == Array([[-2, 0], [0, 2]])
        assert (ETA / 2)[1, 1] == sympy.Rational(1, 2)
        assert (-ETA)[0, 0] == 1
        scalar = Tensor(3, "")
        assert (scalar + 1).components == 4
        assert (1 + scalar).components == 4
        assert (scalar - 1).components == 2
        assert (5 - scalar).components == 2

    def test_a_result_is_named_by_what_it_is_the_value_of(self):  # API-26, CLI-10
        assert (2 * ETA).name is None
        assert (2 * ETA).source == r"2 \eta_{ab}"
        assert (ETA + ETA).source == r"\eta_{ab} + \eta_{ab}"
        v = Tensor([2, 3], "^{b}", (t, x), name="v")
        assert (ETA * v).source == r"\eta_{ab} v^{b}"
        assert (ETA * Tensor([2, 3], "^{b}", (t, x))).source is None

    def test_tensor_plus_number_is_undefined(self):
        with pytest.raises(TypeError):
            ETA + 1

    def test_einstein_rules(self):  # and API-38's grammar
        v = Tensor([1, 2], "_a")
        with pytest.raises(IndexNotationError, match="twice as a lower index"):
            v * v
        with pytest.raises(IndexNotationError, match="appears 3 times"):
            contract(v, v.relabel("_a"), Tensor([1, 2], "^a"))

    def test_relabel_keeps_positions(self):
        assert ETA.relabel("_{c d}").indices.names == ("c", "d")
        with pytest.raises(IndexNotationError, match="positions differ"):
            ETA.relabel("^{c}_{d}")

    def test_component_functions(self):
        f = Tensor([[x**2 - x * x + t, 0], [0, 1]], "_{a b}")
        assert f.simplify()[0, 0] == t
        assert f.subs(t, 5)[0, 0] == 5
        assert f.applyfunc(lambda c: c * 2)[1, 1] == 2
        assert f.diff(t)[0, 0] == 1
        assert f.free_symbols == {t}

    def test_nonzero_uses_coordinate_names(self):
        assert ETA.nonzero() == {("t", "t"): -1, ("x", "x"): 1}

    def test_latex(self):
        assert (
            ETA.latex()
            == "\\begin{aligned}\n\\eta_{tt} &= -1 \\\\\n\\eta_{xx} &= 1\n\\end{aligned}"
        )
        gamma = Tensor(Array.zeros(2, 2, 2), "^{a}_{b c}", (t, x), name="Gamma")
        assert gamma.latex() == r"\Gamma^{a}{}_{bc} = 0"
        assert ETA._repr_latex_().startswith("$\\displaystyle")

    def test_sympy_and_numpy_idioms(self):  # API-24
        assert sympy.latex(ETA) == ETA.latex()
        assert sympy.sympify(ETA) == Array([[-1, 0], [0, 1]])
        assert np.asarray(ETA).shape == (2, 2)
        assert ETA.dtype == np.dtype(object)
        data = Tensor(np.array([1.0, 4.0]), "^{a}", spaces=(Space("s", 2),))
        assert data.dtype == np.float64
        root = np.sqrt(data)
        assert isinstance(root, Tensor)
        assert root.indices == data.indices
        assert np.allclose(root.to_numpy(), [1.0, 2.0])
        assert (np.float64(2.0) * data).to_numpy().tolist() == [2.0, 8.0]
        assert (np.float64(2.0) * ETA)[0, 0] == sympy.Float(-2.0)

    def test_numerical_tensors_across_spaces(self):
        days, assets = Space("days", 6), Space("assets", 3)
        returns = Tensor(np.arange(18.0).reshape(6, 3), "_{t a}", spaces=(days, assets))
        weights = Tensor(np.ones(3), "^{a}", spaces=(assets,))
        portfolio = returns * weights
        assert np.allclose(portfolio.components, returns.components.sum(axis=1))
        assert portfolio.spaces == (days,)
        with pytest.raises(IndexNotationError, match="3 components in one place and 6"):
            returns * Tensor(np.ones(6), "^{a}")
        other = Space("other", 3)
        with pytest.raises(IndexNotationError, match="ranges over assets in one place"):
            returns * Tensor(np.ones(3), "^{a}", spaces=(other,))

    def test_numbers_keep_numerical_components_numbers(self):
        w = Tensor(np.array([1.0, 2.0]), "^{a}")
        for result in (w * 2, 2 * w, w / 2):
            assert result.components.dtype == np.float64
        assert np.allclose((w * 2).components, [2.0, 4.0])

    def test_equal_numerical_tensors_have_equal_hashes(self):
        w = Tensor(np.array([1.0, 2.0]), "^{a}")
        nearly = Tensor(np.array([1.0, 2.0 + 1e-15]), "^{a}")
        assert w == nearly
        assert hash(w) == hash(nearly)
        assert w != Tensor(np.array([1.0]), "^{a}")
        assert w != Tensor([x, 1], "^{a}")  # a symbol is no number

    def test_equality_and_hash(self):
        same = Tensor([[-1, 0], [0, 1]], "_{a b}")
        assert same == ETA
        assert hash(ETA) == hash(same)
        assert Tensor([[-1, 0], [0, 1]], "_{a c}") != ETA
        assert ETA.__eq__(1) is NotImplemented


class TestComponentsByName:  # API-13
    @pytest.fixture
    def gamma(self):
        from relativisticpy import spacetimes

        return spacetimes.schwarzschild().christoffel

    def test_by_coordinate(self, gamma):
        _M, r, _theta = symbols("M r theta")
        assert gamma["r", "t", "t"] == gamma[1, 0, 0]
        assert (
            gamma["theta", "phi", "phi"] == gamma["θ", "φ", "φ"] == gamma[r"\theta", "phi", "phi"]
        )
        assert gamma[r, symbols("t"), symbols("t")] == gamma[1, 0, 0]

    def test_a_slice_is_a_tensor_that_keeps_what_it_fixes(self, gamma):  # API-9
        row = gamma["r", "t", :]
        assert isinstance(row, Tensor)
        assert row.indices == Indices("_{c}")
        assert row.source == r"\Gamma^{r}_{tc}"
        assert display.unicode(row).splitlines()[0].startswith("Γ^r_{tc}")

    def test_an_unknown_name_says_what_the_names_are(self, gamma):
        with pytest.raises(KeyError, match="'z' is not a coordinate; the coordinates are t, r"):
            gamma["z", "t", "t"]

    def test_by_an_element_of_a_basis(self):
        assets = Space("assets", 3, basis=("AAPL", "MSFT", "NVDA"))
        w = Tensor(np.array([0.5, 0.3, 0.2]), "^{a}", spaces=(assets,))
        assert w["MSFT"] == 0.3
        with pytest.raises(KeyError, match="'GOOG' is not an element of the basis"):
            w["GOOG"]

    def test_unlabelled_axes_are_numbered(self):  # API-35
        days = Space("days", 5)
        data = Tensor(np.ones((5, 3)), "_{ta}", spaces=(days, Space("assets", 3)))
        assert next(iter(data.nonzero())) == (0, 0)
        assert data.to_pandas().loc[0].tolist() == [1.0, 1.0, 1.0]
        with pytest.raises(ValueError, match="different sizes, 5 × 3: use shape"):
            data.dim  # noqa: B018


class TestNumbers:  # API-14
    def test_symbolic_components_need_values(self):
        M, r = symbols("M r")
        g = Tensor([[2 * M / r - 1, 0], [0, 1]], "_{ab}", (sympy.Symbol("t"), r), name="g")
        with pytest.raises(EvaluationError, match=r"g_\{ab\} has symbolic components \(in r, M\)"):
            g.to_numpy()
        assert g.to_numpy(M=1, r=4).tolist() == [[-0.5, 0.0], [0.0, 1.0]]

    def test_a_tensor_over_spaces_of_different_sizes(self):  # CLI-4
        t_space, a_space = Space("T", 3), Space("A", 2)
        data = Tensor([[1, 2], [3, 4], [5, 6]], "_{ta}", spaces=(t_space, a_space), name="R")
        assert data.nonzero()[(2, 1)] == 6
        assert "R_{ta} over T × A: 6 components non-zero" in display.unicode(data)
