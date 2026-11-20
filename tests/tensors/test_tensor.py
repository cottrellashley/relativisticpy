"""The Tensor value: index notation in Python."""

import pytest
import sympy
from sympy import Array, symbols

from relativisticpy.errors import IndexNotationError
from relativisticpy.tensors import Index, Indices, Tensor, contract

t, x = symbols("t x")
ETA = Tensor([[-1, 0], [0, 1]], "_{a b}", (t, x), name="eta")


class TestIndices:
    def test_str_is_the_plain_form(self):
        assert str(Indices([Index("a", True), Index("b"), Index("c")])) == "^{a}_{b}_{c}"

    def test_latex_groups_neighbours(self):
        indices = Indices([Index("mu", True), Index("nu"), Index("rho")])
        assert indices.latex() == r"^{\mu}{}_{\nu \rho}"
        assert indices.latex(["t", "t", "r"]) == "^{t}{}_{t r}"

    def test_flip(self):
        assert Index("a").flip() == Index("a", True)

    def test_parsed_from_text(self):
        assert Tensor([1, 2], "^a").indices == Indices([Index("a", True)])
        assert str(Tensor(Array([[1, 0], [0, 1]]), r"^{\mu}_{\nu}").indices) == "^{mu}_{nu}"

    def test_bad_index_text(self):
        with pytest.raises(ValueError, match="cannot read indices"):
            Tensor([1, 2], "+")
        with pytest.raises(ValueError, match="index labels only"):
            Tensor([1, 2], "_{a=0}")


class TestTensor:
    def test_shape(self):
        assert (ETA.rank, ETA.dim, ETA.shape) == (2, 2, (2, 2))
        assert ETA[0, 0] == -1
        assert len(ETA) == 4

    def test_str_is_the_components(self):
        assert str(ETA) == "[[-1, 0], [0, 1]]"
        assert repr(ETA) == "Tensor('_{a}_{b}', [[-1, 0], [0, 1]])"

    def test_shape_must_fit_the_indices(self):
        with pytest.raises(ValueError, match="do not fit"):
            Tensor([1, 2], "_{a b}")
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

    def test_tensor_plus_number_is_undefined(self):
        with pytest.raises(TypeError):
            ETA + 1

    def test_einstein_rules(self):
        v = Tensor([1, 2], "_a")
        with pytest.raises(IndexNotationError, match="twice as an lower index"):
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

    def test_nonzero_uses_coordinate_names(self):
        assert ETA.nonzero() == {("t", "t"): -1, ("x", "x"): 1}

    def test_latex(self):
        assert ETA.latex() == r"\eta_{a b} = \left[\begin{matrix}-1 & 0\\0 & 1\end{matrix}\right]"
        gamma = Tensor(Array.zeros(2, 2, 2), "^{a}_{b c}", (t, x), name="Gamma")
        assert gamma.latex() == r"\Gamma^{a}{}_{b c} = 0"
        assert ETA._repr_latex_().startswith("$\\displaystyle")

    def test_equality_and_hash(self):
        same = Tensor([[-1, 0], [0, 1]], "_{a b}")
        assert same == ETA
        assert hash(ETA) == hash(same)
        assert Tensor([[-1, 0], [0, 1]], "_{a c}") != ETA
        assert ETA.__eq__(1) is NotImplemented
