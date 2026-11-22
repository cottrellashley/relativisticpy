"""Binding data from Python into a session: what a written name binds, the data read from
a value, the space each axis ranges over, and the spaces declared."""

import numpy as np
import pandas as pd
import pytest
import sympy

from relativisticpy.compiler import Session
from relativisticpy.compiler.binding import (
    bind_data,
    data_of,
    data_tensor,
    declare_space,
    in_basis_order,
    labelled_spaces,
    rank_of,
    value_node,
    written_name,
    written_tensor,
)
from relativisticpy.errors import CompileError, DataError, UsageError
from relativisticpy.tensors import Space


def bound(session, written, value):
    """Bind ``value`` to ``written``, as ``wb[written] = value`` binds data."""
    name, entries = written_tensor(written, rank_of(value))
    bind_data(session, written, name, entries, data_of(value, written), "numpy")
    return session.globals.lookup(name)


def spaces(binding):
    return [str(space) for space in binding.node.type.spaces]


class TestWhatIsWritten:
    @pytest.mark.parametrize(
        ("written", "named"),
        [
            ("x", ("x", None)),
            ("R_{ta}", ("R", [("t", False), ("a", False)])),
            ("w^a", ("w", [("a", True)])),
            ("B_a^b", ("B", [("a", False), ("b", True)])),
            ("X^i_j", ("X", [("i", True), ("j", False)])),
        ],
    )
    def test_a_name_or_a_tensor_with_its_index_labels(self, written, named):
        assert written_name(written) == named

    @pytest.mark.parametrize("written", ["x + 1", "R_{aa}", "f(x)"])
    def test_what_is_neither_is_refused(self, written):
        with pytest.raises(DataError, match="write a name, as in x, or a tensor"):
            written_name(written)

    def test_data_with_axes_binds_to_y_t_as_the_tensor_y(self):
        assert written_tensor("y_t", 1) == ("y", [("t", False)])
        assert written_tensor("y_t", 0) is None  # a number, bound to the name y_t
        assert written_tensor("x", 2) is None

    def test_the_rank_of_a_value(self):
        assert [rank_of(value) for value in (2.0, [1, 2], np.ones((2, 3)))] == [0, 1, 2]


class TestData:
    def test_pandas_labels_and_axis_names_are_read_and_the_numbers_copied(self):
        frame = pd.DataFrame(
            [[1, 2]], index=pd.Index(["d1"], name="days"), columns=["AAPL", "MSFT"]
        )
        components, bases, names = data_of(frame, "R_{ta}")
        assert bases == [("d1",), ("AAPL", "MSFT")]
        assert names == ["days", None]
        frame.iloc[0, 0] = 9
        assert components[0, 0] == 1

    def test_text_is_never_data(self):
        with pytest.raises(DataError, match="holds text"):
            data_of([1, "2"], "w^a")

    def test_a_number_or_an_expression_is_a_constant(self):
        session = Session()
        assert value_node(session, "J", 1.5).op.value == 1.5
        assert value_node(session, "J", sympy.Symbol("x") + 1).op.value == sympy.Symbol("x") + 1
        with pytest.raises(DataError, match="an array is bound to a tensor"):
            value_node(session, "J", np.ones(2))
        with pytest.raises(DataError, match="not dict"):
            value_node(session, "J", {})


class TestSpaces:
    def test_like_indices_share_a_space_and_others_have_their_own(self):
        session = Session()
        assert spaces(bound(session, "S_{ab}", np.eye(2))) == ["the space of a"] * 2
        assert spaces(bound(session, "R_{ta}", np.ones((3, 2)))) == [
            "the space of t",
            "the space of a",
        ]

    def test_a_declared_label_ranges_over_its_space(self):
        session = Session()
        assets = declare_space(session, "assets", 2, labels="ab")
        binding = bound(session, "w^a", np.ones(2))
        assert binding.node.type.spaces == (assets,)

    def test_labelled_data_meets_labelled_data_by_label(self):
        session = Session()
        bound(session, "w^a", pd.Series([1.0, 2.0], index=["AAPL", "MSFT"]))
        later = bound(session, "v^b", pd.Series([3.0, 4.0], index=["MSFT", "AAPL"]))
        assert set(labelled_spaces(session, "v", "b", ())) == set(later.node.type.spaces)
        assert spaces(later) == ["the space of a"]
        assert list(later.node.op.value) == [4.0, 3.0]  # in the order of the space's basis

    def test_the_node_of_data_without_binding_it(self):
        session = Session()
        node, indices = data_tensor(
            session, "x_t", "x", [("t", False)], data_of([1, 2, 3], "x_t"), "numpy"
        )
        assert str(indices) == "_{t}"
        assert session.globals.lookup("x") is None
        assert node.type.spaces[0].dim == 3

    def test_data_in_the_order_of_a_basis(self):
        space = Space("assets", 2, basis=("AAPL", "MSFT"))
        ordered = in_basis_order(np.array([1.0, 2.0]), 0, ("MSFT", "AAPL"), space, "w")
        assert list(ordered) == [2.0, 1.0]
        with pytest.raises(CompileError, match="NVDA is not one; MSFT is missing"):
            in_basis_order(np.array([1.0, 2.0]), 0, ("AAPL", "NVDA"), space, "w")


class TestDeclaringASpace:
    def test_a_space_by_its_dimension_or_its_basis(self):
        session = Session()
        assert declare_space(session, "A", 3).dim == 3
        named = declare_space(session, "assets", basis=["AAPL", "MSFT"], labels="ab")
        assert named.basis == ("AAPL", "MSFT")
        assert session.spaces["assets"] is named
        assert session.labels["a"] is named

    def test_a_measure_is_normalised_and_taken_by_label(self):
        session = Session()
        weights = pd.Series([3.0, 1.0], index=["MSFT", "AAPL"])
        space = declare_space(session, "assets", basis=["AAPL", "MSFT"], measure=weights)
        assert list(session.measures[space].op.value) == [0.25, 0.75]

    def test_what_cannot_be_a_space_is_refused(self):
        session = Session()
        with pytest.raises(UsageError, match="give the dimension of the space"):
            declare_space(session, "A")
        with pytest.raises(UsageError, match="real or complex, not 'quaternion'"):
            declare_space(session, "A", 2, field="quaternion")
        with pytest.raises(DataError, match="needs 2 weights, one for each element"):
            declare_space(session, "A", 2, measure=[1.0, 2.0, 3.0])
