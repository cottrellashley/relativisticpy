"""Spaces: what an index ranges over."""

import pytest
from sympy import symbols

from relativisticpy.tensors import Space
from relativisticpy.tensors.spaces import spacetime


def test_a_space_has_a_size_and_may_name_its_basis():
    assets = Space("assets", 3, basis=("AAPL", "MSFT", "NVDA"))
    assert (assets.dim, str(assets)) == (3, "assets")
    assert assets.position("NVDA") == 2
    assert assets.position("GOOG") is None
    assert Space("days", 2520).position("0") is None


def test_spaces_are_values():
    assert Space("assets", 3) == Space("assets", 3)
    assert Space("assets", 3) != Space("assets", 4)
    assert Space("qubit", 2, field="complex") != Space("qubit", 2)


def test_spacetime_is_the_space_of_the_coordinates():
    t, r = symbols("t r")
    space = spacetime((t, r))
    assert space.spacetime
    assert (space.dim, space.basis) == (2, ("t", "r"))
    assert space == spacetime((t, r))


@pytest.mark.parametrize(
    ("dim", "basis", "message"),
    [(-1, None, "cannot have dimension -1"), (2, ("a",), "dimension 2 but 1 names")],
)
def test_what_is_not_a_space(dim, basis, message):
    with pytest.raises(ValueError, match=message):
        Space("bad", dim, basis=basis)
