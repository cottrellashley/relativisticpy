"""The IR graph: interning, retyping and the listing."""

import sympy

from relativisticpy.ir import SCALAR, VALUE, Graph, TensorType, dependencies, ops, render
from relativisticpy.tensors import Index, Indices


def test_equal_computations_are_one_node():
    graph = Graph()
    x = graph.const(sympy.Symbol("x"), SCALAR)
    first = graph.add(ops.Pow(), (x, x), SCALAR)
    second = graph.add(ops.Pow(), (x, x), SCALAR)
    assert first is second
    assert len(graph) == 1


def test_constants_are_never_merged():
    graph = Graph()
    assert graph.const(1, SCALAR) is not graph.const(1, SCALAR)


def test_retyping_keeps_the_computation():
    graph = Graph()
    x = graph.const(sympy.Array([1, 2]), VALUE)
    vector = TensorType(Indices([Index("a")]), 2)
    renamed = TensorType(Indices([Index("b")]), 2)
    node = graph.add(ops.ToArray((2,)), (x,), vector)
    relabelled = graph.retype(node, renamed)
    assert relabelled is not node
    assert relabelled.key == node.key
    assert relabelled.type == renamed
    assert graph.retype(node, vector) is node
    assert graph.retype(x, renamed).key == x.key


def test_dependencies_come_before_their_users():
    graph = Graph()
    x = graph.const(sympy.Symbol("x"), SCALAR)
    two = graph.const(2, SCALAR)
    square = graph.add(ops.Pow(), (x, two), SCALAR)
    total = graph.add(ops.Add((1, 1)), (square, x), SCALAR)
    order = dependencies([total])
    assert order.index(x) < order.index(square) < order.index(total)
    assert len(order) == 4


def test_render():
    graph = Graph()
    x = graph.const(sympy.Symbol("x"), SCALAR)
    square = graph.add(ops.Pow(), (x, x), SCALAR)
    listing = render([square])
    assert "const x" in listing
    assert "pow(" in listing
    assert listing.splitlines()[-1] == f"out {square!r}"


def test_operation_labels():
    assert ops.Contract(((0, 1), (1, 2)), (0, 2), 4).label() == "contract ab,bc->ac"
    assert ops.Select(((0, 1),)).label() == "select 0=1"
    assert ops.Symmetrize((0, 1), antisymmetric=True).label() == "antisymmetrize (0, 1)"
    assert ops.Add((1, -1)).label() == "add + -"
    assert ops.Map(sympy.simplify, "simplify").label() == "map simplify"
    assert ops.Call(sympy.sin).label() == "call sin"
    assert ops.Const("x" * 60).label().endswith("...")
