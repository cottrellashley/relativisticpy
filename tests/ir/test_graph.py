"""The IR graph: interning, retyping, its keys and the listing."""

import pickle

import sympy

from relativisticpy.ir import SCALAR, VALUE, Graph, TensorType, dependencies, ops, render
from relativisticpy.tensors import Index, Indices, Space


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


def test_constants_of_one_value_are_different_computations():
    # Keyed by the value's identity, an array changed in place and bound again would
    # be taken for the old one, and its results with it.
    graph, value = Graph(), [1.0, 2.0]
    assert graph.const(value, VALUE).key != graph.const(value, VALUE).key
    assert graph.const(value, VALUE).key != Graph().const(value, VALUE).key


def test_retyping_keeps_the_computation():
    graph = Graph()
    x = graph.const(sympy.Array([1, 2]), VALUE)
    plane = Space("plane", 2)
    vector = TensorType(Indices([Index("a")]), (plane,))
    renamed = TensorType(Indices([Index("b")]), (plane,))
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
    assert ops.Contract(((0, 1), (1, 2)), (0, 2), (4, 4, 4)).label() == "contract ab,bc->ac"
    assert ops.Select(((0, 1),)).label() == "select 0=1"
    assert ops.Symmetrize((0, 1), antisymmetric=True).label() == "antisymmetrize (0, 1)"
    assert ops.Add((1, -1)).label() == "add + -"
    assert ops.Map("simplify").label() == "map simplify"
    assert ops.Call("sin").label() == "call sin"
    assert ops.Call("core.function", (("name", "f"),)).label() == "call core.function name=f"
    assert ops.Const("x" * 60).label().endswith("...")


def test_a_graph_pickled_numbers_what_it_makes_after_what_it_was_given():
    # A worker that takes in a pickled workbook goes on with its graph: what it makes
    # there meets nothing made before, nor anything of a graph of its own.
    graph = Graph()
    made = [graph.const(n, SCALAR) for n in range(3)]
    carried = pickle.loads(pickle.dumps(graph))
    later = carried.const(3, SCALAR)
    assert later.id == len(made)
    assert later.key not in {node.key for node in made}
    assert Graph().const(0, SCALAR).key not in {node.key for node in made}


def test_a_graph_pickled_keeps_what_it_interned():
    graph = Graph()
    x = graph.const(sympy.Symbol("x"), SCALAR)
    square = graph.add(ops.Pow(), (x, x), SCALAR)
    carried = pickle.loads(pickle.dumps(graph))
    again = carried.add(ops.Pow(), (x, x), SCALAR)
    assert (again.id, again.key) == (square.id, square.key)
    assert len(carried) == len(graph)
