"""The executor: lazy, cached, and parallel when asked."""

import pytest
import sympy
from sympy import Array, Symbol

from relativisticpy import Workbook
from relativisticpy.errors import EvaluationError
from relativisticpy.ir import SCALAR, Graph, ops
from relativisticpy.runtime import Executor


class Counter:
    """A picklable function that counts its calls in this process."""

    calls = 0

    def __call__(self, value):
        Counter.calls += 1
        return value + 1


def test_values_are_computed_once():
    graph, executor = Graph(), Executor()
    counter = Counter()
    one = graph.const(sympy.Integer(1), SCALAR)
    node = graph.add(ops.Call(counter, "count"), (one,), SCALAR)
    Counter.calls = 0
    assert executor.evaluate(node) == 2
    assert executor.evaluate(node) == 2
    assert Counter.calls == 1


def test_only_what_is_needed_is_computed():
    graph, executor = Graph(), Executor()
    counter = Counter()
    one = graph.const(sympy.Integer(1), SCALAR)
    graph.add(ops.Call(counter, "count"), (one,), SCALAR)  # never asked for
    Counter.calls = 0
    assert executor.evaluate(one) == 1
    assert Counter.calls == 0


def test_errors_carry_the_span_of_their_node():
    wb = Workbook()
    with pytest.raises(EvaluationError) as error:
        wb.expr("Coordinates := [x, y]\ng_{ab} := [[1, 1], [1, 1]]\ng^{ab}")
    assert error.value.span is not None
    assert error.value.span.text.startswith("g_{ab} :=")


def test_parallel_map_gives_the_same_results():
    items = [Symbol("x") ** k - Symbol("x") ** k for k in range(40)]
    serial = Executor(1).map(sympy.simplify, items)
    pool = Executor(2, batch=4)
    try:
        assert pool.map(sympy.simplify, items) == serial
    finally:
        pool.close()


def test_a_workbook_with_workers_gives_the_same_physics():
    with Workbook(workers=2) as wb:
        wb.exe(r"g_{\mu\nu} := Schwarzschild(M)")
        assert wb.expr(r"simplify(R_{\mu\nu})").components == Array.zeros(4, 4)
        assert wb.expr("Kretschmann") == 48 * Symbol("M") ** 2 / Symbol("r") ** 6


def test_workers_default_to_the_cpu_count():
    import os

    assert Executor(None).workers == (os.cpu_count() or 1)
    assert Executor(0).workers == 1


def test_close_without_a_pool_is_harmless():
    Executor().close()
