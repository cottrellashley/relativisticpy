"""The executor: lazy, cached, and parallel when asked."""

import subprocess
import sys

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


def test_symbolic_equations_need_no_numpy():
    # Numbers are decided to rounding error in what they were computed from; symbols,
    # which the core computes with SymPy alone, never need NumPy for it.
    script = "\n".join(
        [
            "import sys",
            "sys.modules['numpy'] = None",
            "from relativisticpy import Workbook",
            "wb = Workbook()",
            "wb.exe('Coordinates := [t, x]\\nT_{ab} := [[0, 0], [0, 0]]')",
            "assert wb.expr('T_{ab} = 0')",
            "assert wb.expr('x + x = 2 x')",
        ]
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_two_numbers_are_never_left_unevaluated():
    # API-01: 25.0 = 24 is false, not Eq(25.0, 24), whose truth Python cannot tell
    wb = Workbook()
    wb["v^a"] = [3.0, 4.0]
    assert wb.expr(r"v^a v^b \delta_{ab} = 24") is sympy.false
    assert wb.expr(r"v^a v^b \delta_{ab} = 25") is sympy.true
    assert wb.expr(r"v^a v^b \delta_{ab} = x") == sympy.Eq(25.0, Symbol("x"))  # to solve
