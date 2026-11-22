"""The executor: lazy, cached, and parallel when asked."""

import contextlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

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


#: A run whose workers take long: each writes its process number, then waits.
SLOW_RUN = """
import os
import sys
import time

from relativisticpy.runtime import Executor


def slow(item):
    with open(sys.argv[1], "a") as pids:
        pids.write(f"{os.getpid()}\\n")
    time.sleep(60)
    return item


if __name__ == "__main__":
    Executor(2, batch=1).map(slow, [1, 2, 3, 4])
"""


def _alive(pid: int) -> bool:
    """Whether a process is running: not gone, and not a zombie waiting to be reaped."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    status = Path(f"/proc/{pid}/status")
    with contextlib.suppress(OSError):
        return "\nState:\tZ" not in status.read_text()
    return True


@pytest.mark.skipif(not hasattr(signal, "SIGTERM") or sys.platform == "win32", reason="POSIX")
def test_a_run_stopped_with_sigterm_leaves_no_workers(tmp_path):
    # AGT-18: a harness stops a command that takes too long with SIGTERM; its workers,
    # which nothing would stop, compute on for nobody.
    script, pids = tmp_path / "run.py", tmp_path / "pids"
    script.write_text(SLOW_RUN)
    run = subprocess.Popen([sys.executable, str(script), str(pids)])
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and len(_read(pids)) < 2:
            time.sleep(0.05)
        workers = _read(pids)
        assert len(workers) == 2
        run.send_signal(signal.SIGTERM)
        run.wait(timeout=30)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and any(_alive(pid) for pid in workers):
            time.sleep(0.05)
        assert not [pid for pid in workers if _alive(pid)]
    finally:
        run.kill()
        for pid in _read(pids):
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGKILL)


def _read(pids: Path) -> list[int]:
    return [int(pid) for pid in pids.read_text().split()] if pids.exists() else []


def test_two_numbers_are_never_left_unevaluated():
    # API-01: 25.0 = 24 is false, not Eq(25.0, 24), whose truth Python cannot tell
    wb = Workbook()
    wb["v^a"] = [3.0, 4.0]
    assert wb.expr(r"v^a v^b \delta_{ab} = 24") is sympy.false
    assert wb.expr(r"v^a v^b \delta_{ab} = 25") is sympy.true
    assert wb.expr(r"v^a v^b \delta_{ab} = x") == sympy.Eq(25.0, Symbol("x"))  # to solve


@pytest.mark.parametrize(
    "program",
    [
        "sin('__import__(\"os\").getcwd()')",
        "diff('__import__(\"os\").getcwd()', x)",
        "simplify('__import__(\"os\").getcwd()')",
        "x := '__import__(\"os\").getcwd()'\nsin(x)",
        # what the compiler computes while it reads, the size of a space, a factor
        "\\mathcal{A} := \\mathbb{R}^{'__import__(\"os\").getcwd()'}",
        "G_{\\mu\\nu} = '__import__(\"os\").getcwd()' T_{\\mu\\nu}",
    ],
)
def test_text_in_quotes_runs_nothing(program, monkeypatch):
    # SymPy reads a string it is given as Python, and runs it
    calls = []
    monkeypatch.setattr(os, "getcwd", lambda: calls.append(1) or "/")
    with pytest.raises(EvaluationError, match="text in quotes is not a value"):
        Workbook().exe(program)
    assert calls == []


def test_text_that_names_things_is_taken():
    assert Workbook().expr("symbols('a b')") == (Symbol("a"), Symbol("b"))
    assert Workbook().expr("\\lim_{x \\to 0} \\frac{\\sin x}{x}") == 1  # a side of a limit
