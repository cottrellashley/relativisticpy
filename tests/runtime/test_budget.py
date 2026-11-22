"""The time budget a closed form is looked for in: past it, a sum or a product is kept as
written, and a time limit in force around it keeps what remains of it."""

import gc
import signal
import time

import pytest
import sympy

from relativisticpy.checking import _time_limit
from relativisticpy.runtime import core

pytestmark = pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="no SIGALRM here")


def sleep_within(seconds: float) -> None:
    with core._budget(seconds):
        time.sleep(1)


def test_past_the_budget_the_block_stops():
    with pytest.raises(core._OverBudget):
        sleep_within(0.05)


def test_a_time_limit_around_it_keeps_what_remains_of_it():
    with _time_limit(10):
        with pytest.raises(core._OverBudget):
            sleep_within(0.05)
        remaining, _ = signal.getitimer(signal.ITIMER_REAL)
    assert 9 < remaining < 10


def test_a_time_limit_that_ends_first_decides():
    with pytest.raises(TimeoutError), _time_limit(0.05):
        sleep_within(5)


def test_a_sum_past_its_budget_is_kept_as_written(monkeypatch):
    def slow(*args, **kwargs):
        time.sleep(1)
        raise AssertionError("the budget should have stopped this")

    monkeypatch.setattr(core, "_CLOSED_FORM_BUDGET", 0.05)
    monkeypatch.setattr(core.assumptions, "summed", slow)
    n = sympy.Symbol("n")
    assert core.summation(1 / n**2, n, 1, sympy.oo) == sympy.Sum(1 / n**2, (n, 1, sympy.oo))


def test_an_equation_is_judged_as_written_where_carrying_it_out_runs_past_the_budget(
    monkeypatch,
):
    # an integral SymPy's heuristics search for ever, as in the claim of a light ray's bending
    from relativisticpy.runtime import kernels

    x = sympy.Symbol("x")
    written = sympy.Integral(x, (x, 0, 1))
    monkeypatch.setattr(core, "_CLOSED_FORM_BUDGET", 0.05)
    monkeypatch.setattr(sympy.Integral, "doit", lambda self, **hints: time.sleep(1))
    started = time.monotonic()
    assert kernels._carried_out(written) == written
    assert time.monotonic() - started < 0.5


def test_a_budget_the_computation_lost_is_raised_again():
    # As Python loses an exception raised in a finalizer that the garbage collector runs:
    # a computation that lost the first goes on only until the next.
    lost = []

    def losing_the_first() -> None:
        try:
            time.sleep(1)
        except core._OverBudget:
            lost.append(time.monotonic())
        time.sleep(5)

    started = time.monotonic()
    with pytest.raises(core._OverBudget), core._budget(0.05):
        losing_the_first()
    assert lost
    assert time.monotonic() - started < 0.05 + core._AGAIN + 0.5


def test_after_the_block_nothing_of_the_budget_remains():
    handler = signal.getsignal(signal.SIGALRM)
    with pytest.raises(core._OverBudget):
        sleep_within(0.05)
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
    assert signal.getsignal(signal.SIGALRM) == handler
    time.sleep(core._AGAIN + 0.1)  # nothing is raised after it


def test_a_budget_runs_out_after_a_callback_of_the_garbage_collector():
    # JAX gives the garbage collector one: an exception raised in it would be lost, as
    # unraisable, and the computation would go on for ever.
    held = []

    def callback(phase: str, info: dict) -> None:
        if phase == "start" and not held:
            held.append(phase)
            time.sleep(0.2)  # the budget runs out in here

    def collecting_then_computing() -> None:
        gc.collect()
        while True:
            time.sleep(0.01)

    gc.callbacks.append(callback)
    try:
        started = time.monotonic()
        with pytest.raises(core._OverBudget), core._budget(0.05):
            collecting_then_computing()
    finally:
        gc.callbacks.remove(callback)
    assert held
    # without raising again it would never end; a loaded machine may take a while to
    assert time.monotonic() - started < 10
