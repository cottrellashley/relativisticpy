"""A session's forks: in the same state, changed apart from it, with functions of their own."""

from dataclasses import replace

import sympy

from relativisticpy.compiler import Compiler, Session
from relativisticpy.compiler.scope import Scope
from relativisticpy.language import Source, parse
from relativisticpy.runtime import Executor


def compiled(session, text):
    """What a program shows, compiled in ``session``."""
    return Compiler(session).compile(parse(Source(text)))


def test_a_fork_gives_its_functions_names_of_its_own():
    session = Session()
    compiled(session, "a := 1\nf(x) := x + a")
    fork = session.fork()
    assert fork.globals.lookup("f").closure is fork.globals
    assert session.globals.lookup("f").closure is session.globals
    compiled(fork, "a := 5")
    [forked] = compiled(fork, "f(0)")
    [kept] = compiled(session, "f(0)")
    executor = Executor()
    assert (executor.evaluate(forked.node), executor.evaluate(kept.node)) == (5, 1)


def test_an_operator_and_a_bracket_stay_what_they_are_in_a_fork():
    session = Session()
    compiled(session, r"\hat{p} := -i\hbar \frac{d}{dx}" "\n" r"\langle x|p\rangle := e^{i p x}")
    fork = session.fork()
    for name, binding in session.globals.bindings():
        forked = fork.globals.lookup(name)
        assert type(forked) is type(binding)
        assert forked.closure is fork.globals
    [applied] = compiled(fork, r"\hat{p} \sin(x)")
    x, hbar = sympy.symbols("x hbar")
    assert Executor().evaluate(applied.node) == -sympy.I * hbar * sympy.cos(x)


def test_a_function_defined_in_another_scope_keeps_it():
    session = Session()
    compiled(session, "f(x) := x^2")
    elsewhere = Scope()
    session.globals.bind("g", replace(session.globals.lookup("f"), closure=elsewhere))
    assert session.fork().globals.lookup("g").closure is elsewhere
