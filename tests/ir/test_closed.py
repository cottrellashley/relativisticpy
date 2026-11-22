"""The IR is closed: every operation that calls a function names it in the registry of
implementations, and no operation holds a Python function. And compiling computes
nothing: the executor is never asked for a value while a program compiles.

The programs compiled are the docs' doctests, the curriculum's cases and the programs the
tests run, read from their sources.
"""

from __future__ import annotations

import ast
import dataclasses
import doctest
import importlib.util
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import sympy

from relativisticpy.compiler import Compiler, Session
from relativisticpy.ir import SCALAR, VALUE, Graph, Node, TensorType, ops
from relativisticpy.language import Source, parse
from relativisticpy.runtime import Executor, registry
from relativisticpy.tensors import Indices

ROOT = Path(__file__).parent.parent.parent
#: The methods of a workbook that are given a program.
_RUN = frozenset({"exe", "run", "expr", "each", "ir", "grad"})


# -- the programs ----------------------------------------------------------------------


def _strings(node: ast.expr, constants: dict[str, str]) -> Iterator[str]:
    """The programs an argument holds: text, a name given text, or a list of either."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        yield node.value
    elif isinstance(node, ast.Name) and node.id in constants:
        yield constants[node.id]
    elif isinstance(node, ast.List | ast.Tuple):
        for item in node.elts:
            yield from _strings(item, constants)
    elif isinstance(node, ast.Call) and node.args:  # Source("...", "notes.md", (3, 1))
        yield from _strings(node.args[0], constants)


def _programs_in(tree: ast.AST, constants: dict[str, str]) -> Iterator[str]:
    """The programs a piece of Python gives a workbook, and the text of every case a test
    is parametrized with, in the order they are written."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in _RUN:
            yield from _strings(node.args[0], constants)
        elif isinstance(func, ast.Attribute) and func.attr == "parametrize":
            for argument in node.args[1:]:
                for inner in ast.walk(argument):
                    if isinstance(inner, ast.Constant) and isinstance(inner.value, str):
                        yield inner.value


def _constants(tree: ast.Module) -> dict[str, str]:
    """The names a module or its classes give text, as ``SCHWARZSCHILD = ...``."""
    found = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
            continue
        if isinstance(node.value.value, str):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    found[target.id] = node.value.value
    return found


def _test_programs() -> list[tuple[str, list[str]]]:
    """The programs of each test function, in order."""
    groups = []
    for path in sorted((ROOT / "tests").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        constants = _constants(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.ClassDef):
                programs = list(dict.fromkeys(_programs_in(node, constants)))
                if programs and isinstance(node, ast.FunctionDef):
                    groups.append((f"{path.relative_to(ROOT)}::{node.name}", programs))
    return groups


def _doc_programs() -> list[tuple[str, list[str]]]:
    """The programs of each page of the docs' doctests, in order."""
    groups = []
    parser = doctest.DocTestParser()
    for path in [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]:
        programs = []
        for example in parser.get_examples(path.read_text(encoding="utf-8")):
            try:
                tree = ast.parse(example.source)
            except SyntaxError:  # pragma: no cover - every doctest is Python
                continue
            programs.extend(_programs_in(tree, {}))
        if programs:
            groups.append((str(path.relative_to(ROOT)), programs))
    return groups


def _curriculum_programs() -> list[tuple[str, list[str]]]:
    spec = importlib.util.spec_from_file_location("curriculum", ROOT / "benchmarks/curriculum.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("curriculum", module)
    spec.loader.exec_module(module)
    return [(case.id, [case.program]) for case in module.load() if case.program.strip()]


SOURCES = {
    "docs": _doc_programs(),
    "curriculum": _curriculum_programs(),
    "tests": _test_programs(),
}


# -- compiling them --------------------------------------------------------------------


def compiled(programs: list[str]) -> list[Node]:
    """Every node made compiling ``programs`` in order, in one session, a statement at a
    time; a statement that cannot be compiled is passed over. Any executor asked for a
    value meanwhile fails the test."""
    made: list[Node] = []
    computed: list[str] = []
    add = Graph.add

    def recording(graph: Graph, *args: Any, **kwargs: Any) -> Node:
        node = add(graph, *args, **kwargs)
        made.append(node)
        return node

    def computing(name: str) -> Any:
        def refused(*args: Any, **kwargs: Any) -> Any:
            computed.append(name)
            raise AssertionError(f"compiling called Executor.{name}")

        return refused

    session = Session()
    compiler = Compiler(session)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Graph, "add", recording)
        for name in ("evaluate", "evaluate_with", "_evaluate", "map", "simplify_map"):
            patch.setattr(Executor, name, computing(name))
        for program in programs:
            try:
                statements = parse(Source(program)).statements
            except Exception:  # text that is not a program
                continue
            for statement in statements:
                try:
                    compiler.statement(statement, session.globals)
                except Exception:  # what does not compile makes no operation worth less
                    continue
    assert computed == [], f"compiling {programs} computed with the executor"
    return made


def _functions(value: Any) -> list[Any]:
    """The Python functions in what an operation holds: a function, a method, a class, or
    anything else that is called rather than computed with. Expressions, arrays and
    indices are values, even those SymPy lets be called, as a Lambda."""
    if isinstance(value, str | bytes | int | float | complex | bool | Indices) or value is None:
        return []
    if isinstance(value, sympy.Basic | sympy.NDimArray):
        return []
    if isinstance(value, tuple | list | frozenset | set):
        return [found for item in value for found in _functions(item)]
    if isinstance(value, dict):
        return [found for item in value.items() for found in _functions(item)]
    return [value] if callable(value) else []


def _consistent(node: Node, entry: registry.Implementation) -> bool:
    """Whether the type of a node agrees with what the registry says its operation gives."""
    args = [arg.type for arg in node.args]
    if isinstance(node.op, ops.Map):
        return isinstance(node.type, TensorType) and node.type == args[0]
    if entry.result == "scalar":
        return node.type == SCALAR or (node.type == VALUE and any(a != SCALAR for a in args))
    if entry.result == "value":
        return node.type == VALUE
    if entry.result == "argument":
        return node.type == args[0]
    return isinstance(node.type, TensorType)


def _faults(node: Node) -> list[str]:
    """What is wrong with an operation of the IR, if anything."""
    op = node.op
    if isinstance(op, ops.Const):
        return [f"a constant holds a function: {op.value!r}"] if _functions(op.value) else []
    faults = [
        f"{op.label()} holds a function in {field.name}"
        for field in dataclasses.fields(op)
        if _functions(getattr(op, field.name))
    ]
    if isinstance(op, ops.Call | ops.Map):
        if op.name not in registry.REGISTRY:
            return [*faults, f"{op.label()}: {op.name} is not registered"]
        entry = registry.lookup(op.name)
        count = len(node.args)
        if count < entry.min_args or (entry.max_args is not None and count > entry.max_args):
            faults.append(f"{op.label()} is given {count} arguments")
        if not _consistent(node, entry):
            faults.append(f"{op.label()} gives {node.type}, not a {entry.result}")
    return faults


@pytest.mark.parametrize("source", list(SOURCES))
def test_every_operation_is_named(source):
    faults = {}
    for where, programs in SOURCES[source]:
        for node in compiled(programs):
            for fault in _faults(node):
                faults.setdefault(fault, where)
    assert faults == {}


def test_the_session_has_no_executor():
    assert not hasattr(Session(), "evaluate")
    assert not hasattr(Session(), "executor")


def test_the_programs_are_many():
    assert len(SOURCES["docs"]) > 10
    assert len(SOURCES["curriculum"]) > 400
    assert sum(len(programs) for _, programs in SOURCES["tests"]) > 1_000


def test_a_function_held_or_a_name_not_registered_is_found():
    assert _functions(sympy.sin) == [sympy.sin]
    assert _functions((("name", "f"),)) == []
    assert _functions(sympy.Lambda(sympy.Symbol("x"), 1)) == []
    graph = Graph()
    x = graph.const(sympy.Symbol("x"), SCALAR)
    assert _faults(graph.add(ops.Call("no.such"), (x,), SCALAR)) == [
        "call no.such: no.such is not registered"
    ]
    assert _faults(graph.add(ops.Call("sin"), (x, x), SCALAR)) == ["call sin is given 2 arguments"]
    assert _faults(graph.const(abs, VALUE)) == [
        "a constant holds a function: <built-in function abs>"
    ]
