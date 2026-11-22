"""The curriculum: what the language should read, by domain and by level, each case with
an answer known without RelativisticPy.

A case is a short program written as a textbook writes it, and an oracle: Python, using
SymPy, NumPy or SciPy alone, that computes the answer independently. The curriculum
documents what is easy and what is hard, and the cases that fail say what to build
next. ``curriculum/passing.txt`` records the cases that pass; a case may join it, never
leave it (``tests/test_curriculum.py`` holds them to it).

    uv run python benchmarks/curriculum.py                      # the scoreboard
    uv run python benchmarks/curriculum.py --domain analysis    # one domain
    uv run python benchmarks/curriculum.py --failures           # why each case fails
    uv run python benchmarks/curriculum.py --update             # record what passes now
    uv run python benchmarks/curriculum.py --seeds 0,1,2        # what the hash seed changes

A run fixes Python's hash seed as the command line does, starting again with
``PYTHONHASHSEED=0`` where the variable is not set, so that its scoreboard is the same
every time (``relativisticpy.__main__``). ``--seeds`` runs the cases once under each seed
given, each in a process of its own, and lists those whose outcome differs between them:
findings, which do not fail the run.

The format of a case is described in ``curriculum/README.md``.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import sympy

from relativisticpy import Workbook
from relativisticpy.__main__ import fix_hash_seed
from relativisticpy.checking import _time_limit
from relativisticpy.errors import RelativisticPyError

ROOT = Path(__file__).resolve().parent.parent / "curriculum"
PASSING = ROOT / "passing.txt"
LEVELS = {
    1: "notation",
    2: "computation",
    3: "derivation",
    4: "specialist",
    5: "frontier",
}
TIME_LIMIT = 30.0  # seconds for each case
_HEADING = re.compile(r"^## (?P<title>.+?)\s*$", re.MULTILINE)
_METADATA = re.compile(
    r"^\*\*Level (?P<level>\d)\*\*\s*·\s*`(?P<id>[^`]+)`(?:\s*·\s*(?P<source>.+?))?\s*$",
    re.MULTILINE,
)
_FENCE = re.compile(
    r"^```(?P<info>[\w-]*)[ \t]*\n(?P<body>.*?)^```[ \t]*$", re.MULTILINE | re.DOTALL
)
_PROGRAMS = {"latex": "LaTeX", "relativisticpy": "plain text"}
_EXPECTATIONS = ("expected", "outputs", "error")


@dataclass(frozen=True)
class Case:
    """One case: a program, the oracle that knows its answer, and where it is written."""

    id: str
    title: str
    domain: str
    topic: str
    level: int
    source: str
    program: str
    oracle: str
    path: Path
    line: int
    scope: str = ""  # Python run first, whose names the program may read: \texttt{prices}

    @property
    def where(self) -> str:
        return f"{self.path.relative_to(ROOT.parent)}:{self.line}"


@dataclass
class Result:
    """How a case went: ``passed``, ``wrong`` (a value, but not the answer), ``refused`` (an
    error from RelativisticPy), ``timeout``, ``crashed`` (an error that is not
    RelativisticPy's), or ``broken`` (the case itself is at fault)."""

    case: Case
    outcome: str
    reason: str = ""
    detail: str = ""
    seconds: float = 0.0

    @property
    def passed(self) -> bool:
        return self.outcome == "passed"

    def describe(self) -> str:
        head = f"{self.case.id} [{self.outcome}] {self.reason}".rstrip()
        lines = [head, f"  {self.case.where}"]
        if self.detail:
            lines += [f"  {line}" for line in self.detail.splitlines()]
        return "\n".join(lines)


# -- reading the curriculum -------------------------------------------------------------


def load(root: Path = ROOT) -> list[Case]:
    """Every case under ``root``, in the order of its domains, topics and sections."""
    cases = []
    for path in sorted(root.glob("*/*.md")):
        cases.extend(read(path))
    return cases


def read(path: Path) -> list[Case]:
    """The cases of one topic: each ``##`` section with a ``**Level N**`` line."""
    text = path.read_text(encoding="utf-8")
    headings = list(_HEADING.finditer(text))
    cases = []
    for number, heading in enumerate(headings):
        end = headings[number + 1].start() if number + 1 < len(headings) else len(text)
        section = text[heading.end() : end]
        metadata = _METADATA.search(section)
        if metadata is None:
            continue  # prose, not a case
        fences = list(_FENCE.finditer(section))
        programs = [f["body"] for f in fences if f["info"] in _PROGRAMS]
        oracles = [f["body"] for f in fences if f["info"] == "python"]
        scopes = [f["body"] for f in fences if f["info"] == "python-scope"]
        cases.append(
            Case(
                id=metadata["id"],
                title=heading["title"],
                domain=path.parent.name,
                topic=path.stem,
                level=int(metadata["level"]),
                source=(metadata["source"] or "").strip(),
                program=programs[0] if programs else "",
                oracle=oracles[0] if oracles else "",
                path=path,
                line=text.count("\n", 0, heading.start()) + 1,
                scope=scopes[0] if scopes else "",
            )
        )
    return cases


def passing(path: Path = PASSING) -> set[str]:
    """The cases recorded as passing."""
    if not path.exists():
        return set()
    lines = path.read_text(encoding="utf-8").splitlines()
    return {line.strip() for line in lines if line.strip() and not line.startswith("#")}


def problems(cases: Sequence[Case]) -> list[str]:
    """What is wrong with the cases themselves: duplicate identifiers, levels out of
    range, a program or oracle missing, an oracle that fails or expects nothing."""
    found = []
    counts = collections.Counter(case.id for case in cases)
    found += [f"{case_id} is used {n} times" for case_id, n in counts.items() if n > 1]
    for case in cases:
        if case.level not in LEVELS:
            found.append(f"{case.where}: {case.id} has level {case.level}, not 1 to 5")
        if not case.program.strip():
            found.append(f"{case.where}: {case.id} has no ```latex or ```relativisticpy program")
        if not case.oracle.strip():
            found.append(f"{case.where}: {case.id} has no ```python oracle")
            continue
        try:
            expectation = oracle(case)
        except Exception as error:
            found.append(f"{case.where}: {case.id}'s oracle fails: {type(error).__name__}: {error}")
            continue
        if not any(name in expectation for name in _EXPECTATIONS):
            names = ", ".join(_EXPECTATIONS)
            found.append(f"{case.where}: {case.id}'s oracle sets none of {names}")
    return found


# -- running a case -----------------------------------------------------------------------


@contextlib.contextmanager
def _beside(case: Case) -> Iterator[None]:
    """Work in the case's folder, so that a table it reads, ``tables/signal.csv``, is
    found beside it, by the program and the oracle alike."""
    before = os.getcwd()
    os.chdir(case.path.parent)
    try:
        yield
    finally:
        os.chdir(before)


def oracle(case: Case) -> dict[str, Any]:
    """Run a case's oracle, with SymPy's names in scope, and NumPy as ``np``, in the case's
    folder: what it expects of the program."""
    namespace: dict[str, Any] = {}
    exec("from sympy import *", namespace)
    try:
        import numpy

        namespace["np"] = numpy
    except ImportError:  # pragma: no cover - NumPy is a development dependency
        pass
    with _beside(case):
        exec(compile(case.oracle, case.where, "exec"), namespace)
    names = (*_EXPECTATIONS, "message", "tolerance")
    return {name: namespace[name] for name in names if name in namespace}


def run(case: Case, time_limit: float | None = TIME_LIMIT) -> Result:
    """Run a case's program in a fresh workbook and compare what it gives with its oracle."""
    try:
        expectation = oracle(case)
    except Exception as error:
        return Result(case, "broken", f"the oracle fails: {type(error).__name__}: {error}")
    tolerance = float(expectation.get("tolerance", 1e-9))
    start = time.perf_counter()
    try:
        with _beside(case):
            workbook = _workbook(case)
    except Exception as error:
        return Result(case, "broken", f"the scope fails: {type(error).__name__}: {error}")
    try:
        with _beside(case), workbook, _time_limit(time_limit):
            values = workbook.exe(case.program)
    except TimeoutError:
        seconds = time.perf_counter() - start
        return Result(case, "timeout", f"over {time_limit:g} s", seconds=seconds)
    except RelativisticPyError as error:
        seconds = time.perf_counter() - start
        wanted = expectation.get("error")
        if wanted is not None and _is_error(error, wanted, expectation.get("message")):
            return Result(case, "passed", seconds=seconds)
        return Result(case, "refused", _headline(error), str(error), seconds)
    except Exception as error:  # a failure of RelativisticPy's own
        reason = f"{type(error).__name__}: {error}".splitlines()[0]
        return Result(case, "crashed", reason, seconds=time.perf_counter() - start)
    seconds = time.perf_counter() - start
    if "error" in expectation:
        detail = f"expected {expectation['error']}, but it gave {_shown(values)}"
        return Result(case, "wrong", "should have been refused", detail, seconds)
    if "outputs" in expectation:
        wanted_all = list(expectation["outputs"])
        if len(values) != len(wanted_all):
            detail = f"expected {len(wanted_all)} outputs, got {len(values)}: {_shown(values)}"
            return Result(case, "wrong", "a different number of outputs", detail, seconds)
        pairs = list(zip(values, wanted_all, strict=True))
    else:
        if not values:
            return Result(case, "wrong", "no output", "the program shows nothing", seconds)
        pairs = [(values[-1], expectation["expected"])]
    for number, (actual, wanted) in enumerate(pairs, start=1):
        try:
            same = equal(actual, wanted, tolerance)
        except Exception as error:  # comparing failed: the value is not what was expected
            same, why = False, f"cannot compare: {type(error).__name__}: {error}"
        else:
            why = ""
        if not same:
            label = f"output {number}: " if len(pairs) > 1 else ""
            detail = f"{label}expected {_shown(wanted)}\n{label}got      {_shown(actual)}"
            return Result(case, "wrong", why or "not the answer", detail, seconds)
    return Result(case, "passed", seconds=seconds)


def _workbook(case: Case) -> Workbook:
    """A fresh workbook for a case, given the Python scope its ``python-scope`` block
    defines, if it has one: objects defined outside the language for it to read."""
    if not case.scope.strip():
        return Workbook()
    namespace: dict[str, Any] = {}
    exec(compile(case.scope, case.where, "exec"), namespace)
    namespace.pop("__builtins__", None)
    return Workbook(scope=namespace)


def _is_error(error: RelativisticPyError, wanted: str, message: str | None) -> bool:
    kinds = {cls.__name__ for cls in type(error).__mro__}
    return wanted in kinds and (message is None or message in error.message)


def _headline(error: RelativisticPyError) -> str:
    return f"{type(error).__name__}: {error.message.splitlines()[0]}"


# -- comparing values -----------------------------------------------------------------------


def equal(actual: Any, expected: Any, tolerance: float = 1e-9) -> bool:
    """Whether RelativisticPy's value is the oracle's.

    - ``True`` or ``False``: an equation that holds, or does not.
    - A mapping from components, by coordinate name, to values: those components of a tensor.
    - A list, a SymPy array or matrix, or a NumPy array: every component.
    - A number or SymPy expression: the same value, symbols compared by name.
    - Solutions, where either side is a mapping from unknowns to values or a list of them:
      the same solutions, in any order (see :func:`_same_solutions`).
    """
    if isinstance(expected, bool) or expected in (sympy.true, sympy.false):
        return _holds(actual) is bool(expected)
    if expected is None:
        return actual is None
    if isinstance(actual, dict) or _mappings(actual) or _mappings(expected):
        return _same_solutions(actual, expected, tolerance)
    if isinstance(expected, dict):
        return all(
            equal(_component(actual, key), value, tolerance) for key, value in expected.items()
        )
    if _is_array(expected) or (_is_array(actual) and not _is_scalar(expected)):
        mine, theirs = _flat(actual), _flat(expected)
        if mine is None or theirs is None or len(mine[1]) != len(theirs[1]):
            return False
        if _vector_shape(mine[0]) != _vector_shape(theirs[0]):  # a column is a vector
            return False
        return all(equal(a, b, tolerance) for a, b in zip(mine[1], theirs[1], strict=True))
    return _same_expression(actual, expected, tolerance)


def _vector_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    """A shape without its axes of one element: a column of three numbers, (3, 1), is the
    vector (3,) the oracle may write as a list, as books write a vector either way."""
    return tuple(size for size in shape if size != 1)


def _mappings(value: Any) -> bool:
    """Whether a value is a list of solutions, each a mapping from unknowns to values."""
    return (
        isinstance(value, list | tuple) and bool(value) and all(isinstance(v, dict) for v in value)
    )


def _solutions(value: Any) -> list[tuple[list[str] | None, list[Any]]]:
    """The solutions of a ``solve``, each as the names of its unknowns (``None`` where they
    are not named) and their values in order, whatever form the answer takes: a mapping
    ``{p: 2/3}``, a list of mappings, a list of tuples ``[(8, sqrt(3))]``, or a list of the
    values of one unknown, ``[2/3]``."""
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list | tuple):
        return [(None, [value])]
    solutions: list[tuple[list[str] | None, list[Any]]] = []
    for solution in value:
        if isinstance(solution, dict):
            solutions.append(([str(key) for key in solution], list(solution.values())))
        elif isinstance(solution, list | tuple):
            solutions.append((None, list(solution)))
        else:
            solutions.append((None, [solution]))
    return solutions


def _same_solutions(actual: Any, expected: Any, tolerance: float) -> bool:
    r"""Whether two answers of a ``solve`` give the same solutions, in any order. The values
    of a solution are matched by the names of the unknowns where both sides name them, and
    otherwise by position: a program names its unknowns in order, as in
    ``solve(..., [K, \omega])``, and an oracle writes their values in that order."""
    mine, theirs = _solutions(actual), _solutions(expected)
    if len(mine) != len(theirs):
        return False
    unmatched = list(mine)
    for names, values in theirs:
        for candidate in unmatched:
            if _same_solution(candidate, (names, values), tolerance):
                unmatched.remove(candidate)
                break
        else:
            return False
    return True


def _same_solution(
    mine: tuple[list[str] | None, list[Any]],
    theirs: tuple[list[str] | None, list[Any]],
    tolerance: float,
) -> bool:
    (my_names, my_values), (their_names, their_values) = mine, theirs
    if len(my_values) != len(their_values):
        return False
    if my_names is not None and their_names is not None:
        if sorted(my_names) != sorted(their_names):
            return False
        named = dict(zip(my_names, my_values, strict=True))
        my_values = [named[name] for name in their_names]
    return all(equal(a, b, tolerance) for a, b in zip(my_values, their_values, strict=True))


def _holds(value: Any) -> bool | None:
    """Whether an equation-like value holds, by whatever it says: True, False or None."""
    holds = getattr(value, "holds", None)
    if callable(holds):
        result = holds()
        return None if result is None else bool(result)
    if value in (sympy.true, sympy.false) or isinstance(value, bool):
        return bool(value)
    if type(value).__name__ == "TensorEquation":
        return bool(value)
    return None


def _is_scalar(value: Any) -> bool:
    return isinstance(value, int | float | complex | sympy.Basic) and not _is_array(value)


def _is_array(value: Any) -> bool:
    if isinstance(value, list | tuple):
        return True
    if isinstance(value, sympy.NDimArray | sympy.MatrixBase):
        return True
    return (
        hasattr(value, "shape")
        and getattr(value, "shape", ()) != ()
        and not isinstance(value, sympy.Basic)
    )


def _flat(value: Any) -> tuple[tuple[int, ...], list[Any]] | None:
    """A value's shape and its components in order, for a tensor, an array or a list."""
    components = getattr(value, "components", value)
    if callable(components):
        components = value
    if isinstance(components, sympy.MatrixBase):
        components = sympy.Array(components.tolist())
    if isinstance(components, sympy.NDimArray):
        return tuple(components.shape), list(components.reshape(len(components)))
    if isinstance(components, list | tuple):
        array = sympy.Array(components)
        return tuple(array.shape), list(array.reshape(len(array)))
    if hasattr(components, "shape") and hasattr(components, "tolist"):
        dense = components.toarray() if hasattr(components, "toarray") else components
        import numpy

        array = numpy.asarray(dense)
        return tuple(array.shape), list(array.reshape(-1))
    return None


def _component(value: Any, key: Any) -> Any:
    """A tensor's component by coordinate names, as ``("t", "r")`` or ``"t,r"``."""
    names = key.split(",") if isinstance(key, str) else [str(k) for k in key]
    coordinates = [str(c) for c in (getattr(value, "coordinates", None) or ())]
    position = tuple(coordinates.index(name.strip()) for name in names)
    components = getattr(value, "components", value)
    return components[position]


def _plain(expr: Any) -> Any:
    """``expr`` with every symbol replaced by a plain one of the same name, so that
    assumptions made by one side and not the other do not tell them apart."""
    expr = sympy.sympify(expr)
    return expr.xreplace({s: sympy.Symbol(s.name) for s in expr.free_symbols})


#: What a result may leave undone only where the answer does too: the language is to
#: compute an integral, a sum or a limit, not hand it back for the comparison to finish.
_UNEVALUATED = (
    sympy.Integral,
    sympy.Sum,
    sympy.Product,
    sympy.Limit,
    sympy.Derivative,
    sympy.Piecewise,
)


def _left_undone(actual: Any, expected: Any) -> set[str]:
    """The kinds of unevaluated operation in ``actual`` that ``expected`` has none of."""
    kinds = {type(node).__name__ for node in actual.atoms(*_UNEVALUATED)}
    return kinds - {type(node).__name__ for node in expected.atoms(*_UNEVALUATED)}


def _same_expression(actual: Any, expected: Any, tolerance: float) -> bool:
    if isinstance(actual, str) or isinstance(expected, str):
        return str(actual) == str(expected)
    a, b = _plain(actual), _plain(expected)
    if a == b:
        return True
    if isinstance(a, sympy.Equality) and isinstance(b, sympy.Equality):
        return _same_equation(a, b, tolerance)
    if isinstance(a, sympy.Piecewise) and not b.has(sympy.Piecewise):
        return _removable(a, b, tolerance)
    if _left_undone(a, b):  # an integral left as written is not its value
        return False
    if a.is_number and b.is_number:
        x, y = complex(a.evalf()), complex(b.evalf())
        return abs(x - y) <= tolerance * max(1.0, abs(y))
    difference = a - b
    try:
        if sympy.simplify(difference) == 0:
            return True
    except Exception:  # SymPy could not simplify it: decide by numbers
        pass
    return _agrees_numerically(difference, tolerance)


def _same_equation(actual: Any, expected: Any, tolerance: float) -> bool:
    """Whether two equations say the same: the differences of their sides agree, up to the
    sides being written the other way round."""
    mine, theirs = actual.lhs - actual.rhs, expected.lhs - expected.rhs
    return _same_expression(mine, theirs, tolerance) or _same_expression(mine, -theirs, tolerance)


def _removable(actual: Any, expected: Any, tolerance: float) -> bool:
    """Whether a Piecewise is the expected value with its removable points filled in: one
    branch is the value, as sin(πs)/(πs) for s ≠ 0, and every other is its limit at the
    point the condition names, as 1 at s = 0. Anything else is a condition the answer
    does not have."""
    pieces = list(actual.args)
    for number, (value, condition) in enumerate(pieces):
        if not _same_expression(value, expected, tolerance):
            continue
        points = _points(condition)
        others = pieces[:number] + pieces[number + 1 :]
        if not points or not others:
            return False
        for other, other_condition in others:
            at = _points(other_condition) or points  # "otherwise" holds where the first does not
            if at is None or not all(
                _same_expression(other, sympy.limit(expected, symbol, point), tolerance)
                for symbol, point in at
            ):
                return False
        return True
    return False


def _points(condition: Any) -> list[tuple[Any, Any]] | None:
    """The points a condition excludes or names: s ≠ 0 or s = 0 gives [(s, 0)], and
    -∞ < s < ∞, the whole line, none."""
    if condition is sympy.true:
        return None
    if isinstance(condition, sympy.Ne | sympy.Eq) and isinstance(condition.lhs, sympy.Symbol):
        return [(condition.lhs, condition.rhs)]
    if isinstance(condition, sympy.core.relational.Relational) and any(
        side.is_infinite for side in condition.args
    ):
        return []
    if isinstance(condition, sympy.And):
        found = [_points(part) for part in condition.args]
        if all(part is not None for part in found):
            return [point for part in found for point in part or []]
    return None


def _agrees_numerically(difference: Any, tolerance: float) -> bool:
    """Whether a difference vanishes at several random points, for every function of the
    symbols too: strong evidence of an identity where simplification gives up."""
    rng = random.Random(0)
    symbols = sorted(difference.free_symbols, key=str)
    functions = sorted(difference.atoms(sympy.core.function.AppliedUndef), key=str)
    for _ in range(3):
        values = {s: sympy.Rational(rng.randrange(300, 2000), 1009) for s in symbols}
        concrete = difference.xreplace(
            {f: sympy.Rational(rng.randrange(300, 2000), 997) for f in functions}
        )
        number = complex(sympy.N(concrete.subs(values), 30))
        if abs(number) > tolerance:
            return False
    return True


def _shown(value: Any) -> str:
    text = repr(value) if isinstance(value, list) else str(value)
    text = " ".join(text.split())
    return text if len(text) <= 300 else text[:299] + "…"


# -- the scoreboard ---------------------------------------------------------------------------


@dataclass
class Board:
    results: list[Result] = field(default_factory=list)

    def table(self) -> str:
        domains = sorted({r.case.domain for r in self.results})
        header = f"{'domain':28}" + "".join(f"{'L' + str(n):>9}" for n in LEVELS) + f"{'all':>11}"
        lines = [header]
        for domain in [*domains, None]:
            row = [r for r in self.results if domain is None or r.case.domain == domain]
            cells = []
            for level in LEVELS:
                at = [r for r in row if r.case.level == level]
                cells.append(f"{sum(r.passed for r in at)}/{len(at)}" if at else "-")
            total = f"{sum(r.passed for r in row)}/{len(row)}"
            name = domain if domain is not None else "all"
            lines.append(f"{name:28}" + "".join(f"{cell:>9}" for cell in cells) + f"{total:>11}")
        return "\n".join(lines)

    def blockers(self, top: int = 15) -> str:
        """What fails most, by reason: the next things to build."""
        reasons: dict[str, list[str]] = collections.defaultdict(list)
        for result in self.results:
            if result.outcome == "refused":
                reasons[_generalised(result.reason)].append(result.case.id)
        lines = []
        for reason, ids in sorted(reasons.items(), key=lambda item: -len(item[1]))[:top]:
            more = f" (+{len(ids) - 3} more)" if len(ids) > 3 else ""
            lines.append(f"{len(ids):4d}  {reason}\n        {', '.join(ids[:3])}{more}")
        return "\n".join(lines)


def _generalised(reason: str) -> str:
    """A reason with its particulars removed, so that like failures are counted together."""
    reason = re.sub(r"'[^']*'", "'…'", reason)
    return re.sub(r"\\?[A-Za-z]+(?:_\{[^}]*\}|\^\{[^}]*\})* (is not|has|needs)", r"X \1", reason)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--domain", action="append", help="only this domain (repeatable)")
    parser.add_argument("--level", type=int, action="append", help="only this level (repeatable)")
    parser.add_argument("--case", action="append", help="only this case (repeatable)")
    parser.add_argument("--failures", action="store_true", help="say why each case fails")
    parser.add_argument("--update", action="store_true", help="record the cases that pass")
    parser.add_argument("--time-limit", type=float, default=TIME_LIMIT, help="seconds per case")
    parser.add_argument(
        "--seeds",
        type=_seeds,
        help="run the cases once under each of these hash seeds, as 0,1,2, each in a process "
        "of its own, and list those whose outcome differs: findings, which do not fail the run",
    )
    parser.add_argument(
        "--outcomes",
        type=Path,
        metavar="FILE",
        help="write the outcome of each case to FILE: a line each, with its reason",
    )
    if argv is None:
        fix_hash_seed()
    args = parser.parse_args(argv)
    if args.seeds:
        return _under_seeds(args)

    cases = load()
    broken = problems(cases)
    if broken:
        print("The curriculum has problems of its own:", file=sys.stderr)
        for problem in broken:
            print(f"  {problem}", file=sys.stderr)
        return 1
    chosen = _chosen(cases, args)
    board = Board()
    with _recorded(args.outcomes) as record:
        for case in chosen:
            board.results.append(run(case, args.time_limit))
            record(board.results[-1])
    passed = sum(r.passed for r in board.results)
    share = passed / len(chosen) if chosen else 0
    print(f"{len(chosen)} cases, {passed} pass ({share:.0%})\n")
    print(board.table())
    outcomes = collections.Counter(r.outcome for r in board.results if not r.passed)
    if outcomes:
        print("\n" + ", ".join(f"{n} {outcome}" for outcome, n in outcomes.most_common()))
    wrong = [r for r in board.results if r.outcome in ("wrong", "crashed", "broken")]
    if wrong:
        print("\nWrong answers and crashes, the most serious:")
        for result in wrong:
            print(result.describe())
    if board.blockers():
        print("\nWhat is refused most, the next things to build:")
        print(board.blockers())
    if args.failures:
        print("\nEvery failure:")
        for result in board.results:
            if not result.passed:
                print(result.describe())
    if args.update:
        _record(cases, board.results)
    regressions = sorted(passing() & {r.case.id for r in board.results if not r.passed})
    if regressions:
        print(f"\n{len(regressions)} cases recorded as passing now fail: {', '.join(regressions)}")
        return 1
    return 0


def _chosen(cases: Sequence[Case], args: argparse.Namespace) -> list[Case]:
    """The cases a command line chooses, by domain, level and identifier, in order."""
    return [
        case
        for case in cases
        if (not args.domain or case.domain in args.domain)
        and (not args.level or case.level in args.level)
        and (not args.case or case.id in args.case)
    ]


@contextlib.contextmanager
def _recorded(path: Path | None) -> Iterator[Callable[[Result], None]]:
    """What records the outcome of each case in ``path``, a line each, as soon as it is
    known, so that a run stopped part of the way has said how far it got."""
    if path is None:
        yield lambda result: None
        return
    with path.open("w", encoding="utf-8") as file:

        def record(result: Result) -> None:
            file.write(f"{result.case.id}\t{result.outcome}\t{' '.join(result.reason.split())}\n")
            file.flush()

        yield record


def _record(cases: Iterable[Case], results: Iterable[Result]) -> None:
    """Record the cases that pass, keeping those not run this time."""
    ran = {r.case.id: r.passed for r in results}
    known = {case.id for case in cases}
    kept = {case_id for case_id in passing() if case_id in known and case_id not in ran}
    now = kept | {case_id for case_id, ok in ran.items() if ok}
    header = "# Cases that pass. Written by benchmarks/curriculum.py --update.\n"
    PASSING.write_text(header + "".join(f"{case_id}\n" for case_id in sorted(now)))
    print(f"\nrecorded {len(now)} passing cases in {PASSING.relative_to(ROOT.parent)}")


# -- under several hash seeds -------------------------------------------------------------

#: The longest a run of the whole curriculum under one seed may take, in seconds: a case
#: that runs past its own time limit, where no signal stops it, is stopped here.
SEEDED_RUN_LIMIT = 45 * 60.0

#: The outcome of a case that a run was still computing when it was stopped.
UNFINISHED = "unfinished"


def _seeds(text: str) -> list[str]:
    """Hash seeds, as ``0,1,2``: two or more whole numbers."""
    seeds = [seed.strip() for seed in text.split(",")]
    if len(seeds) < 2 or not all(seed.isdigit() for seed in seeds):
        raise argparse.ArgumentTypeError(f"takes two or more whole numbers, as 0,1,2, not {text}")
    return seeds


def _under_seeds(args: argparse.Namespace) -> int:
    """Run the chosen cases once under each hash seed, each in a process of its own, and
    list those whose outcome differs between them: findings, which do not fail the run.
    In GitHub Actions, each is an annotation too, a warning on its case."""
    options = ["--time-limit", str(args.time_limit)]
    for option, values in (("--domain", args.domain), ("--level", args.level)):
        options += [part for value in values or [] for part in (option, str(value))]
    options += [part for case in args.case or [] for part in ("--case", case)]
    chosen = _chosen(load(), args)
    outcomes: dict[str, dict[str, tuple[str, str]]] = {}
    with tempfile.TemporaryDirectory() as folder:
        for seed in args.seeds:
            under = _under_seed(seed, options, Path(folder) / f"{seed}.tsv", chosen)
            if under is None:
                return 1
            outcomes[seed] = under
    found = differing(outcomes)
    if not found:
        print(f"\nEvery case has the same outcome under seeds {', '.join(args.seeds)}.")
        return 0
    print(f"\nThe outcome of {len(found)} cases depends on the hash seed:")
    where = {case.id: case for case in chosen}
    annotate = os.environ.get("GITHUB_ACTIONS") == "true"
    for case_id, seeds_of in found.items():
        said = "; ".join(
            f"{outcome} under {', '.join(seeds)}" for outcome, seeds in seeds_of.items()
        )
        case = where[case_id]
        print(f"{case_id}: {said}\n  {case.where}")
        for seed in args.seeds:
            outcome, reason = outcomes[seed][case_id]
            if reason:
                print(f"  under {seed}, {outcome}: {reason}")
        if annotate:
            path = case.path.relative_to(ROOT.parent)
            title = "The outcome depends on the hash seed"
            print(f"::warning file={path},line={case.line},title={title}::{case_id}: {said}")
    return 0


def _under_seed(
    seed: str, options: list[str], written: Path, chosen: Sequence[Case]
) -> dict[str, tuple[str, str]] | None:
    """The outcome of each case, run under one hash seed in a process of its own. Where the
    run stops before its end, at its limit or by a failure of its own, the case it was
    computing is unfinished, and those after it are not there; ``None`` where the
    curriculum does not run at all."""
    command = [sys.executable, str(Path(__file__).resolve()), *options, "--outcomes", str(written)]
    environment = {**os.environ, "PYTHONHASHSEED": seed}
    try:
        ran = subprocess.run(
            command,
            env=environment,
            capture_output=True,
            text=True,
            timeout=SEEDED_RUN_LIMIT,
            check=False,
        )
    except subprocess.TimeoutExpired:
        stopped, said = f"stopped after {SEEDED_RUN_LIMIT:g} s", ""
    else:
        stopped, said = f"stopped with exit status {ran.returncode}", ran.stdout + ran.stderr
    if not written.is_file():
        print(f"the curriculum did not run under seed {seed}: {stopped}\n{said}", file=sys.stderr)
        return None
    found = _outcomes(written)
    unfinished = next((case for case in chosen if case.id not in found), None)
    if unfinished is None:
        print(f"under seed {seed}: {(said.splitlines() or [stopped])[0]}")
        return found
    found[unfinished.id] = (UNFINISHED, stopped)
    print(f"under seed {seed}: {stopped}, at {unfinished.id}, case {len(found)} of {len(chosen)}")
    return found


def _outcomes(path: Path) -> dict[str, tuple[str, str]]:
    """The outcome of each case, and its reason, as ``--outcomes`` writes them."""
    found = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        case_id, outcome, reason = line.split("\t", 2)
        found[case_id] = (outcome, reason)
    return found


def differing(
    outcomes: Mapping[str, Mapping[str, tuple[str, str]]],
) -> dict[str, dict[str, list[str]]]:
    """The cases whose outcome differs between the runs under several hash seeds, given the
    outcome of each case under each seed: for each, the seeds under which it met each
    outcome, in the order of the cases. A case that a run stopped before is left out."""
    seeds = list(outcomes)
    cases = list(dict.fromkeys(case_id for each in outcomes.values() for case_id in each))
    found = {}
    for case_id in cases:
        if not all(case_id in outcomes[seed] for seed in seeds):
            continue
        seeds_of: dict[str, list[str]] = {}
        for seed in seeds:
            seeds_of.setdefault(outcomes[seed][case_id][0], []).append(seed)
        if len(seeds_of) > 1:
            found[case_id] = seeds_of
    return found


if __name__ == "__main__":
    sys.exit(main())
