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

The format of a case is described in ``curriculum/README.md``.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import os
import random
import re
import sys
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import sympy

from relativisticpy import Workbook
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
        with _beside(case), Workbook() as workbook, _time_limit(time_limit):
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
    """
    if isinstance(expected, bool) or expected in (sympy.true, sympy.false):
        return _holds(actual) is bool(expected)
    if expected is None:
        return actual is None
    if isinstance(expected, dict):
        return all(
            equal(_component(actual, key), value, tolerance) for key, value in expected.items()
        )
    if _is_array(expected) or (_is_array(actual) and not _is_scalar(expected)):
        mine, theirs = _flat(actual), _flat(expected)
        if mine is None or theirs is None or len(mine[1]) != len(theirs[1]):
            return False
        if mine[0] != theirs[0]:
            return False
        return all(equal(a, b, tolerance) for a, b in zip(mine[1], theirs[1], strict=True))
    return _same_expression(actual, expected, tolerance)


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


def _same_expression(actual: Any, expected: Any, tolerance: float) -> bool:
    if isinstance(actual, str) or isinstance(expected, str):
        return str(actual) == str(expected)
    a, b = _plain(actual), _plain(expected)
    if a == b:
        return True
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
    args = parser.parse_args(argv)

    cases = load()
    broken = problems(cases)
    if broken:
        print("The curriculum has problems of its own:", file=sys.stderr)
        for problem in broken:
            print(f"  {problem}", file=sys.stderr)
        return 1
    chosen = [
        case
        for case in cases
        if (not args.domain or case.domain in args.domain)
        and (not args.level or case.level in args.level)
        and (not args.case or case.id in args.case)
    ]
    board = Board()
    for case in chosen:
        board.results.append(run(case, args.time_limit))
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


def _record(cases: Iterable[Case], results: Iterable[Result]) -> None:
    """Record the cases that pass, keeping those not run this time."""
    ran = {r.case.id: r.passed for r in results}
    known = {case.id for case in cases}
    kept = {case_id for case_id in passing() if case_id in known and case_id not in ran}
    now = kept | {case_id for case_id, ok in ran.items() if ok}
    header = "# Cases that pass. Written by benchmarks/curriculum.py --update.\n"
    PASSING.write_text(header + "".join(f"{case_id}\n" for case_id in sorted(now)))
    print(f"\nrecorded {len(now)} passing cases in {PASSING.relative_to(ROOT.parent)}")


if __name__ == "__main__":
    sys.exit(main())
