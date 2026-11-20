r"""The :class:`Workbook`: a session in which RelativisticPy programs run."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any

import sympy
from sympy import Symbol

from relativisticpy import checking
from relativisticpy.compiler import Compiler, Session
from relativisticpy.compiler.compiler import _public
from relativisticpy.compiler.scope import Context, Output, TensorBinding, VariableBinding
from relativisticpy.diffgeom import Metric
from relativisticpy.documents import Document, find_math_blocks, math_blocks
from relativisticpy.errors import RelativisticPyError
from relativisticpy.ir import EquationType, ScalarType, TensorType, render
from relativisticpy.language import Source, Span, parse, tokenize
from relativisticpy.language import syntax as ast
from relativisticpy.language.syntax import Program
from relativisticpy.language.tokens import Token
from relativisticpy.runtime import Executor
from relativisticpy.tensors import Tensor, arrays
from relativisticpy.tensors.equation import TensorEquation
from relativisticpy.tensors.simplification import is_zero, tidy


class Workbook:
    r"""A session for RelativisticPy programs, like the cells of a notebook.

    Definitions persist from one call to the next::

        >>> wb = Workbook()
        >>> wb.expr(r'''
        ...     Coordinates := [t, r, \theta, \phi]
        ...     g_{\mu\nu} := Schwarzschild(M)
        ... ''')
        >>> wb.expr(r"R_{abcd} R^{abcd}")
        48*M**2/r**6

    :meth:`expr` returns the value of the last expression and :meth:`exe` the
    value of every expression. Scalars are SymPy expressions; tensors are
    :class:`~relativisticpy.tensors.Tensor` values carrying their indices and
    coordinates.

    ``workers`` sets how many processes share component-wise work such as
    simplification (``None`` means one per CPU). Only large batches use them.
    """

    def __init__(self, *, workers: int | None = 1) -> None:
        self._workers = workers
        self._executor = Executor(workers)
        self._session = Session(executor=self._executor)
        self._compiler = Compiler(self._session)

    # -- running programs --------------------------------------------------------

    def exe(self, source: str | Source, *, name: str = "<input>") -> list[Any]:
        """Run a program and return the value of each expression it contains."""
        outputs = self._compiler.compile(self._parse(source, name))
        return [self._present(output) for output in outputs]

    run = exe

    def each(self, source: str | Source, *, name: str = "<input>") -> Iterator[Step]:
        """Run a program one statement at a time, carrying on after errors.

        Each :class:`Step` holds what one statement showed or the error it raised. A
        chain of relations, ``a = b = c``, gives a step for each link. Errors in the
        program's syntax are raised at once, before any statement runs.

            >>> wb = Workbook()
            >>> [(step.values, step.error) for step in wb.each("x := 2; y; x + 1")][1:]
            [([y], None), ([3], None)]
        """
        program = self._parse(source, name)
        for statement in program.statements:
            for piece in _pieces(statement):
                try:
                    outputs = self._compiler.statement(piece, self._session.globals)
                    values = [self._present(output) for output in outputs]
                except RelativisticPyError as error:
                    yield Step(piece.span, piece, error=error)
                else:
                    yield Step(piece.span, piece, values, sides=self._sides(piece))

    def _sides(self, statement: ast.Statement) -> tuple[Any, Any] | None:
        """The values of an equation's two sides, for saying how they differ."""
        if not (isinstance(statement, ast.Evaluate) and isinstance(statement.expr, ast.Equation)):
            return None
        ctx = Context(latex=statement.latex)
        scope = self._session.globals
        try:
            lhs, rhs = (
                self._executor.evaluate(self._compiler.value(side, scope, ctx))
                for side in (statement.expr.lhs, statement.expr.rhs)
            )
        except RelativisticPyError:
            return None
        return lhs, rhs

    def check(
        self,
        document: str | Path | Document,
        *,
        lines: tuple[int, int] | None = None,
        time_limit: float | None = None,
    ) -> checking.Report:
        """Run the equations of a LaTeX document, or of a Markdown notebook, in order and
        say what became of each: whether it holds, defines something, or could not be
        read. ``lines`` limits the check to part of the document, such as one chapter,
        and ``time_limit`` each statement's time in seconds.
        See :mod:`relativisticpy.checking`."""
        return checking.check(self, document, lines=lines, time_limit=time_limit)

    def expr(self, source: str | Source, *, name: str = "<input>") -> Any:
        """Run a program and return the value of its last expression (``None`` if it has none)."""
        results = self.exe(source, name=name)
        return results[-1] if results else None

    def run_file(self, path: str | Path) -> list[Any]:
        """Run a program from a file. In a Markdown file, the ``$$...$$`` blocks are the program."""
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() in (".md", ".markdown"):
            return [result for block in math_blocks(text, str(path)) for result in self.exe(block)]
        return self.exe(text, name=str(path))

    def markdown(self, text: str, *, name: str = "<markdown>") -> str:
        """Run the ``$$...$$`` blocks of a Markdown document in order, and return the
        document with each block's results typeset after it."""
        pieces: list[str] = []
        done = 0
        for block, end in find_math_blocks(text, name):
            pieces.append(text[done:end])
            pieces.extend(f"\n\n$$\n{latex(result)}\n$$" for result in self.exe(block))
            done = end
        pieces.append(text[done:])
        return "".join(pieces)

    # -- looking inside ----------------------------------------------------------

    def tokens(self, source: str) -> list[Token]:
        """The tokens of ``source``."""
        return tokenize(source)

    def parse(self, source: str) -> Program:
        """The syntax tree of ``source``."""
        return parse(source)

    def ir(self, source: str | Source) -> str:
        """The IR that ``source`` compiles to, as text; nothing is computed."""
        outputs = self._compiler.compile(self._parse(source, "<input>"))
        return render([output.node for output in outputs])

    def __getitem__(self, name: str) -> Any:
        """The value bound to ``name``: ``wb["x"]``, or a tensor such as ``wb["T"]``."""
        binding = self._session.globals.lookup(name)
        if isinstance(binding, TensorBinding | VariableBinding):
            return self._present(Output(binding.node, binding.node.span or _nowhere(), name))
        raise KeyError(name)

    @property
    def coordinates(self) -> tuple[Symbol, ...] | None:
        return self._session.coordinates

    @property
    def metric(self) -> Metric | None:
        """The metric in force, as a :class:`~relativisticpy.diffgeom.Metric`."""
        if self._session.metric is None or self._session.coordinates is None:
            return None
        components = self._executor.evaluate(self._session.metric.node)
        return Metric(components, self._session.coordinates, check=False)

    # -- lifecycle ---------------------------------------------------------------

    def reset(self) -> None:
        """Forget every definition."""
        self._executor.close()
        self._executor = Executor(self._workers)
        self._session = Session(executor=self._executor)
        self._compiler = Compiler(self._session)

    def close(self) -> None:
        """Stop any worker processes."""
        self._executor.close()

    def __enter__(self) -> Workbook:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    # -- internals -----------------------------------------------------------------

    @staticmethod
    def _parse(source: str | Source, name: str) -> Program:
        return parse(source if isinstance(source, Source) else Source(source, name))

    def _present(self, output: Output) -> Any:
        value = self._executor.evaluate(output.node)
        kind = output.node.type
        coordinates: Sequence[Symbol] = self._session.coordinates or ()
        if isinstance(kind, TensorType):
            components = arrays.apply(value, tidy)
            return Tensor(components, _public(kind.indices), coordinates, name=output.name)
        if isinstance(kind, EquationType) and isinstance(kind.of, TensorType):
            assert isinstance(value, TensorEquation)
            return TensorEquation(value.lhs, value.rhs, _public(kind.of.indices), coordinates)
        if isinstance(kind, ScalarType):
            return tidy(value)
        return value


@dataclass(frozen=True)
class Step:
    """One statement of a program, run by :meth:`Workbook.each`: the values it showed
    (none for a definition), or the error it raised."""

    span: Span
    statement: ast.Statement
    values: list[Any] | None = None
    error: RelativisticPyError | None = None
    sides: tuple[Any, Any] | None = None  # an equation's two sides, as computed

    @property
    def zero(self) -> bool:
        """Whether this is an equation whose two sides are both zero."""
        return self.sides is not None and all(_zero(side) for side in self.sides)


def _zero(value: Any) -> bool:
    if isinstance(value, sympy.NDimArray):
        return all(is_zero(component) for component in arrays.flatten(value))
    return isinstance(value, sympy.Expr) and is_zero(value)


def _pieces(statement: ast.Statement) -> list[ast.Statement]:
    """A statement, or each link of a chain of relations as a statement of its own."""
    if isinstance(statement, ast.Evaluate) and isinstance(statement.expr, ast.Chain):
        return [
            ast.Evaluate(link, span=link.span, latex=statement.latex)
            for link in statement.expr.links()
        ]
    return [statement]


def latex(value: Any) -> str:
    """Typeset a result: a tensor, a tensor equation, or any SymPy value."""
    if hasattr(value, "latex") and callable(value.latex):
        return str(value.latex())
    return str(sympy.latex(value))


def _nowhere() -> Any:
    return Source("").__class__("")
