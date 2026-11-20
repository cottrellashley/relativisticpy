r"""The :class:`Workbook`: a session in which RelativisticPy programs run."""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any

import sympy
from sympy import Symbol

from relativisticpy.compiler import Compiler, Session
from relativisticpy.compiler.compiler import _public
from relativisticpy.compiler.scope import Output, TensorBinding, VariableBinding
from relativisticpy.diffgeom import Metric
from relativisticpy.ir import EquationType, ScalarType, TensorType, render
from relativisticpy.language import Source, parse, tokenize
from relativisticpy.language.syntax import Program
from relativisticpy.language.tokens import Token
from relativisticpy.runtime import Executor
from relativisticpy.tensors import Tensor, arrays
from relativisticpy.tensors.equation import TensorEquation
from relativisticpy.tensors.simplification import tidy

# Display maths in Markdown, $$...$$, outside code. Fenced code blocks and code
# spans are matched only so that what they contain is skipped, as when rendered.
_MARKDOWN = re.compile(
    r"^ {0,3}(?P<fence>`{3,}|~{3,}).*?^ {0,3}(?P=fence)[ \t]*$"
    r"|(?P<ticks>`+).+?(?P=ticks)"
    r"|\$\$(?P<math>.+?)\$\$",
    re.DOTALL | re.MULTILINE,
)


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
        for block, end in _blocks(text, name):
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


def math_blocks(text: str, name: str = "<markdown>") -> list[Source]:
    """The ``$$...$$`` blocks of a Markdown document, outside code, each a source that
    knows where it is in the document, so that errors point at the right line."""
    return [block for block, _ in _blocks(text, name)]


def _blocks(text: str, name: str) -> Iterator[tuple[Source, int]]:
    """Each block, and the offset just after its closing ``$$``."""
    for match in _MARKDOWN.finditer(text):
        if match.group("math") is not None:
            start = match.start("math")
            origin = (text.count("\n", 0, start) + 1, start - text.rfind("\n", 0, start))
            yield Source(match.group("math"), name, origin), match.end()


def latex(value: Any) -> str:
    """Typeset a result: a tensor, a tensor equation, or any SymPy value."""
    if hasattr(value, "latex") and callable(value.latex):
        return str(value.latex())
    return str(sympy.latex(value))


def _nowhere() -> Any:
    return Source("").__class__("")
