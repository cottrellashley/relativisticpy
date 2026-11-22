r"""The :class:`Workbook`: a session in which RelativisticPy programs run."""

from __future__ import annotations

import functools
import numbers
import re
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, TypeVar

import sympy
from sympy import Symbol
from sympy.core.relational import Relational

from relativisticpy import checking, display
from relativisticpy.compiler import Compiler, Session
from relativisticpy.compiler.scope import (
    Check,
    Context,
    FunctionBinding,
    Output,
    Scope,
    TensorBinding,
    VariableBinding,
)
from relativisticpy.diffgeom import Metric
from relativisticpy.documents import Document, find_math_blocks, math_blocks
from relativisticpy.errors import (
    CompileError,
    DataError,
    EvaluationError,
    FileError,
    InternalError,
    NotComputedError,
    ParseError,
    RelativisticPyError,
    UnknownNameError,
    UsageError,
    debugging,
    place,
    translate,
)
from relativisticpy.ir import SCALAR, EquationType, ScalarType, TensorType, ops, render
from relativisticpy.language import Source, Span, parse, tokenize
from relativisticpy.language import syntax as ast
from relativisticpy.language.parser import split_index_labels
from relativisticpy.language.syntax import Program
from relativisticpy.language.tokens import Token
from relativisticpy.language.vocabulary import GREEK, GREEK_VARIANTS
from relativisticpy.reading.readings import Explanation
from relativisticpy.runtime import Executor
from relativisticpy.runtime.kernels import differs
from relativisticpy.tensors import Index, Indices, Space, Tensor, arrays, numerical
from relativisticpy.tensors.equation import TensorEquation, UndecidedError, _asymptotic, holds
from relativisticpy.tensors.indices import latex_label
from relativisticpy.tensors.simplification import is_zero, tidy
from relativisticpy.typing.typer import _public

_F = TypeVar("_F", bound=Callable[..., Any])
_T = TypeVar("_T")


def _from_python(method: _F) -> _F:
    """A method called from Python, whose errors show the caller's line and the
    diagnostic, not RelativisticPy's frames or the library exceptions behind it
    (``RELATIVISTICPY_DEBUG=1`` keeps them). A failure of RelativisticPy's own keeps its
    traceback, which a report needs."""

    @functools.wraps(method)
    def call(*args: Any, **kwargs: Any) -> Any:
        try:
            return method(*args, **kwargs)
        except RelativisticPyError as error:
            if isinstance(error, InternalError) or debugging():
                raise
            raise error.with_traceback(None) from None

    return call  # type: ignore[return-value]


#: The built-in scalars of a metric, which a program computes rather than defines.
_BUILT_IN_SCALARS = frozenset({"R", "Ric", "Kretschmann"})

#: How the compiler words a statement it reads but does not compute, as a relation such
#: as ≈. Until the compiler raises :class:`~relativisticpy.errors.NotComputedError` for
#: one itself, an error of these words is given that type here, where a step of
#: :meth:`Workbook.each` takes the compiler's error, so that a check judges it by its
#: type. This goes when the compiler raises the type.
_NOT_COMPUTED = "is read but not computed"


def _typed(error: RelativisticPyError) -> RelativisticPyError:
    """An error of the compiler's, with the type that says what it is."""
    if type(error) is CompileError and _NOT_COMPUTED in error.message:
        typed = NotComputedError(error.message, span=error.span, hint=error.hint, notes=error.notes)
        typed.cause = error.cause
        return typed
    return error


class Workbook:
    r"""A session for RelativisticPy programs, like the cells of a notebook.

    Start here:

    - :meth:`expr` runs a program and returns the value of its last expression;
    - :meth:`run`, also called :meth:`exe`, runs a program and returns the value of
      every expression in it, in order;
    - :meth:`each` runs a program one statement at a time, as its steps are asked for,
      and carries on after an error: each :class:`Step` holds what one statement gave,
      or the error it raised;
    - :meth:`check` runs the equations of a paper and says which hold.

    A program is text in the language, in LaTeX, plain text or Unicode;
    ``relativisticpy help reference`` prints the whole language on one page. ``:=``
    always defines; ``=`` defines only what has no meaning yet, and is otherwise a claim,
    which is checked, or an equation. Write LaTeX in raw strings, ``r"\frac{1}{2}"``.

    Definitions persist from one call to the next::

        >>> wb = Workbook()
        >>> wb.expr(r'''
        ...     Coordinates := [t, r, \theta, \phi]
        ...     g_{\mu\nu} := Schwarzschild(M)
        ... ''')
        >>> wb.expr(r"R_{abcd} R^{abcd}")
        48*M**2/r**6

    ``wb["x"]`` reads what a program defined, ``wb["w^a"] = data`` binds data from
    Python, and ``"x" in wb`` and :meth:`names` say what is defined. Scalars are SymPy
    expressions; tensors are :class:`~relativisticpy.tensors.Tensor` values carrying
    their indices and coordinates. Every error is a
    :class:`~relativisticpy.errors.RelativisticPyError`.

    ``workers`` sets how many processes share component-wise work such as
    simplification (``None`` means one per CPU). Batches of a few components or more
    use them. ``backend="jax"`` keeps data bound from Python as JAX arrays, so that it
    is computed with JAX, on a GPU if there is one; a SciPy sparse matrix stays sparse.
    It turns on JAX's 64-bit numbers, ``jax_enable_x64``, for the whole process.
    """

    def __init__(self, *, workers: int | None = 1, backend: str = "numpy") -> None:
        if backend not in ("numpy", "jax"):
            raise UsageError(f"the backend is numpy or jax, not {backend!r}")
        if workers is not None and not isinstance(workers, int):
            raise UsageError(f"workers is a number of processes, not {workers!r}")
        self._backend = backend
        self._workers = workers
        self._executor = Executor(workers)
        self._session = Session()
        self._compiler = Compiler(self._session)
        self._metric: tuple[Any, Metric] | None = None  # the metric in force, by its node
        self._revision = 0  # counts what may have changed what it defines, for its copies

    # -- running programs --------------------------------------------------------

    @_from_python
    def exe(self, source: str | Source, *, name: str = "<input>") -> list[Any]:
        """Run a program and return the value of every expression in it, in order: a
        definition gives none, an expression one, and a chain of relations,
        ``a = b = c``, one for each link. An error stops the program and is raised;
        what came before it is kept. :meth:`run` is the same method."""
        outputs = self._statements(self._parse(source, name, "exe"))
        return [self._present(output) for output in outputs]

    run = exe

    def each(
        self, source: str | Source, *, name: str = "<input>", recover: bool = False
    ) -> Iterator[Step]:
        r"""Run a program one statement at a time, carrying on after errors.

        It is lazy: a statement runs when its :class:`Step` is asked for, as by a ``for``
        loop or ``list(...)``, so nothing runs until then. Each step holds what one
        statement showed, or the error it raised, with how the statement was read where
        that could surprise. A chain of relations, ``a = b = c``, gives a step for each
        link. Errors in the program's syntax are raised as soon as the first step is
        asked for, before any statement runs.

            >>> wb = Workbook()
            >>> [(step.values, step.error) for step in wb.each("x := 2; y; x + 1")][1:]
            [([y], None), ([3], None)]

        With ``recover``, a statement that cannot be read is a step of its own instead,
        in its place: its span is the statement, and its error the
        :class:`~relativisticpy.errors.ParseError` that says why. The statements before
        and after it run, so that every error of a program is reported in order::

            >>> steps = Workbook().each("x := 2\n\\frca{1}{2}\nx + 1", recover=True)
            >>> [step.error.message if step.error else step.values for step in steps]
            [[], 'unknown LaTeX command \\frca', [3]]
        """
        try:
            program = self._parse(source, name, "each", recover=recover)
        except RelativisticPyError as error:
            if debugging():
                raise
            raise error.with_traceback(None) from None
        for piece, span in _steps_of(program):
            if isinstance(piece, ParseError):  # a statement that cannot be read
                yield Step(span, ast.Statement(span=span), error=piece)
                continue
            try:
                compile = functools.partial(self._compiler.statement, piece, self._session.globals)
                outputs = self._run(compile)
                values = [self._present(output) for output in outputs]
            except Exception as error:  # an error of the input's, or a failure of ours
                failure = _typed(translate(error, span=piece.span))
                yield Step(piece.span, piece, error=failure)
            else:
                notes = [output.note for output in outputs if output.note]
                sides = self._sides(piece)
                yield Step(piece.span, piece, values, sides=sides, notes=notes)

    def _sides(self, statement: ast.Statement) -> tuple[Any, Any] | None:
        """The values of an equation's two sides, for saying how they differ."""
        if not (isinstance(statement, ast.Evaluate) and isinstance(statement.expr, ast.Equation)):
            return None
        ctx = Context(latex=statement.latex)
        scope = self._session.globals
        try:
            lhs, rhs = (
                self._executor.evaluate(
                    self._run(functools.partial(self._compiler.value, side, scope, ctx))
                )
                for side in (statement.expr.lhs, statement.expr.rhs)
            )
        except Exception:  # the sides only say how an equation differs: without them, it does
            return None
        return lhs, rhs

    @_from_python
    def check(
        self,
        document: str | Path | Document,
        *,
        lines: tuple[int | None, int | None] | int | None = None,
        time_limit: float | None = None,
    ) -> checking.Report:
        """Run the equations of a LaTeX document, or of a Markdown notebook, in order and
        say what became of each: whether it holds, differs, defines something, or could
        not be read.

        ``document`` is a :class:`~relativisticpy.documents.Document`, the path of a file,
        or the text of a document itself: text with a line break or a ``\\begin`` that
        is not the path of a file. ``lines`` checks part of the document, such as one
        chapter: ``(first, last)``, either of which may be ``None`` for an open end, or
        one line; what comes before it still runs, so that its definitions are known.
        ``time_limit`` is each statement's time in seconds. See
        :mod:`relativisticpy.checking`."""
        return checking.check(self, document, lines=lines, time_limit=time_limit)

    @_from_python
    def expr(self, source: str | Source, *, name: str = "<input>") -> Any:
        """Run a program and return the value of its last expression (``None`` if it has
        none, as when it only defines). An error stops the program and is raised."""
        results = self.exe(source, name=name)
        return results[-1] if results else None

    @_from_python
    def run_file(self, path: str | Path) -> list[Any]:
        """Run a program from a file, as :meth:`run` does, and return the value of every
        expression in it, in order. In a Markdown notebook (``.md``), its blocks of maths
        are the program, without the results ``relativisticpy --render`` typeset in it
        (see :func:`read_program`). A LaTeX document, whose equations are checked rather
        than run, and a Jupyter notebook are refused, with what to do instead."""
        sources = read_program(_path(path, "run_file"))
        return [result for source in sources for result in self.exe(source)]

    @_from_python
    def markdown(self, text: str, *, name: str = "<markdown>") -> str:
        """Run the blocks of maths of a Markdown notebook in order, and return the notebook
        with each block's results typeset after it, as ``relativisticpy --render`` does.

        The results go between :data:`RESULT_START` and :data:`RESULT_END`, so that a
        notebook rendered again has its results replaced, not run. A block's notes are
        typeset with its results, and a statement that fails has its error typeset in its
        place, and the blocks after it still run: :meth:`each` gives the errors as
        values."""
        if not isinstance(text, str):
            raise UsageError(f"markdown takes the document as a str, not {_kind(text)}")
        rendered, _ = render_notebook(self, text, name)
        return rendered

    # -- looking inside ----------------------------------------------------------

    def tokens(self, source: str) -> list[Token]:
        """The tokens of ``source``."""
        return tokenize(source)

    def parse(self, source: str) -> Program:
        """The syntax tree of ``source``."""
        return parse(source)

    @_from_python
    def ir(self, source: str | Source | Iterable[str | Source]) -> str:
        r"""The IR that ``source`` compiles to, as text. Nothing is computed, and the
        workbook is left as it was: a definition in ``source`` is not made.

        Several sources, such as the blocks of a Markdown notebook, compile in order in
        one copy of the workbook, so that each sees what those before it define. The IR
        of each is headed by where it is, and lists what its outputs add to the
        computation (a definition adds nothing until something uses it)::

            >>> print(Workbook().ir([Source("y := 2", "notes.md", (3, 1)), "y^2"]))
            # notes.md:3
            # <input>:1
            %0 = const 2                                             Scalar
            %1 = const 2                                             Scalar
            %2 = pow(%0, %1)                                         Scalar
            out %2
        """
        if isinstance(source, str | Source):
            compiler = Compiler(self._session.fork())
            outputs = compiler.compile(self._parse(source, "<input>", "ir"))
            return render([output.node for output in outputs])
        if not isinstance(source, Iterable):
            raise UsageError(f"ir takes a program, or a list of them, not {_kind(source)}")
        compiler = Compiler(self._session.fork())
        shown: set[str] = set()
        listing: list[str] = []
        for block in source:
            program = self._parse(block, "<input>", "ir")
            outputs = compiler.compile(program)
            text = program.span.source
            listing.append(f"# {text.name}:{text.origin[0]}")
            for line in render([output.node for output in outputs]).splitlines():
                node = line.split(" ", 1)[0]
                if line.startswith("out ") or node not in shown:
                    listing.append(line)
                    shown.add(node)
        return "\n".join(listing)

    @_from_python
    def explain(self, source: str | Source) -> Explanation:
        r"""How each statement of ``source`` is read, and why: how each name is resolved,
        against what the workbook defines, its coordinates and the built-in names; each
        ambiguous form decided, as a definition or a claim, notation or components, a
        product or a call, an index or a power, a run of letters, a constant or a symbol;
        and the notes said where the less likely reading is taken. Nothing is computed, and
        the workbook is left as it was, as with :meth:`ir`: a definition in ``source`` is
        not made. What it returns is the readings, one for each statement (see
        :class:`~relativisticpy.reading.readings.Reading`), and ``str()`` gives them as
        text, values shown as :mod:`relativisticpy.display` shows them::

            >>> print(Workbook().explain(r"x := 2" "\n" r"\pi x"))
            line 1: x := 2
              definition or claim: x := ... is a definition, since it is written with :=
            line 2: \pi x
              constant or symbol: \pi is a constant, since nothing defines \pi
              \pi: a constant, π
              x: a value, defined on line 1, 2
        """
        compiler = Compiler(self._session.fork())
        program = self._parse(source, "<input>", "explain")
        return Explanation([compiler.read(statement) for statement in program.statements])

    @_from_python
    def __getitem__(self, written: str) -> Any:
        r"""What ``written`` names in this workbook, as a program would see it:

        - a name a program defined, ``wb["x"]``, or a tensor, by its name, ``wb["T"]``;
        - a tensor written with indices, as it was bound, ``wb["w^a"]``, or with others,
          ``wb["w_a"]``, which are lowered as a program would lower them;
        - a Greek name in any of its forms: ``wb[r"\Sigma"]``, ``wb["Σ"]``, ``wb["Sigma"]``;
        - ``wb["g"]``, the metric in force (:attr:`metric`);
        - a function, ``wb["f"]``, as a SymPy ``Lambda``, which can be called;
        - a coordinate, as its symbol, and a space, as a
          :class:`~relativisticpy.tensors.Space`.

        A name nothing defines raises :class:`~relativisticpy.errors.UnknownNameError`,
        a ``KeyError``. The built-in tensors, such as ``R`` and ``\Gamma``, are computed
        by a program, as in ``wb.expr(r"\Gamma^a_{bc}")``."""
        name, entries = self._key(written)
        session = self._session
        binding = session.globals.lookup(name)
        if binding is None and name == session.names["metric"] and self.metric is not None:
            return self.metric if entries is None else self.expr(_spelled(name, entries))
        if isinstance(binding, TensorBinding | VariableBinding):
            if entries is not None:  # with indices, which may be in other positions
                return self.expr(_spelled(name, entries))
            return self._present(Output(binding.node, binding.node.span or _nowhere(), name))
        if isinstance(binding, FunctionBinding):
            return self._function(name, binding)
        for coordinate in session.coordinates or ():
            if coordinate.name == name and entries is None:
                return coordinate
        if name in session.spaces and entries is None:
            return session.spaces[name]
        raise self._unknown(name, written)

    def __contains__(self, written: object) -> bool:
        """Whether ``written`` names something in this workbook, as ``wb[written]``
        would find it: ``"x" in wb``."""
        if not isinstance(written, str):
            return False
        try:
            name, _ = self._key(written)
        except RelativisticPyError:
            return False
        return name in self.names()

    def __iter__(self) -> Iterator[str]:
        """The names this workbook defines, as :meth:`names` lists them."""
        return iter(self.names())

    def names(self) -> dict[str, Definition]:
        r"""What this workbook defines, by name, in the order it was defined: its
        coordinates, the metric and every name a program defined or Python bound, with
        what kind of thing each is and how it is written::

            >>> wb = Workbook()
            >>> wb.exe(r"Coordinates := [t, x]; f(y) := y^2; V^{a} := [1, t]; c := 3")
            []
            >>> wb.names()
            {'t': <coordinate t>, 'x': <coordinate x>, 'f': <function f(y)>,
             'V': <tensor V^{a}>, 'c': <value c>}
        """
        found: dict[str, Definition] = {}
        session = self._session
        for coordinate in session.coordinates or ():
            found[coordinate.name] = Definition("coordinate", latex_label(coordinate.name))
        if session.metric is not None:
            name = session.names["metric"]
            span = session.metric.node.span
            found[name] = Definition("metric", f"{name}_{{\\mu\\nu}}", span)
        for name, binding in session.globals.bindings():
            found[name] = _definition(name, binding)
        for name, space in session.spaces.items():
            found.setdefault(name, Definition("space", str(space.name)))
        return found

    def __repr__(self) -> str:
        """``<Workbook: coordinates t, r, θ, φ; metric g_{\\mu\\nu} := Schwarzschild(M);
        defined u^{\\mu}, \\Sigma>``"""
        parts = []
        session = self._session
        if session.coordinates:
            shown = ", ".join(display.unicode(c) for c in session.coordinates)
            parts.append(f"coordinates {shown}")
        if session.metric is not None:
            parts.append(f"metric {_shortened(session.metric.node.span)}")
        defined = [
            definition.written
            for definition in self.names().values()
            if definition.kind not in ("coordinate", "metric")
        ]
        if defined:
            parts.append("defined " + ", ".join(defined))
        return f"<Workbook: {'; '.join(parts) or 'nothing defined'}>"

    def _key(self, written: str) -> tuple[str, list[tuple[str, bool]] | None]:
        """A name as it may be written to read it, and its indices, if any are written."""
        if not isinstance(written, str):
            raise UsageError(f"a name is a str, as in wb['x'], not {_kind(written)}")
        try:
            target = _written(written)
        except RelativisticPyError:
            raise UnknownNameError(
                f"{written} is not a name",
                hint="write a name, as in wb['x'], or a tensor with its indices, as in "
                "wb['w^a']; to compute an expression, use wb.expr",
            ) from None
        if isinstance(target, ast.Name):
            return target.id, None
        return target

    def _knows_tensor(self, name: str, rank: int) -> bool:
        """Whether ``name`` with ``rank`` indices is a tensor this workbook knows: one a
        program or Python defined, the metric, or a built-in one, as the Ricci tensor."""
        binding = self._session.globals.lookup(name)
        if isinstance(binding, TensorBinding):
            return True
        if isinstance(binding, VariableBinding):
            return isinstance(binding.node.type, TensorType) or binding.node.type != SCALAR
        if name == self._session.names["metric"]:
            return rank == 2
        return self._compiler._builtin_tensor(name, rank) is not None

    def _unknown(self, name: str, written: str) -> UnknownNameError:
        """The error for a name nothing in this workbook defines."""
        defined = [definition.written for definition in self.names().values()]
        notes = [f"defined here: {', '.join(defined)}"] if defined else []
        session = self._session
        builtin = name in _BUILT_IN_SCALARS or any(
            self._compiler._builtin_tensor(name, rank) for rank in range(1, 5)
        )
        if name == session.names["metric"]:
            hint = "define the metric first, as in wb.expr(r'g_{\\mu\\nu} := Schwarzschild(M)')"
        elif builtin:
            hint = f"a built-in quantity is computed by a program, as in wb.expr(r'{written}')"
        elif session.metric is not None and name in session.metric.names:
            hint = f"{written} is a symbol the metric is written with, and has no value"
        else:
            hint = f"define it first, as in wb[{written!r}] = 1 or wb.expr(r'{written} := 1')"
        return UnknownNameError(f"{written} is not defined", notes=notes, hint=hint)

    def _function(self, name: str, binding: FunctionBinding) -> sympy.Lambda:
        """A function a program defined, as a SymPy ``Lambda`` of its parameters."""
        definition = binding.definition
        parameters = [sympy.Dummy(parameter) for parameter in definition.params]
        session = self._session.fork()
        compiler = Compiler(session)
        scope = session.globals.child()
        for parameter, symbol in zip(definition.params, parameters, strict=True):
            scope.bind(parameter, VariableBinding(session.graph.const(symbol, SCALAR)))
        call = ast.Call(
            ast.Name(name, span=_nowhere()),
            tuple(ast.Name(p, span=_nowhere()) for p in definition.params),
            span=_nowhere(),
        )
        compile = functools.partial(compiler.value, call, scope, Context())
        value = self._executor.evaluate(self._run(compile, compiler))
        return sympy.Lambda(tuple(parameters), value)

    @_from_python
    def __setitem__(self, written: str, value: Any) -> None:
        r"""Bind data from Python: ``wb["R_{ta}"] = returns`` binds a tensor whose first
        index ranges over the rows of ``returns`` and its second over the columns, and
        ``wb["J"] = 1.5`` a number.

        The data may be a NumPy array, a pandas DataFrame or Series, a SciPy sparse
        matrix, a SymPy array or nested lists, and is copied. A name may be bound to a
        number, a SymPy expression, or text, which is read as the language reads it, and
        never run as Python: ``wb["J"] = "x^2 + 1"``. Each axis ranges over:

        - the space of its index, if the index was declared, as by
          ``a, b \in \mathcal{A}`` or :meth:`space`;
        - the space a pandas axis's name names, if there is one;
        - the space of a like index of the same data: both indices of a covariance
          :math:`\Sigma_{ab}` range over one space, but :math:`t` and :math:`a` of
          returns :math:`R_{ta}` over two, as indices from the same run of the alphabet
          (a to h, i to n, o to z, or Greek) are alike;
        - the space of the same index of a tensor already bound, as ``w^a`` takes the
          assets of ``R_{ta}``;
        - otherwise a space of its own, the size of the axis, named after the pandas axis
          if it has a name, with the pandas labels as its basis.

        A declared or named space must be the axis's size, and the others are shared
        only where they are. Data labelled by pandas is put in the order of its space's
        basis, by label. Binding declares nothing: the labels of ``R_{ta}`` range over its
        spaces in statements that write them on it, and are free everywhere else, so that
        ``t`` may still be a coordinate in ``\Gamma^{t}_{tr}``::

            >>> import numpy as np
            >>> wb = Workbook()
            >>> wb["w^a"] = np.array([0.5, 0.25, 0.25])
            >>> wb.expr("w^{a} w^{b} \\delta_{ab}")
            0.375

        Data that cannot be bound raises :class:`~relativisticpy.errors.DataError`, a
        ``ValueError``.
        """
        if not isinstance(written, str):
            raise UsageError(f"a name is a str, as in wb['x'] = 2, not {_kind(written)}")
        self._revision += 1
        target = _written(written)
        session = self._session
        if isinstance(target, ast.Name) and "_" in target.id and _rank_of(value):
            base, _, tail = target.id.partition("_")  # y_t: the tensor y, with index t
            target = (base, [(label, False) for label in split_index_labels(tail)])
        if isinstance(target, ast.Name):
            session.globals.bind(target.id, VariableBinding(self._scalar(written, value)))
            return
        name, entries = target
        components, bases, axis_names = _data(value, written)
        shape = tuple(getattr(components, "shape", ()))
        if len(shape) != len(entries):
            raise DataError(
                f"{written} has {_count(len(entries), 'index', 'indices')}, but the data "
                f"has {_count(len(shape), 'axis', 'axes')}",
                hint="write one index for each axis of the data, as in R_{ta} for a table",
            )
        complex_data = numerical.is_numerical(components) and numerical.numpy().iscomplexobj(
            components
        )
        field: Literal["real", "complex"] = "complex" if complex_data else "real"
        spaces: list[Space] = []
        for axis, ((label, _), size, basis, axis_name) in enumerate(
            zip(entries, shape, bases, axis_names, strict=True)
        ):
            earlier = [(entries[k][0], space) for k, space in enumerate(spaces)]
            space = _axis_space(session, name, label, size, basis, axis_name, field, earlier)
            if space.dim != size:
                raise DataError(
                    f"the index {label} ranges over {space}, of dimension {space.dim}, "
                    f"but this axis of the data has {size} components"
                )
            if space.name == axis_name:  # a space a pandas axis names, for later data
                session.spaces.setdefault(axis_name, space)
            if basis is not None and space.basis is not None and basis != space.basis:
                what = f"axis {axis + 1} of {written}"
                components = _in_basis_order(components, axis, basis, space, what)
            spaces.append(space)
        if self._backend == "jax" and numerical.is_numerical(components):
            import jax
            import jax.numpy as jnp

            _double_precision(jax)
            if not numerical.is_sparse(components):  # a sparse matrix stays SciPy's
                components = jnp.asarray(components)
        indices = Indices(Index(label, upper) for label, upper in entries)
        node = session.graph.const(components, TensorType(indices, tuple(spaces)))
        session.globals.bind(name, TensorBinding(node, indices))

    def _scalar(self, written: str, value: Any) -> Any:
        """The node of a value bound to a name from Python: a number or a SymPy
        expression as it is, and text as the language reads it, which never runs Python."""
        session = self._session
        if isinstance(value, str):
            source = Source(value, f"<the value of {written}>")
            try:
                statements = parse(source).statements
                if len(statements) != 1 or not isinstance(statements[0], ast.Evaluate):
                    raise DataError("give one expression", span=Span(source, 0, len(value)))
                ctx = Context(latex=statements[0].latex)
                expr = statements[0].expr
                return self._run(
                    functools.partial(self._compiler.value, expr, session.globals, ctx)
                )
            except RelativisticPyError as error:
                raise DataError(
                    f"the text bound to {written} is not an expression: {error.message}",
                    span=error.span,
                    hint=f"write it in the language, as in wb[{written!r}] = 'x^2 + 1'",
                ) from None
        if numerical.is_numerical(value) and getattr(value, "shape", ()) != ():
            raise DataError(
                f"an array is bound to a tensor written with its indices, as in {written}^{{a}}"
            )
        if numerical.is_numerical(value):
            scalar = value[()] if isinstance(value, numerical.numpy().ndarray) else value
            return session.graph.const(scalar, SCALAR)  # a view of nothing
        if isinstance(value, sympy.Basic | numbers.Number):  # never text, which SymPy would run
            return session.graph.const(sympy.sympify(value, strict=True), SCALAR)
        raise DataError(
            f"{written} can be bound to a number, a SymPy expression or text in the "
            f"language, not {_kind(value)}"
        )

    @_from_python
    def grad(self, expression: str, wrt: str) -> Tensor:
        r"""The gradient of a scalar ``expression`` with respect to the data bound to the
        tensor ``wrt``, by automatic differentiation with JAX: the derivative of every
        operation the expression compiles to, contractions and linear solves included::

            >>> import numpy as np
            >>> wb = Workbook()
            >>> wb["w^a"] = np.array([1.0, 2.0])
            >>> wb["S_{ab}"] = np.array([[2.0, 0.0], [0.0, 3.0]])
            >>> wb.grad(r"w^a S_{ab} w^b", "w").to_numpy()        # 2 S_{ab} w^b
            array([ 4., 12.])

        The gradient has the indices of ``wrt`` in the other position: it is a
        covector where ``wrt`` is a vector. JAX is set to compute in 64 bits.
        """
        try:
            import jax
            import jax.numpy as jnp
        except ImportError as error:  # pragma: no cover - depends on the environment
            raise ImportError('gradients need JAX: pip install "relativisticpy[jax]"') from error
        outputs = self._statements(self._parse(expression, "<gradient>", "grad"))
        if not outputs or isinstance(outputs[-1].node.type, TensorType | EquationType):
            raise UsageError("a gradient is of a scalar expression, such as w^a S_{ab} w^b")
        node = outputs[-1].node
        binding = self._session.globals.lookup(wrt)
        if not isinstance(binding, TensorBinding):
            data = [
                name
                for name, other in self._session.globals.bindings()
                if isinstance(other, TensorBinding) and isinstance(other.node.op, ops.Const)
            ]
            raise UnknownNameError(
                f"{wrt} is not bound to data",
                notes=[f"the tensors bound to data are {', '.join(data)}"] if data else [],
                hint="bind it first, as in wb['w^a'] = np.array([1.0, 2.0])",
            )
        key = binding.node.key
        data = numerical.dense(numerical.as_array(self._executor.evaluate(binding.node)))
        _double_precision(jax)
        gradient = jax.grad(lambda x: self._executor.evaluate_with(node, {key: x}))(
            jnp.asarray(data, dtype=jnp.float64)
        )
        assert isinstance(binding.node.type, TensorType)
        indices = Indices(index.flip() for index in binding.indices)
        return Tensor(
            numerical.numpy().asarray(gradient),
            _public(indices),
            name=f"grad {wrt}",
            spaces=binding.node.type.spaces,
        )

    @_from_python
    def space(
        self,
        name: str,
        dim: int | None = None,
        *,
        basis: Iterable[Any] | None = None,
        labels: Iterable[str] = (),
        field: str = "real",
        measure: Any = None,
    ) -> Space:
        r"""Declare a space for indices to range over, as ``\mathcal{A} :=
        \mathbb{R}^{500}`` does: its ``dim``, or the names of its ``basis``; index
        ``labels`` that range over it; and a ``measure``, weights for its expectations
        (uniform otherwise), taken by label from a pandas Series::

            >>> wb = Workbook()
            >>> wb.space("assets", basis=["AAPL", "MSFT"], labels="ab")
            Space(name='assets', dim=2, basis=('AAPL', 'MSFT'), field='real', spacetime=False)
        """
        self._revision += 1
        names = tuple(str(b) for b in basis) if basis is not None else None
        size = len(names) if names is not None else dim
        if size is None:
            raise UsageError("give the dimension of the space, or the names of its basis")
        if field not in ("real", "complex"):
            raise UsageError(f"a space's field is real or complex, not {field!r}")
        kind: Literal["real", "complex"] = "complex" if field == "complex" else "real"
        try:
            space = Space(name, size, basis=names, field=kind)
        except ValueError as error:
            raise UsageError(str(error)) from None
        session = self._session
        if measure is not None:
            weights, labelled, _ = _data(measure, f"the measure on {name}")
            weights = numerical.numpy().asarray(numerical.dense(weights), dtype=float)
            if weights.shape != (size,):
                given = len(weights) if weights.ndim == 1 else None
                raise DataError(
                    f"a measure on {name} needs {size} weights, one for each element, but "
                    + (
                        f"{given} {'is' if given == 1 else 'are'} given"
                        if given is not None
                        else f"the weights have {_count(weights.ndim, 'axis', 'axes')}"
                    )
                )
            if labelled[0] is not None and names is not None:
                what = f"the measure on {name}"
                weights = _in_basis_order(weights, 0, labelled[0], space, what)
            session.measures[space] = session.graph.const(weights / weights.sum(), SCALAR)
        session.spaces[name] = space
        for label in labels:
            session.labels[label] = space
        return space

    @property
    def coordinates(self) -> tuple[Symbol, ...] | None:
        """The coordinates in force, declared or read from a line element; ``None``
        before there are any."""
        return self._session.coordinates

    @property
    def metric(self) -> Metric | None:
        """The metric in force, as a :class:`~relativisticpy.diffgeom.Metric`: the same
        one until the metric is defined again, so that what it computes, as its
        ``christoffel``, is computed once. Its components are those a program shows for
        ``g_{\\mu\\nu}``."""
        state, coordinates = self._session.metric, self._session.coordinates
        if state is None or coordinates is None:
            return None
        key = (state.node.key, coordinates)
        if self._metric is None or self._metric[0] != key:
            components = arrays.apply(self._executor.evaluate(state.node), tidy)
            metric = Metric(
                components, coordinates, check=False, simplify=self._executor.simplify_map
            )
            self._metric = (key, metric)
        return self._metric[1]

    # -- lifecycle ---------------------------------------------------------------

    def copy(self) -> Workbook:
        """A workbook in the state this one is in, whose definitions are its own from then
        on: what either defines afterwards, the other does not see. The copy knows what
        this one has computed, and what either computes, the other need not compute
        again::

            >>> wb = Workbook()
            >>> wb.expr("x := 2")
            >>> trial = wb.copy()
            >>> trial.expr("x := 3; x^2")
            9
            >>> wb.expr("x^2")
            4

        A copy computes with this workbook's worker processes, if it has any."""
        copied = object.__new__(type(self))
        copied.__dict__.update(self.__dict__)
        copied._session = self._session.fork()
        copied._compiler = Compiler(copied._session)
        scope = copied._session.globals
        for name, binding in scope.bindings():  # a function sees the names of its own copy
            if isinstance(binding, FunctionBinding) and binding.closure is self._session.globals:
                scope.bind(name, FunctionBinding(binding.definition, scope))
        return copied

    def _portable(self) -> tuple[str, Session, dict[Any, Any]]:
        """What another process needs to carry on from this workbook as it is: its backend,
        its session and the values it has computed, by computation (see :meth:`_carried`)."""
        return self._backend, self._session, dict(self._executor._values)

    @classmethod
    def _carried(cls, backend: str, session: Session, values: dict[Any, Any]) -> Workbook:
        """A workbook that carries on from what :meth:`_portable` gave in another process."""
        workbook = cls(backend=backend)
        workbook._session = session
        workbook._compiler = Compiler(session)
        workbook._executor._values.update(values)
        return workbook

    def reset(self) -> None:
        """Forget every definition."""
        self._executor.close()
        self._executor = Executor(self._workers)
        self._session = Session()
        self._compiler = Compiler(self._session)
        self._metric = None
        self._revision += 1

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
    def _parse(
        source: str | Source, name: str, method: str = "exe", *, recover: bool = False
    ) -> Program:
        if not isinstance(source, str | Source):
            raise UsageError(f"{method} takes the program as a str, not {_kind(source)}")
        if not isinstance(name, str):
            raise UsageError(f"the name of a program is a str, not {_kind(name)}")
        text = source if isinstance(source, Source) else Source(source, name)
        return parse(text, recover=recover)

    def _statements(self, program: Program) -> list[Output]:
        """Compile a program a statement at a time, running each statement's checks before
        the next is compiled: what it shows, in order."""
        outputs: list[Output] = []
        for statement in program.statements:
            compile = functools.partial(self._compiler.statement, statement, self._session.globals)
            outputs.extend(self._run(compile))
        return outputs

    def _run(self, compile: Callable[[], _T], compiler: Compiler | None = None) -> _T:
        """Compile, with ``compile``, then compute what the definitions compiled need of the
        values they make, as their checks say (see :class:`~relativisticpy.compiler.scope.Check`):
        so a metric that cannot be one is refused where it is defined, and what the
        statement defined is undone, as if it had not compiled. Compiling computes nothing;
        this is where the statement runs. Where the statement does not compile, what is
        written before the error is computed first, in order, so that an error of its own
        is said first, as written."""
        compiler = compiler or self._compiler
        self._revision += 1
        saved = compiler.save()
        try:
            compiled = compile()
        except RelativisticPyError:  # what is written before the error runs first
            for check in compiler.take_checks():
                self._verify(check)
            raise
        finally:
            checks = compiler.take_checks()
        try:
            for check in checks:
                self._verify(check)
        except BaseException:  # over a time limit too: what it defined is undone
            compiler.restore(saved)
            raise
        return compiled

    def _verify(self, check: Check) -> None:
        """Compute what a check needs. Where it cannot be computed, the error is its
        diagnosis's, if that says why, at the definition, or else the computation's own;
        with how the statement was read."""
        try:
            self._executor.evaluate(check.node)
        except RelativisticPyError as error:
            failure = self._diagnosed(check) or error
            if isinstance(failure, EvaluationError):
                failure.notes.extend(check.context)
                if check.hint is not None:
                    failure.hint = check.hint
            failure.notes.extend(note for note in check.notes if note not in failure.notes)
            raise failure from None

    def _diagnosed(self, check: Check) -> RelativisticPyError | None:
        """The error a check's diagnosis gives, where the definition is written; ``None``
        where it gives none, or where what it is computed from cannot be computed either."""
        if check.diagnosis is None:
            return None
        try:
            found = self._executor.evaluate(check.diagnosis)
        except Exception:  # the components cannot be computed: their own error says why
            return None
        if not isinstance(found, RelativisticPyError):
            return None
        return type(found)(found.message, span=check.span, hint=found.hint, notes=found.notes)

    def _present(self, output: Output) -> Any:
        """The value of an output as it is shown. What cannot be shown is a
        :class:`~relativisticpy.errors.RelativisticPyError` where the output was written,
        never a Python exception that would stop a run, or the check of a paper, at one
        statement."""
        value = self._executor.evaluate(output.node)
        if value is NotImplemented:  # an operation SymPy declined, as [1, 2] < [3, 4]
            raise EvaluationError("this is not defined for these values", span=output.span)
        if (undecided := _undecided(output, value)) is not None:
            raise undecided
        kind = output.node.type
        coordinates: Sequence[Symbol] = self._session.coordinates or ()
        try:
            if isinstance(kind, TensorType):
                components = value if numerical.is_numerical(value) else arrays.apply(value, tidy)
                return Tensor(
                    components,
                    _public(kind.indices),
                    coordinates,
                    name=output.name,
                    spaces=kind.spaces,
                    source=output.span.text,
                )
            if isinstance(kind, EquationType) and isinstance(kind.of, TensorType):
                assert isinstance(value, TensorEquation)
                return TensorEquation(
                    value.lhs,
                    value.rhs,
                    _public(kind.of.indices),
                    coordinates,
                    kind.of.spaces,
                    scale=value.scale,
                    source=output.span.text,
                )
            if numerical.is_numerical(value) and getattr(value, "shape", ()) == ():
                return value.item()  # a plain Python number
            if isinstance(kind, ScalarType):
                return tidy(value)
            return value
        except Exception as error:  # in RelativisticPy's words, or an internal error
            failure = translate(error, span=output.span)
            if failure is error or isinstance(failure, InternalError):
                raise failure from failure.__cause__  # its traceback, which a report needs
            raise failure from None


@dataclass(frozen=True)
class Step:
    """One statement of a program, run by :meth:`Workbook.each`: the values it showed
    (none for a definition), or the error it raised, and the notes on how it was read."""

    span: Span
    statement: ast.Statement
    values: list[Any] | None = None
    error: RelativisticPyError | None = None
    sides: tuple[Any, Any] | None = None  # an equation's two sides, as computed
    notes: list[str] = field(default_factory=list)  # how the statement was read

    @property
    def zero(self) -> bool:
        """Whether this is an equation whose two sides are both zero."""
        return self.sides is not None and all(_zero(side) for side in self.sides)

    def __repr__(self) -> str:
        """``Step(line 1: R_{ab} → CompileError: the Ricci tensor needs coordinates)``,
        or ``Step(line 1: x + 1 → [x + 1])``."""
        text = " ".join(self.span.text.split())
        if len(text) > 60:
            text = text[:59] + "…"
        if self.error is not None:
            outcome = f"{self.error.kind}: {self.error.message}"
        elif self.values:
            outcome = "[" + ", ".join(_shown(value) for value in self.values) + "]"
        else:
            outcome = "defined"
        return f"Step(line {self.span.line}: {text} → {outcome})"


def _shown(value: Any) -> str:
    """A value as a repr shows it; never a failure."""
    try:
        return display.unicode(value)
    except Exception:  # a repr must not fail, whatever the value
        return f"<a {type(value).__name__}>"


@dataclass(frozen=True)
class Definition:
    """Something a workbook defines: its ``kind``, one of ``coordinate``, ``metric``,
    ``value``, ``tensor``, ``function`` and ``space``; how it is ``written``, with its
    indices or parameters; and, if a statement defined it, its ``span``."""

    kind: str
    written: str
    span: Span | None = field(default=None, compare=False)

    def __repr__(self) -> str:
        return f"<{self.kind} {self.written}>"

    def to_dict(self) -> dict[str, Any]:
        """The definition as data: its kind, how it is written and where."""
        where = place(self.span)
        return {
            "kind": self.kind,
            "written": self.written,
            "file": where["file"],
            "line": where["line"],
            "column": where["column"],
        }


def _definition(name: str, binding: Any) -> Definition:
    """What a binding of ``name`` is, as :meth:`Workbook.names` lists it."""
    written = latex_label(name)
    if isinstance(binding, FunctionBinding):
        parameters = ", ".join(latex_label(p) for p in binding.definition.params)
        return Definition("function", f"{written}({parameters})", binding.definition.span)
    span = binding.node.span
    if isinstance(binding, TensorBinding):
        return Definition("tensor", written + _public(binding.indices).latex(), span)
    if isinstance(binding.node.type, TensorType):
        return Definition("tensor", written + _public(binding.node.type.indices).latex(), span)
    return Definition("value", written, span)


def _spelled(name: str, entries: Sequence[tuple[str, bool]]) -> str:
    """A tensor written with its indices, unambiguously: ``w^{a}_{b}``."""
    return name + "".join(f"{'^' if upper else '_'}{{{label}}}" for label, upper in entries)


def _shortened(span: Span | None, width: int = 60) -> str:
    """A statement on one line, cut short if long."""
    text = " ".join(span.text.split()) if span is not None else ""
    return text if len(text) <= width else text[: width - 1] + "…"


def _kind(value: Any) -> str:
    """What kind of Python value ``value`` is, as a message says it: ``int``, ``None``."""
    return "None" if value is None else type(value).__name__


def _count(number: int, one: str, many: str) -> str:
    """``1 index``, ``2 indices``."""
    return f"{number} {one if number == 1 else many}"


def _path(path: Any, method: str) -> Path:
    if not isinstance(path, str | Path):
        raise UsageError(f"{method} takes the path of a file, not {_kind(path)}")
    return Path(path)


def _not_utf8(path: Path, error: UnicodeDecodeError) -> RelativisticPyError:
    """The error for a file that is not UTF-8 text."""
    text = error.object[: error.start]
    line = text.count(b"\n") + 1
    column = error.start - (text.rfind(b"\n") + 1) + 1
    return FileError(
        f"{path} is not UTF-8 text: the byte 0x{error.object[error.start]:02X} at line "
        f"{line}, column {column} is not",
        hint=f"save it as UTF-8, as with iconv -f latin1 -t utf-8 {path}",
    )


def _unreadable(path: Path, error: OSError) -> RelativisticPyError:
    """The error for a file that cannot be read."""
    if isinstance(error, FileNotFoundError):
        return FileError(f"no file {path}")
    if isinstance(error, IsADirectoryError):
        return FileError(f"{path} is a folder, not a file")
    return FileError(f"cannot read {path}: {error.strerror or error}")


# -- the files a run reads ---------------------------------------------------------------

#: The comments around the results that ``relativisticpy --render`` typesets in a
#: Markdown notebook, each on a line of its own: rendering the notebook again replaces
#: what is between them, and running or rendering it never runs it as input.
RESULT_START = "<!-- relativisticpy:result -->"
RESULT_END = "<!-- /relativisticpy:result -->"

_RESULT = re.compile(
    r"(?:\n[ \t]*){0,2}^" + re.escape(RESULT_START) + r"[ \t]*\n.*?^" + re.escape(RESULT_END),
    re.DOTALL | re.MULTILINE,
)
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})[^\n]*\n.*?^ {0,3}\1[ \t]*$", re.DOTALL | re.MULTILINE)
_MARKDOWN_SUFFIXES = (".md", ".markdown")
_LATEX_SUFFIXES = (".tex", ".ltx", ".latex")
_LATEX_DOCUMENT = re.compile(r"\\documentclass\b|\\begin\s*\{document\}")


def rendered_results(text: str) -> list[tuple[int, int]]:
    r"""Where a Markdown notebook holds results that ``relativisticpy --render``
    typeset: the start and the end of each, from the line break before it, outside
    code blocks::

        >>> rendered_results("$$ x^2 $$\n\n" + RESULT_START + "\n$$ x^{2} $$\n" + RESULT_END)
        [(9, 85)]
    """
    fences = [match.span() for match in _FENCE.finditer(text)]
    found = []
    for match in _RESULT.finditer(text):
        marker = text.index(RESULT_START, match.start())
        if not any(start <= marker < end for start, end in fences):
            found.append(match.span())
    return found


def notebook_blocks(text: str, name: str = "<markdown>") -> list[Source]:
    """The blocks a Markdown notebook runs, in order, as
    :func:`~relativisticpy.documents.math_blocks` finds them, but not the results that
    ``relativisticpy --render`` typeset in it, which are not input."""
    return math_blocks(_blanked(text, rendered_results(text)), name)


def _blanked(text: str, spans: Sequence[tuple[int, int]]) -> str:
    """``text`` with each span made blank, its line breaks and every offset kept, so that
    the lines of what follows are still the file's."""
    pieces, done = [], 0
    for start, end in spans:
        pieces.append(text[done:start])
        pieces.append(re.sub(r"[^\n]", " ", text[start:end]))
        done = end
    pieces.append(text[done:])
    return "".join(pieces)


def render_notebook(
    workbook: Workbook, text: str, name: str = "<markdown>"
) -> tuple[str, list[RelativisticPyError]]:
    """A Markdown notebook with the results of each block typeset after it, and the
    errors of the blocks that failed, each typeset in its place too.

    The results are put between :data:`RESULT_START` and :data:`RESULT_END`, so that
    rendering the document again replaces them rather than running them, and gives the
    same document. ``relativisticpy --render`` and :meth:`Workbook.markdown` both render
    with this."""
    old = rendered_results(text)
    blanked = _blanked(text, old)
    pieces: list[str] = []
    errors: list[RelativisticPyError] = []
    done = 0
    for block, end in find_math_blocks(blanked, name):
        body: list[str] = []
        for step in workbook.each(block, recover=True):
            if step.error is not None:
                errors.append(step.error)
                body.append(_fenced(_placeless(step.error.render())))
                continue
            body.extend(f"$$\n{latex(value)}\n$$" for value in step.values or [])
            body.extend(f"> note: {note}" for note in step.notes)
        pieces.append(_without(text, done, end, old))
        if body:
            pieces.append(f"\n\n{RESULT_START}\n" + "\n\n".join(body) + f"\n{RESULT_END}")
        done = end
    pieces.append(_without(text, done, len(text), old))
    return "".join(pieces), errors


def _without(text: str, start: int, end: int, spans: Sequence[tuple[int, int]]) -> str:
    """``text[start:end]`` without the spans in it."""
    pieces = []
    for first, last in spans:
        if last <= start or first >= end:
            continue
        pieces.append(text[start:first])
        start = last
    pieces.append(text[start:end])
    return "".join(pieces)


def _placeless(diagnostic: str) -> str:
    """A diagnostic without the lines of the file it points at, as it is typeset under
    its block: where it is goes without saying there, and its lines change as results
    are typeset above it, which a document rendered again must not."""
    lines = []
    for line in diagnostic.split("\n"):
        if re.match(r"\s*(?:-->|:::) ", line):
            continue
        gutter = re.match(r"\s*\d*\s?\|", line)
        lines.append("  |" + line[gutter.end() :] if gutter else line)
    return "\n".join(lines)


def _fenced(text: str) -> str:
    """Text in a fenced block of code, whose fence nothing in the text closes."""
    longest = max((len(run) for run in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}text\n{text}\n{fence}"


def read_program(path: str | Path) -> list[Source]:
    r"""What running a file runs: the text of a program, or the blocks of a Markdown
    notebook (``.md``, ``.markdown``), as :func:`notebook_blocks` finds them. The file
    is UTF-8 text, with a byte order mark or without.

    A file that cannot be read is a :class:`~relativisticpy.errors.FileError`, and so is
    a LaTeX document (``.tex``, or text with ``\documentclass``), whose equations are
    checked rather than run, and a Jupyter notebook; each says what to do instead."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".ipynb":
        raise FileError(
            f"{path} is a Jupyter notebook, which RelativisticPy does not run",
            hint=f"export it as Markdown, as with jupyter nbconvert --to markdown {path}, "
            "and run that",
        )
    try:
        text = path.read_text(encoding="utf-8-sig")  # with a byte order mark or without
    except UnicodeDecodeError as error:
        raise _not_utf8(path, error) from None
    except OSError as error:
        raise _unreadable(path, error) from None
    if suffix in _MARKDOWN_SUFFIXES:
        return notebook_blocks(text, str(path))
    if suffix in _LATEX_SUFFIXES or _LATEX_DOCUMENT.search(text):
        raise FileError(
            f"{path} is a LaTeX document, not a program",
            hint=f"to check its equations: relativisticpy check {path} (in Python, "
            f"wb.check({str(path)!r}))",
        )
    return [Source(text, str(path))]


def _double_precision(jax: Any) -> None:
    """Turn on JAX's 64-bit numbers, which numerical work needs: by default JAX
    computes in 32 bits."""
    if not jax.config.read("jax_enable_x64"):
        jax.config.update("jax_enable_x64", True)


def _written(written: str) -> ast.Name | tuple[str, list[tuple[str, bool]]]:
    """What ``wb[written] = ...`` binds: a name, or a tensor with its index labels."""
    try:
        program = parse(Source(written, "<binding>"))
    except RelativisticPyError:
        program = None
    statements = program.statements if program is not None else ()
    expr = (
        statements[0].expr
        if len(statements) == 1 and isinstance(statements[0], ast.Evaluate)
        else None
    )
    if isinstance(expr, ast.Name):
        return expr
    if isinstance(expr, ast.Scripted) and isinstance(expr.base, ast.Name):
        name = expr.base.id
        entries: list[tuple[str, bool]] = []
        head, underscore, tail = name.partition("_")
        if underscore and tail.isalpha():  # B_a^b reads as the name B_a; it is B_{a}{}^{b}
            name = head
            entries.extend((label, False) for label in split_index_labels(tail))
        count = len(entries)
        for script in expr.scripts:
            for item in script.indices.items if script.indices is not None else ():
                count += 1
                if item.label is None or item.value is not None or item.derivative is not None:
                    continue
                # X^i_j reads as X^{i_j}; as a binding it can only mean X^{i}{}_{j}
                first, *rest = str(item.label).split("_")
                entries.append((first, script.kind == "^"))
                entries.extend((label, False) for label in rest if label)
                count += sum(1 for label in rest if label)
        if entries and len(entries) == count == len({label for label, _ in entries}):
            return name, entries
    raise DataError(
        f"cannot bind {written!r}: write a name, as in x, or a tensor with distinct index "
        "labels, as in R_{ta}"
    )


def _rank_of(value: Any) -> int:
    shape = getattr(value, "shape", None)
    if shape is None and isinstance(value, list | tuple):
        return 1
    return len(shape) if shape is not None else 0


def _data(value: Any, written: str) -> tuple[Any, list[tuple[str, ...] | None], list[str | None]]:
    """Components, the names of each axis's elements and each axis's name, from a pandas
    object; just the components from anything else. The components are a copy, so that
    data changed in place afterwards changes nothing bound. Text is refused, never read
    as SymPy would read it, as Python."""
    module = type(value).__module__ or ""
    if module.startswith("pandas"):
        axes = [value.index] if value.ndim == 1 else [value.index, value.columns]
        bases = [None if _is_range(axis) else tuple(str(x) for x in axis) for axis in axes]
        names = [str(axis.name) if axis.name is not None else None for axis in axes]
        return _copy(_numbers(value.to_numpy(), written)), bases, names
    if numerical.is_numerical(value):
        rank = len(getattr(value, "shape", ()))
        return _copy(_numbers(value, written)), [None] * rank, [None] * rank
    if isinstance(value, sympy.NDimArray | sympy.MatrixBase):
        array = arrays.to_array(value)
        return array, [None] * array.rank(), [None] * array.rank()
    if not isinstance(value, list | tuple):
        raise DataError(
            f"the data for {written} is {_kind(value)}: bind an array, a pandas object, a "
            "SciPy sparse matrix, a SymPy array or nested lists"
        )
    _refuse_text(value, written)
    try:
        array = numerical.numpy().array(value)
    except ValueError:
        where = _ragged(value) or "its rows are not all of one length"
        raise DataError(f"the data for {written} is ragged: {where}") from None
    if array.dtype == object:  # SymPy expressions among the numbers
        array = arrays.to_array(value)
        return array, [None] * array.rank(), [None] * array.rank()
    return array, [None] * array.ndim, [None] * array.ndim


def _numbers(array: Any, written: str) -> Any:
    """A numerical array, refused if it holds text."""
    dtype = getattr(array, "dtype", None)
    if dtype is not None and dtype.kind in "USa":
        raise DataError(f"the data for {written} is text: bind numbers")
    if dtype is not None and dtype.kind == "O":
        _refuse_text(array.tolist(), written)
    return array


def _refuse_text(value: Any, written: str) -> None:
    """Refuse text anywhere in nested lists: SymPy would read it as Python, and run it."""
    if isinstance(value, str | bytes):
        raise DataError(
            f"the data for {written} holds text, {value[:40]!r}: bind numbers or SymPy expressions",
            hint="to compute with an expression written as text, write it in a program",
        )
    if isinstance(value, list | tuple):
        for item in value:
            _refuse_text(item, written)


def _ragged(value: Sequence[Any]) -> str:
    """Where nested lists are not rectangular, as a message says it; empty if they are."""
    nested = [isinstance(item, list | tuple) for item in value]
    if not any(nested):
        return ""
    if not all(nested):
        odd = (
            nested.index(True) if nested.count(True) < nested.count(False) else nested.index(False)
        )
        what = "a list" if nested[odd] else "a number"
        others = "numbers" if nested[odd] else "lists"
        return f"entry {odd + 1} is {what}, the others are {others}"
    lengths = [len(item) for item in value]
    for position, length in enumerate(lengths):
        if length != lengths[0]:
            return f"entry {position + 1} has {length} entries, but entry 1 has {lengths[0]}"
    for position, item in enumerate(value):
        inner = _ragged(item)
        if inner:
            return f"in entry {position + 1}, {inner}"
    return ""


def _copy(array: Any) -> Any:
    """A numerical array of one's own: a copy, except of a JAX array, which cannot change,
    with small integers widened so that arithmetic cannot overflow them."""
    if numerical.is_jax(array):
        return array
    if numerical.is_sparse(array):
        return numerical.widened(array.copy())
    return numerical.widened(numerical.numpy().array(array))


def _axis_space(
    session: Session,
    name: str,
    label: str,
    size: int,
    basis: tuple[str, ...] | None,
    axis_name: str | None,
    field: Literal["real", "complex"],
    earlier: Sequence[tuple[str, Space]],
) -> Space:
    """The space an axis of the data bound to the tensor ``name`` ranges over, given its
    index ``label`` and the name of its pandas axis, after the ``earlier`` axes' labels
    and spaces: the rules of :meth:`Workbook.__setitem__`."""
    declared = session.labels.get(label)
    if declared is not None:
        return declared
    if axis_name is not None:
        named = session.spaces.get(axis_name)
        return named if named is not None else Space(axis_name, size, basis=basis, field=field)
    candidates = [space for other, space in earlier if _alike(other, label)]
    candidates += _spaces_of_label(session.globals, name, label)
    for space in candidates:
        if space.dim == size and _same_elements(space.basis, basis):
            return space
    return Space(f"the space of {label}", size, basis=basis, field=field)


def _alike(one: str, other: str) -> bool:
    """Whether two index labels are of a kind: from the same run of the alphabet, a to h
    as assets a and b are, i to n as nodes i and j, o to z as times s and t; or Greek."""
    return _run(one) == _run(other)


def _run(label: str) -> str:
    if label in GREEK or label in GREEK_VARIANTS:
        return "Greek"
    first = label[:1].lower()
    for start, end in (("a", "h"), ("i", "n"), ("o", "z")):
        if start <= first <= end:
            return start
    return label  # alike only to itself


def _spaces_of_label(scope: Scope, name: str, label: str) -> list[Space]:
    """The spaces the index ``label`` ranges over in tensors already bound, other than
    spacetime: in the tensor ``name`` first, then in the others, the latest first."""
    bindings = scope.bindings()
    ordered = [b for n, b in bindings if n == name] + [b for n, b in bindings[::-1] if n != name]
    return [
        space
        for binding in ordered
        if isinstance(binding, TensorBinding) and isinstance(binding.node.type, TensorType)
        for index, space in zip(binding.indices, binding.node.type.spaces, strict=True)
        if index.name == label and not space.spacetime
    ]


def _same_elements(one: tuple[str, ...] | None, other: tuple[str, ...] | None) -> bool:
    """Whether two bases may name the same elements: they do, or one has no names."""
    return one is None or other is None or sorted(one) == sorted(other)


def _in_basis_order(
    components: Any, axis: int, basis: tuple[str, ...], space: Space, what: str
) -> Any:
    """Data labelled by pandas along ``axis``, put in the order of its space's basis; an
    error if its labels are not the space's elements."""
    assert space.basis is not None
    if sorted(basis) != sorted(space.basis):
        foreign = [element for element in basis if element not in space.basis]
        missing = [element for element in space.basis if element not in basis]
        details = []
        if foreign:
            details.append(f"{_listed(foreign)} {'is' if len(foreign) == 1 else 'are'} not one")
        if missing:
            details.append(f"{_listed(missing)} {'is' if len(missing) == 1 else 'are'} missing")
        raise CompileError(
            f"the labels of {what} are not the elements of {space}: "
            + ("; ".join(details) if details else "some appear more than once")
        )
    position = {element: number for number, element in enumerate(basis)}
    order = [position[element] for element in space.basis]
    return numerical.numpy().take(components, order, axis=axis)


def _listed(names: Sequence[str]) -> str:
    shown = ", ".join(names[:3])
    return shown + (f" and {len(names) - 3} more" if len(names) > 3 else "")


def _is_range(axis: Any) -> bool:
    """Whether a pandas axis is the default numbering, 0, 1, 2, ..., not names."""
    return type(axis).__name__ == "RangeIndex" and axis.start == 0 and axis.step == 1


#: The relations a statement claims, as ``e^x \geq 1 + x``, besides equations.
_RELATIONS = frozenset({"core.lt", "core.gt", "core.le", "core.ge", "core.eq", "core.ne"})


def _undecided(output: Output, value: Any) -> UndecidedError | None:
    r"""The error for a claim whose truth is not decided, as ``holds`` judges it: an equation
    SymPy could not show to hold, though numbers do not show its sides to differ, or an
    inequality between expressions SymPy leaves as written. Shown as itself, it would read
    as an answer (guard 5 of wave 0, curriculum/ROADMAP.md). An equation that numbers show
    not to hold stays an equation, to solve; and an inequality between rational functions
    of names, as ``a > 0`` or ``x \geq y + 1``, states a condition, and stays as written."""
    op = output.node.op
    claim = isinstance(op, ops.Equation) or (isinstance(op, ops.Call) and op.name in _RELATIONS)
    if not claim or _a_condition(value):
        return None
    try:
        verdict = holds(value)
    except TypeError:  # no claim after all, as an equation of lists
        return None
    if verdict is not None:
        return None
    written = " ".join(output.span.text.split())
    if isinstance(value, TensorEquation):
        analysis = value.analysis()
        why = (
            f"in {analysis.failing} of {analysis.total} components, SymPy could not show the "
            "sides equal, nor numbers that they differ"
        )
    elif isinstance(value, sympy.Equality) and _asymptotic(value):
        why = "it has an order term, O(...), which no value at a point decides"
    elif isinstance(value, sympy.Equality) and differs(value.lhs, value.rhs) is False:
        why = "the sides agree numerically, but this was not proven"
    elif isinstance(value, sympy.Equality):
        why = "the sides could not be computed as numbers"
    else:
        why = "SymPy decides an inequality only where it can compare its sides"
    return UndecidedError(
        f"whether {written} holds is not decided: {why}",
        span=output.span,
        notes=["shown as it is, an undecided claim would read as an answer"],
        hint="to see what is left, write the difference of its sides alone, as an expression",
    )


def _a_condition(value: Any) -> bool:
    r"""Whether a value is an inequality between rational functions of names, as ``a > 0``,
    ``r' < r`` or ``k_B T > h c B``: a condition the program states, as books state the
    regime they work in, rather than a claim to decide, as ``e^x \geq 1 + x`` is."""
    if not isinstance(value, Relational) or isinstance(value, sympy.Equality):
        return False
    return all(isinstance(side, sympy.Expr) and side.is_rational_function() for side in value.args)


def _zero(value: Any) -> bool:
    if numerical.is_numerical(value):
        return numerical.close(value, 0)
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


def _steps_of(program: Program) -> list[tuple[ast.Statement | ParseError, Span]]:
    """The steps :meth:`Workbook.each` gives for a program, in order, with their spans:
    each statement, each link of a chain as a statement of its own, and each statement
    that cannot be read, as its error."""
    found: list[tuple[ast.Statement | ParseError, Span]] = []
    for item, span in _in_order(program):
        if isinstance(item, ParseError):
            found.append((item, span))
        else:
            found.extend((piece, piece.span) for piece in _pieces(item))
    return found


#: What separates statements, and the space around them: what a statement that cannot be
#: read does not begin or end with.
_BETWEEN = " \t\r\n;"


def _in_order(program: Program) -> list[tuple[ast.Statement | ParseError, Span]]:
    """The statements of a program and the errors of those that could not be read, in
    the order they are written, each with the span of its statement.

    A statement that cannot be read is the text between what comes before it and what
    comes after it: from the end of the statement before it, or, after another that
    cannot be read either, from the start of the line its error is on; to the start of
    what follows. Separators, space and comments are left out at either end."""
    if not program.errors:
        return [(statement, statement.span) for statement in program.statements]
    items = program.in_order()
    source = program.span.source
    text = source.text

    def own(item: ast.Statement | ParseError) -> Span | None:
        """The span of an item, if it is in this program's text."""
        span = item.span
        return span if span is not None and span.source is source else None

    starts: list[int] = []
    for position, item in enumerate(items):
        previous = items[position - 1] if position else None
        before = own(previous) if previous is not None else None
        done = before.end if before is not None else 0
        span = own(item)
        if not isinstance(item, ParseError):
            starts.append(span.start if span is not None else done)
        elif isinstance(previous, ParseError) and span is not None:
            line = text.rfind("\n", 0, span.start) + 1  # its own statement starts on its line
            starts.append(max(line, min(done, span.start)))
        else:
            starts.append(done)
    found: list[tuple[ast.Statement | ParseError, Span]] = []
    for position, item in enumerate(items):
        if not isinstance(item, ParseError):
            found.append((item, item.span))
            continue
        end = starts[position + 1] if position + 1 < len(items) else len(text)
        start = _skip_between(text, starts[position], end)
        while end > start and text[end - 1] in _BETWEEN:
            end -= 1
        if text.endswith("\\\\", start, end):  # a row break ends a row, not a statement
            end -= 2
            while end > start and text[end - 1] in _BETWEEN:
                end -= 1
        span = own(item)
        if end <= start and span is not None:  # nothing but the error itself
            start, end = span.start, span.end
        found.append((item, Span(source, start, max(start, end))))
    return found


def _skip_between(text: str, start: int, end: int) -> int:
    """Where a statement starts, from ``start``: after separators, space, row breaks and
    lines that are comments."""
    while start < end:
        if text[start] in _BETWEEN:
            start += 1
        elif text.startswith("\\\\", start):
            start += 2
        elif text[start] == "%":  # a comment, to the end of its line
            newline = text.find("\n", start)
            start = end if newline == -1 else newline + 1
        else:
            break
    return min(start, end)


def latex(value: Any) -> str:
    """Typeset a result, as :func:`relativisticpy.display.latex` does."""
    return display.latex(value)


def _nowhere() -> Span:
    """The span of something written nowhere, as data bound from Python is."""
    return Span(Source(""), 0, 0)
