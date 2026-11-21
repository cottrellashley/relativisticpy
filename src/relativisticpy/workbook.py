r"""The :class:`Workbook`: a session in which RelativisticPy programs run."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Literal

import sympy
from sympy import Symbol

from relativisticpy import checking
from relativisticpy.compiler import Compiler, Session
from relativisticpy.compiler.compiler import _public
from relativisticpy.compiler.scope import Context, Output, Scope, TensorBinding, VariableBinding
from relativisticpy.diffgeom import Metric
from relativisticpy.documents import Document, find_math_blocks, math_blocks
from relativisticpy.errors import CompileError, EvaluationError, RelativisticPyError
from relativisticpy.ir import SCALAR, EquationType, ScalarType, TensorType, render
from relativisticpy.language import Source, Span, parse, tokenize
from relativisticpy.language import syntax as ast
from relativisticpy.language.parser import split_index_labels
from relativisticpy.language.syntax import Program
from relativisticpy.language.tokens import Token
from relativisticpy.language.vocabulary import GREEK, GREEK_VARIANTS
from relativisticpy.runtime import Executor
from relativisticpy.tensors import Index, Indices, Space, Tensor, arrays, numerical
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
    simplification (``None`` means one per CPU). Batches of a few components or more
    use them. ``backend="jax"`` keeps data bound from Python as JAX arrays, so that it
    is computed with JAX, on a GPU if there is one.
    """

    def __init__(self, *, workers: int | None = 1, backend: str = "numpy") -> None:
        if backend not in ("numpy", "jax"):
            raise ValueError(f"the backend is numpy or jax, not {backend!r}")
        self._backend = backend
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
                except Exception as error:  # a failure of RelativisticPy's: the next goes on
                    message = f"{type(error).__name__}: {error}" if str(error) else repr(error)
                    failure = EvaluationError(message, span=piece.span)
                    yield Step(piece.span, piece, error=failure)
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
        except Exception:  # the sides only say how an equation differs: without them, it does
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
        text = path.read_text(encoding="utf-8-sig")  # with a byte order mark or without
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
        """The IR that ``source`` compiles to, as text. Nothing is computed, and the
        workbook is left as it was: a definition in ``source`` is not made."""
        compiler = Compiler(self._session.fork())
        outputs = compiler.compile(self._parse(source, "<input>"))
        return render([output.node for output in outputs])

    def __getitem__(self, name: str) -> Any:
        """The value bound to ``name``: ``wb["x"]``, or a tensor such as ``wb["T"]``."""
        binding = self._session.globals.lookup(name)
        if isinstance(binding, TensorBinding | VariableBinding):
            return self._present(Output(binding.node, binding.node.span or _nowhere(), name))
        raise KeyError(name)

    def __setitem__(self, written: str, value: Any) -> None:
        r"""Bind data from Python: ``wb["R_{ta}"] = returns`` binds a tensor whose first
        index ranges over the rows of ``returns`` and its second over the columns, and
        ``wb["J"] = 1.5`` a number.

        The data may be a NumPy array, a pandas DataFrame or Series, a SciPy sparse
        matrix, a SymPy array or nested lists, and is copied. Each axis ranges over:

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

        Each needs the axis's size. Data labelled by pandas is put in the order of its
        space's basis, by label. Binding declares nothing: the labels of ``R_{ta}`` range
        over its spaces in statements that write them on it, and are free everywhere
        else, so that ``t`` may still be a coordinate in ``\Gamma^{t}_{tr}``::

            >>> import numpy as np
            >>> wb = Workbook()
            >>> wb["w^a"] = np.array([0.5, 0.25, 0.25])
            >>> wb.expr("w^{a} w^{b} \\delta_{ab}")
            0.375
        """
        target = _written(written)
        session = self._session
        if isinstance(target, ast.Name) and "_" in target.id and _rank_of(value):
            base, _, tail = target.id.partition("_")  # y_t: the tensor y, with index t
            target = (base, [(label, False) for label in split_index_labels(tail)])
        if isinstance(target, ast.Name):
            session.globals.bind(
                target.id, VariableBinding(session.graph.const(_scalar(value), SCALAR))
            )
            return
        name, entries = target
        components, bases, axis_names = _data(value)
        shape = tuple(getattr(components, "shape", ()))
        if len(shape) != len(entries):
            raise CompileError(
                f"{written} has {len(entries)} indices but the data has {len(shape)} axes"
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
                raise CompileError(
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
            if not numerical.is_sparse(components):  # sparse stays SciPy's, and sparse
                components = jnp.asarray(components)
        indices = Indices(Index(label, upper) for label, upper in entries)
        node = session.graph.const(components, TensorType(indices, tuple(spaces)))
        session.globals.bind(name, TensorBinding(node, indices))

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
        outputs = self._compiler.compile(self._parse(expression, "<gradient>"))
        if not outputs or isinstance(outputs[-1].node.type, TensorType | EquationType):
            raise ValueError("a gradient is of a scalar expression, such as w^a S_{ab} w^b")
        node = outputs[-1].node
        binding = self._session.globals.lookup(wrt)
        if not isinstance(binding, TensorBinding):
            raise KeyError(f"{wrt} is not a tensor bound to data")
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
        names = tuple(str(b) for b in basis) if basis is not None else None
        size = len(names) if names is not None else dim
        if size is None:
            raise ValueError("give the dimension of the space, or the names of its basis")
        kind: Literal["real", "complex"] = "complex" if field == "complex" else "real"
        space = Space(name, size, basis=names, field=kind)
        session = self._session
        if measure is not None:
            weights, labelled, _ = _data(measure)
            weights = numerical.numpy().asarray(numerical.dense(weights), dtype=float)
            if weights.shape != (size,):
                raise ValueError(f"a measure on {name} has {size} weights")
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
        """The value of an output as it is shown. What cannot be shown is an
        :class:`EvaluationError` where the output was written, never a Python exception
        that would stop a run, or the check of a paper, at one statement."""
        value = self._executor.evaluate(output.node)
        if value is NotImplemented:  # an operation SymPy declined, as [1, 2] < [3, 4]
            raise EvaluationError("this is not defined for these values", span=output.span)
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
                )
            if isinstance(kind, EquationType) and isinstance(kind.of, TensorType):
                assert isinstance(value, TensorEquation)
                return TensorEquation(
                    value.lhs, value.rhs, _public(kind.of.indices), coordinates, kind.of.spaces
                )
            if numerical.is_numerical(value) and getattr(value, "shape", ()) == ():
                return value.item()  # a plain Python number
            if isinstance(kind, ScalarType):
                return tidy(value)
            return value
        except RelativisticPyError:
            raise
        except Exception as error:
            message = str(error) or type(error).__name__
            raise EvaluationError(message, span=output.span) from error


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
        entries: list[tuple[str, bool]] = []
        count = 0
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
            return expr.base.id, entries
    raise CompileError(
        f"cannot bind {written!r}: write a name, as in x, or a tensor with distinct index "
        "labels, as in R_{ta}"
    )


def _rank_of(value: Any) -> int:
    shape = getattr(value, "shape", None)
    if shape is None and isinstance(value, list | tuple):
        return 1
    return len(shape) if shape is not None else 0


def _scalar(value: Any) -> Any:
    if numerical.is_numerical(value) and getattr(value, "shape", ()) != ():
        raise CompileError("an array is bound to a tensor written with its indices, as in w^{a}")
    if not numerical.is_numerical(value):
        return sympy.sympify(value)
    return value[()] if isinstance(value, numerical.numpy().ndarray) else value  # not a view


def _data(value: Any) -> tuple[Any, list[tuple[str, ...] | None], list[str | None]]:
    """Components, the names of each axis's elements and each axis's name, from a pandas
    object; just the components from anything else. The components are a copy, so that
    data changed in place afterwards changes nothing bound."""
    module = type(value).__module__ or ""
    if module.startswith("pandas"):
        axes = [value.index] if value.ndim == 1 else [value.index, value.columns]
        bases = [None if _is_range(axis) else tuple(str(x) for x in axis) for axis in axes]
        names = [str(axis.name) if axis.name is not None else None for axis in axes]
        return _copy(value.to_numpy()), bases, names
    if numerical.is_numerical(value):
        rank = len(getattr(value, "shape", ()))
        return _copy(value), [None] * rank, [None] * rank
    if isinstance(value, sympy.NDimArray | sympy.MatrixBase):
        array = arrays.to_array(value)
        return array, [None] * array.rank(), [None] * array.rank()
    array = numerical.numpy().array(value)
    if array.dtype == object:
        array = arrays.to_array(value)
        return array, [None] * array.rank(), [None] * array.rank()
    return array, [None] * array.ndim, [None] * array.ndim


def _copy(array: Any) -> Any:
    """A numerical array of one's own: a copy, except of a JAX array, which cannot change."""
    if numerical.is_jax(array):
        return array
    if numerical.is_sparse(array):
        return array.copy()
    return numerical.numpy().array(array)


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
    bindings = list(scope._bindings.items())
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


def latex(value: Any) -> str:
    """Typeset a result: a tensor, a tensor equation, or any SymPy value."""
    if hasattr(value, "latex") and callable(value.latex):
        return str(value.latex())
    return str(sympy.latex(value))


def _nowhere() -> Any:
    return Source("").__class__("")
