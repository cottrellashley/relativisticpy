r"""Python objects in a program: ``\texttt{prices}`` is the object called ``prices`` in the
Python scope the workbook was given, read as what it becomes (see
:mod:`relativisticpy.integrations._objects`) before anything else of the statement is read,
since what ``\texttt`` names is never a name of the language's.

- A number or an expression is a scalar, its symbols the program's, by name.
- Data is a tensor whose indices no one wrote, each axis over a space: the space whose
  elements are its labels, if one is in use, or one named after its axis, or after the
  object. Defined with indices, as ``P_{ta} := \texttt{prices}``, it is bound as
  ``wb["P_{ta}"] = prices`` binds it, so that data from two objects meets by label.
- A vector or a matrix is a list, as the language's own are.
- A function is called, as ``\texttt{hann}(n, N)``: on scalars it gives a scalar, and on
  tensors a tensor with their indices, elementwise. A name defined as one,
  ``w := \texttt{hann}``, is called as it is.

A workbook given no scope, as a document checked and a program sent to the MCP server are
not, refuses a reference, saying how to give one.
"""

from __future__ import annotations

import re
from typing import Any, Literal

import sympy

from relativisticpy.compiler.passes import Term
from relativisticpy.compiler.scope import Context, Scope, VariableBinding
from relativisticpy.compiler.session import Session
from relativisticpy.errors import CompileError, IndexNotationError, RelativisticPyError
from relativisticpy.integrations._objects import (
    Array,
    Becomes,
    Data,
    PythonFunction,
    Reference,
    Scalar,
    describe,
    reference,
)
from relativisticpy.ir import SCALAR, VALUE, EquationType, Node, TensorType, ops
from relativisticpy.language import syntax as ast
from relativisticpy.language.source import Span
from relativisticpy.lowering.quantum import Quantum
from relativisticpy.tensors import Index, Indices, Space, numerical
from relativisticpy.typing.syntax import _plain_label

#: The name the lexer gives a Python object: the command, and what its braces hold.
_WRITTEN = re.compile(r"\\texttt\{(?P<written>.*)\}", re.DOTALL)

#: Where a program reaches nothing in Python, and why.
NO_SCOPE = (
    "a document checked with relativisticpy check, and a program sent to the MCP server, "
    "are given no Python scope, so that they never reach into Python"
)

#: The decision of whether a name followed by brackets is a call.
CALL = "product or call"


class PythonObjects(Quantum):
    r"""The reading of the Python objects a statement names, as ``\texttt{prices}``: before
    any other form, since what ``\texttt`` names is never the language's."""

    def __init__(self, session: Session) -> None:
        super().__init__(session)
        # What a definition read of an object it did not bind itself, for the walk of its
        # value to take, rather than read the object again.
        self._objects_read: dict[int, Becomes] = {}

    def term(self, node: ast.Expr, scope: Scope, ctx: Context) -> Term:
        read = self._python_object(node, scope, ctx)
        return read if read is not None else super().term(node, scope, ctx)

    def _define_now(self, target: ast.Expr, value: ast.Expr, scope: Scope, ctx: Context) -> None:
        try:
            if not self._defines_with_object(target, value, scope):
                super()._define_now(target, value, scope, ctx)
        finally:
            self._objects_read.pop(id(value), None)

    def _check_statistic_name(self, factor: ast.Expr, scope: Scope) -> None:
        r"""A Python object with a subscript, as ``\texttt{cov}_{ab}``, is no statistic
        misspelt, nor an operator: it is read as data with its indices."""
        if not (isinstance(factor, ast.Scripted) and _named_object(factor.base)):
            super()._check_statistic_name(factor, scope)

    # -- reading ------------------------------------------------------------------------

    def _python_object(self, node: ast.Expr, scope: Scope, ctx: Context) -> Term | None:
        """``node`` read as a Python object, if it names one, called or not, written with
        indices, or calls a name defined as a Python function; ``None`` for anything else."""
        if _decorated(node):
            raise CompileError(
                f"{node.span.text}: a prime or an accent on a Python object is not read",
                span=node.span,
                hint=r"write what is meant, as \frac{d}{dx} \texttt{f} for a derivative, or "
                "give it a name of its own in Python",
            )
        joined = node.func if isinstance(node, ast.Call) else node
        if _joined_object(joined):  # \texttt{J}_0(x), joined as J_0(x) is
            raise CompileError(
                f"{joined.span.text}: what follows a Python object's name is not part of it",
                span=joined.span,
                hint=r"write its whole name inside the braces, as \texttt{J_0}",
            )
        if isinstance(node, ast.Scripted) and _named_object(node.base):
            return self._object_scripted(node, scope, ctx)
        if _named_object(node):
            assert isinstance(node, ast.Name)
            made = self._objects_read.pop(id(node), None)
            if made is None:
                _, made = self._read_object(node)
            if isinstance(made, PythonFunction):
                raise CompileError(
                    f"{node.span.text} is a Python function, so it is called, as "
                    f"{node.span.text}(x)",
                    span=node.span,
                )
            return self._object_node(made, node)
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            return None
        if _named_object(node.func):
            return self._object_called(node, scope, ctx)
        binding = scope.lookup(node.func.id)
        if isinstance(binding, VariableBinding) and _function_of(binding.node) is not None:
            why = f"{node.func.id} is a Python function, defined on line {_line(binding.node)}"
            self._decide(CALL, node.span, f"a call of {node.func.id}", why)
            return self._call_object(binding.node, node.func.id, node.args, node.span, scope, ctx)
        return None

    def _read_object(self, node: ast.Name) -> tuple[Reference, Becomes]:
        r"""The object ``\texttt{...}`` names, and what it becomes, from the workbook's scope;
        refused, saying how to give one, where there is none."""
        held = written_object(node)
        assert held is not None
        written = reference(held, node.span)
        reader = self.session.objects
        if reader is None:
            raise CompileError(
                f"{node.span.text} names a Python object, but no Python scope was given to "
                "read it from",
                span=node.span,
                notes=[NO_SCOPE],
                hint="give the workbook one, as Workbook(scope=globals()), or the object to "
                f"show, as show(source, {written.name}={written.name})",
            )
        try:
            made = reader.read(written)
        except RelativisticPyError as error:
            if error.span is None:
                error.span = node.span
            raise
        self._resolve(node.span, f"the Python object {written.shown()}, {describe(made)}")
        return written, made

    def _object_node(self, made: Becomes, node: ast.Name) -> Node:
        """The value an object became, as a node of the IR."""
        if isinstance(made, Scalar):
            return self.graph.const(self._as_written(made.value), SCALAR, node.span)
        if isinstance(made, Array):
            return self.graph.const(self._as_written(made.value), VALUE, node.span)
        if isinstance(made, Data):
            return self._data_node(made, node)
        return self.graph.const(made, VALUE, node.span)

    def _as_written(self, value: Any) -> Any:
        """A SymPy value from Python with its symbols the program's, by name, so that the x of
        an expression is the x the program writes, with what it declared of it."""
        if not isinstance(value, sympy.Basic):
            return value
        named = {
            symbol: self.session.symbol(symbol.name)
            for symbol in value.free_symbols
            if type(symbol) is sympy.Symbol and self.session.symbol(symbol.name) != symbol
        }
        return value.xreplace(named) if named else value

    # -- data ---------------------------------------------------------------------------

    def _data_node(self, made: Data, node: ast.Name) -> Node:
        """Data as a tensor whose indices no one wrote, each axis over the space its labels
        meet, or one named after its axis or the object."""
        from relativisticpy.workbook import _double_precision, _in_basis_order

        components = made.components
        complex_data = numerical.is_numerical(components) and numerical.numpy().iscomplexobj(
            components
        )
        field: Literal["real", "complex"] = "complex" if complex_data else "real"
        spaces: list[Space] = []
        rank = len(made.shape)
        for axis, (size, basis, name) in enumerate(
            zip(made.shape, made.bases, made.names, strict=True)
        ):
            written = written_object(node) or node.span.text
            space = self._object_space(written, axis, rank, size, basis, name, field)
            if basis is not None and space.basis is not None and basis != space.basis:
                what = f"axis {axis + 1} of {node.span.text}"
                components = _in_basis_order(components, axis, basis, space, what)
            spaces.append(space)
        reader = self.session.objects
        if reader is not None and reader.backend == "jax" and not numerical.is_sparse(components):
            import jax
            import jax.numpy as jnp

            _double_precision(jax)
            components = jnp.asarray(components)
        indices = Indices(Index(_unwritten(axis)) for axis in range(rank))
        return self.graph.const(components, TensorType(indices, tuple(spaces)), node.span)

    def _object_scripted(self, node: ast.Scripted, scope: Scope, ctx: Context) -> Term:
        r"""A Python object with scripts: data with the indices written on it, as
        ``\texttt{signal}_t`` or ``\texttt{weights}^a``; and the power of anything else, as
        ``\texttt{c}^2``. A subscript on what is no data is refused, never made a name."""
        base = node.base
        assert isinstance(base, ast.Name)
        _, made = self._read_object(base)
        below = any(script.kind == "_" for script in node.scripts)
        labelled = all(
            script.indices is not None and all(_plain_label(i) for i in script.indices.items)
            for script in node.scripts
        )
        if isinstance(made, Data | Array) and (below or labelled):
            return self._object_indexed(node, made)
        if below:
            raise CompileError(
                f"{base.span.text} is {describe(made)}, which has no indices, so a subscript on "
                "it is not read",
                span=node.span,
                hint=r"write its whole name inside the braces, as \texttt{c_1}",
            )
        self._objects_read[id(base)] = made  # its power reads it from here
        try:
            return super().term(node, scope, ctx)
        finally:
            self._objects_read.pop(id(base), None)

    def _object_indexed(self, node: ast.Scripted, made: Data | Array) -> Node:
        r"""``\texttt{signal}_t``: data read from an object, with the indices written on it,
        each axis over a space as ``x_t := \texttt{signal}`` would have it; an index written
        twice is summed, as on any tensor."""
        from relativisticpy.workbook import _data_tensor

        base = node.base
        assert isinstance(base, ast.Name)
        if isinstance(made, Array):
            components = _numbers_or_symbols(made.value)
            made = Data(components, (None,) * len(components.shape))
        entries = self._entries(node, "the indices of a Python object")
        if not all(_plain_label(item) for item, _ in entries):
            raise IndexNotationError(
                f"write index labels on {base.span.text}, as {base.span.text}_t; to take a "
                r"component, define the data first, as x_t := \texttt{signal}, then x_{t=0}",
                span=node.span,
            )
        written = [(str(item.label), upper) for item, upper in entries]
        reader = self.session.objects
        backend = reader.backend if reader is not None else "numpy"
        data = (made.components, made.bases, made.names)
        tensor, indices = _data_tensor(
            self.session, node.span.text, base.id, written, data, backend, node.span, self._labels
        )
        assert isinstance(tensor.type, TensorType)
        for index, space in zip(indices, tensor.type.spaces, strict=True):
            self._check_label(index.name, space, node.span)
        return self._trace(tensor, node.span)

    def _object_space(
        self,
        written: str,
        axis: int,
        rank: int,
        size: int,
        basis: tuple[str, ...] | None,
        name: str | None,
        field: Literal["real", "complex"],
    ) -> Space:
        """The space an axis of data read from an object ranges over: one in use whose
        elements are its labels, the one its name names, or one of its own, named after its
        axis, or the object's."""
        from relativisticpy.workbook import _labelled_spaces

        if basis is not None:
            for space in _labelled_spaces(self.session, "", "#", ()):
                if space.dim == size and sorted(space.basis or ()) == sorted(basis):
                    return space
        if name is not None:
            named = self.session.spaces.get(name)
            if named is not None and named.dim == size:
                return named
            return Space(name, size, basis=basis, field=field)
        parts = {1: ("entries",), 2: ("rows", "columns")}.get(rank)
        part = parts[axis] if parts is not None else f"axis {axis + 1}"
        return Space(f"the {part} of {written}", size, basis=basis, field=field)

    def _defines_with_object(self, target: ast.Expr, value: ast.Expr, scope: Scope) -> bool:
        r"""Define ``target`` with an object, if ``value`` names one and it is data the
        target writes indices for, as ``P_{ta} := \texttt{prices}``, bound as
        ``wb["P_{ta}"] = prices`` binds it, or a function a name is defined as, as
        ``w := \texttt{hann}``. What else it read, the walk of ``value`` takes."""
        if not _named_object(value):
            return False
        assert isinstance(value, ast.Name)
        _, made = self._read_object(value)
        if isinstance(made, PythonFunction) and isinstance(target, ast.Name):
            scope.bind(target.id, VariableBinding(self.graph.const(made, VALUE, value.span)))
            return True
        if isinstance(made, Data | Array) and self._bind_object(target, made, value.span):
            return True
        self._objects_read[id(value)] = made
        return False

    def _bind_object(self, target: ast.Expr, made: Data | Array, span: Span) -> bool:
        """Bind data, or a vector or a matrix, to a target written with indices, by the rules
        by which ``wb[...] = data`` binds it; whether the target writes indices."""
        from relativisticpy.workbook import _bind_data, _tensor_written

        if isinstance(made, Data):
            data = (made.components, made.bases, made.names)
        else:
            components = _numbers_or_symbols(made.value)
            rank = len(getattr(components, "shape", ()))
            data = (components, (None,) * rank, (None,) * rank)
        written = target.span.text
        try:
            tensor = _tensor_written(written, len(getattr(data[0], "shape", ())))
        except RelativisticPyError:  # no target to bind data to: the definition says why
            return False
        if tensor is None:
            return False
        name, entries = tensor
        reader = self.session.objects
        backend = reader.backend if reader is not None else "numpy"
        _bind_data(self.session, written, name, entries, data, backend, span)
        return True

    # -- calling ------------------------------------------------------------------------

    def _object_called(self, node: ast.Call, scope: Scope, ctx: Context) -> Node:
        r"""``\texttt{hann}(n, N)``: a Python function called; a number or an expression
        followed by brackets is a product, as ``\texttt{G}(M + m)``, or the expression itself
        where the brackets hold the names it is written in, as ``\texttt{f}(x)``."""
        func = node.func
        assert isinstance(func, ast.Name)
        _, made = self._read_object(func)
        if isinstance(made, PythonFunction):
            function = self.graph.const(made, VALUE, func.span)
            return self._call_object(function, func.span.text, node.args, node.span, scope, ctx)
        if not isinstance(made, Scalar):
            raise CompileError(
                f"{func.span.text} is {describe(made)}, not a function, so it is not called",
                span=node.span,
                hint=r"give data its indices in a definition, as in x_t := \texttt{signal}",
            )
        value = self._as_written(made.value)
        names = [arg.id for arg in node.args if isinstance(arg, ast.Name)]
        symbols = {str(symbol) for symbol in getattr(value, "free_symbols", ())}
        if names and len(names) == len(node.args) and set(names) <= symbols:
            why = f"{func.span.text} is an expression of {', '.join(names)}"
            self._decide(CALL, node.span, f"the value of {func.span.text}", why)
            return self.graph.const(value, SCALAR, func.span)
        why = f"{func.span.text} is {describe(made)}, not a function"
        self._decide(CALL, node.span, "a product", why)
        group = node.args[0] if len(node.args) == 1 else ast.Tuple(node.args, span=node.span)
        factors = [self.graph.const(value, SCALAR, func.span), self.value(group, scope, ctx)]
        return self._multiply(factors, node.span)

    def _call_object(
        self,
        function: Node,
        written: str,
        args: tuple[ast.Expr, ...],
        span: Span,
        scope: Scope,
        ctx: Context,
    ) -> Node:
        """A Python function called on the values of ``args``: a scalar where they are
        scalars, and, where any is a tensor, a tensor with its indices, elementwise, the
        other tensors aligned with it."""
        values = [self.value(arg, scope, ctx) for arg in args]
        for value, arg in zip(values, args, strict=True):
            if isinstance(value.type, EquationType):
                raise CompileError(
                    f"{written} is called on numbers and data, not on an equation",
                    span=arg.span,
                )
        tensors = [value for value in values if isinstance(value.type, TensorType)]
        if not tensors:
            kind = SCALAR if all(value.type == SCALAR for value in values) else VALUE
            return self._call("python.call", [function, *values], kind, span)
        first = tensors[0].type
        assert isinstance(first, TensorType)
        aligned = [
            self._align(value, first.indices, arg.span, what=f"the arguments of {written}")
            if isinstance(value.type, TensorType)
            else value
            for value, arg in zip(values, args, strict=True)
        ]
        return self._call("python.call", [function, *aligned], first, span)


def written_object(node: ast.Node) -> str | None:
    r"""What the braces of ``\texttt{...}`` hold, where ``node`` is a name written so, which
    names a Python object; ``None`` for any other node."""
    if not (isinstance(node, ast.Name) and node.verbatim):
        return None
    whole = _WRITTEN.fullmatch(node.id)
    return whole["written"] if whole is not None else None


def _named_object(node: ast.Node) -> bool:
    r"""Whether ``node`` is a name written ``\texttt{...}``, which names a Python object."""
    return written_object(node) is not None


def _joined_object(node: ast.Node) -> bool:
    r"""Whether ``node`` is a name the parser joined of a Python object's and what follows
    it, as ``\texttt{J}_0`` of ``\texttt{J}_0(x)``, as it joins ``J_0`` of ``J_0(x)``."""
    return (
        isinstance(node, ast.Name)
        and node.verbatim
        and node.id.startswith("\\texttt{")
        and written_object(node) is None
    )


def _decorated(node: ast.Node) -> bool:
    r"""Whether ``node`` is a Python object with a prime or an accent, as ``\texttt{f}'`` or
    ``\hat{\texttt{x}}``, called or not, which the language would read as a name of its own."""
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Prime):
        return _named_object(node.func)
    return isinstance(node, ast.Accent) and _named_object(node.base)


def _function_of(node: Node) -> PythonFunction | None:
    """The Python function a node holds, if it is one."""
    if isinstance(node.op, ops.Const) and isinstance(node.op.value, PythonFunction):
        return node.op.value
    return None


def _unwritten(axis: int) -> str:
    """The index no one wrote on an axis of data, shown as the letter of its position."""
    return f"#{'ijklmn'[axis]}" if axis < 6 else f"#o{axis}"


def _line(node: Node) -> int | str:
    return node.span.line if node.span is not None else "?"


def _numbers_or_symbols(array: Any) -> Any:
    """The entries of a vector or a matrix as data: a NumPy array where they are numbers,
    as ``wb[...] = [0.5, 0.25]`` binds them, and as they are otherwise."""
    if all(entry.is_number for entry in sympy.flatten(array.tolist())):
        return numerical.as_array(array)
    return array
