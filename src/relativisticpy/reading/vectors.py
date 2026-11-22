r"""Vectors as books on vector calculus write them, read by their operands
(``design/notation.md``, 8, 9, 16 and 25), each reading recorded with why, and a note where
another was possible:

- a list, a tuple or a column of two or three numbers is a vector, as ``\mathbf{v} := (1,
  2, 3)`` makes one; ``\cdot`` between vectors is their dot product and ``\times`` their
  cross product, and between numbers both are multiplication;
- ``\hat{\mathbf{x}}``, ``\hat{x}``, ``\hat{\imath}``, ``\mathbf{i}``, ``\mathbf{j}`` and
  ``\mathbf{k}`` are the unit vectors of Cartesian coordinates where nothing defines them,
  and ``\hat{\mathbf{r}}``, ``\hat{\boldsymbol\theta}`` and ``\mathbf{e}_r`` those of the
  coordinates in force; a plain hat in a statement that writes one on a letter that is no
  coordinate, as ``\hat{p}``, is an operator's, as before;
- the frame of vector calculus is that of the coordinates in force: a line element or a
  metric gives their scale factors; coordinates declared without one take them from the
  Cartesian coordinates the program writes in them, as ``x := r \sin\theta \cos\phi``, or
  from their names, spherical, cylindrical or polar; without coordinates, the frame is
  Cartesian, in x, y and z.

What a vector is is known without computing anything, as a matrix's shape is (see
:mod:`relativisticpy.reading.matrices`): its components are those written, or those of
the arithmetic, the derivatives and the integrals of vectors, component by component.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import sympy
from sympy import Symbol

from relativisticpy.compiler.scope import Scope, VariableBinding
from relativisticpy.errors import CompileError
from relativisticpy.ir import SCALAR, VALUE, Node, ops
from relativisticpy.language import syntax as ast
from relativisticpy.language.source import Span
from relativisticpy.language.vocabulary import GREEK_VARIANTS
from relativisticpy.reading.reader import Reader

#: The decision of whether a form is of vectors or of numbers.
VECTOR = "vector or number"

#: The commands that set a letter in bold, as books write a vector.
BOLD = ("\\mathbf", "\\boldsymbol", "\\bm", "\\pmb", "\\vb")

#: The Cartesian coordinates, and the letters of their unit vectors i, j and k.
CARTESIAN = ("x", "y", "z")
_IJK = {"i": "x", "j": "y", "k": "z"}

#: The coordinates of the frames known by name that are not Cartesian, whose symbols in a
#: function of Cartesian coordinates are a dependence on x, y and z that would be lost.
CURVILINEAR = frozenset({"r", "theta", "phi", "rho"})

#: The operations whose value is a vector, of those the frame computes, with how many
#: components it has: as many as the coordinates.
_OF_THE_FRAME = frozenset({"vectors.gradient", "vectors.curl"})

#: Coordinates known by their names, with the Cartesian coordinates written in them, as
#: physics books name them: θ from the z axis, φ about it (Arfken, Weber and Harris
#: (2013), sections 3.9 and 3.10).
_SYSTEMS: dict[tuple[str, ...], tuple[str, str]] = {
    ("r", "theta", "phi"): ("spherical", "r sin θ cos φ, r sin θ sin φ, r cos θ"),
    ("rho", "phi", "z"): ("cylindrical", "ρ cos φ, ρ sin φ, z"),
    ("s", "phi", "z"): ("cylindrical", "s cos φ, s sin φ, z"),
    ("r", "phi", "z"): ("cylindrical", "r cos φ, r sin φ, z"),
    ("r", "theta"): ("polar", "r cos θ, r sin θ"),
    ("r", "phi"): ("polar", "r cos φ, r sin φ"),
    ("rho", "phi"): ("polar", "ρ cos φ, ρ sin φ"),
}

#: A hat in LaTeX, with the letter under it and whether that is bold.
_HAT = re.compile(
    r"\\(?:hat|widehat)\s*\{?\s*(?P<bold>\\(?:mathbf|boldsymbol|bm|pmb)\s*\{?\s*)?"
    r"\\?(?P<letter>[A-Za-z]+)"
)


def bold(node: ast.Node) -> bool:
    """Whether a name is set in bold, as books set a vector: ``\\mathbf{v}``,
    ``\\boldsymbol{\\omega}`` or ``{\\bf v}``."""
    if node.span.text.lstrip().startswith(BOLD):
        return True
    before = node.span.source.text[: node.span.start].rstrip()
    return before.endswith(("{\\bf", "\\bf"))


#: The dotless letters, which books set under a hat for the unit vectors i and j.
_DOTLESS = {"imath": "i", "jmath": "j"}


def _letter(name: str) -> str:
    """A letter as a coordinate is named: ``phi`` of ``\\varphi``, ``i`` of ``\\imath``."""
    return _DOTLESS.get(name, GREEK_VARIANTS.get(name, name))


def hatted(node: ast.Expr) -> tuple[str, bool] | None:
    """The letter a hat sits on, and whether it marks a vector as bold does: ``("x", True)``
    of ``\\hat{\\mathbf{x}}``, ``\\mathbf{\\hat{x}}`` or ``\\hat{\\imath}``, whose dotless i
    is set only under a hat; ``None`` for anything else."""
    if not (isinstance(node, ast.Accent) and node.accent == "hat"):
        return None
    base = node.base
    if not isinstance(base, ast.Name):
        return None
    dotless = base.span.text.strip().lstrip("\\") in _DOTLESS
    return _letter(base.id), bold(base) or bold(node) or dotless


@dataclass(frozen=True)
class Frame:
    """The coordinates vector calculus works in, and how their scale factors are known:
    from ``metric``, the node of their metric's components, or, where it is ``None``,
    as Cartesian coordinates, each of scale 1. ``written`` names them for messages, and
    ``system`` says which coordinates they are, where their names told."""

    coordinates: tuple[Symbol, ...]
    metric: Node | None
    written: str
    system: str | None = None

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(coordinate.name for coordinate in self.coordinates)

    @property
    def cartesian(self) -> bool:
        """Whether the frame is Cartesian, its unit vectors the same everywhere: named so,
        x, y and z, or without a metric to say otherwise."""
        return self.metric is None or set(self.names) <= set(CARTESIAN)


class Vectors(Reader):
    """How vectors, unit vectors and the frame of vector calculus are read."""

    # -- what a vector is ----------------------------------------------------------------

    def _vector_components(self, node: Node) -> list[Node] | None:
        """The components of a vector, known without computing: those of a list, a tuple
        or a column written out, of the arithmetic of vectors, by numbers and with each
        other, and of what the frame computes, as a gradient; ``None`` for any other value,
        as a number, a matrix or a list of anything but numbers."""
        if node.type != VALUE:
            return None
        op, args = node.op, list(node.args)
        if isinstance(op, ops.Call) and op.name in ("core.list", "core.tuple"):
            scalars = len(args) >= 2 and all(arg.type == SCALAR for arg in args)
            return args if scalars else None
        if isinstance(op, ops.MakeArray):
            column = len(op.shape) == 1 or (len(op.shape) == 2 and op.shape[1] == 1)
            return args if column and len(args) >= 2 else None
        if isinstance(op, ops.Call) and op.name in _OF_THE_FRAME:
            coordinates = dict(op.attributes)["coordinates"]
            assert isinstance(coordinates, tuple)
            return [self._component(node, position) for position in range(len(coordinates))]
        if isinstance(op, ops.Mul):
            vectors = [arg for arg in args if arg.type != SCALAR]
            if len(vectors) != 1 or (each := self._vector_components(vectors[0])) is None:
                return None
            return [
                self.graph.add(
                    ops.Mul(), [part if arg is vectors[0] else arg for arg in args], SCALAR
                )
                for part in each
            ]
        if isinstance(op, ops.Add):
            terms = [self._vector_components(arg) for arg in args]
            sizes = {len(term) for term in terms if term is not None}
            if any(term is None for term in terms) or len(sizes) != 1:
                return None
            return [
                self.graph.add(ops.Add(op.signs), [term[k] for term in terms if term], SCALAR)
                for k in range(sizes.pop())
            ]
        return None

    def _component(self, vector: Node, position: int) -> Node:
        """The component of a vector computed as a whole, at ``position``, counted from 0."""
        at = self.graph.const(position, VALUE)
        return self.graph.add(ops.Call("vectors.component"), (vector, at), SCALAR, vector.span)

    def _vector_of(self, components: Sequence[Node], span: Span) -> Node:
        """A vector of ``components``, as a list."""
        return self._call("core.list", list(components), VALUE, span)

    def _vector_like(self, vector: Node, components: Sequence[Node], span: Span) -> Node:
        """A vector of ``components`` written as ``vector`` is: a column, as of
        ``\\begin{pmatrix} 1 \\\\ 2 \\end{pmatrix}``, where it is one, and else a list."""
        if _column(vector):
            return self.graph.add(
                ops.MakeArray((len(components), 1)), list(components), VALUE, span
            )
        return self._vector_of(components, span)

    # -- the frame -------------------------------------------------------------------------

    def _vector_frame(self, span: Span, variables: Sequence[str] | None = None) -> Frame | None:
        """The frame vector calculus works in at ``span``: the coordinates in force, in two
        or three dimensions, with their metric or the Cartesian coordinates written in
        them; without coordinates, Cartesian ones, the ``variables`` a function is written
        in or else x, y and z. ``None`` where the coordinates in force are no such frame,
        as those of spacetime."""
        session = self.session
        coordinates = session.coordinates
        if coordinates is None:
            names = tuple(variables) if variables else CARTESIAN
            symbols = tuple(session.symbol(name) for name in names)
            return Frame(symbols, None, self._coordinates_written(names))
        names = tuple(coordinate.name for coordinate in coordinates)
        if len(names) not in (2, 3):
            return None
        written = self._coordinates_written(names)
        if session.metric is not None:
            return Frame(coordinates, session.metric.node, written)
        if set(names) <= set(CARTESIAN):
            return Frame(coordinates, None, written)
        embedding = self._embedding(coordinates, span)
        if embedding is None:
            return None
        metric, system = embedding
        return Frame(coordinates, metric, written, system)

    def _coordinates_written(self, names: Sequence[str]) -> str:
        """Coordinates as the statement writes them: ``(r, \\theta, \\phi)``."""
        return "(" + ", ".join(self._written_label(name) for name in names) + ")"

    def _embedding(self, coordinates: tuple[Symbol, ...], span: Span) -> tuple[Node, str] | None:
        """The metric of coordinates declared without one, from the Cartesian coordinates
        written in them: those the program defines, as ``x := r \\sin\\theta \\cos\\phi``,
        or those their names give, with a note; ``None`` where neither tells."""
        scope = self.session.globals
        size = len(coordinates)
        written = [scope.lookup(name) for name in CARTESIAN[:size]]
        if all(isinstance(binding, VariableBinding) for binding in written):
            cartesian = [
                binding.node for binding in written if isinstance(binding, VariableBinding)
            ]
            system = "the Cartesian coordinates the program writes in them"
        else:
            names = tuple(coordinate.name for coordinate in coordinates)
            known = _SYSTEMS.get(names)
            if known is None:
                return None
            kind, formulas = known
            system = f"{kind} coordinates, x, y{', z' if size == 3 else ''} = {formulas}"
            self._note(f"{self._coordinates_written(names)} are read as {system}")
            cartesian = [
                self.graph.const(value, SCALAR) for value in _cartesian_of(kind, coordinates)
            ]
        attributes = (("coordinates", coordinates),)
        listed = self._vector_of(cartesian, span)
        metric = self.graph.add(ops.Call("vectors.induced_metric", attributes), (listed,), VALUE)
        return metric, system

    def _frame_or_refuse(self, span: Span, what: str) -> Frame:
        """The frame of vector calculus, or the error that there is none here."""
        frame = self._vector_frame(span)
        if frame is not None:
            return frame
        coordinates = self.session.coordinates or ()
        names = [coordinate.name for coordinate in coordinates]
        written = self._coordinates_written(names)
        if len(names) not in (2, 3):
            reason = f"the coordinates in force, {written}, are {len(names)}"
            hint = "with indices, write the derivative as \\nabla_\\mu"
        else:
            reason = f"nothing gives the scale factors of the coordinates {written}"
            hint = "write their line element, as ds^2 = dr^2 + r^2 d\\theta^2"
        raise CompileError(
            f"{what} is of vector calculus, in two or three dimensions, and {reason}",
            span=span,
            hint=hint,
        )

    # -- unit vectors ------------------------------------------------------------------------

    def _unit_vector(
        self, letter: str, node: ast.Expr, scope: Scope, *, written_bold: bool
    ) -> Node | None:
        """The unit vector along the coordinate ``letter`` names, where nothing defines it:
        of the frame in force, with i, j and k for x, y and z. A plain hat is one only in a
        statement whose plain hats all sit on coordinates; elsewhere, as in ``[\\hat{x},
        \\hat{p}]``, it is an operator's, and ``None`` leaves it as before. ``None`` too
        where the letter names no coordinate of the frame, as ``\\hat{\\beta}`` does."""
        frame = self._vector_frame(node.span)
        if frame is None:
            return None
        names = frame.names
        along = letter if letter in names else _IJK.get(letter) if frame.cartesian else None
        if along not in names:
            if written_bold and letter in CARTESIAN + tuple(_IJK) and not frame.cartesian:
                raise CompileError(
                    f"{node.span.text} is a unit vector of Cartesian coordinates, and the "
                    f"coordinates in force are {frame.written}",
                    span=node.span,
                    hint=f"write the vector with the unit vectors of {frame.written}",
                )
            return None
        if not written_bold and not self._hats_on_coordinates(names):
            return None
        assert along is not None
        position = names.index(along)
        shown = self._written_label(along)
        reading = f"the unit vector along {shown}"
        why = f"{node.span.text} names nothing, and {shown} is a coordinate of {frame.written}"
        note = None
        if not written_bold:
            note = (
                f"{node.span.text} is read as the unit vector along {shown}; to name an "
                f"operator or an estimate, define {node.span.text}"
            )
            self._note(note)
        self._decide(VECTOR, node.span, reading, why, note)
        one, zero = sympy.Integer(1), sympy.Integer(0)
        values = [one if k == position else zero for k in range(len(names))]
        return self._vector_of([self.graph.const(value, SCALAR) for value in values], node.span)

    def _coordinate_before_brackets(
        self, func: ast.Expr, args: Sequence[ast.Expr], scope: Scope
    ) -> str | None:
        """Why x in ``2x(y + 1)`` multiplies the brackets rather than being called, where
        nothing defines it: it is a Cartesian coordinate, of the frame vector calculus works
        in, in a statement that writes its unit vectors or ∇, and the brackets hold more
        than a name (decision 28 of design/notation.md), while x(t) is a path's coordinate;
        ``None`` for any other name."""
        if not isinstance(func, ast.Name) or func.id not in CARTESIAN:
            return None
        if len(args) == 1 and isinstance(args[0], ast.Name):
            return None
        session = self.session
        if scope.lookup(func.id) is not None or func.id in session.functions:
            return None
        if session.coordinates is not None:
            return None  # coordinates in force are read so already
        letters = {_letter(match["letter"]) for match in _HAT.finditer(self._text)}
        vectors = letters & (set(CARTESIAN) | set(_IJK)) or "\\nabla" in self._text
        if not vectors and "∇" not in self._text:
            return None
        return f"{func.span.text} is a Cartesian coordinate, in a statement of vector calculus"

    def _hats_on_coordinates(self, names: Sequence[str]) -> bool:
        """Whether every plain hat the statement writes sits on a coordinate of the frame,
        so that none of them is an operator's, as ``\\hat{p}`` is."""
        for match in _HAT.finditer(self._text):
            if match["bold"] or match["letter"] in _DOTLESS:
                continue
            letter = _letter(match["letter"])
            if letter not in names and _IJK.get(letter) not in names:
                return False
        return True


def _column(vector: Node) -> bool:
    """Whether a vector is written as a column, or made of one by arithmetic."""
    op = vector.op
    if isinstance(op, ops.MakeArray):
        return len(op.shape) == 2
    if isinstance(op, ops.Mul | ops.Add):
        return any(_column(arg) for arg in vector.args if arg.type != SCALAR)
    return False


def _cartesian_of(kind: str, coordinates: Sequence[Symbol]) -> list[sympy.Expr]:
    """The Cartesian coordinates written in coordinates of a ``kind`` known by name."""
    if kind == "spherical":
        r, theta, phi = coordinates
        return [
            r * sympy.sin(theta) * sympy.cos(phi),
            r * sympy.sin(theta) * sympy.sin(phi),
            r * sympy.cos(theta),
        ]
    if kind == "cylindrical":
        rho, phi, z = coordinates
        return [rho * sympy.cos(phi), rho * sympy.sin(phi), z]
    radius, angle = coordinates
    return [radius * sympy.cos(angle), radius * sympy.sin(angle)]
