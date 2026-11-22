"""The compiler: from syntax tree to IR, a statement at a time, in three passes over one
walk of its syntax: the reader (:mod:`relativisticpy.reading`), the typer
(:mod:`relativisticpy.typing`) and the lowerer (:mod:`relativisticpy.lowering`). The
compiler orchestrates them, and keeps the session and the scope."""

from relativisticpy.compiler.compiler import Compiler
from relativisticpy.compiler.session import Session

__all__ = ["Compiler", "Session"]
