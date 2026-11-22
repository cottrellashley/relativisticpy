"""The executor: computes the values of IR nodes."""

from __future__ import annotations

import multiprocessing
import os
import sys
from collections.abc import Callable, Hashable, Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Any, TypeVar

from relativisticpy.errors import (
    InternalError,
    RelativisticPyError,
    debugging,
    translate,
)
from relativisticpy.ir import Node, ops
from relativisticpy.runtime.kernels import execute, refuse_quoted, refuse_text
from relativisticpy.tensors import numerical
from relativisticpy.tensors.simplification import simplify_component

_T = TypeVar("_T")


def _context() -> multiprocessing.context.BaseContext | None:
    """How worker processes start: the platform's way, unless JAX is loaded, whose
    threads a forked process could deadlock on; then from a clean server process, which,
    as with spawning, needs a script to keep its main code under
    ``if __name__ == "__main__":``."""
    if "jax" in sys.modules and "forkserver" in multiprocessing.get_all_start_methods():
        return multiprocessing.get_context("forkserver")
    return None


_R = TypeVar("_R")

#: How often, in seconds, a worker process looks for the process that started it.
_WATCH = 0.25


def _outlive_nothing() -> None:
    """Start a worker process that ends when the process that started it does, however
    that ended: a run stopped with SIGTERM, as a harness stops a command that takes too
    long, or with SIGKILL, has no Python left in it to stop its workers, which would
    compute on for nobody. When its parent is gone, a worker is adopted by another
    process, and so changes parent. (A worker started by a fork server has the server as
    its parent, which ends when the workbook's process does.)"""
    import threading
    import time

    parent = os.getppid()

    def watch() -> None:
        while os.getppid() == parent:
            time.sleep(_WATCH)
        os._exit(1)  # nothing of the worker's is worth finishing

    threading.Thread(target=watch, name="relativisticpy-parent-watch", daemon=True).start()


class Executor:
    """Computes node values on demand and remembers them.

    Only the nodes an output depends on are computed, and each at most once:
    asking twice for the Ricci tensor, or for two expressions that share it,
    costs one computation.

    Component-wise work (simplifying every component of a tensor, the curvature
    kernels) goes through :meth:`map`. With ``workers`` above one, batches of
    ``batch`` items or more are spread over a pool of worker processes, started when
    first needed. Even the ten components of a Ricci tensor are worth spreading: one
    of Kerr's takes seconds to simplify. A worker never outlives the process that
    started it, even one stopped by a signal.

    A failure of SymPy or NumPy while computing a node is a
    :class:`~relativisticpy.errors.RelativisticPyError` where the node was written, in
    RelativisticPy's own words (see :func:`~relativisticpy.errors.translate`).
    """

    def __init__(self, workers: int | None = 1, *, batch: int = 4) -> None:
        self.workers = (os.cpu_count() or 1) if workers is None else max(1, workers)
        self.batch = batch
        self._values: dict[Hashable, Any] = {}  # by computation key
        self._pool: ProcessPoolExecutor | None = None

    def evaluate(self, node: Node) -> Any:
        """The value of ``node``, computing whatever it depends on first. Text written in
        quotes in a program is never given out as a value, as to the compiler, which
        would hand it to SymPy, which reads text as Python."""
        value = self._evaluate(node, self._values)
        refuse_quoted(node, value)
        return value

    def evaluate_with(self, node: Node, values: dict[Hashable, Any]) -> Any:
        """The value of ``node`` with some nodes given ``values`` instead, by their keys,
        computed afresh without remembering anything: for automatic differentiation,
        which evaluates the same graph at traced values."""
        return self._evaluate(node, dict(values))

    def _evaluate(self, node: Node, known: dict[Hashable, Any]) -> Any:
        stack: list[tuple[Node, bool]] = [(node, False)]
        while stack:
            current, ready = stack.pop()
            if current.key in known:
                continue
            if not ready:
                stack.append((current, True))
                stack.extend((arg, False) for arg in current.args if arg.key not in known)
                continue
            args = [known[arg.key] for arg in current.args]
            try:
                refuse_text(current.op, args, current.args)  # SymPy would run it as Python
                if isinstance(current.op, ops.Equation) and numerical.involves(args):
                    # numbers are decided to rounding error in what they were computed from
                    with numerical.operands(_operand_scale(current, known)):
                        known[current.key] = execute(current.op, args, self)
                else:
                    known[current.key] = execute(current.op, args, self)
            except RelativisticPyError as error:
                if error.span is None:
                    error.span = current.span
                raise
            except Exception as error:  # SymPy's or NumPy's text is never shown as it is
                failure = translate(error, span=current.span)
                if isinstance(failure, InternalError) or debugging():
                    raise failure from error  # whose traceback a report needs
                raise failure from None  # the library's exception is its cause
        return known[node.key]

    def map(self, func: Callable[[_T], _R], items: Sequence[_T]) -> list[_R]:
        """``[func(item) for item in items]``, in parallel for large batches."""
        if self.workers == 1 or len(items) < self.batch:
            return [func(item) for item in items]
        if self._pool is None:
            self._pool = ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=_context(),
                initializer=_outlive_nothing,
            )
        chunk = max(1, len(items) // (4 * self.workers))
        return list(self._pool.map(func, items, chunksize=chunk))

    def simplify_map(self, items: Sequence[Any]) -> list[Any]:
        """Simplify a batch of components: a
        :data:`~relativisticpy.diffgeom.curvature.SimplifyMap` that uses the workers."""
        return self.map(simplify_component, items)

    def close(self) -> None:
        """Stop the worker processes, if any were started, dropping work not yet begun."""
        if self._pool is not None:
            self._pool.shutdown(cancel_futures=True)
            self._pool = None


def _operand_scale(equation: Node, known: dict[Hashable, Any]) -> float:
    """The magnitude of the numbers an equation's sides were computed from, which their
    rounding error is relative to: the sides, and the terms of a side that is a sum, so
    that ``a - b = 0`` holds where ``a = b`` does."""
    scale = 0.0
    pending = list(equation.args)
    while pending:
        node = pending.pop()
        scale = max(scale, numerical.magnitude(known.get(node.key)))
        if isinstance(node.op, ops.Add | ops.Permute):  # the terms, in any order of axes
            pending.extend(node.args)
        elif isinstance(node.op, ops.Scale):  # a multiple of a sum: its terms, not the factor
            pending.append(node.args[-1])
    return scale
