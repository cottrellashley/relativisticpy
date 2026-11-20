"""The executor: computes the values of IR nodes."""

from __future__ import annotations

import os
from collections.abc import Callable, Hashable, Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Any, TypeVar

from relativisticpy.errors import EvaluationError, RelativisticPyError
from relativisticpy.ir import Node
from relativisticpy.runtime.kernels import execute
from relativisticpy.tensors.simplification import simplify_component

_T = TypeVar("_T")
_R = TypeVar("_R")


class Executor:
    """Computes node values on demand and remembers them.

    Only the nodes an output depends on are computed, and each at most once:
    asking twice for the Ricci tensor, or for two expressions that share it,
    costs one computation.

    Component-wise work (simplifying every component of a tensor, the curvature
    kernels) goes through :meth:`map`. With ``workers`` above one, large batches
    are spread over a pool of worker processes, started when first needed.
    """

    def __init__(self, workers: int | None = 1, *, batch: int = 16) -> None:
        self.workers = (os.cpu_count() or 1) if workers is None else max(1, workers)
        self.batch = batch
        self._values: dict[Hashable, Any] = {}  # by computation key
        self._pool: ProcessPoolExecutor | None = None

    def evaluate(self, node: Node) -> Any:
        """The value of ``node``, computing whatever it depends on first."""
        stack: list[tuple[Node, bool]] = [(node, False)]
        while stack:
            current, ready = stack.pop()
            if current.key in self._values:
                continue
            if not ready:
                stack.append((current, True))
                stack.extend((arg, False) for arg in current.args if arg.key not in self._values)
                continue
            args = [self._values[arg.key] for arg in current.args]
            try:
                self._values[current.key] = execute(current.op, args, self)
            except RelativisticPyError as error:
                if error.span is None:
                    error.span = current.span
                raise
            except Exception as error:
                raise EvaluationError(
                    str(error) or type(error).__name__, span=current.span
                ) from error
        return self._values[node.key]

    def map(self, func: Callable[[_T], _R], items: Sequence[_T]) -> list[_R]:
        """``[func(item) for item in items]``, in parallel for large batches."""
        if self.workers == 1 or len(items) < self.batch:
            return [func(item) for item in items]
        if self._pool is None:
            self._pool = ProcessPoolExecutor(max_workers=self.workers)
        chunk = max(1, len(items) // (4 * self.workers))
        return list(self._pool.map(func, items, chunksize=chunk))

    def simplify_map(self, items: Sequence[Any]) -> list[Any]:
        """Simplify a batch of components: a
        :data:`~relativisticpy.diffgeom.curvature.SimplifyMap` that uses the workers."""
        return self.map(simplify_component, items)

    def close(self) -> None:
        """Stop the worker processes, if any were started."""
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None
