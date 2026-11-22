r"""Where the editor computes: a process of its own, so that the notebook's kernel never
waits on a computation, and one that runs too long is stopped.

:class:`Computations`, on the notebook's side, sends the editor's jobs to a worker
process one at a time, from a thread of its own, and hands each answer back. A run makes
the jobs before it stale: those waiting are dropped, and the one in hand is interrupted,
as Ctrl-C interrupts one. Each job has a time limit, which the worker keeps softly, as
``relativisticpy check`` does, in its main thread: the statement over it is stopped and
reported. A computation that does not stop when asked, as one in a library's compiled
code may not, has its worker stopped, and the next job starts another. The worker is
given the workbook the editor was given, as it is, whenever it has changed: pickled, with
the values it has computed.

Where no process can be started, as in a browser's Python, :class:`InPlace` computes in
the notebook's own, with no time limit.

The worker, :func:`main`, does each job with an :class:`~relativisticpy.live.engine.Engine`.
Messages are pickled, each after its length, on its standard input and output.
"""

from __future__ import annotations

import ast
import contextlib
import copyreg
import io
import itertools
import json
import os
import pickle
import queue
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import IO, Any

from relativisticpy.errors import (
    EvaluationError,
    InternalError,
    NotComputedError,
    RelativisticPyError,
    internal_error,
)
from relativisticpy.live.engine import Engine
from relativisticpy.workbook import Workbook

#: The longest a job may run, in seconds, unless the editor is given another limit.
TIME_LIMIT = 30.0
#: The memory a worker may use, in MiB, where the system enforces it, as Linux does.
MEMORY_LIMIT = 4096
#: How long a worker has, past a job's time limit or being interrupted, to answer.
GRACE = 5.0
#: How long a new worker has to start: the first, after an installation, compiles.
STARTUP = 120.0

#: What the editor's programs are called where their errors say where they are.
NAME = "<editor>"

#: Where a job's answer goes: its kind, ``ran`` or ``computed``, and its data.
Deliver = Callable[[str, dict[str, Any]], None]


@dataclass
class Job:
    """Something the editor asked for: a ``run`` of a program, or to ``compute`` a part of
    one, with its arguments, and where its answer goes. ``state`` is the workbook pickled,
    as it was when the job was asked for, and ``revision`` says which state that is."""

    kind: str
    arguments: dict[str, Any]
    deliver: Deliver
    revision: int = 0
    state: bytes | None = None
    stale: bool = field(default=False, compare=False)

    @property
    def answer(self) -> str:
        """The kind of answer it is given: ``ran`` or ``computed``."""
        return "ran" if self.kind == "run" else "computed"

    def give(self, kind: str, data: dict[str, Any]) -> None:
        """Hand its answer on. A widget closed since has nowhere to show it, and the
        thread that hands it on goes on to the next job."""
        with contextlib.suppress(Exception):
            self.deliver(kind, data)


# == the notebook's side =====================================================================


def portable(workbook: Workbook) -> bytes:
    """A workbook as it is, as bytes another process carries on from: its definitions and
    what it has computed."""
    buffer = io.BytesIO()
    pickler = pickle.Pickler(buffer, protocol=pickle.HIGHEST_PROTOCOL)
    pickler.dispatch_table = {**copyreg.dispatch_table, itertools.count: _count}
    pickler.dump(workbook._portable())
    return buffer.getvalue()


def _count(counter: Any) -> tuple[Any, tuple[Any, ...]]:
    """A counter as the count it has reached, since Python 3.14 pickles no counter of
    itertools: ``count(5)`` is made again as ``count(5)``."""
    reached = ast.literal_eval(repr(counter)[len("count") :])
    return itertools.count, reached if isinstance(reached, tuple) else (reached,)


class Computations:
    """Computes the editor's jobs in a worker process, one at a time, from a thread of its
    own, and hands each answer to its job's ``deliver``, from that thread."""

    def __init__(
        self, *, time_limit: float = TIME_LIMIT, memory_limit: int | None = MEMORY_LIMIT
    ) -> None:
        self.time_limit = time_limit
        self.memory_limit = memory_limit
        self._events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._pending: list[Job] = []
        self._current: Job | None = None
        self._deadline = 0.0
        self._worker: _Worker | None = None
        self._generations = itertools.count(1)
        self._loaded: int | None = None  # which state of the workbook the worker has
        self._sending: int | None = None  # which state the job in hand took to it
        self._state: tuple[int, bytes] | None = None  # the latest state the editor gave
        self._thread = threading.Thread(target=self._serve, name="relativisticpy-live", daemon=True)
        self._thread.start()

    def submit(self, job: Job) -> None:
        """Ask for a job: a run makes every job before it stale, and a part to compute
        makes the parts before it stale."""
        self._events.put(("job", job))

    def warm(self) -> None:
        """Start a worker before it is needed, so that the first job need not wait."""
        self._events.put(("warm", None))

    def close(self) -> None:
        """Stop the worker, dropping what has not been done."""
        self._events.put(("close", None))

    # -- the thread ------------------------------------------------------------------

    def _serve(self) -> None:
        while True:
            try:
                kind, payload = self._events.get(timeout=self._waiting())
            except queue.Empty:
                self._expired()
                continue
            if kind == "close":
                self._stop_worker()
                return
            if kind == "job":
                self._queue(payload)
            elif kind == "warm" and self._worker is None:
                self._start_worker()
            elif kind == "message":
                self._received(*payload)
            elif kind == "lost":
                self._lost(payload)
            self._next()

    def _waiting(self) -> float | None:
        """How long to wait for the next event before a deadline passes: that of the job
        in hand, or of a worker starting."""
        if self._current is None and (self._worker is None or self._worker.ready):
            return None
        return max(0.0, self._deadline - time.monotonic())

    def _queue(self, job: Job) -> None:
        if job.state is not None:
            self._state = (job.revision, job.state)
        if job.kind == "run":
            self._pending.clear()
        else:
            self._pending = [waiting for waiting in self._pending if waiting.kind == "run"]
        current = self._current
        overtaken = current is not None and (job.kind == "run" or current.kind != "run")
        if current is not None and overtaken and not current.stale:
            current.stale = True
            self._interrupt()
        self._pending.append(job)

    def _next(self) -> None:
        """Start the next job, if the worker is free; start a worker, if there is none."""
        if self._current is not None or not self._pending:
            return
        if self._worker is None:
            self._start_worker()
        if self._worker is None or not self._worker.ready:
            return
        job = self._pending.pop(0)
        state = self._state if self._state is not None and self._state[0] != self._loaded else None
        arguments = {**job.arguments, "limit": self.time_limit, "name": NAME}
        try:
            self._worker.send((job.kind, arguments, state[1] if state is not None else None))
        except OSError:  # it stopped as it was asked: the job waits for the next worker
            self._pending.insert(0, job)
            self._lost(self._worker.generation)
            return
        self._current = job
        self._sending = state[0] if state is not None else None
        self._deadline = time.monotonic() + self.time_limit + GRACE

    def _received(self, generation: int, message: tuple[str, Any]) -> None:
        worker = self._worker
        if worker is None or generation != worker.generation:
            return  # from a worker stopped since
        kind, data = message
        if kind == "ready":
            worker.ready = True
            return
        job, self._current = self._current, None
        if job is None:
            return
        if kind != "failed" and self._sending is not None:  # it took the workbook in
            self._loaded = self._sending
        if job.stale or kind == "stopped":
            return
        if kind == "failed":
            job.give(job.answer, _failure(job, data))
        else:
            job.give(kind, data)

    def _lost(self, generation: int) -> None:
        """The worker stopped, or was stopped: the job in hand fails, saying so; and if it
        never started, every job waiting fails too, until another is asked for."""
        worker = self._worker
        if worker is None or generation != worker.generation:
            return
        why = worker.why()
        self._stop_worker()
        if not worker.ready:
            self._fail_waiting(_did_not_start(why))
            return
        job, self._current = self._current, None
        if job is not None and not job.stale:
            failure = EvaluationError(
                "RelativisticPy's worker process stopped while computing this",
                notes=[why] if why else [],
                hint="it may have run out of memory: compute this in steps, or give the "
                "editor more, as with Editor(memory_limit=8192)",
            )
            job.give(job.answer, _failure(job, _data(failure)))

    def _expired(self) -> None:
        """A deadline passed: a worker did not start, or a job did not stop."""
        worker, job = self._worker, self._current
        if worker is not None and not worker.ready:
            why = worker.why()
            self._stop_worker()
            self._fail_waiting(_did_not_start(why))
            return
        if job is None:
            return
        self._stop_worker()
        self._current = None
        if not job.stale:
            failure = NotComputedError(
                f"over the time limit of {self.time_limit:g} s, and it did not stop when "
                "asked to, so its worker was stopped",
                hint="give the editor longer, as with Editor(time_limit=120), or compute "
                "this in steps, defining its parts first",
            )
            job.give(job.answer, _failure(job, _data(failure)))

    def _fail_waiting(self, failure: RelativisticPyError) -> None:
        waiting, self._pending = self._pending, []
        for job in waiting:
            job.give(job.answer, _failure(job, _data(failure)))

    def _interrupt(self) -> None:
        """Interrupt the job in hand, which has gone stale; stop its worker if it cannot be
        interrupted, or does not answer soon."""
        if self._worker is None:
            return
        if not self._worker.interrupt():
            self._stop_worker()
            self._current = None
            return
        self._deadline = min(self._deadline, time.monotonic() + GRACE)

    def _start_worker(self) -> None:
        self._worker = _Worker(next(self._generations), self._events, self.memory_limit)
        self._loaded = None
        self._deadline = time.monotonic() + STARTUP
        try:
            self._worker.start()
        except OSError as error:
            self._worker = None
            self._fail_waiting(_did_not_start(str(error)))

    def _stop_worker(self) -> None:
        worker, self._worker = self._worker, None
        self._loaded = None
        if worker is not None:
            worker.stop()


def _did_not_start(why: str | None) -> InternalError:
    return InternalError(
        "the editor's worker process did not start",
        notes=[why] if why else [],
        hint="run python -c 'import relativisticpy' to see why",
    )


def _data(error: RelativisticPyError) -> dict[str, Any]:
    """An error as the editor's data has it: as data, and as the command line prints it."""
    return {**error.to_dict(), "start": 0, "end": 0, "rendered": error.render()}


def _failure(job: Job, error: dict[str, Any]) -> dict[str, Any]:
    """The answer of a job that failed as a whole: its program as one statement, or its
    part, with the error."""
    if job.kind == "run":
        text = job.arguments.get("text", "")
        error = {**error, "end": len(text)}
        entry = {
            "statement": text,
            "start": 0,
            "end": len(text),
            "line": 1,
            "column": 1,
            "values": [],
            "defines": [],
            "notes": [],
            "read": False,
            "typeset": text,
            "definition": None,
            "error": error,
        }
        return {"source": text, "statements": [entry], "notes": [], "ok": False, "seconds": 0.0}
    return {
        "action": job.arguments.get("action", "compute"),
        "start": job.arguments.get("start", 0),
        "end": job.arguments.get("end", 0),
        "part": "",
        "values": [],
        "notes": [],
        "holds": None,
        "error": error,
    }


class _Worker:
    """A worker process, as the notebook sees it: a job goes to its standard input, and its
    answer comes back on its standard output, read by a thread of its own."""

    def __init__(
        self, generation: int, events: queue.Queue[tuple[str, Any]], memory_limit: int | None
    ) -> None:
        self.generation = generation
        self.ready = False
        self._events = events
        self._memory_limit = memory_limit
        self._process: subprocess.Popen[bytes] | None = None
        self._said: deque[str] = deque(maxlen=20)  # the last lines it wrote to stderr

    def start(self) -> None:
        from relativisticpy.mcp import _environment, _quiet_folder

        settings = json.dumps({"memory_limit": self._memory_limit})
        code = "from relativisticpy.live.worker import main; main()"
        self._process = subprocess.Popen(
            [sys.executable, "-c", code, settings],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(_quiet_folder()),
            env=_environment(),
            start_new_session=sys.platform != "win32",  # a terminal's Ctrl-C is the kernel's
        )
        answers, said = self._process.stdout, self._process.stderr
        assert answers is not None
        assert said is not None
        for target, pipe in ((self._read, answers), (self._listen, said)):
            thread = threading.Thread(target=target, args=(pipe,), name="relativisticpy-live")
            thread.daemon = True
            thread.start()

    def send(self, message: Any) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise OSError("the worker has stopped")
        _send(process.stdin, message)

    def interrupt(self) -> bool:
        """Interrupt the computation in hand, as Ctrl-C would; whether it could be."""
        process = self._process
        if process is None or sys.platform == "win32" or process.poll() is not None:
            return False
        with contextlib.suppress(ProcessLookupError):
            process.send_signal(signal.SIGINT)
        return True

    def stop(self) -> None:
        """Stop the process at once."""
        process, self._process = self._process, None
        if process is None:
            return
        with contextlib.suppress(ProcessLookupError):
            process.kill()
        with contextlib.suppress(Exception):
            process.wait(timeout=GRACE)
        for pipe in (process.stdin, process.stdout, process.stderr):
            with contextlib.suppress(Exception):
                if pipe is not None:
                    pipe.close()

    def why(self) -> str | None:
        """What the process last wrote on its standard error, if anything."""
        said = " ".join(line.strip() for line in self._said).strip()
        return f"the worker wrote: {said}" if said else None

    def _read(self, pipe: IO[bytes]) -> None:
        try:
            while True:
                self._events.put(("message", (self.generation, _receive(pipe))))
        except Exception:  # its end of the pipe closed, or what it sent cannot be read
            self._events.put(("lost", self.generation))

    def _listen(self, pipe: IO[bytes]) -> None:
        with contextlib.suppress(Exception):
            for line in pipe:
                self._said.append(line.decode("utf-8", "replace").rstrip())


class InPlace:
    """Computes the editor's jobs in the notebook's own process, where no worker can be
    started, as in a browser's Python, or be given the workbook: at once, in the thread
    that asks, which a time limit stops only if it is the main thread. ``note`` says why,
    under each run."""

    def __init__(
        self, workbook: Workbook, *, time_limit: float = TIME_LIMIT, note: str | None = None
    ) -> None:
        self.workbook = workbook
        self.time_limit = time_limit
        self.note = note
        self.engine = Engine(workbook)
        self._loaded = workbook._revision

    def submit(self, job: Job) -> None:
        if job.revision != self._loaded:
            self.engine.load(self.workbook)
            self._loaded = job.revision
        arguments = {**job.arguments, "limit": self.time_limit, "name": NAME}
        try:
            kind, data = _do(self.engine, job.kind, arguments)
        except Exception as error:  # a failure of our own: reported as the job's answer
            failure = internal_error(error, during="computing this")
            job.give(job.answer, _failure(job, _data(failure)))
            return
        if self.note and kind == "ran":
            data = {**data, "notes": [*data["notes"], self.note]}
        job.give(kind, data)

    def warm(self) -> None:
        """Nothing to start."""

    def close(self) -> None:
        """Nothing to stop."""


# == the worker's side =======================================================================


def _send(wire: IO[bytes], message: Any) -> None:
    data = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
    wire.write(len(data).to_bytes(8, "big") + data)
    wire.flush()


def _receive(wire: IO[bytes]) -> Any:
    size = int.from_bytes(_read(wire, 8), "big")
    return pickle.loads(_read(wire, size))


def _read(wire: IO[bytes], size: int) -> bytes:
    data = b""
    while len(data) < size:
        chunk = wire.read(size - len(data))
        if not chunk:
            raise EOFError("the other end has closed the pipe")
        data += chunk
    return data


def _do(engine: Engine, kind: str, arguments: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Do a job with an engine: the kind of its answer, and the answer."""
    limit = arguments.get("limit")
    if kind == "run":
        return "ran", engine.run(arguments["text"], arguments["name"], limit)
    return "computed", engine.compute(
        arguments["text"],
        arguments["start"],
        arguments["end"],
        arguments["action"],
        limit,
        arguments["name"],
    )


_computing = False  # whether an interruption stops what the worker is doing


def _interrupted(signum: int, frame: Any) -> None:
    if _computing:
        raise KeyboardInterrupt


def _numbered_apart() -> None:
    """Number what this process makes apart from what the notebook's process made: the
    constants of the IR and the dummy indices of index notation are numbered by counters
    of the process, and the workbook this one is given comes with the notebook's numbers."""
    from relativisticpy import abstract
    from relativisticpy.ir import graph

    graph._TOKENS = itertools.count(1 << 62)
    abstract._FRESH = itertools.count(1 << 62)


def main() -> None:
    """Do jobs until the notebook closes the pipe: the worker's main function."""
    from relativisticpy.mcp import _claim_standard_output, _limit_memory, _watch_parent

    global _computing
    settings: dict[str, Any] = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    wire = _claim_standard_output()
    _watch_parent()
    _limit_memory(settings.get("memory_limit"))
    _numbered_apart()
    signal.signal(signal.SIGINT, _interrupted)
    engine = Engine()
    _send(wire, ("ready", os.getpid()))
    while True:
        try:
            kind, arguments, state = _receive(sys.stdin.buffer)
        except EOFError:
            return  # the notebook is done
        reply: tuple[str, Any]
        try:
            if state is not None:
                engine.load(Workbook._carried(*pickle.loads(state)))
            _computing = True
            try:
                reply = _do(engine, kind, arguments)
            finally:
                _computing = False
        except KeyboardInterrupt:
            reply = ("stopped", {})
        except Exception as error:  # a failure of the worker's own: reported, not fatal
            failure = error if isinstance(error, RelativisticPyError) else internal_error(error)
            reply = ("failed", _data(failure))
        _send(wire, reply)
