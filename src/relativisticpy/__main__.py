"""``python -m relativisticpy``, and the ``relativisticpy`` command, whose entry point is
:func:`main`: it fixes Python's hash seed, then runs :mod:`relativisticpy.cli`.

Python orders a set of names by their hashes, which change from one run to the next unless
``PYTHONHASHSEED`` fixes them, and SymPy's algorithms walk such sets, so that a few take
another path in another run: the same program must give the same answer every time
(design/interface.md). So, before it imports anything that hashes a name, the command
starts itself again with ``PYTHONHASHSEED=0`` where the variable is not set, with
:func:`os.execve`, which keeps its process, its arguments and its standard streams; the
processes it starts for its work, the workers of ``-j`` and of ``relativisticpy mcp``,
inherit the seed. A seed already set is kept, so ``PYTHONHASHSEED=random`` asks for one
of Python's choosing, and a command started again is marked, ``RELATIVISTICPY_RESTARTED``,
so that it never starts itself a third time.

Windows has no ``execve`` that keeps the process: there, the command keeps the seed
Python chose, and only the processes it starts are given 0. Set ``PYTHONHASHSEED=0``
before it runs to reproduce a run exactly there.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence

#: Set in the environment of a command started again with a fixed seed, so that it never
#: starts itself again, whatever becomes of the seed.
RESTARTED = "RELATIVISTICPY_RESTARTED"


def fix_hash_seed() -> None:
    """Fix Python's hash seed, where ``PYTHONHASHSEED`` does not: give it 0 for every
    process this one starts, then, on a platform that can replace a process, start this
    one again with it, as it was started: from a script or a module. Returns only where it
    does not start again."""
    if os.environ.get("PYTHONHASHSEED") or os.environ.get(RESTARTED):
        return
    os.environ["PYTHONHASHSEED"] = "0"
    started_from_a_file = os.path.isfile(sys.argv[0]) and len(sys.orig_argv) > 1
    if sys.platform == "win32" or sys.flags.ignore_environment or not started_from_a_file:
        return
    environment = {**os.environ, RESTARTED: "1"}
    try:
        os.execve(sys.executable, [sys.executable, *sys.orig_argv[1:]], environment)
    except OSError:  # the interpreter cannot be started again: keep the seed Python chose
        return


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command with ``argv``, by default ``sys.argv[1:]``, and return its exit
    status. Started with the command line's own arguments, it first fixes the hash seed;
    given them, as tests give them, it runs in this process as it is."""
    if argv is None:
        fix_hash_seed()
    from relativisticpy.cli import main as run

    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
