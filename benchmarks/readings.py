"""How every statement the repository writes is read: a line each. Two runs compared say
which statements a change to the language reads otherwise, and only those.

Each program of the docs, the curriculum and the tests is compiled a statement at a time,
in a session of its own, as ``tests/ir/test_closed.py`` gathers them, and nothing is
computed. A statement's reading is the operations it compiles to, written out, with the
notes on how it was read, or the error it is refused with::

    uv run python benchmarks/readings.py --write before.tsv
    # ... change the lexer, the parser or the reader ...
    uv run python benchmarks/readings.py --write after.tsv
    uv run python benchmarks/readings.py --compare before.tsv after.tsv

Every statement listed by ``--compare`` is read otherwise than before: each should be one
the change meant to change.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from relativisticpy.compiler.compiler import Compiler
from relativisticpy.compiler.session import Session
from relativisticpy.ir.graph import Node
from relativisticpy.language import Source, parse
from relativisticpy.runtime.executor import Executor

ROOT = Path(__file__).resolve().parent.parent

#: How much of a reading a line shows; the whole is compared by its digest.
SHOWN = 240


def sources() -> dict[str, list[tuple[str, list[str]]]]:
    """The programs of the docs, the curriculum and the tests, by where they are written,
    as ``tests/ir/test_closed.py`` gathers them."""
    sys.path.insert(0, str(ROOT / "tests"))
    spec = importlib.util.spec_from_file_location("_closed", ROOT / "tests/ir/test_closed.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SOURCES  # type: ignore[no-any-return]


def written(node: Node, seen: dict[int, str]) -> str:
    """A node as the operations that make it, each written once: a node met again is
    named by the order it was first met in."""
    if node.id in seen:
        return seen[node.id]
    label = node.op.label()
    arguments = ", ".join(written(arg, seen) for arg in node.args)
    text = f"{label}({arguments})" if node.args else label
    seen[node.id] = f"#{len(seen)}"
    return text


def reading(compiler: Compiler, statement: Any, scope: Any) -> str:
    """How a statement is read: what it shows, and the notes on how it was read; or the
    error it is refused with."""
    try:
        outputs = compiler.statement(statement, scope)
    except Exception as error:  # refused, or failed: either is how it was read
        message = getattr(error, "message", None) or str(error)
        return f"! {type(error).__name__}: {message.splitlines()[0] if message else ''}"
    seen: dict[int, str] = {}
    parts = [written(output.node, seen) for output in outputs]
    notes = [note for output in outputs if output.note for note in output.note.split("\n")]
    notes += compiler.take_notes()
    text = " ; ".join(parts) or "-"
    if notes:
        text += "  [" + " | ".join(notes) + "]"
    return re.sub(r" at 0x[0-9a-f]+", "", text)


def read_all(groups: dict[str, list[tuple[str, list[str]]]]) -> list[tuple[str, str, str]]:
    """Every statement, with where it is written and how it is read, in order."""
    rows: list[tuple[str, str, str]] = []
    refusing = {"evaluate", "evaluate_with", "_evaluate", "map", "simplify_map"}

    def computing(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("compiling computed with the executor")

    saved = {name: getattr(Executor, name) for name in refusing}
    for name in refusing:
        setattr(Executor, name, computing)
    try:
        for source, programs_of in groups.items():
            for where, programs in programs_of:
                session = Session()
                compiler = Compiler(session)
                count = 0
                for program in programs:
                    try:
                        statements = parse(Source(program)).statements
                    except Exception as error:  # what does not parse is read so too
                        first = program.strip().splitlines()[0] if program.strip() else ""
                        message = getattr(error, "message", str(error))
                        rows.append((f"{source}:{where}#{count}", first, f"! {message}"))
                        count += 1
                        continue
                    for statement in statements:
                        text = " ".join(statement.span.text.split())
                        rows.append(
                            (
                                f"{source}:{where}#{count}",
                                text,
                                reading(compiler, statement, session.globals),
                            )
                        )
                        count += 1
    finally:
        for name, method in saved.items():
            setattr(Executor, name, method)
    return rows


def write(path: Path) -> None:
    started = time.monotonic()
    rows = read_all(sources())
    with path.open("w", encoding="utf-8") as file:
        for key, text, read in rows:
            digest = hashlib.sha1(read.encode()).hexdigest()[:12]
            line = "\t".join(
                (key, text.replace("\t", " "), digest, read[:SHOWN].replace("\t", " "))
            )
            file.write(line + "\n")
    refused = sum(1 for _, _, read in rows if read.startswith("!"))
    print(
        f"{len(rows)} statements read, {refused} refused, in {time.monotonic() - started:.1f} s:"
        f" {path}"
    )


def load(path: Path) -> dict[tuple[str, str], tuple[str, str]]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, text, digest, shown = [*line.split("\t"), "", "", ""][:4]
        rows[(key, text)] = (digest, shown)
    return rows


def compare(before_path: Path, after_path: Path, limit: int) -> int:
    before, after = load(before_path), load(after_path)
    changed = [key for key in before if key in after and before[key][0] != after[key][0]]
    gone = [key for key in before if key not in after]
    new = [key for key in after if key not in before]
    kinds: Counter[str] = Counter()
    for key in changed:
        was, now = before[key][1], after[key][1]
        kinds[
            f"{'refused' if was.startswith('!') else 'read'} -> "
            f"{'refused' if now.startswith('!') else 'read'}"
        ] += 1
    for key in changed[:limit]:
        where, text = key
        print(f"{where}\n  {text[:150]}")
        print(f"  before {before[key][1][:200]}\n  after  {after[key][1][:200]}")
    if len(changed) > limit:
        print(f"... and {len(changed) - limit} more")
    moved = ", ".join(f"{count} {kind}" for kind, count in kinds.items()) or "none"
    print(
        f"\n{len(changed)} statements read otherwise ({moved}),"
        f" {len(gone)} gone, {len(new)} new, of {len(after)}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--write", type=Path, metavar="FILE", help="read every statement into FILE")
    parser.add_argument("--compare", type=Path, nargs=2, metavar=("BEFORE", "AFTER"))
    parser.add_argument("--limit", type=int, default=200, help="statements shown by --compare")
    args = parser.parse_args(argv)
    if args.write:
        write(args.write)
    if args.compare:
        return compare(*args.compare, args.limit)
    if not args.write and not args.compare:
        parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
