"""How much real general-relativity LaTeX RelativisticPy reads, and checks.

Fetches the LaTeX source of papers from arXiv, finds every display equation,
and reports how many parse and what checking them finds, with the commonest
reasons equations fail to parse. The corpus is a list of arXiv identifiers:
papers are fetched into ~/.cache/relativisticpy/arxiv, never stored here.

    uv run python benchmarks/corpus.py                          # every paper
    uv run python benchmarks/corpus.py gr-qc/9712019 --kinds 20    # one paper
"""

from __future__ import annotations

import argparse
import collections
import io
import tarfile
import urllib.request
from pathlib import Path

from relativisticpy import Workbook
from relativisticpy.checking import Verdict
from relativisticpy.documents import Document
from relativisticpy.errors import RelativisticPyError
from relativisticpy.language import parse

CORPUS = {
    "gr-qc/9712019": "Carroll, Lecture Notes on General Relativity (1997)",
    # Six papers submitted to gr-qc in November 2026, chosen only for being recent.
    "2611.26560": "gr-qc 2611.26560",
    "2611.26576": "gr-qc 2611.26576",
    "2611.26585": "gr-qc 2611.26585",
    "2611.26591": "gr-qc 2611.26591",
    "2611.26594": "gr-qc 2611.26594",
    "2611.26620": "gr-qc 2611.26620",
}
CACHE = Path.home() / ".cache" / "relativisticpy" / "arxiv"


def fetch(identifier: str) -> Path:
    """The paper's LaTeX source, all its .tex files in one, downloaded once."""
    folder = CACHE / identifier.replace("/", "-")
    combined = folder / "combined.tex"
    if combined.exists():
        return combined
    folder.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        f"https://arxiv.org/e-print/{identifier}",
        headers={"User-Agent": "relativisticpy-corpus"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read()
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        archive.extractall(folder / "src", filter="data")
    # Packages first, for their macros, then the documents.
    files = sorted((folder / "src").rglob("*.sty")) + sorted((folder / "src").rglob("*.tex"))
    combined.write_text("\n".join(path.read_text(errors="replace") for path in files))
    return combined


def reason(error: Exception) -> str:
    """An error's kind, for counting: its first line, without positions."""
    if not isinstance(error, RelativisticPyError):
        return f"crash: {type(error).__name__}: {error}"
    return str(error).splitlines()[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("papers", nargs="*", default=list(CORPUS), help="arXiv identifiers")
    parser.add_argument("--kinds", type=int, default=12, help="failure kinds to list")
    parser.add_argument("--examples", type=int, default=1, help="examples of each kind")
    args = parser.parse_args()

    failures: collections.Counter[str] = collections.Counter()
    examples: dict[str, list[str]] = collections.defaultdict(list)
    print(f"{'paper':52} {'equations':>9} {'parse':>7} {'holds':>6} {'differs':>8}")
    for identifier in args.papers:
        path = fetch(identifier)
        document = Document.read(path)
        equations = document.equations()
        parsed = 0
        for equation in equations:
            try:
                parse(equation.source)
                parsed += 1
            except RelativisticPyError as error:
                kind = reason(error)
                failures[kind] += 1
                if len(examples[kind]) < args.examples:
                    text = " ".join(equation.source.text.split())
                    examples[kind].append(f"line {equation.line}: {text[:150]}")
        with Workbook() as workbook:
            report = workbook.check(document, time_limit=20)
        title = CORPUS.get(identifier, identifier)[:52]
        share = parsed / len(equations) if equations else 0
        holds, differs = report.count(Verdict.HOLDS), report.count(Verdict.DIFFERS)
        print(f"{title:52} {len(equations):>9} {share:>7.1%} {holds:>6} {differs:>8}")
    for kind, count in failures.most_common(args.kinds):
        print(f"\n{count:5d}  {kind}")
        for example in examples[kind]:
            print(f"       {example}")


if __name__ == "__main__":
    main()
