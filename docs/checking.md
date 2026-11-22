# Checking papers

A paper is a program whose statements are its display equations.
`relativisticpy check` runs them in order, as a reader would, and says what
became of each: which hold, which do not, which relate quantities the paper never
defines, and which it could not read.

```bash
relativisticpy check notes.tex                    # the whole document
relativisticpy check notes.tex --lines 9058:9112  # one passage
relativisticpy check notes.tex --all              # every statement, with reasons
relativisticpy check notes.tex --time-limit 60    # seconds for each statement; inf for none
relativisticpy check notes.tex -j 4               # four processes for component-wise work
relativisticpy check notes.tex --strict           # fail on what is listed as failing
relativisticpy check notebook.md                  # a Markdown notebook
```

A statement gets 30 seconds by default, to be computed and judged; one that takes
longer is not computed. The command exits with status 0 whatever it finds, unless
it is given `--strict`: then an equation that differs, notation that is ill-formed,
a statement that cannot be parsed or is over its time limit, or a document in which
no equation was found, makes it exit with status 1, for a continuous integration job
to act on.

## An example

Here is a short passage in the style of a textbook, with a mistake in one of its
equations:

```python
>>> from relativisticpy import Workbook
>>> from relativisticpy.documents import Document
>>> paper = Document(r"""
... \newcommand{\be}{\begin{equation}}
... \newcommand{\ee}{\end{equation}}
... A static, spherically symmetric spacetime has the line element
... \be ds^2 = -e^{2\alpha(r)}{\rm d} t^2 + e^{2\beta(r)}{\rm d} r^2 + r^2 d\Omega^2 \ , \ee
... whose connection includes
... \begin{eqnarray}
...   \Gamma^1_{00} &=& e^{2(\alpha-\beta)}\partial_1\alpha \ , \cr
...   \Gamma^1_{22} &=& -r e^{-2\beta} \ , \qquad \Gamma^3_{23} = {\cos\theta \over \sin\theta} \ .
... \end{eqnarray}
... Its curvature has
... \be R_{22} = e^{-2\beta}[r(\partial_1\alpha - \partial_1\beta) - 1] + 1 \ . \ee
... A body in orbit has
... \be \omega^2 = \frac{G M}{r^3} \ . \ee
... """, "paper.tex")
>>> report = Workbook().check(paper)
>>> print(report.describe())
paper.tex: 4 equations, 6 statements, checked in ... s
  ✓ holds               3   proven equal
  ✗ differs             1   of known quantities, whose sides are not equal: a mistake, or a condition
  ↔ relates             1   relations between quantities nothing defines, not identities
  = defines             1   definitions
<BLANKLINE>
paper.tex:12:5: ✗ differs  R_{22} = e^{-2\beta}[r(\partial_1\alpha - \partial_1\beta) - 1] + 1
    the left side minus the right is (2r β'(r) − 2r α'(r)) e^(−2 β(r))
<BLANKLINE>
Not listed: 3 holds, 1 relates, 1 defines. --all lists every statement.

```

Nothing was declared. The line element named the coordinates t, r, θ and φ,
`\alpha(r)` made `\alpha` a function of r from then on, `0` and `1` in
`\Gamma^1_{00}` are the coordinates t and r, and the `\qquad` separated two
equations. The sign in R₂₂ should be the other way round: every quantity in it is
known, so the difference of its sides is a mistake, or a condition the passage has
not stated. The orbit's ω, G and M are defined nowhere, so that equation relates
them; it is not an identity, and not a mistake either.

`Workbook.check` takes a `Document`, the text of a document, or the path of a LaTeX
file or a Markdown notebook; text with a line break or a `\begin` that is not the
path of a file is read as a document. A file must be UTF-8. It takes optionally
`lines`, a part of the document, and a `time_limit` in seconds for each statement.
The document runs in a workbook of its own, with the same workers, so the workbook's
definitions take no part and are kept. Whatever fails in one statement is a finding
about it, and the check goes on.

## Verdicts

| | Verdict | Meaning | Fails `--strict` |
|---|---|---|---|
| ✓ | holds | proven equal: component by component, or, for an equation in [index notation](notation.md) without components, as notation. If both sides are zero the finding says so: such an equation may hold only because space is flat there, or its quantities are not yet defined | no |
| ✗ | differs | every quantity is known, and the sides are not equal: a mistake, or a condition such as an equation of motion the text has not stated. The finding shows the difference of the sides, or, for tensors, in how many components they differ and the first few | yes |
| ↔ | relates | the sides involve quantities nothing defines: not a coordinate, not a name the metric is written with, not declared or defined before. It is a relation to solve, such as a law or Einstein's equation, not an identity. The finding names those quantities | no |
| ✗ | ill-formed | the notation breaks the rules of index notation, as written: an index three times in one term, a Greek index twice up or twice down, or terms or sides whose free indices differ. Only plain notation is judged, so the paper is never blamed for what RelativisticPy may have read otherwise | yes |
| = | defines | a definition, which the equations after it use | no |
| → | computes | an expression, computed | no |
| ~ | not computed | a relation that RelativisticPy reads but does not decide, such as ≈, ∝ or →; a condition on a coordinate; an equation that is neither proven nor shown to differ; a division by zero; or a statement over its time limit | yes, for the time limit |
| ? | not understood | something RelativisticPy cannot do yet, such as the derivative of an abstract tensor, with the reason. A failure of RelativisticPy's own is an internal error, which invites a report | no |
| ! | not parsed | not read, with the reason, the place and a hint | yes |

An equation holds when it is proven. Its sums, products and integrals are carried
out, and the difference of its sides must vanish, exactly or, between
floating-point numbers, to a relative 10⁻¹²: `0.1 + 0.2 = 0.3` holds. When no
proof is found, numbers tell the other two cases apart. The symbols are given
positive values, and functions that nothing defines, such as `\alpha(r)`, random
polynomials. An equation whose sides differ wherever they are tried differs, or
relates if a quantity in it is defined nowhere; one whose sides agree, as
\(\sqrt{x^2} = x\) does for positive x, or cannot be computed, is not decided. A
side that is exactly zero has no size for a floating-point number to be compared
with, so `0.1 + 0.2 - 0.3 = 0` differs.

## The report

The report begins with a summary: how many statements met each verdict, what each
verdict means, the commonest reasons statements were not understood, and how long
the check took. It then lists what fails, each finding located as `paper.tex:12:5`,
which terminals and editors link to, with the statement as the file has it, the
reason, the hint and, for a statement that could not be read, the place in it where
reading failed. It ends by counting what it did not list. Nothing is cut short,
unless the report is given the width of a terminal, to wrap its lines to.

```python
>>> print(report.describe(all=True))
paper.tex: 4 equations, 6 statements, checked in ... s
  ✓ holds               3   proven equal
  ✗ differs             1   of known quantities, whose sides are not equal: a mistake, or a condition
  ↔ relates             1   relations between quantities nothing defines, not identities
  = defines             1   definitions
<BLANKLINE>
paper.tex:5:5: = defines  ds^2 = -e^{2\alpha(r)}{\rm d} t^2 + e^{2\beta(r)}{\rm d} r^2 + r^2 d\Omega^2
paper.tex:8:3: ✓ holds  \Gamma^1_{00} &=& e^{2(\alpha-\beta)}\partial_1\alpha
paper.tex:9:3: ✓ holds  \Gamma^1_{22} &=& -r e^{-2\beta}
paper.tex:9:47: ✓ holds  \Gamma^3_{23} = {\cos\theta \over \sin\theta}
paper.tex:12:5: ✗ differs  R_{22} = e^{-2\beta}[r(\partial_1\alpha - \partial_1\beta) - 1] + 1
    the left side minus the right is (2r β'(r) − 2r α'(r)) e^(−2 β(r))
paper.tex:14:5: ↔ relates  \omega^2 = \frac{G M}{r^3}
    nothing defines G, M and ω: the left side minus the right is ω² − G M/r³

```

From Python, a `Report` is a sequence of `Finding`s, and a `Verdict` is its name,
so that `report.count("differs")` counts. `report.ok` says whether the check passes
`--strict`, and `report.failed(...)` whether it fails a policy of one's own, such
as only mistakes. `report.to_dict()` gives the whole report as data, for JSON:

```python
>>> report.count("differs"), report.ok, report.failed({"ill-formed", "not parsed"})
(1, False, False)
>>> [finding.verdict for finding in report]
['defines', 'holds', 'holds', 'holds', 'differs', 'relates']
>>> mistake = report.findings[4]
>>> mistake.location, mistake.text
('paper.tex:12:5', 'R_{22} = e^{-2\\beta}[r(\\partial_1\\alpha - \\partial_1\\beta) - 1] + 1')
>>> sorted(report.to_dict())
['context', 'counts', 'equations', 'file', 'findings', 'lines', 'ok', 'reasons', 'seconds', 'statements', 'warnings']

```

Each finding, as data, has its verdict, its file, line and column, the statement
as written, the reason, the hint and notes, and the difference of the sides in each
of the forms results are shown in.

## Part of a document

`--lines FIRST:LAST`, or `lines=(first, last)` from Python, checks a passage. What
comes before it in its section still runs first, so that its definitions are known,
as a reader who opens a book at a page has read the pages before it; only the
statements in the lines are reported. An environment that reaches into the lines
counts, as the `eqnarray` above, which starts on line 7, does for its rows:

```python
>>> passage = Workbook().check(paper, lines=(9, 13))
>>> [(finding.line, finding.verdict) for finding in passage]
[(9, 'holds'), (9, 'holds'), (12, 'differs')]
>>> print(passage.describe().splitlines()[-1])
Also run, for their definitions, and not reported: 2 statements outside lines 9–13.

```

From Python, `lines` may also be one line, `lines=12`, or open at either end,
`lines=(9, None)`.

## How a document is read

- **Display equations**: `equation`, `eqnarray`, `align`, `gather`,
  `multline`, `$$...$$` and `\[...\]`, including the shorthands authors define
  for them, such as `\be ... \ee`.
- **Macros**: the document's `\newcommand`, `\def` (with arguments),
  `\DeclareMathOperator` and `\let` are expanded, as are those of its local
  packages and the files it includes.
- **Order and scope**: equations run in order, so definitions carry forward.
  Each `\section` or `\chapter` starts afresh, because texts reuse symbols:
  Ω is a solid angle in one chapter and a density in the next.
- **Positions**: findings give the file's own line numbers.

## Reading physics as it is written

Texts leave a great deal implicit, and RelativisticPy reads it the way a
physicist does:

- **A line element names its coordinates**: `-dt^2 + dr^2 + r^2 d\Omega^2` is
  in t, r, θ and φ, unless coordinates were declared.
- **A function written once with its arguments**, `\alpha(t, r)`, is that
  function wherever `\alpha` appears afterwards.
- **Numbered indices are components**: `\Gamma^0_{00}`, `R^1{}_{212}`,
  `\partial_0 \alpha`. `\partial_0^2 \beta` is `\partial_0 \partial_0 \beta`.
- **Derivations are chains**: `a = b = c`, and rows that continue with `&=&`,
  give one claim per link.
- **Letters are symbols**: `re^{-2\beta}` is \(r\,e^{-2\beta}\), `2Gm(r)` is
  \(2G\,m(r)\), and with r a coordinate, `r(\partial_1\beta - 1)` is a product.
- **Sentences around equations**: closing punctuation is ignored, while commas,
  `\quad`, `\qquad` and `\Rightarrow` separate equations.

The [LaTeX page](latex.md#notation-from-papers) lists the notation in full.

## Carroll's lecture notes

Sean Carroll's *Lecture Notes on General Relativity*
([arXiv:gr-qc/9712019](https://arxiv.org/abs/gr-qc/9712019)) are the
benchmark for this feature. Of their 797 display equations, 786 (98.6%)
parse. The other eleven are mostly words laid out as matrices, and integrals
written in ways that leave their variables unclear. Checked from end to end, in
about five seconds, 116 statements hold, 75 of them with sides that are not both
zero.

The notes' chapter on the Schwarzschild solution makes 25 claims about the
general spherically symmetric metric: twelve Christoffel symbols, eight
components of the Riemann tensor and five of the Ricci tensor, in equations
(7.14) to (7.16). RelativisticPy confirms 24 of them. The 25th is an error in
the notes. In R₀₀, the bracket of time derivatives
\(\partial_0^2\beta + (\partial_0\beta)^2 - \partial_0\alpha\,\partial_0\beta\)
should carry a minus sign. A calculation from first principles, independent of
RelativisticPy's own curvature code, agrees. The slip is harmless: the notes
show next that \(\partial_0\beta = 0\), which removes the bracket.

A few pages later, (7.30) gives the Kretschmann scalar of the Schwarzschild
metric as \(12G^2M^2/r^6\). The check finds \(48G^2M^2/r^6\), the established
value. The conclusion drawn from it, that \(r = 0\) is a true singularity,
stands.

In all, of the equations whose sides are not equal, 20 involve only what the
notes have defined, and so differ, these two among them; most of the others are
steps of a derivation whose conditions are stated in the prose, as
\(\partial_0\beta = 0\) is. The other 175 relate quantities the notes never
define, as laws, equations of motion and the interval
\(s^2 = (\Delta x)^2 + (\Delta y)^2\) do. One equation is ill-formed: (1.56)
writes \(\eta_{\rho\nu}\eta^{\nu\mu} = \delta^\rho_\mu\), whose free indices call
for \(\delta^\mu_\rho\).

Most of the notes concern tensors on general manifolds, without components.
Their equations are read as [index notation](notation.md), and those that are
identities are verified as such: four, among them
\(g^{\mu\nu} g_{\nu\sigma} = \delta^\mu_\sigma\) and the definition of the
Einstein tensor. Laws, such as Einstein's equation, are read but are not
identities: they relate tensors, such as the stress–energy tensor, that nothing
defines. Many more statements take derivatives of tensors without
components, which index notation does not take yet, and so are not understood.
Two conventions that would let more of them be checked are not read yet either:
\(x^\mu\) as the coordinates themselves, and coordinates as functions along a
curve, \(x^\mu(\lambda)\), in the geodesic equation.

## The corpus

`benchmarks/corpus.py` measures all this. It fetches papers from arXiv by their
identifiers and reports, for each, how many equations parse and what checking
them finds. The papers themselves are never stored in the repository.

```bash
uv run python benchmarks/corpus.py
```

| Paper | Equations | Parse | Hold | Differ | Relate |
|---|---:|---:|---:|---:|---:|
| Carroll, *Lecture Notes on General Relativity* | 797 | 98.6% | 116 | 20 | 175 |
| gr-qc 2611.26560 | 31 | 77.4% | 1 | 1 | 10 |
| gr-qc 2611.26576 | 305 | 46.6% | 0 | 1 | 58 |
| gr-qc 2611.26585 | 270 | 63.0% | 0 | 1 | 24 |
| gr-qc 2611.26591 | 27 | 92.6% | 0 | 0 | 8 |
| gr-qc 2611.26594 | 63 | 87.3% | 1 | 0 | 4 |
| gr-qc 2611.26620 | 47 | 74.5% | 0 | 0 | 10 |

Most of what the recent papers write between quantities they introduce relates
them, as a paper's results do; few of their equations involve only what is
defined. They lean on their own notation, from physics packages to macros built on
TeX internals, so they are where the language grows next.
