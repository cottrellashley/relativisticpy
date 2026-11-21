# Checking papers

A paper is a program whose statements are its display equations.
`relativisticpy check` runs them in order, as a reader would, and says what
became of each: which hold, which do not, and which it could not read.

```bash
relativisticpy check notes.tex                    # the whole document
relativisticpy check notes.tex --lines 9058:9112  # one passage
relativisticpy check notes.tex --all              # every statement, with reasons
relativisticpy check notes.tex --time-limit 60    # seconds for each statement; inf for none
relativisticpy check notes.tex -j 4               # four processes for component-wise work
relativisticpy check notes.tex --strict           # fail if an equation differs or is not parsed
relativisticpy check notebook.md                  # a Markdown notebook
```

A statement gets 30 seconds by default, to be computed and judged; one that takes
longer is not computed. The command exits with status 0 whatever it finds, unless
it is given `--strict`: then an equation that differs, or one that cannot be
parsed, makes it exit with status 1, for a continuous integration job to act on.

## An example

Here is a short passage in the style of a textbook, with a mistake in its last
equation:

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
... """, "paper.tex")
>>> report = Workbook().check(paper)
>>> print(report.summary())
paper.tex: 3 equations, 5 statements
  ✓ holds               3
  ✗ differs             1
  = defines             1
>>> for finding in report.findings[-2:]:
...     print(finding.describe())
line 9      ✓ holds           \Gamma^3_{23} = {\cos\theta \over \sin\theta
line 12     ✗ differs         R_{22} = e^{-2\beta}[r(\partial_1\alpha - \partial_1\beta) - 1] + 1
              the left side minus the right is (-2*r*Derivative(alpha(r), r) + 2*r*Derivative(beta(r), r))*exp(-2*beta(r))

```

Nothing was declared. The line element named the coordinates t, r, θ and φ,
`\alpha(r)` made `\alpha` a function of r from then on, `0` and `1` in
`\Gamma^1_{00}` are the coordinates t and r, and the `\qquad` separated two
equations. The sign in R₂₂ should be the other way round.

`Workbook.check` takes a `Document`, a path to a LaTeX file or a Markdown
notebook, and optionally `lines=(first, last)` and a `time_limit` in seconds
for each statement. The document runs in a workbook of its own, with the same
workers, so the workbook's definitions take no part and are kept. Whatever
fails in one statement is a finding about it, and the check goes on.

## Verdicts

| | Verdict | Meaning |
|---|---|---|
| ✓ | holds | an identity, proven: component by component, or, for an equation in [index notation](notation.md) without components, as notation. If both sides are zero the finding says so: such an equation may hold only because space is flat there, or its quantities are not yet defined |
| ✗ | differs | the two sides are not identically equal: an error, or an equation to solve, such as a condition or an equation of motion. The finding shows the difference |
| = | defines | a definition, which the equations after it use |
| → | computes | an expression, computed, or an equation in index notation that is not an identity, such as a field equation |
| ~ | not computed | a relation that RelativisticPy reads but does not decide, such as ≈, ∝ or →; an equation that is neither proven nor shown to differ; or a statement over its time limit |
| ? | not understood | read, but needing context that RelativisticPy does not have, such as the components of an abstract tensor, or failing in RelativisticPy itself |
| ! | not parsed | not yet RelativisticPy |

An equation holds when it is proven. Its sums, products and integrals are carried
out, and the difference of its sides must vanish, exactly or, between
floating-point numbers, to a relative 10⁻¹²: `0.1 + 0.2 = 0.3` holds. When no
proof is found, numbers tell the other two cases apart. The symbols are given
positive values, and functions that nothing defines, such as `\alpha(r)`, random
polynomials. An equation whose sides differ wherever they are tried differs; one
whose sides agree, as \(\sqrt{x^2} = x\) does for positive x, or cannot be
computed, is not decided. A side that is exactly zero has no size for a
floating-point number to be compared with, so `0.1 + 0.2 - 0.3 = 0` differs.

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
benchmark for this feature. Of their 797 display equations, 787 (98.7%)
parse. The other ten are mostly words laid out as matrices. Checked from end
to end, in under five seconds, 95 statements hold, 52 of them with sides that
are not both zero.

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

Most of the notes concern tensors on general manifolds, without components.
Their equations are read as [index notation](notation.md), and those that are
identities are verified as such: four, among them
\(g^{\mu\nu} g_{\nu\sigma} = \delta^\mu_\sigma\) and the definition of the
Einstein tensor. Laws, such as Einstein's equation, are read but are not
identities. Many more statements take derivatives of tensors without
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

| Paper | Equations | Parse | Hold | Differ |
|---|---:|---:|---:|---:|
| Carroll, *Lecture Notes on General Relativity* | 797 | 98.7% | 95 | 132 |
| gr-qc 2611.26560 | 31 | 71.0% | 1 | 11 |
| gr-qc 2611.26576 | 305 | 43.0% | 0 | 61 |
| gr-qc 2611.26585 | 270 | 63.0% | 2 | 30 |
| gr-qc 2611.26591 | 27 | 88.9% | 0 | 10 |
| gr-qc 2611.26594 | 63 | 87.3% | 1 | 9 |
| gr-qc 2611.26620 | 47 | 72.3% | 0 | 8 |

The recent papers lean on their own notation, from physics packages to
macros built on TeX internals, so they are where the language grows next.
