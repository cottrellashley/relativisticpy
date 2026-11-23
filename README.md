<p align="center">
  <img src="https://raw.githubusercontent.com/cottrellashley/relativisticpy/main/assets/logo.gif" alt="RelativisticPy" width="280">
</p>

<h1 align="center">RelativisticPy</h1>

<p align="center">
  <b>RelativisticPy reads mathematics as it is written, in LaTeX, plain text or Unicode, and computes it.</b><br>
  Calculus, linear algebra, probability, signals, quantum mechanics and General Relativity, in the notation of their books; data from NumPy and pandas; and the equations of LaTeX papers, which it checks.
</p>

<p align="center">
  <img alt="Python 3.10 or newer" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-blue"></a>
</p>

---

Write an integral, a matrix, a random variable or a line element as a book writes it,
and get its value. Write index notation over NumPy or pandas data. Run
`relativisticpy check paper.tex` to see which of a paper's equations hold. RelativisticPy
is a Python package and a language: you write the mathematics, it computes it, and it
says how it read each line.

## Install

RelativisticPy needs Python 3.10 or newer. This is version 0.1.0, the first release of
RelativisticPy as a language. PyPI still serves the releases of an earlier
implementation, 1.1.9 to 1.2.9, which pip would choose, and the last of which fails to
import; until they are withdrawn there, install it from GitHub:

```bash
pip install "relativisticpy @ git+https://github.com/cottrellashley/relativisticpy"
```

Its only dependency is SymPy. Extras add what numerical work needs:

| Extra | Adds | For |
|---|---|---|
| `numeric` | NumPy and SciPy | numbers, NumPy functions and geodesics |
| `data` | NumPy, SciPy, pandas and scikit-learn | data in index notation |
| `jax` | NumPy, SciPy and JAX | computing with JAX, on a GPU, and gradients |
| `mcp` | the MCP SDK | serving RelativisticPy to AI agents, `relativisticpy mcp` |
| `live` | anywidget | the editor that computes mathematics as it is typed, in marimo and Jupyter |

```bash
pip install "relativisticpy[data] @ git+https://github.com/cottrellashley/relativisticpy"
```

## A first program

```python
from relativisticpy import Workbook

wb = Workbook()
wb.expr(r"""
    ds^2 = -\left(1 - \frac{2GM}{c^2 r}\right) c^2 dt^2
           + \frac{dr^2}{1 - \frac{2GM}{c^2 r}} + r^2 d\Omega^2
""")
print(wb.expr(r"\Gamma^{r}_{tt}"))  # a Christoffel symbol
print(wb.expr(r"R_{\mu\nu} = 0"))  # the vacuum field equations
print(wb.expr(r"R_{abcd} R^{abcd}"))  # the Kretschmann scalar
```

prints

```text
G*M*(-2*G*M + c**2*r)/(c**2*r**3)
R_{μν} = 0: holds in all 10 components
48*G**2*M**2/(c**4*r**6)
```

The line element is written as a paper writes it, and names its own coordinates: t, r, θ
and φ. In Python, write LaTeX in raw strings, `r"..."`, as here: in an ordinary string,
Python turns `\f` in `\frac`, and `\t` in `\theta`, into control characters.

## Mathematics, as its books write it

Each field is read in its own notation, by the names its books use. Every block here is
a session of its own.

Calculus, series and differential equations:

```python
>>> from relativisticpy import Workbook
>>> wb = Workbook()
>>> wb.expr(r"\int_0^\infty e^{-x^2} \, dx")
sqrt(pi)/2
>>> wb.expr(r"\sum_{n=1}^{\infty} \frac{1}{n^2}")
pi**2/6
>>> wb.expr(r"dsolve(y'' + y = 0, y(x))")
Eq(y(x), C1*sin(x) + C2*cos(x))

```

Special functions, and transforms:

```python
>>> wb = Workbook()
>>> wb.expr(r"\Gamma\left(\tfrac{1}{2}\right)")
sqrt(pi)
>>> wb.expr(r"{}_2F_1(1, 1; 2; z)")
-log(1 - z)/z
>>> wb.expr(r"\mathcal{L}\{t^2 e^{-t}\}(s)")
2/(s + 1)**3

```

Matrices and vectors:

```python
>>> wb = Workbook()
>>> wb.exe(r"A := \begin{pmatrix} 2 & 1 \\ 1 & 3 \end{pmatrix}")
[]
>>> wb.expr(r"\operatorname{tr} A"), wb.expr(r"\det A"), wb.expr(r"\lambda_{\max}(A)")
(5, 5, sqrt(5)/2 + 5/2)
>>> wb.expr(r"\nabla \times (-y, x, 0)")
[0, 0, 2]

```

Probability, with random variables declared as books declare them:

```python
>>> wb = Workbook()
>>> wb.exe(r"X \sim \mathcal{N}(0, 1)")
[]
>>> wb.expr(r"\operatorname{Var}(3X + 1)")
9
>>> wb.expr(r"\Pr(X > 1)")
1/2 - erf(sqrt(2)/2)/2

```

Optima, with their constraints:

```python
>>> wb = Workbook()
>>> wb.expr(r"\min_{x, y} \left( x^2 + y^2 \right) \quad \text{subject to} \quad x + y = 1")
1/2

```

Sequences, sums and number theory:

```python
>>> wb = Workbook()
>>> wb.exe(r"x[n] := \left(\tfrac{1}{2}\right)^n u[n]")
[]
>>> wb.expr(r"\mathcal{Z}\{x[n]\}(z)")
2*z/(2*z - 1)
>>> wb.expr(r"\varphi(36)"), wb.expr(r"7^{-1} \bmod 26"), wb.expr(r"\sum_{d \mid 12} d")
(12, 15, 28)

```

Quantum mechanics, with operators and Dirac's notation:

```python
>>> wb = Workbook()
>>> wb.exe(r"\hat{p} := -i\hbar \frac{d}{dx}")
[]
>>> wb.expr(r"[\hat{x}, \hat{p}] = i\hbar")
True
>>> wb.exe(r"|+\rangle := \frac{1}{\sqrt{2}} \begin{pmatrix} 1 \\ 1 \end{pmatrix}")
[]
>>> wb.expr(r"\langle +|+\rangle")
1

```

[The language](docs/language.md) has each of these in full, and `relativisticpy help NAME`
describes any function: how it is written, what it takes, and what computes it.

## It says how it read you

Notation is ambiguous, and RelativisticPy never guesses silently. Where it assumes what
a book would, a note says so:

```python
>>> step = next(iter(Workbook().each(r"\int_0^\infty e^{-s t} \, dt")))
>>> step.values, step.notes
([1/s], ['assuming s > 0, as books take a scale to be; the program does not say what s is'])

```

Write `s > 0` before it, and that is an assumption for the rest of the program.
`explain` shows how a statement is read, computing nothing:

```python
>>> print(Workbook().explain(r"\Gamma\left(\tfrac{1}{2}\right)"))
line 1: \Gamma\left(\tfrac{1}{2}\right)
  book name: \Gamma\left(\tfrac{1}{2}\right) is gamma, Euler's gamma function, since \Gamma(x) is how books write it, and nothing defines \Gamma
  note: \Gamma is read as Euler's gamma function, gamma; for a function of your own, define \Gamma first

```

What it cannot read, or cannot decide, it refuses, saying why, where, and what to write
instead. A claim that numbers support but nothing proves is not shown as true:

```python
>>> Workbook().expr(r"\arctan x + \arctan \frac{1}{x} = \frac{\pi}{2}")
Traceback (most recent call last):
  ...
relativisticpy.tensors.equation.UndecidedError: whether \arctan x + \arctan \frac{1}{x} = \frac{\pi}{2} holds is not decided: the sides agree numerically, but this was not proven
 --> <input>:1:1
  |
1 | \arctan x + \arctan \frac{1}{x} = \frac{\pi}{2}
  | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
note: shown as it is, an undecided claim would read as an answer
hint: to see what is left, write the difference of its sides alone, as an expression

```

## General Relativity

Derive the black hole yourself. Start from a general static, spherically symmetric
metric and let the Ricci tensor do the work:

```python
>>> from relativisticpy import Workbook
>>> Workbook().expr("""
...     Coordinates := [t, r, theta, phi]
...     g_{mu nu} := [[-A(r), 0, 0, 0], [0, B(r), 0, 0], [0, 0, r**2, 0], [0, 0, 0, r**2*sin(theta)**2]]
...
...     eq0 := Ric_{mu=0 nu=0}
...     eq1 := Ric_{mu=1 nu=1}
...     eq2 := Ric_{mu=2 nu=2}
...
...     B1 := RHS(dsolve((eq0*B(r) + eq1*A(r))*(r*B(r)), B(r)))
...     RHS(dsolve(simplify(subs(eq2, B(r), B1)), A(r)))
... """)
C1 + C2/r

```

That is Schwarzschild: `A(r) = C1 + C2/r`, with the constants fixed by the Newtonian
limit.

- **Index notation that checks itself.** Einstein summation over repeated indices;
  raising and lowering with the metric; traces, symmetrisation `T_{(ab)}` and
  antisymmetrisation `F_{[ab]}`, even across tensors as in the Bianchi identity.
  Mistakes are caught *before* anything is computed:

  ```python
  >>> wb = Workbook()
  >>> wb.expr(r"g_{\mu\nu} := Schwarzschild(M)")
  >>> wb.expr(r"R_{\mu\nu} + g^{\mu\nu}")
  Traceback (most recent call last):
    ...
  relativisticpy.errors.IndexNotationError: the free indices of a sum must agree: _{\mu\nu} and ^{\mu\nu}
   --> <input>:1:14
    |
  1 | R_{\mu\nu} + g^{\mu\nu}
    |              ^^^^^^^^^^

  ```

- **Geometry, cached and fast.** Christoffel symbols, the Riemann, Ricci, Einstein and
  Weyl tensors, the Ricci and Kretschmann scalars, geodesic equations, line elements,
  and a catalogue of spacetimes: `Minkowski`, `Schwarzschild`, `ReissnerNordstrom`,
  `Kerr`, `KerrNewman`, `FLRW`, `DeSitter` and `AntiDeSitter`. Checking that Kerr is a
  vacuum solution takes seconds.

- **Tensor equations, solved as tensor equations.** Before there is a metric, an
  equation between tensors is kept as one, and `solve`, `trace`, `simplify` and `subs`
  rearrange it as you would on paper, with the identities of General Relativity built
  in. Solving Einstein's equations for the Ricci tensor reverses their trace:

  ```python
  >>> wb = Workbook()
  >>> wb.exe(r"EFE := G_{\mu\nu} + \Lambda g_{\mu\nu} = 8\pi G T_{\mu\nu}")
  []
  >>> wb.expr(r"solve(EFE, R_{\mu\nu})")
  IndexEquation(R_{μν} = 8π G T_{μν} + (Λ − 4π G T) g_{μν})

  ```

  With a metric, the solution has components: for an FLRW universe, `T_{tt}` is the
  Friedmann equation's energy density.

- **Numbers when you need them.** Compile any result to a NumPy function, or send a
  particle around a black hole with SciPy:

  ```python
  >>> from relativisticpy import spacetimes
  >>> from relativisticpy.numeric import four_velocity, geodesic
  >>> g = spacetimes.schwarzschild()                       # geometric units, M = 1
  >>> start = [0, 10, 3.141592653589793 / 2, 0]            # t, r, θ, φ
  >>> u = four_velocity(g, start, [0, 0, (1 / 700) ** 0.5], M=1)
  >>> orbit = geodesic(g, start, u, (0, 1000), M=1)
  >>> print(orbit["r"].min().round(6), orbit["r"].max().round(6))   # circular, at r = 10M
  10.0 10.0

  ```

## Data

Index notation never needed spacetime: an index ranges over a space, of coordinates,
assets, days or the states of a qubit. Bind data from NumPy or pandas and write
statistics, spectra and linear algebra as on a page; RelativisticPy computes them with
NumPy, SciPy, scikit-learn and JAX. The minimum-variance portfolio of two assets, by
Cholesky solves rather than an inverse:

```python
>>> import numpy as np
>>> market = Workbook()
>>> market["R_{ta}"] = np.array([[0.01, 0.02], [0.03, -0.01], [-0.02, 0.01], [0.00, 0.03]])
>>> market.exe(r"\Sigma_{ab} := \operatorname{Cov}_t(R_{ta}, R_{tb})")
[]
>>> market.expr(r"\frac{(\Sigma^{-1})^{ab} 1_b}{1_c (\Sigma^{-1})^{cd} 1_d}").to_numpy().round(4).tolist()
[0.4351, 0.5649]

```

Python objects are read by name. Give a workbook a scope, and `\texttt{name}` is the
object of that name in it: a NumPy array, a pandas table, whose labels become the bases
of its spaces, a SymPy expression or a Python function. Here a sampled tone, and the bin
where its spectrum peaks:

```python
>>> signal = np.sin(2 * np.pi * 5 * np.arange(64) / 64)
>>> wb = Workbook(scope={"signal": signal})
>>> wb.exe(r"x_t := \texttt{signal}")
[]
>>> wb.expr(r"\operatorname*{arg\,max}_k \left| \operatorname{fft}(x)_k \right|")
5

```

A paper that is checked, and a program an AI agent sends, are given no scope: neither
can reach into Python. Portfolios, quantum systems, graphs and regressions are worked
through in [Beyond spacetime](docs/spaces.md).

## In notebooks

Results are typeset in Jupyter and marimo. `show` runs a program and shows each statement
with what came of it:

```python
>>> from relativisticpy.live import show
>>> show(r"""
... f(x) := x^2 e^{-x}
... f'(x)
... \int_0^\infty f(x) \, dx
... """)
f(x) := x^2 e^{-x}
  defines f(x)
f'(x)
  (2x − x²) e^(−x)
\int_0^\infty f(x) \, dx
  2

```

In a notebook the same call shows them typeset. `Editor` is a widget that computes
mathematics as it is typed, underlines an error where it is, and computes any part of a
statement from a menu; in marimo, a cell that passes its data to `show` runs again when
the data changes. [Live mathematics](docs/live.md) has both, and they need the `live`
extra.

## Checking papers

`relativisticpy check paper.tex` runs the display equations of a LaTeX document in
order, as a reader would, and says which hold, which do not, which relate quantities the
paper never defines, and which it could not read. It reads 98.6% of the equations in Sean
Carroll's *Lecture Notes on General Relativity* as written, confirms 24 of the 25 claims
his notes make about the connection and curvature of a spherically symmetric spacetime,
and finds the 25th, and his Kretschmann scalar, to be mistaken.
[Checking papers](docs/checking.md) shows how to reproduce this.

Markdown notebooks run too: `relativisticpy notes.md --render` runs the `$$...$$` blocks
of a Markdown file and typesets the results beneath them.

## One language, three notations

RelativisticPy reads plain text, LaTeX and Unicode, and they mix freely. Copy an
equation out of a paper and it parses; type it quickly by hand and it means the same.

| Plain text | LaTeX | Meaning |
|---|---|---|
| `g_{mu nu} := [[...]]` | `g_{\mu\nu} := \begin{pmatrix} ... \end{pmatrix}` | define the metric |
| `C^{a}_{b c}` | `\Gamma^{\lambda}_{\mu\nu}` | Christoffel symbols |
| `R^{a}_{b c d}` | `R^{\rho}{}_{\sigma\mu\nu}` | Riemann tensor |
| `Ric_{a b}` | `R_{\mu\nu}` | Ricci tensor (and `Ric`, `R` the scalar) |
| `G_{a b}` | `G_{\mu\nu}` | Einstein tensor |
| `d_{a} * T^{b}` | `\partial_\mu T^{\nu}`, `T^{\nu}{}_{,\mu}` | partial derivative |
| `D_{a} * T^{b}` | `\nabla_\mu T^{\nu}`, `T^{\nu}{}_{;\mu}` | covariant derivative |
| `Ric_{mu=0 nu=0}` | `R_{tt}` | a single component |
| `diff(f(x), x, 2)` | `f''(x)`, `\frac{d^2}{dx^2} f(x)` | derivatives of a function |
| `integrate(f, (x, 0, 1))` | `\int_0^1 f \, dx` | integrals |
| `f(x) \|_{x = 2}` | `\left. f(x) \right\|_{x=2}` | evaluation |
| `f(x) := x^2` | `f(x) \equiv x^2` | definitions |

`:=` always defines. `=` defines only what has no meaning yet: a new name, a new
function, a new tensor, a line element. Otherwise it states a claim, which is checked,
or an equation to solve.

## For AI agents

Everything needed to use RelativisticPy is in the installed package, offline.

- **Call it as a tool.** `relativisticpy mcp` serves RelativisticPy over the Model
  Context Protocol to Claude Code, Claude Desktop or any MCP client: `compute` runs
  mathematics written as in a paper, `verify` checks a claimed step and gives the
  difference of its sides in LaTeX, `check` checks a paper, and `help` is the
  documentation. Install it with the `mcp` extra, and connect it:

  ```bash
  pip install "relativisticpy[mcp] @ git+https://github.com/cottrellashley/relativisticpy"
  claude mcp add relativisticpy -- relativisticpy mcp
  ```

  [The MCP server](docs/mcp.md) has the configuration of other clients, every tool, and
  the bounds the server keeps: no Python, files only in folders allowed, and a time limit
  on every call.

- **Learn the language.** `relativisticpy help` lists the pages of the documentation,
  and `relativisticpy help TOPIC` prints one. `relativisticpy help reference` prints the
  whole language on one page of about 11,000 tokens: every statement, command,
  environment, constant, statistic and verdict, and every function by name, which
  `relativisticpy help NAME` describes in full, as `relativisticpy help besselj` does:
  its forms, its arguments, its examples with their answers, and what computes it. From
  Python:

  ```python
  >>> from relativisticpy.help import page, topics
  >>> ("reference", "The language reference") in topics()
  True
  >>> print(page("reference").splitlines()[0])
  # The language reference

  ```

- **Read results as data.** `--format json` gives the command's results, and the
  report of a check, as JSON. Each value says whether it is an answer: one kept in part
  as written, as an integral with no closed form, has `"complete": false` and its
  `reasons`, and one with floating-point numbers in it is `approximate`. In Python,
  `display.data(value)` describes any result, and `report.to_dict()` and
  `error.to_dict()` give a check and an error:

  ```python
  >>> from relativisticpy import RelativisticPyError, display
  >>> display.data(Workbook().expr(r"\frac{48 M^2}{r^6}"))["text"]
  '48*M^2/r^6'
  >>> try:
  ...     Workbook().expr(r"\frac{1}{2")
  ... except RelativisticPyError as error:
  ...     found = error.to_dict()
  >>> found["type"], found["line"], found["column"], found["hint"]
  ('ParseError', 1, 9, "add '}' where the argument ends")

  ```

- **Exit status** of the `relativisticpy` command:

  | Status | Meaning |
  |---|---|
  | 0 | done, and everything succeeded |
  | 1 | a statement failed, or a check failed by `--strict` |
  | 2 | the command line was wrong: an unknown option or a missing file |
  | 130 | interrupted with Ctrl-C |
  | 143 | stopped with SIGTERM |

- **Pitfalls.**
  1. In Python, write LaTeX in raw strings, `r"\frac{1}{2}"`.
  2. `:=` always defines; `=` defines only what has no meaning yet, and is otherwise a
     claim, which is checked.
  3. In LaTeX a run of letters is a product, as it is typeset: `\frac{2GM}{r}` is
     2·G·M/r. Write `\mathrm{name}` for a name of several letters.
  4. A script without braces takes one letter, as in TeX: write `\Gamma^{r}_{tt}`, not
     `\Gamma^r_tt`.
  5. A name right before a bracket calls it, as in `A(r)`; to multiply, leave a space or
     write `\,`.
  6. `%` starts a comment, as in LaTeX.

The documentation is also served for language models as
[llms.txt](https://relativisticpy.org/llms.txt), an index, and
[llms-full.txt](https://relativisticpy.org/llms-full.txt), every page in one file.

## Extending it

Every function of the language is a capability that an integration declares: a module
of `relativisticpy/integrations/` that follows one pattern, with examples whose answers
come from outside it. A conformance suite holds every capability to the pattern, and
says in words what to fix. So adding a library, by a person or by an agent, is writing
one module: there is no configuration. SymPy, SciPy, NumPy and JAX are integrated this
way, and [Integrations](docs/integrations.md) shows how to add another.

What the language should read is measured by a [curriculum](curriculum/README.md): 734
problems across 32 fields at five levels, each written as its book writes it, with an
answer computed without RelativisticPy. The test suite keeps every problem that passes
passing, and the ones that fail say what to build next.

## Documentation

The documentation is in [docs](docs/), and `relativisticpy help` prints it offline:
[Getting started](docs/getting-started.md), [the language](docs/language.md),
[LaTeX](docs/latex.md), [tensors and geometry](docs/tensors.md),
[solving tensor equations](docs/notation.md), [checking papers](docs/checking.md),
[beyond spacetime](docs/spaces.md), [numerics](docs/numerics.md),
[the language reference](docs/reference.md), [the Python API](docs/python.md),
[the command line](docs/cli.md), [the MCP server](docs/mcp.md),
[integrations](docs/integrations.md), [troubleshooting](docs/troubleshooting.md),
[coming from other tools](docs/coming-from.md), [coming from 1.x](docs/upgrading.md),
[Jupyter and notebooks](docs/jupyter.md), [live mathematics](docs/live.md),
[how it works](docs/architecture.md) and the [roadmap](docs/roadmap.md). The [examples](examples/) are Markdown notebooks and
Python scripts, shown with their results in [Examples](docs/examples.md).

## Development

```bash
uv sync                     # install with the development tools
uv run pytest               # the test suite, including every example in the docs
uv run ruff check           # lint
uv run mypy                 # type-check
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT, see [LICENSE](LICENSE).
