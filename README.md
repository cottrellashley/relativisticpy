<p align="center">
  <img src="https://raw.githubusercontent.com/cottrellashley/relativisticpy/main/assets/logo.gif" alt="RelativisticPy" width="280">
</p>

<h1 align="center">RelativisticPy</h1>

<p align="center">
  <b>RelativisticPy reads mathematics as it is written, in LaTeX, plain text or Unicode, and computes it.</b><br>
  General Relativity with SymPy, numerical data in index notation over any space, and the equations of LaTeX papers, which it checks.
</p>

<p align="center">
  <img alt="Python 3.10 or newer" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-blue"></a>
</p>

---

Type a line element and get its curvature. Write index notation over NumPy or pandas
data. Run `relativisticpy check paper.tex` to see which of a paper's equations hold.
RelativisticPy is a Python package and a language: you write the mathematics, and it
computes it.

## Install

RelativisticPy 2 needs Python 3.10 or newer. PyPI still has version 1, whose last
release, 1.2.9, fails to import; until version 2 is published there, install it from
GitHub:

```bash
pip install "relativisticpy @ git+https://github.com/cottrellashley/relativisticpy@v2"
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
pip install "relativisticpy[data] @ git+https://github.com/cottrellashley/relativisticpy@v2"
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

- **The rest of mathematics.** Every derivative notation, integrals, limits, sums and
  products, `solve`, `dsolve`, `series`, `simplify` and more, piecewise functions,
  matrices, complex numbers, and functions of your own:

  ```python
  >>> wb.expr(r"\int_0^\infty e^{-x^2} \, dx")
  sqrt(pi)/2
  >>> wb.expr(r"""
  ...     f(x) := x^2 e^{-x}
  ...     f'(x)
  ... """)
  (-x**2 + 2*x)*exp(-x)

  ```

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

Portfolios, quantum systems, graphs and regressions are worked through in
[Beyond spacetime](docs/spaces.md).

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
  pip install "relativisticpy[mcp] @ git+https://github.com/cottrellashley/relativisticpy@v2"
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
  report of a check, as JSON. In Python, `display.data(value)` describes any result,
  and `report.to_dict()` and `error.to_dict()` give a check and an error:

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

## Documentation

The documentation is in [docs](docs/), and `relativisticpy help` prints it offline:
[Getting started](docs/getting-started.md), [the language](docs/language.md),
[LaTeX](docs/latex.md), [tensors and geometry](docs/tensors.md),
[solving tensor equations](docs/notation.md), [checking papers](docs/checking.md),
[beyond spacetime](docs/spaces.md), [numerics](docs/numerics.md),
[the language reference](docs/reference.md), [the Python API](docs/python.md),
[troubleshooting](docs/troubleshooting.md), [coming from other tools](docs/coming-from.md),
[upgrading from 1.x](docs/upgrading.md), [Jupyter and notebooks](docs/jupyter.md) and
[how it works](docs/architecture.md). The [examples](examples/) are Markdown notebooks and
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
