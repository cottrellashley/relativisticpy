<p align="center">
  <img src="https://raw.githubusercontent.com/cottrellashley/relativisticpy/main/assets/logo.gif" alt="RelativisticPy" width="280">
</p>

<h1 align="center">RelativisticPy</h1>

<p align="center">
  <b>General Relativity for physicists in a hurry.</b><br>
  Write tensor equations the way they look in a paper, or paste the LaTeX, and compute them.
</p>

<p align="center">
  <a href="https://pypi.org/project/relativisticpy/"><img alt="PyPI" src="https://img.shields.io/pypi/v/relativisticpy"></a>
  <a href="https://github.com/cottrellashley/relativisticpy/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/cottrellashley/relativisticpy/actions/workflows/ci.yml/badge.svg"></a>
  <img alt="Python" src="https://img.shields.io/pypi/pyversions/relativisticpy">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-blue"></a>
</p>

---

Give RelativisticPy a line element, straight from a paper:

```python
>>> from relativisticpy import Workbook
>>> wb = Workbook()
>>> wb.expr(r"""
...     Coordinates := [t, r, \theta, \phi]
...     ds^2 = -\left(1 - \frac{2GM}{c^2 r}\right) c^2 dt^2
...            + \frac{dr^2}{1 - \frac{2GM}{c^2 r}} + r^2 d\Omega^2
... """)

```

and ask it anything, in the same notation:

```python
>>> wb.expr(r"\Gamma^{\theta}_{\phi\phi}")
-sin(theta)*cos(theta)
>>> wb.expr(r"\Gamma^{r}_{tt}")
G*M*(-2*G*M + c**2*r)/(c**2*r**3)
>>> wb.expr(r"R_{\mu\nu} = 0")          # the vacuum field equations hold
TensorEquation(_{mu}_{nu}, 0 equations)
>>> wb.expr(r"R_{abcd} R^{abcd}")       # the Kretschmann scalar
48*G**2*M**2/(c**4*r**6)

```

Or derive the black hole yourself. Start from a general static, spherically
symmetric metric and let the Ricci tensor do the work:

```python
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

That is Schwarzschild: `A(r) = C1 + C2/r`, with the constants fixed by the
Newtonian limit.

## One language, two notations

RelativisticPy reads plain-text maths and LaTeX, and the two mix freely. Copy
an equation out of a paper and it parses; type it quickly by hand and it means
the same thing.

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
| `diff(f(x), x, 2)` | `\frac{d^2 f}{dx^2}`, `f''(x)`, `\pdv[2]{f}{x}` | derivatives |
| `integrate(f, (x, 0, 1))` | `\int_0^1 f \, dx` | integrals |
| `f(x) \|_{x = 2}` | `\left. f(x) \right\|_{x=2}` | evaluation |
| `f(x) := x^2` | `f(x) \equiv x^2` | definitions |

## What's inside

- **Index notation that checks itself.** Einstein summation over repeated
  indices; raising and lowering with the metric; traces, symmetrisation
  `T_{(ab)}` and antisymmetrisation `F_{[ab]}`, even across tensors as in the
  Bianchi identity. Mistakes are caught *before* anything is computed:

  ```python
  >>> wb.expr(r"R_{\mu\nu} + g^{\mu\nu}")
  Traceback (most recent call last):
    ...
  relativisticpy.errors.IndexNotationError: the free indices of a sum must agree: _{mu}_{nu} and ^{mu}^{nu}
   --> <input>:1:14
    |
  1 | R_{\mu\nu} + g^{\mu\nu}
    |              ^^^^^^^^^^

  ```

- **Geometry, cached and fast.** Christoffel symbols, the Riemann, Ricci,
  Einstein and Weyl tensors, the Ricci and Kretschmann scalars, geodesic
  equations, line elements, and a catalogue of spacetimes: `Minkowski`,
  `Schwarzschild`, `ReissnerNordstrom`, `Kerr`, `KerrNewman`, `FLRW`, `DeSitter`,
  `AntiDeSitter`. Checking that Kerr is a vacuum solution takes seconds.

- **The rest of mathematics.** Every derivative notation, integrals, limits,
  sums and products, `solve`, `dsolve`, `series`, `simplify` and friends,
  piecewise functions and `cases`, matrices, complex numbers, and functions
  of your own with proper lexical scope:

  ```python
  >>> wb.expr(r"\int_0^\infty e^{-x^2} \, dx")
  sqrt(pi)/2
  >>> wb.expr(r"\lim_{x \to 0} \frac{\sin x}{x}")
  1
  >>> wb.expr(r"f(x) := x^2 e^{-x}" "\n" r"f'(x)")
  (-x**2 + 2*x)*exp(-x)

  ```

- **Numbers when you need them.** Compile any result to a NumPy function, or
  send a particle around a black hole with SciPy:

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

- **Papers that check themselves.** `relativisticpy check paper.tex` runs the
  display equations of a LaTeX document in order, as a reader would, and says
  which hold, which do not, and which it could not read. It reads 98.7% of the
  equations in Sean Carroll's *Lecture Notes on General Relativity* as written,
  confirms 24 of the 25 claims his notes make about the connection and
  curvature of a spherically symmetric spacetime, and finds the 25th, and his
  Kretschmann scalar, to be mistaken. See
  [Checking papers](https://relativisticpy.org/checking/).

- **Tensor equations, solved as tensor equations.** Before there is a metric,
  an equation between tensors is kept as one, and `solve`, `trace`, `simplify`
  and `subs` rearrange it as you would on paper, with the identities of General
  Relativity built in. Solving Einstein's equations for the Ricci tensor
  reverses their trace:

  ```python
  >>> wb.exe(r"EFE := G_{\mu\nu} + \Lambda g_{\mu\nu} = 8\pi G T_{\mu\nu}")
  []
  >>> wb.expr(r"solve(EFE, R_{\mu\nu})")
  IndexEquation(R_{\mu\nu} = 8 \pi G T_{\mu\nu} + \left(\Lambda - 4 \pi G T\right) g_{\mu\nu})

  ```

  With a metric, the solution has components: for an FLRW universe, \(T_{tt}\)
  is the Friedmann equation's energy density. See
  [Solving tensor equations](https://relativisticpy.org/notation/).

- **Notebooks in Markdown.** `relativisticpy notes.md --render` runs every
  `$$...$$` block of a Markdown file and typesets the results beneath it. The
  equations you write up *are* the computation.

## How it works

```
source text ─▶ lexer ─▶ parser ─▶ syntax tree ─▶ compiler ─▶ IR graph ─▶ executor ─▶ results
               plain text        every reading     meaning, index     interned,     lazy, cached,
               and LaTeX         of ambiguous      checks, lowering   typed         parallel
               become one        notation kept     to a small core
               token stream
```

The parser records what was *written*; the compiler, which knows what every
name refers to, decides what it *means* (is `x^{a}` a power, or a component of
the vector `x`?) and checks the index notation. Programs compile to a typed
graph in which equal computations are shared, and the executor computes only
what the results need, spreading component-wise work over processes with
`Workbook(workers=N)`. SymPy does the algebra. Read more in
[the architecture guide](https://relativisticpy.org/architecture/).

## Philosophy

1. **Ease of use.** Einstein's equations should look exactly as they do in
   papers, with an engine that can actually compute them.
2. **Lightweight.** The only dependency is SymPy, which does the symbolic
   work. NumPy and SciPy are optional, for numerics.

## Install

```bash
pip install relativisticpy               # symbolic
pip install "relativisticpy[numeric]"    # with NumPy and SciPy for numerics
```

Python 3.10 or newer.

## Documentation

The guide at **[relativisticpy.org](https://relativisticpy.org)** covers
[the language](https://relativisticpy.org/language/),
[tensors and geometry](https://relativisticpy.org/tensors/),
[solving tensor equations](https://relativisticpy.org/notation/),
[LaTeX](https://relativisticpy.org/latex/),
[checking papers](https://relativisticpy.org/checking/),
[numerics](https://relativisticpy.org/numerics/) and
[the Python API](https://relativisticpy.org/python/). The
[examples](https://github.com/cottrellashley/relativisticpy/tree/main/examples) are Markdown notebooks you can read on GitHub and run
with `relativisticpy`.

## Development

```bash
uv sync                     # install with the development tools
uv run pytest               # the test suite, including every example in the docs
uv run ruff check           # lint
uv run mypy                 # type-check
```

See [CONTRIBUTING.md](https://github.com/cottrellashley/relativisticpy/blob/main/CONTRIBUTING.md).

## License

MIT, see [LICENSE](https://github.com/cottrellashley/relativisticpy/blob/main/LICENSE).
