# Getting started

> **For:** everyone, first. **You will learn:** how to install RelativisticPy, run a
> workbook from Python, and run a program, a notebook and a check from the command line.

## Install

RelativisticPy needs Python 3.10 or newer. PyPI still serves the releases of an earlier
implementation, 1.1.9 to 1.2.9, which pip would choose; until they are withdrawn there,
install it from GitHub:

```bash
pip install "relativisticpy @ git+https://github.com/cottrellashley/relativisticpy"
```

Its only dependency is [SymPy](https://www.sympy.org), which does the algebra. Extras
add what numerical work needs:

| Extra | Adds | For |
|---|---|---|
| `numeric` | NumPy and SciPy | numbers, NumPy functions and geodesics: [Numerics](numerics.md) |
| `data` | NumPy, SciPy, pandas and scikit-learn | data in index notation: [Beyond spacetime](spaces.md) |
| `jax` | NumPy, SciPy and JAX | computing with JAX, on a GPU, and gradients |
| `live` | anywidget | the editor, which computes as you type: [Live mathematics in notebooks](live.md) |

```bash
pip install "relativisticpy[data] @ git+https://github.com/cottrellashley/relativisticpy"
```

## Raw strings

> **Write LaTeX in Python in raw strings:** `r"\frac{1}{2}"`, not `"\frac{1}{2}"`. In an
> ordinary string, Python turns `\a`, `\b`, `\f`, `\n`, `\r`, `\t` and `\v` into control
> characters before RelativisticPy sees them, so `"\theta"` arrives as a tab and `heta`.
> RelativisticPy refuses such text, and says to write a raw string.

## Your first workbook

A `Workbook` is a session, like a notebook: definitions persist from one call to the
next. `expr` runs a program and returns the value of its last expression; `exe` returns
the value of every expression.

```python
>>> from relativisticpy import Workbook
>>> wb = Workbook()
>>> wb.expr(r"""
...     Coordinates := [t, r, \theta, \phi]
...     g_{\mu\nu} := diag(-(1 - 2M/r), 1/(1 - 2M/r), r^2, r^2 \sin^2\theta)
... """)
>>> wb.exe(r"""
...     R
...     R_{abcd} R^{abcd}
... """)
[0, 48*M**2/r**6]

```

`:=` defines. `=` defines too, but only what has no meaning yet: with a metric in force,
`R = 0` asks whether the Ricci scalar vanishes, and says so. Scalars come back as SymPy
expressions. Tensors come back as [`Tensor`](python.md#tensor) values, which know their
indices and coordinates:

```python
>>> christoffel = wb.expr(r"\Gamma^{a}_{bc}")
>>> christoffel.indices
Indices('^{a}_{bc}')
>>> christoffel.nonzero()[("theta", "phi", "phi")]
-sin(theta)*cos(theta)

```

Mistakes come back as errors that point at the source:

```python
>>> wb.expr(r"R^{a}_{bc}")
Traceback (most recent call last):
  ...
relativisticpy.errors.IndexNotationError: R has 2 or 4 indices, but 3 are given
 --> <input>:1:1
  |
1 | R^{a}_{bc}
  | ^^^^^^^^^^

```

## A program

A program is a text file, by convention ending in `.relpy`. Write one:

```bash
cat > schwarzschild.relpy <<'EOF'
g_{\mu\nu} := Schwarzschild(M)
\Gamma^{r}_{tt}
R_{abcd} R^{abcd}
EOF
```

`relativisticpy schwarzschild.relpy` runs it and prints each result. From Python,
`Workbook().run_file("schwarzschild.relpy")` runs the file, and `exe` the same program
given as text:

```python
>>> Workbook().exe(r"""
...     g_{\mu\nu} := Schwarzschild(M)
...     \Gamma^{r}_{tt}
...     R_{abcd} R^{abcd}
... """)
[M*(-2*M + r)/r**3, 48*M**2/r**6]

```

The command does more:

```bash
relativisticpy -c '\int_0^1 x^2 \, dx'          # run code given on the command line
relativisticpy --latex schwarzschild.relpy      # print the results as LaTeX
relativisticpy -j 8 schwarzschild.relpy         # share component-wise work among 8 processes
relativisticpy                                  # an interactive prompt, where definitions persist
relativisticpy --help                           # every option
```

## A notebook

A Markdown file whose equations are in displays, `$$ ... $$`, is a notebook: GitHub
typesets it, and RelativisticPy runs its displays in order, as one program.

```bash
cat > sphere.md <<'EOF'
# The 2-sphere

A sphere of radius $a$:

$$
ds^2 = a^2 \left(d\theta^2 + \sin^2\theta\, d\phi^2\right)
$$

Its Ricci scalar:

$$
R
$$
EOF
relativisticpy sphere.md            # print the results
relativisticpy sphere.md --render   # print the document, with each result typeset under its display
```

[Jupyter and notebooks](jupyter.md) says more, and the [examples](examples.md) are
notebooks of this kind.

## A check

`relativisticpy check` runs the display equations of a LaTeX document, or a notebook, in
order, as a reader would, and gives each a verdict: which hold, which differ, which
define something, and which it could not read.

```bash
cat > sphere.tex <<'EOF'
\begin{equation} ds^2 = a^2 (d\theta^2 + \sin^2\theta\, d\phi^2) \end{equation}
The sphere is curved: \begin{equation} R = \frac{2}{a^2} \end{equation}
and not by half as much: \begin{equation} R = \frac{1}{a^2} \end{equation}
EOF
relativisticpy check sphere.tex
```

From Python, `wb.check` takes a file, or the text of a document:

```python
>>> report = Workbook().check(r"""
... \begin{equation} ds^2 = a^2 (d\theta^2 + \sin^2\theta\, d\phi^2) \end{equation}
... The sphere is curved: \begin{equation} R = \frac{2}{a^2} \end{equation}
... and not by half as much: \begin{equation} R = \frac{1}{a^2} \end{equation}
... """)
>>> [(finding.line, finding.verdict) for finding in report]
[(2, 'defines'), (3, 'holds'), (4, 'differs')]

```

[Checking papers](checking.md) explains every verdict.

## Where next

- [The language](language.md): statements, `:=` and `=`, functions, equations and
  calculus; and the [language reference](reference.md), all of it on one page.
- [Tensors and geometry](tensors.md), for General Relativity.
- [Beyond spacetime](spaces.md), for data.
- [Checking papers](checking.md), for LaTeX documents.
- [Troubleshooting](troubleshooting.md), when a result surprises you.

Offline, `relativisticpy help` lists these pages and prints any of them.
