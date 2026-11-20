# RelativisticPy

**General Relativity for physicists in a hurry.** Write tensor equations the
way they look in a paper, or paste the LaTeX, and compute them.

```python
>>> from relativisticpy import Workbook
>>> wb = Workbook()
>>> wb.expr(r"""
...     Coordinates := [t, r, \theta, \phi]
...     ds^2 = -\left(1 - \frac{2M}{r}\right) dt^2 + \frac{dr^2}{1 - \frac{2M}{r}} + r^2 d\Omega^2
... """)
>>> wb.expr(r"\Gamma^{r}_{tt}")
M*(-2*M + r)/r**3
>>> wb.expr(r"G_{\mu\nu} = 0")
TensorEquation(_{mu}_{nu}, 0 equations)
>>> wb.expr(r"R_{abcd} R^{abcd}")
48*M**2/r**6

```

## Why

Tensor calculations in General Relativity are long, mechanical and easy to get
wrong by hand. Computer algebra systems can do them, but each asks you to
translate your equations into its own vocabulary of functions and objects first.

RelativisticPy asks nothing of the kind. Its input *is* the notation of the
field: indices up and down, Einstein summation, Christoffel symbols written as
\(\Gamma^\lambda{}_{\mu\nu}\), derivatives written as \(\partial_\mu\) or
\(\nabla_\mu\). It reads both quick plain-text maths and LaTeX, so an equation
copied from a paper or your own notes runs as it is.

## Where to go next

- [Getting started](getting-started.md): install, your first workbook, the
  command line and Markdown notebooks.
- [The language](language.md): definitions, functions, equations, calculus.
- [Tensors and geometry](tensors.md): index notation, the built-in tensors and
  their conventions, line elements, spacetimes.
- [LaTeX](latex.md): how LaTeX is read, and what is supported.
- [Numerics](numerics.md): NumPy functions and geodesic integration.
- [Python API](python.md): using the library directly from Python.
- [How it works](architecture.md): lexer, parser, compiler, IR and executor.
