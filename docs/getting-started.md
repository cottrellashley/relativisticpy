# Getting started

## Install

```bash
pip install relativisticpy               # symbolic
pip install "relativisticpy[numeric]"    # with NumPy and SciPy for numerics
```

RelativisticPy needs Python 3.10 or newer. Its only dependency is
[SymPy](https://www.sympy.org), which does the algebra.

## Your first workbook

A `Workbook` is a session, like a notebook: definitions persist from one call
to the next. `expr` runs a program and returns the value of its last
expression; `exe` returns the value of every expression.

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

Scalars come back as SymPy expressions. Tensors come back as
[`Tensor`](python.md#tensor) values that know their indices and coordinates:

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

## The command line

The `relativisticpy` command runs a program file, code given with `-c`, or a
Markdown document:

```bash
relativisticpy schwarzschild.relpy           # print each result
relativisticpy -c "\int_0^1 x^2 \, dx"        # 1/3
relativisticpy --ir schwarzschild.relpy       # show the IR instead of computing
relativisticpy -j 8 kerr.relpy                # simplify with eight processes
relativisticpy                                # an interactive session
```

Tensors print as their nonzero components:

```text
$ relativisticpy -c 'g_{\mu\nu} := Schwarzschild(M)
\Gamma^{a}_{bc}'
Gamma^{a}_{b}_{c}:
  [t, t, r] = M/(r*(-2*M + r))
  [t, r, t] = M/(r*(-2*M + r))
  [r, t, t] = M*(-2*M + r)/r**3
  ...
```

## Markdown notebooks

Write your derivation in Markdown, with the equations in `$$...$$` blocks as
usual. GitHub and most editors typeset them; RelativisticPy runs them, in
order, as one program:

```bash
relativisticpy notes.md            # print the results
relativisticpy notes.md --render   # print the document with results typeset under each block
```

From Python, `Workbook.markdown(text)` does the same as `--render` and
`Workbook.run_file(path)` the same as running the file. The
[examples](https://github.com/cottrellashley/relativisticpy/tree/main/examples)
are written this way.
