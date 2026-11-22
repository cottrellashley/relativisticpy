# Upgrading from 1.x

> **For:** users of RelativisticPy 1.x. **You will learn:** what still works in version
> 2, what changed, and the program of version 1's README, rewritten.

Version 2 is a new implementation, from the reading of the text to the computing of its
results. Most programs of version 1 run, and most run as before; the changes below are
the ones to look for.

## Installing

Version 1.2.9, the last on PyPI, fails to import, because it needs a package, `loguru`,
that it does not declare. Version 2 needs only SymPy, and Python 3.10 or newer. Until it
is on PyPI, install it from GitHub:

```bash
pip install "relativisticpy @ git+https://github.com/cottrellashley/relativisticpy@v2"
```

## What still works

- `import relativisticpy as rel` and `rel.Workbook()`, and its `expr` method.
- Index notation in version 1's style: `g_{mu}_{nu}`, `Ric_{mu}_{nu}`, and components
  chosen with a colon, `Ric_{mu:0}_{nu:0}` and `C^{a:1}_{b:0 c:0}`.
- `Coordinates := [t, r, theta, phi]`, metrics as nested lists, and `**` for powers.
- `RHS`, `dsolve`, `subs`, `simplify` and `tsimplify`, and tensors defined from
  expressions, such as `Gamma^{a}_{c f} := (1/2)*g^{a b}*(\ldots)`.

## What changed

- **`:=` and `=`.** In version 1, `eq0 = Ric_{mu:0}_{nu:0}` named a value. In version 2,
  `:=` defines, and `=` defines only what has no meaning yet; otherwise it is a claim,
  which is checked. Write `:=` for every definition.
- **Names and unknown functions.** `B` and the unknown function `B(r)` are one name: once
  `B` has a value, `B(r)` no longer means the unknown function. Give a solution a name of
  its own, as `B_1` below.
- **Results.** `exe` returns a list of every value; `expr` returns the last. Arithmetic
  is exact, so `1/2` is a half, not `0.5`.
- **Errors** are raised as exceptions, every one a `RelativisticPyError`, with the place
  in the source and a hint, rather than returned as text.
- **LaTeX** runs as well as plain text: `\Gamma^{\lambda}_{\mu\nu}`, `\frac{1}{2}`,
  `\partial_\mu`, line elements and whole environments.
- **Modules.** The modules of version 1, such as `relativisticpy.core`,
  `relativisticpy.gr` and `relativisticpy.symengine`, are gone. Import from
  `relativisticpy` itself; [the Python API](python.md) lists what it holds.

## Version 1's README, in version 2

Version 1's README derived the Schwarzschild solution in these steps:

```text
eq0 = Ric_{mu:0}_{nu:0}
eq1 = Ric_{mu:1}_{nu:1}
eq2 = Ric_{mu:2}_{nu:2}
eq5 = (eq0*B(r) + eq1*A(r))*(r*B(r))
B = RHS( dsolve(eq5, B(r)) )
eq6 = simplify( subs(eq2, B(r), B) )
A = RHS( dsolve(eq6, A(r)) )
```

In version 2 it defines with `:=`, and names the solution for `B(r)` apart from the
function:

```python
>>> from relativisticpy import Workbook
>>> Workbook().expr("""
...     Coordinates := [t, r, theta, phi]
...     g_{mu}_{nu} := [[-A(r), 0, 0, 0], [0, B(r), 0, 0], [0, 0, r**2, 0], [0, 0, 0, r**2*sin(theta)**2]]
...     eq0 := Ric_{mu:0}_{nu:0}
...     eq1 := Ric_{mu:1}_{nu:1}
...     eq2 := Ric_{mu:2}_{nu:2}
...     B_1 := RHS(dsolve((eq0*B(r) + eq1*A(r))*(r*B(r)), B(r)))
...     RHS(dsolve(simplify(subs(eq2, B(r), B_1)), A(r)))
... """)
C1 + C2/r

```

That is Schwarzschild's A(r), with the constants that the Newtonian limit fixes.

**Next:** [Getting started](getting-started.md), for what version 2 adds, and
[the language](language.md).
