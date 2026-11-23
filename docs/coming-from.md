# Coming from other tools

> **For:** users of SymPy, Mathematica and xAct, or EinsteinPy. **You will learn:** how
> what you write there is written here, and how RelativisticPy and SymPy work together.

RelativisticPy is not a library of functions to call but a language: you write the
mathematics, in LaTeX or plain text, and it computes it. Its results are SymPy values,
so whatever SymPy does to them still works.

```python
>>> import sympy
>>> from relativisticpy import Workbook
>>> wb = Workbook()
>>> wb.expr(r"g_{\mu\nu} := Schwarzschild(M)")
>>> K = wb.expr(r"R_{abcd} R^{abcd}")
>>> K
48*M**2/r**6
>>> sympy.diff(K, sympy.Symbol("r"))
-288*M**2/r**7

```

## SymPy

| SymPy | RelativisticPy |
|---|---|
| `x = sympy.Symbol("x")` | nothing: a name nothing defines is a symbol |
| `sympy.Symbol("a", positive=True)` | `a > 0`, or `Positive := [a]` |
| `f = sympy.Function("f")` | nothing: `f(x)` is an unknown function |
| `sympy.diff(f(x), x, 2)` | `f''(x)`, `\frac{d^2}{dx^2} f(x)`, or `diff(f(x), x, 2)` |
| `sympy.integrate(x**2, (x, 0, 1))` | `\int_0^1 x^2 \, dx`, or `integrate(x^2, (x, 0, 1))` |
| `sympy.Eq(x**2, 4)` | `x^2 = 4` |
| `sympy.solve(sympy.Eq(x**2, 4), x)` | `solve(x^2 = 4, x)` |
| `expr.subs(x, 2)` | `subs(expr, x, 2)`, or an evaluation bar |
| `sympy.Matrix([[1, 2], [3, 4]])` | `\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}` |
| `sympy.tensor.tensor` | index notation, as written: `R_{\mu\nu} - \frac{1}{2} R g_{\mu\nu}` |

A SymPy value can be bound to a name from Python, and used in the language:

```python
>>> x = sympy.Symbol("x")
>>> wb["f"] = sympy.sin(x) ** 2
>>> wb.expr(r"\frac{d}{dx} f")
2*sin(x)*cos(x)

```

## Mathematica and xAct

Mathematica's functions have their counterparts in the language, written as in LaTeX:

| Mathematica | RelativisticPy |
|---|---|
| `D[f[x], {x, 2}]` | `f''(x)` |
| `Integrate[f, {x, 0, 1}]` | `\int_0^1 f \, dx` |
| `Limit[Sin[x]/x, x -> 0]` | `\lim_{x \to 0} \frac{\sin x}{x}` |
| `Series[f, {x, 0, 5}]` | `series(f, x, 0, 6)`: SymPy counts the order of the remainder |
| `Solve[x^2 == 4, x]` | `solve(x^2 = 4, x)` |
| `DSolve[y'[x] == y[x], y[x], x]` | `dsolve(y'(x) = y(x), y(x))` |
| `Simplify[expr]`, `FullSimplify[expr]` | `simplify(expr)` |
| `Sum[1/n^2, {n, 1, Infinity}]` | `\sum_{n=1}^{\infty} \frac{1}{n^2}`, its closed form where there is one |

xAct's tensors are abstract until a chart gives them components. RelativisticPy works
at both levels too, without the definitions xAct asks for first:

| xAct | RelativisticPy |
|---|---|
| `DefManifold[M, 4, {a, b, c, d}]` | nothing: any letter, Latin or Greek, is an index |
| `DefMetric[-1, g[-a, -b], CD]` | `g_{\mu\nu} := \ldots`, or a line element, `ds^2 = \ldots` |
| `RiemannCD[-a, -b, -c, -d]` | `R_{abcd}` |
| `RicciCD[-a, -b]`, `RicciScalarCD[]` | `R_{ab}`, `R` |
| `EinsteinCD[-a, -b]` | `G_{ab}` |
| `ChristoffelCD[a, -b, -c]` | `\Gamma^{a}_{bc}` |
| `CD[-a][T[a, b]]` | `\nabla_a T^{ab}` |
| `ToCanonical[expr]` | `simplify(expr)`, for tensors without components |
| components with xCoba | components come with the metric: `\Gamma^{r}_{\theta\theta}` |

Without a metric, an equation between tensors stays notation, and `simplify` brings it
to a canonical form with the symmetries of the Riemann tensor, as `ToCanonical` does:

```python
>>> Workbook().expr(r"simplify(R_{\rho\sigma\mu\nu} + R_{\rho\mu\nu\sigma} + R_{\rho\nu\sigma\mu})")
IndexExpression(0)

```

## EinsteinPy

EinsteinPy builds a tensor object for each quantity from a metric object.
RelativisticPy computes what you write, and caches it for the metric in force:

| EinsteinPy | RelativisticPy |
|---|---|
| `MetricTensor(components, symbols)` | `g_{\mu\nu} := [[\ldots]]` after `Coordinates := [\ldots]`, or a line element |
| `einsteinpy.symbolic.predefined.Schwarzschild()` | `g_{\mu\nu} := Schwarzschild(M)` |
| `ChristoffelSymbols.from_metric(m).tensor()` | `\Gamma^{a}_{bc}` |
| `RiemannCurvatureTensor.from_metric(m)` | `R^{a}_{bcd}` |
| `RicciTensor.from_metric(m)` | `R_{\mu\nu}` |
| `RicciScalar.from_metric(m).expr` | `R` |
| `EinsteinTensor.from_metric(m)` | `G_{\mu\nu}` |
| `WeylTensor.from_metric(m)` | `C_{abcd}` |
| `.change_config("uu")` | the indices where you want them: `R^{\mu\nu}` |
| `einsteinpy.geodesic` | `relativisticpy.numeric.geodesic`; see [Numerics](numerics.md) |

Components are taken by coordinate name, and equations between tensors say whether
they hold:

```python
>>> wb.expr(r"\Gamma^{r}_{\theta\theta}")
2*M - r
>>> wb.expr(r"R_{\mu\nu} = 0")
R_{μν} = 0: holds in all 10 components

```

**Next:** [Getting started](getting-started.md), or the
[language reference](reference.md) for everything at once. From RelativisticPy 1.x, see
[Coming from 1.x](upgrading.md).
