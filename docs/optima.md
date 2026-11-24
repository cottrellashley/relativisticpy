# Optima

> **For:** anyone who maximises or minimises. **You will learn:** how to write an
> optimum and its constraints, how it is found, and when it is global.
> **Before this:** [The language](language.md).

```python
>>> from relativisticpy import Workbook

```

## Writing an optimum

The greatest and the least value of what follows, and where it is, are written as books
write them, with the variables beneath the operator: `\max_{x}`, `\min_{x \in S}`,
`\sup_{0 \le x < 1}` and `\inf_{x > 0}`, and `\arg\max_{x}`, `\operatorname*{arg\,min}_{w}`
or `\underset{x}{\min}`, which give the point, not the value. In plain text, `min_{x}` and
`argmax_{x >= 0}`. A set or a condition beneath the operator says what the variables range
over; constraints follow the words "subject to" or "s.t.", on the same line or on the
lines after it, and names before a comma share the relation after them, as in
`x, y \geq 0`. An optimum is of the whole sum it begins, up to a relation or "subject to",
as `\min_x x^2 - 4x + 7` is, and of its factor alone where it is a factor, as in
`2 \max_x f(x)`. `\max(a, b)` is still the greater of two values:

```python
>>> optima = Workbook()
>>> optima.exe(r"""
...     \min_{x \in \mathbb{R}} \left(x^2 - 4x + 7\right)
...     \arg\max_{x \geq 0} \; x e^{-x}
...     \max_{0 \leq x \leq 3} \left(x^3 - 3x\right)
...     \sup_{x \geq 0} \frac{x}{1 + x}
...     \min_{x, y} \; x + y \quad \text{subject to} \quad x^2 + y^2 = 2
...     \max_{x, y} \; 3x + 5y \quad \text{subject to} \quad x \leq 4, \quad 2y \leq 12, \quad 3x + 2y \leq 18, \quad x, y \geq 0
... """)
[3, 1, 18, 1, -2, 36]

```

## How an optimum is found

An optimum over variables is found as the program runs, by the
[optimizers](reference.md#optimizers) the integrations declare, in turn. SymPy's come
first, and find closed forms: from the stationary points and the ends of the domain of a
function of one variable, the Lagrange and KKT conditions of several, the simplex method of
a linear program, or by trying each point of a finite domain, as of `x \in \{0, 1\}^3`.
SciPy's come next, where the problem has no parameters, and find numbers; a problem written
with decimals, as data is, goes to them first. A note says which optimizer found the
optimum, how, and whether it is proven global:

```python
>>> [step.notes for step in optima.each(r"\min_{x, y} \; x + y \quad \text{subject to} \quad x^2 + y^2 = 2")]
[['the global minimum, found exactly by SymPy, from the Lagrange conditions; the constraints bound the region (kkt)']]

```

An optimum is said to be global only where that is proven: of a function of one variable,
where every stationary point in its domain is found, those of a periodic function too, and
every point where it is not defined, toward which its values are approached and never
reached; of several, of a convex problem, or of a polynomial one over a bounded region.
A search by SciPy gives an optimum only where it converged and nothing lower is found far
away, and it is not said to be global unless the problem is convex. Where more than one
point reaches the optimum, the note says which, an interval or a curve of them too, and
`\arg\min` gives the least:

```python
>>> [step.notes for step in optima.each(r"\arg\min_x \left(|x - 1| + |x + 1|\right)")]
[['the global minimum, found exactly by SymPy, from its stationary points, the points where it has no derivative, and the ends of its domain (extrema); every point of [−1, 1] reaches it, and −1 is given']]

```

## Variables and parameters

The variables are real, or whole numbers, as `n \in \mathbb{N}` says. A variable is a
vector where its set is `\mathbb{R}^n` or `\{0, 1\}^n`, or a matrix it is multiplied by
gives its length, as `A \mathbf{x}` does, and its components may be written `x_1, x_2`; a
tensor, where it is written with an index of a space. What the program assumes of its
parameters is used, as `a > 0`, and so is what it assumes of the variables themselves, as
`x \in [0, 1]` before `\min_x`, which a note says; where the answer depends on signs it does
not say, it is found as books find it, with the parameters real and the fewest of them
positive, which a note says (decision 1 of
[the notation decisions](https://github.com/cottrellashley/relativisticpy/blob/main/design/notation.md)).
A maximum that no point reaches, or that is infinite, is refused, and `\sup` gives the
bound:

```python
>>> optima.exe(r"""
...     A := \begin{pmatrix} 1 & 1 & 0 \\ 0 & 1 & 1 \end{pmatrix}
...     \mathbf{b} := \begin{pmatrix} 1 \\ 2 \end{pmatrix}
...     \arg\min_{\mathbf{x}} \|\mathbf{x}\|_2 \quad \text{subject to} \quad A\mathbf{x} = \mathbf{b}
...     \max_{x \in \{0, 1\}^3} \; 60x_1 + 100x_2 + 120x_3 \quad \text{subject to} \quad 10x_1 + 20x_2 + 30x_3 \leq 50
...     L(v) = -m c^2 \sqrt{1 - \frac{v^2}{c^2}}
...     H(p) = \max_v \left(p v - L(v)\right)
...     H(p)
... """)
[[[0], [1], [1]], 220, c*sqrt(c**2*m**2 + p**2)]
>>> optima.expr(r"\max_x x^2")
Traceback (most recent call last):
  ...
relativisticpy.errors.EvaluationError: \max_x has no maximum: what it is of grows without bound, to ∞
 --> <input>:1:1
  |
1 | \max_x x^2
  | ^^^^^^^^^^
hint: the bound it never reaches is \sup, as in \sup_{x} f(x)

```

## Over data, and matrices

Over an index of data, an optimum binds the index as a sum does, and is the greatest or
the least value along it, and `\arg\max` its label: see
[Beyond spacetime](spaces.md#optima-over-an-index). `A \succ 0` and `A \succeq 0` say that
a symmetric matrix is positive definite or semidefinite, and between vectors, as in the
constraint `w \succeq 0`, hold of each component.

**Next:** [Number theory and sequences](discrete.md), or [Beyond spacetime](spaces.md),
for optima over data.
