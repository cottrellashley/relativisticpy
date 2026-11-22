# Inequalities and epsilon–delta statements

Inequalities that hold for every value of their variables, a greatest lower bound, and the
quantified statements of an epsilon–delta proof: what an analysis course proves rather than
computes. Stewart (2015) is *Calculus: Early Transcendentals*, 8th edition; Rudin (1976) is
*Principles of Mathematical Analysis*, 3rd edition.

## The exponential lies above its tangent

**Level 3** · `analysis.inequalities.exp-tangent` · standard

An inequality between functions that holds for every real \(x\). Deciding it means proving
it, here from the minimum of \(e^x - 1 - x\), which is 0.

```latex
e^x \geq 1 + x
```

```python
x = symbols("x", real=True)
assert minimum(exp(x) - 1 - x, x, S.Reals) == 0
expected = True
```

## The arithmetic and geometric means

**Level 3** · `analysis.inequalities.am-gm` · standard

An inequality under assumptions stated as a book states them, \(a, b > 0\): it holds
because the difference of the two sides is \((\sqrt{a} - \sqrt{b})^2 / 2\).

```latex
a, b > 0
\frac{a + b}{2} \geq \sqrt{ab}
```

```python
a, b = symbols("a b", positive=True)
assert simplify((a + b) / 2 - sqrt(a * b) - (sqrt(a) - sqrt(b))**2 / 2) == 0
expected = True
```

## An infimum that is not attained

**Level 4** · `analysis.inequalities.infimum` · Rudin (1976), Example 1.9

The greatest lower bound of a set written in set-builder notation: every \(1/n\) is
positive, but they come arbitrarily close to 0.

```latex
\inf \left\{ \frac{1}{n} : n = 1, 2, 3, \ldots \right\}
```

```python
n = symbols("n", integer=True, positive=True)
expected = limit(1 / n, n, oo)  # the terms decrease towards their infimum
assert expected == 0
```

## An epsilon–delta proof

**Level 5** · `analysis.inequalities.epsilon-delta` · Stewart (2015), §2.4, Example 2

The definition of \(\lim_{x \to 3} (4x - 5) = 7\), with its quantifiers: for every
\(\varepsilon\) there is a \(\delta\), here \(\varepsilon / 4\).

```latex
\forall \varepsilon > 0 \ \exists \delta > 0 : 0 < |x - 3| < \delta \implies |(4x - 5) - 7| < \varepsilon
```

```python
x = symbols("x", real=True)
# |(4x - 5) - 7| = 4|x - 3|, so delta = epsilon/4 serves every epsilon
assert simplify(Abs((4 * x - 5) - 7) - 4 * Abs(x - 3)) == 0
expected = True
```
