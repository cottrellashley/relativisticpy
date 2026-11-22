# Hypergeometric functions

Gauss's hypergeometric function \({}_2F_1(a, b; c; z)\), with its parameters separated by
semicolons, and the Pochhammer symbol \((a)_n\) of its series. Abramowitz and Stegun (1964)
is the *Handbook of Mathematical Functions*, chapter 15; DLMF is the NIST *Digital Library
of Mathematical Functions*.

## The Pochhammer symbol

**Level 4** · `special.hypergeometric.pochhammer` · DLMF §5.2(iii)

The rising factorial \((a)_n = a(a+1)\cdots(a+n-1)\), written as a bracket with a subscript,
the building block of hypergeometric series.

```latex
\left(\tfrac{1}{2}\right)_3
```

```python
expected = rf(Rational(1, 2), 3)
assert expected == Rational(1, 2) * Rational(3, 2) * Rational(5, 2) == Rational(15, 8)
```

## A logarithm as a hypergeometric function

**Level 4** · `special.hypergeometric.logarithm` · Abramowitz and Stegun (1964), 15.1.3

An elementary case of \({}_2F_1\), as an identity for every \(z\).

```latex
{}_2F_1(1, 1; 2; z) = -\frac{\ln(1 - z)}{z}
```

```python
z = symbols("z")
assert simplify(hyperexpand(hyper([1, 1], [2], z)) + log(1 - z) / z) == 0
expected = True
```

## Gauss's summation theorem

**Level 4** · `special.hypergeometric.gauss` · Abramowitz and Stegun (1964), 15.1.20

\({}_2F_1\) at \(z = 1\), which Gauss's theorem gives as a ratio of Gamma functions,
\(\Gamma(c)\Gamma(c - a - b) / \Gamma(c - a)\Gamma(c - b)\).

```latex
{}_2F_1\left(\tfrac{1}{2}, \tfrac{1}{2}; 2; 1\right)
```

```python
a, b, c = Rational(1, 2), Rational(1, 2), 2
expected = gamma(c) * gamma(c - a - b) / (gamma(c - a) * gamma(c - b))
assert expected == 4 / pi == hyperexpand(hyper([a, b], [c], 1))
```

## The inverse sine as a hypergeometric function

**Level 4** · `special.hypergeometric.arcsin` · Abramowitz and Stegun (1964), 15.1.6

\({}_2F_1\) of an argument \(z^2\), multiplied by \(z\), is \(\arcsin z\).

```latex
\arcsin z = z \, {}_2F_1\left(\tfrac{1}{2}, \tfrac{1}{2}; \tfrac{3}{2}; z^2\right)
```

```python
z = symbols("z")
assert (
    simplify(
        hyperexpand(z * hyper([Rational(1, 2), Rational(1, 2)], [Rational(3, 2)], z**2)) - asin(z)
    )
    == 0
)
expected = True
```
