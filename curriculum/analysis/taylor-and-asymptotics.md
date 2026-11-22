# Taylor's theorem and asymptotics

Taylor expansions with an order term, Taylor coefficients, the remainder of Taylor's
formula, and Stirling's formula. Apostol (1967) is *Calculus*, volume 1, 2nd edition; Rudin
(1976) is *Principles of Mathematical Analysis*, 3rd edition.

## An expansion to a given order

**Level 2** · `analysis.taylor.big-o` · standard

A Taylor expansion written as an equation with an order term: it holds because both sides
agree up to \(x^4\).

```latex
e^x \sin x = x + x^2 + \frac{x^3}{3} + O(x^5)
```

```python
x = symbols("x")
assert series(exp(x) * sin(x), x, 0, 5) == x + x**2 + x**3 / 3 + O(x**5)
expected = True
```

## A Taylor coefficient

**Level 2** · `analysis.taylor.nth-derivative` · Abramowitz and Stegun (1964), 4.3.69

The coefficient of \(x^4\) in the Maclaurin series of \(\sec x\), from the notation
\(f^{(n)}\) for the \(n\)-th derivative.

```latex
f(x) := \sec x
\frac{f^{(4)}(0)}{4!}
```

```python
x = symbols("x")
expected = diff(sec(x), x, 4).subs(x, 0) / factorial(4)
assert expected == Rational(5, 24) == series(sec(x), x, 0, 6).removeO().coeff(x, 4)
```

## Taylor's formula with an integral remainder

**Level 3** · `analysis.taylor.integral-remainder` · Apostol (1967), §7.5

Taylor's formula for \(e^x\) to second order: a Taylor polynomial built from derivatives
at 0, and the error at \(x = 1\) written as an integral of the third derivative.

```latex
f(x) := e^x
T_2(x) := f(0) + f'(0) x + \frac{f''(0)}{2!} x^2
f(1) - T_2(1) = \int_0^1 \frac{f'''(t)}{2!} (1 - t)^2 \, dt
```

```python
t = symbols("t")
difference = E - (1 + 1 + Rational(1, 2))
remainder = integrate(exp(t) / factorial(2) * (1 - t)**2, (t, 0, 1))
assert simplify(difference - remainder) == 0
expected = True
```

## Stirling's formula

**Level 4** · `analysis.taylor.stirling` · Rudin (1976), §8.22

Asymptotic equivalence, \(\sim\): the ratio of the two sides tends to 1 as \(n \to \infty\).

```latex
n! \sim \sqrt{2\pi n} \left(\frac{n}{e}\right)^n
```

```python
n = symbols("n", integer=True, positive=True)
assert limit(factorial(n) / (sqrt(2 * pi * n) * (n / E)**n), n, oo) == 1
expected = True
```
