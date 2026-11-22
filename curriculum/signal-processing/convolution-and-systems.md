# Convolution and discrete-time systems

Sequences written as signal processing writes them, \(x[n]\) with square brackets, built
from the unit impulse \(\delta[n]\) and the unit step \(u[n]\); the convolution sum; the
stability of a system from its impulse response; convolution of continuous-time
signals; and what it means for a system to be linear.

## The convolution sum

**Level 2** · `sig.convolution.sequences` · Oppenheim and Schafer (2010), chapter 2

\(y[n] = \sum_k x[k] h[n - k]\) for two short sequences given by impulses: the output of
a two-point summer. \(y = (1, 3, 5, 3)\), and \(y[2] = 5\).

```latex
x[n] := \delta[n] + 2\delta[n-1] + 3\delta[n-2]
h[n] := \delta[n] + \delta[n-1]
y[n] := \sum_{k=-\infty}^{\infty} x[k] h[n-k]
y[2]
```

```python
y = np.convolve([1, 2, 3], [1, 1])
assert list(y) == [1, 3, 5, 3]
expected = int(y[2])
assert expected == sum([1, 2, 3][k] * [1, 1][2 - k] for k in range(3) if 0 <= 2 - k <= 1)
```

## Stability from the impulse response

**Level 2** · `sig.systems.bibo` · Oppenheim and Schafer (2010), chapter 2

A system is stable, bounded inputs giving bounded outputs, when its impulse response is
absolutely summable. For \(h[n] = (-0.8)^n u[n]\) the sum of \(|h[n]|\) is
\(1/(1 - 0.8) = 5\).

```latex
h[n] := (-0.8)^n u[n]
\sum_{n=-\infty}^{\infty} |h[n]|
```

```python
n = symbols("n", integer=True, nonnegative=True)
expected = summation(Rational(4, 5) ** n, (n, 0, oo))  # u[n] removes the negative n
assert expected == 5
assert abs(np.sum(np.abs((-0.8) ** np.arange(400))) - 5) < 1e-12
```

## The convolution of two exponentials

**Level 3** · `sig.convolution.exponentials` · Oppenheim, Willsky and Nawab (1997), chapter 2

The convolution of \(a^n u[n]\) with \(b^n u[n]\), for \(a \neq b\): the steps cut the sum
to \(0 \leq k \leq n\), and the geometric sum gives
\(\frac{a^{n+1} - b^{n+1}}{a - b}\) for \(n \geq 0\).

```latex
n \in \mathbb{Z}
n \geq 0
a \neq b
x[n] := a^n u[n]
h[n] := b^n u[n]
\sum_{k=-\infty}^{\infty} x[k] h[n-k]
```

```python
a, b = symbols("a b")
n = symbols("n", integer=True, nonnegative=True)
expected = (a ** (n + 1) - b ** (n + 1)) / (a - b)
for m in range(7):  # the sum over 0 <= k <= m, against the closed form
    assert cancel(sum(a**k * b ** (m - k) for k in range(m + 1)) - expected.subs(n, m)) == 0
for A, B in ((0.5, -0.3), (2.0, 0.7)):
    full = np.convolve(A ** np.arange(8), B ** np.arange(8))[:8]
    assert np.allclose(full, [(A ** (m + 1) - B ** (m + 1)) / (A - B) for m in range(8)])
```

## Convolution of continuous-time signals

**Level 4** · `sig.convolution.continuous` · Oppenheim, Willsky and Nawab (1997), chapter 2

The asterisk between two signals is their convolution, \(\int x(\tau) h(t - \tau)\,d\tau\),
not their product: \(e^{-t}u(t) * e^{-2t}u(t) = e^{-t} - e^{-2t}\) for \(t > 0\).

```latex
t > 0
\left(e^{-t} u(t)\right) * \left(e^{-2t} u(t)\right)
```

```python
t, tau = symbols("t tau", positive=True)
expected = integrate(exp(-tau) * exp(-2 * (t - tau)), (tau, 0, t))  # the steps cut it to [0, t]
assert simplify(expected - (exp(-t) - exp(-2 * t))) == 0
from scipy.integrate import quad

for at in (0.5, 2.0):  # the convolution integral over the whole line, numerically
    integrand = lambda s, at=at: np.exp(-s) * (s > 0) * np.exp(-2 * (at - s)) * (at > s)
    value = quad(integrand, -1, 4, points=[0, at])
    assert abs(value[0] - (np.exp(-at) - np.exp(-2 * at))) < 1e-8
```

## Is a squarer linear?

**Level 5** · `sig.systems.linearity` · Oppenheim and Schafer (2010), chapter 2

A system is linear when superposition holds for every pair of inputs and every pair of
constants. For the squarer \(T\{x[n]\} = x[n]^2\) it does not: one counterexample, two
times an impulse, is enough. A claim quantified over all sequences, with a system as an
operator on them, is beyond any design of the language yet.

```latex
T\{x[n]\} := x[n]^2
\forall\, x_1, x_2, a, b: \quad T\{a x_1[n] + b x_2[n]\} = a T\{x_1[n]\} + b T\{x_2[n]\}
```

```python
square = lambda sequence: [value**2 for value in sequence]
impulse, nothing = [1, 0, 0], [0, 0, 0]
left = square([2 * p + 0 * q for p, q in zip(impulse, nothing, strict=True)])
right = [2 * p + 0 * q for p, q in zip(square(impulse), square(nothing), strict=True)]
assert left == [4, 0, 0] and right == [2, 0, 0]  # superposition fails
expected = False
```
