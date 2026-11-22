# One-dimensional maps

Iterated maps \(x_{n+1} = f(x_n)\): the fixed points of the logistic map and their
stability, its cycle of period two, the Lyapunov exponent of the tent map as the average
of \(\ln|f'|\) along an orbit, and the Feigenbaum constant of the cascade of period
doublings.

## The stability of the logistic map's fixed point

**Level 2** · `dyn.maps.logistic-fixed-point` · Strogatz (2015), section 10.3

The fixed point \(x^* = 1 - \frac{1}{r}\) of \(f(x) = rx(1 - x)\) is stable when
\(|f'(x^*)| < 1\). The multiplier is \(f'(x^*) = 2 - r\), so the fixed point is stable for
\(1 < r < 3\).

```latex
f(x) := r x (1 - x)
x^* := 1 - \frac{1}{r}
f'(x^*)
```

```python
x, r = symbols("x r")
f = r * x * (1 - x)
fixed = 1 - 1 / r
assert simplify(f.subs(x, fixed) - fixed) == 0
expected = simplify(diff(f, x).subs(x, fixed))
assert expected == 2 - r
```

## The cycle of period two

**Level 3** · `dyn.maps.period-two` · Strogatz (2015), section 10.3

For \(r > 3\) the logistic map has a 2-cycle,
\(p, q = \frac{r + 1 \pm \sqrt{(r - 3)(r + 1)}}{2r}\). Its multiplier
\(f'(p)f'(q) = 4 + 2r - r^2\) is less than one in size for \(3 < r < 1 + \sqrt{6}\): the
product of two radicals must cancel.

```latex
r > 3
f(x) := r x (1 - x)
p := \frac{r + 1 + \sqrt{(r - 3)(r + 1)}}{2r}
q := \frac{r + 1 - \sqrt{(r - 3)(r + 1)}}{2r}
f'(p) f'(q)
```

```python
x = symbols("x")
r = symbols("r", positive=True)
f = r * x * (1 - x)
root = sqrt((r - 3) * (r + 1))
p, q = (r + 1 + root) / (2 * r), (r + 1 - root) / (2 * r)
assert simplify(f.subs(x, p) - q) == 0 and simplify(f.subs(x, q) - p) == 0  # a 2-cycle
expected = expand(simplify(diff(f, x).subs(x, p) * diff(f, x).subs(x, q)))
assert expected == 4 + 2 * r - r**2
assert solve(expected + 1, r) == [1 + sqrt(6)]  # where it loses stability
```

## The Lyapunov exponent of the tent map

**Level 4** · `dyn.maps.tent-lyapunov` · Strogatz (2015), section 10.5

The Lyapunov exponent, the average of \(\ln|f'(x_n)|\) along an orbit, for the tent map of
slope \(r\): \(|f'| = r\) wherever the derivative exists, so \(\lambda = \ln r\) for almost every
starting point, positive, and the map chaotic, for \(r > 1\).

```latex
0 < r \leq 2
f(x) := \begin{cases} r x & 0 \leq x \leq \frac{1}{2} \\ r (1 - x) & \frac{1}{2} < x \leq 1 \end{cases}
x_{n+1} := f(x_n)
\lim_{n \to \infty} \frac{1}{n} \sum_{i=0}^{n-1} \ln |f'(x_i)|
```

```python
r = symbols("r", positive=True)
expected = log(r)
for slope in (1.5, 2.0):
    point, total = 0.1234567, 0.0
    for _ in range(2000):  # the orbit stays off the corner, where f' is not defined
        total += np.log(slope)
        point = slope * point if point <= 0.5 else slope * (1 - point)
    assert abs(total / 2000 - float(expected.subs(r, slope))) < 1e-12
```

## The Feigenbaum constant

**Level 5** · `dyn.maps.feigenbaum` · Strogatz (2015), section 10.6

The logistic map doubles its period at \(r_1 = 3\), \(r_2 = 1 + \sqrt{6}\), \(r_3 = 3.5441\),
\(\ldots\), ever faster: the ratio of successive intervals tends to Feigenbaum's universal
\(\delta = 4.6692016\). A sequence defined in words, by where each cycle is born, is beyond
any design of the language yet.

```latex
f(x) := r x (1 - x)
r_n := \text{the value of } r \text{ at which the } 2^n\text{-cycle of } f \text{ is born}
\lim_{n \to \infty} \frac{r_n - r_{n-1}}{r_{n+1} - r_n}
```

```python
import mpmath

mpmath.mp.dps = 30


def returns(r, n):  # where the 2^n-cycle passes through 1/2: the superstable parameters
    x = mpmath.mpf(1) / 2
    for _ in range(2**n):
        x = r * x * (1 - x)
    return x - mpmath.mpf(1) / 2


R = [mpmath.mpf(2), 1 + mpmath.sqrt(5)]
for n in range(2, 11):
    R.append(mpmath.findroot(lambda r, n=n: returns(r, n), R[-1] + (R[-1] - R[-2]) / 4.669))
ratios = [(R[i] - R[i - 1]) / (R[i + 1] - R[i]) for i in range(1, len(R) - 1)]
expected = Float("4.669201609102990671853203820466", 30)  # Feigenbaum (1978)
assert abs(ratios[-1] - expected) < 1e-6  # the same ratios converge to the same constant
assert abs(1 + mpmath.sqrt(6) - 3.449489743) < 1e-9  # r_2, from the 2-cycle's multiplier
tolerance = 1e-6
```
