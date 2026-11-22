# Interest, annuities and bonds

The time value of money: present values, rates under different compounding and the
limit that gives continuous compounding, perpetuities and annuities, written in
actuarial notation too, and bonds, their prices, yields and durations, as Hull and
Brealey, Myers and Allen write them.

## A present value

**Level 1** · `econ.interest.present-value` · Brealey, Myers and Allen (2020), chapter 2

A thousand dollars in five years, discounted at 6 percent a year: \(\frac{1000}{1.06^5} =
747.26\) dollars today.

```latex
\frac{1000}{(1 + 0.06)^5}
```

```python
expected = 1000 / Rational(106, 100) ** 5
assert abs(float(expected) - 747.26) < 0.005
```

## An equivalent continuous rate

**Level 1** · `econ.interest.continuous-equivalent` · Hull (2018), section 4.2

Ten percent a year compounded twice a year is the same as \(R_c = m\ln(1 + R_m/m) =
2\ln 1.05 = 9.758\) percent compounded continuously.

```latex
2 \ln\left(1 + \frac{0.10}{2}\right)
```

```python
expected = 2 * log(Rational(105, 100))
assert abs(float(expected) - 0.09758) < 5e-6  # the book's 9.758 percent
assert abs(np.exp(float(expected)) - 1.05**2) < 1e-15
```

## The time to double

**Level 1** · `econ.interest.doubling-time` · standard

At 8 percent a year, money doubles in \(\frac{\ln 2}{\ln 1.08} = 9.006\) years, close to the
rule of 72's \(72/8 = 9\).

```latex
\frac{\ln 2}{\ln 1.08}
```

```python
expected = log(2) / log(Rational(108, 100))
assert abs(float(expected) - 9.006) < 5e-4
assert abs(Rational(108, 100) ** expected - 2) < 1e-12
```

## A perpetuity

**Level 2** · `econ.interest.perpetuity` · Brealey, Myers and Allen (2020), chapter 2

A payment \(C\) at the end of every year for ever, discounted at \(r > 0\): the geometric
series sums to \(\frac{C}{r}\).

```latex
r > 0
\sum_{t=1}^{\infty} \frac{C}{(1 + r)^t}
```

```python
C = symbols("C")
r = symbols("r", positive=True)
t = symbols("t", integer=True, positive=True)
expected = C / r
assert simplify(summation(C / (1 + r) ** t, (t, 1, oo)) - expected) == 0
assert abs(sum(100 / 1.05**k for k in range(1, 2000)) - 100 / 0.05) < 1e-9
```

## Continuous compounding as a limit

**Level 2** · `econ.interest.continuous-limit` · Hull (2018), section 4.2

Compounding \(m\) times a year for \(T\) years at the rate \(R\), with \(m\) without bound: the
growth factor \((1 + R/m)^{mT}\) tends to \(e^{RT}\), for every rate and term, symbolically.

```latex
R > 0
T > 0
\lim_{m \to \infty} \left(1 + \frac{R}{m}\right)^{m T}
```

```python
R, T, m = symbols("R T m", positive=True)
expected = exp(R * T)
assert limit((1 + R / m) ** (m * T), m, oo) == expected
assert abs((1 + 0.1 / 10**6) ** (10**6 * 2.5) - np.exp(0.25)) < 1e-6
```

## The price of a bond

**Level 2** · `econ.bonds.price` · Hull (2018), chapter 4

A two-year bond with a principal of 100 dollars and coupons of 6 percent a year paid every
six months, discounted at the continuously compounded yield of 6.76 percent: 98.39
dollars.

```latex
\sum_{k=1}^{4} 3 e^{-0.0676 \cdot 0.5 k} + 100 e^{-0.0676 \cdot 2}
```

```python
y = Rational(676, 10000)
expected = sum(3 * exp(-y * Rational(k, 2)) for k in range(1, 5)) + 100 * exp(-2 * y)
assert abs(float(expected) - 98.39) < 0.005  # the book's price
```

## The internal rate of return

**Level 3** · `econ.interest.irr` · Brealey, Myers and Allen (2020), chapter 5

The rate that makes the net present value of \(-100, 60, 60\) zero: in \(v = \frac{1}{1 + r}\) a
quadratic, whose positive root is \(r = \frac{\sqrt{69} - 7}{10} = 13.07\) percent. The other
root, below \(-1\), is not a rate, and the stated \(r > 0\) excludes it.

```latex
r > 0
solve(-100 + \frac{60}{1 + r} + \frac{60}{(1 + r)^2} = 0, r)
```

```python
from scipy.optimize import brentq

r = symbols("r", positive=True)
expected = solve(-100 + 60 / (1 + r) + 60 / (1 + r) ** 2, r)
assert expected == [(sqrt(69) - 7) / 10]
assert (
    abs(brentq(lambda x: -100 + 60 / (1 + x) + 60 / (1 + x) ** 2, 0, 1) - float(expected[0]))
    < 1e-12
)
```

## Duration

**Level 3** · `econ.bonds.duration` · Hull (2018), chapter 4

A three-year bond with 10 percent coupons paid every six months, at a continuously
compounded yield of 12 percent: its price \(B = \sum_i c_i e^{-yt_i} = 94.213\), and its
duration, the mean time of its payments weighted by their present values,
\(D = \frac{1}{B}\sum_i t_i c_i e^{-yt_i} = 2.653\) years. The cash flows and the times are
lists, indexed from 1.

```latex
c := (5, 5, 5, 5, 5, 105)
t := (0.5, 1, 1.5, 2, 2.5, 3)
y := 0.12
B := \sum_{i=1}^{6} c_i e^{-y t_i}
\frac{1}{B} \sum_{i=1}^{6} t_i c_i e^{-y t_i}
```

```python
times = [Rational(i, 2) for i in range(1, 7)]
cash = [5, 5, 5, 5, 5, 105]
price = sum(c * exp(-Rational(12, 100) * t) for c, t in zip(cash, times))
expected = sum(t * c * exp(-Rational(12, 100) * t) for c, t in zip(cash, times)) / price
assert abs(float(price) - 94.213) < 5e-4 and abs(float(expected) - 2.653) < 5e-4  # the book's
h = 1e-7  # duration is minus the relative sensitivity of the price to the yield
bond = lambda yield_: sum(c * np.exp(-yield_ * float(t)) for c, t in zip(cash, times))
assert abs(-(bond(0.12 + h) - bond(0.12 - h)) / (2 * h) / bond(0.12) - float(expected)) < 1e-6
```

## An annuity in actuarial notation

**Level 4** · `econ.interest.annuity-notation` · Kellison (2009), chapter 3

\(a_{\overline{n}|i}\), the actuaries' "a angle n at i": the present value of one paid at the
end of each of \(n\) periods at the rate \(i\), \(\frac{1 - (1 + i)^{-n}}{i}\). For ten years at 5
percent it is 7.7217.

```latex
a_{\overline{10}|0.05}
```

```python
n, i = 10, Rational(5, 100)
expected = (1 - (1 + i) ** -n) / i
assert expected == sum((1 + i) ** -k for k in range(1, n + 1))
assert abs(float(expected) - 7.7217) < 5e-5
```
