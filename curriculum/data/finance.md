# Interest and options

Compound interest, and the Black–Scholes–Merton price of a European option: its
formula, the partial differential equation it solves, and the hedging argument that
leads to that equation, as Hull writes them.

## Monthly compounding

**Level 1** · `data.finance.compounding` · Hull (2018), section 4.2

A hundred dollars invested for a year at 10 percent a year, compounded monthly, grow to
110.47.

```latex
100 \left(1 + \frac{0.1}{12}\right)^{12}
```

```python
expected = 100 * (1 + Rational(1, 120)) ** 12
assert abs(float(expected) - 110.47) < 0.005  # the book's table of compounding frequencies
```

## Continuous compounding

**Level 1** · `data.finance.continuous-compounding` · Hull (2018), section 4.2

Compounding \(m\) times a year, with \(m\) without bound: \(100 e^{0.1} = 110.52\).

```latex
\lim_{m \to \infty} 100 \left(1 + \frac{0.1}{m}\right)^{m}
```

```python
m = symbols("m", positive=True)
expected = 100 * exp(Rational(1, 10))
assert limit(100 * (1 + Rational(1, 10) / m) ** m, m, oo) == expected
assert abs(float(expected) - 110.52) < 0.005
```

## A European call by Black–Scholes

**Level 2** · `data.finance.black-scholes-call` · Hull (2018), chapter 15

Hull's example: a stock at 42 dollars, a strike of 40, six months to expiry, a risk-free
rate of 10 percent and a volatility of 20 percent. With \(N(x)\) the cumulative
distribution function of the standard normal distribution, as Hull writes it, the call
is worth 4.76 dollars.

```latex
S_0 := 42
K := 40
r := 0.1
\sigma := 0.2
T := 0.5
d_1 := \frac{\ln(S_0/K) + (r + \sigma^2/2) T}{\sigma \sqrt{T}}
d_2 := d_1 - \sigma \sqrt{T}
S_0 N(d_1) - K e^{-r T} N(d_2)
```

```python
from scipy.stats import norm

S0, K, r, sigma, T = 42, 40, 0.1, 0.2, 0.5
d1 = (np.log(S0 / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
d2 = d1 - sigma * np.sqrt(T)
assert abs(d1 - 0.7693) < 5e-5 and abs(d2 - 0.6278) < 5e-5  # the book's d_1 and d_2
expected = float(S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
assert abs(expected - 4.76) < 0.005
```

## The Black–Scholes–Merton equation

**Level 3** · `data.finance.black-scholes-pde` · Hull (2018), chapter 15

The price of a call, with \(N\) defined as the normal distribution function, satisfies
the Black–Scholes–Merton differential equation at every time \(t\) before expiry.

```latex
N(x) := \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{x} e^{-u^2/2} \, du
d_1 := \frac{\ln(S/K) + (r + \sigma^2/2)(T - t)}{\sigma \sqrt{T - t}}
d_2 := d_1 - \sigma \sqrt{T - t}
c := S N(d_1) - K e^{-r (T - t)} N(d_2)
\frac{\partial c}{\partial t} + r S \frac{\partial c}{\partial S} + \frac{1}{2} \sigma^2 S^2 \frac{\partial^2 c}{\partial S^2} = r c
```

```python
S, K, r, sigma, T, t = symbols("S K r sigma T t", positive=True)
N_ = lambda x: (1 + erf(x / sqrt(2))) / 2
d1 = (log(S / K) + (r + sigma**2 / 2) * (T - t)) / (sigma * sqrt(T - t))
d2 = d1 - sigma * sqrt(T - t)
c = S * N_(d1) - K * exp(-r * (T - t)) * N_(d2)
residual = diff(c, t) + r * S * diff(c, S) + sigma**2 * S**2 * diff(c, S, 2) / 2 - r * c
for point in (
    {S: 42, K: 40, r: Rational(1, 10), sigma: Rational(1, 5), T: 1, t: Rational(1, 2)},
    {S: Rational(37, 2), K: 21, r: Rational(3, 100), sigma: Rational(7, 20), T: 2, t: Rational(1, 3)},
):
    assert abs(N(residual.subs(point), 50)) < 1e-40
expected = True
```

## The Black–Scholes–Merton equation from a hedged portfolio

**Level 5** · `data.finance.black-scholes-derivation` · Hull (2018), chapter 15

The derivation itself: the stock follows geometric Brownian motion, the derivative's
price \(f\) follows from Itô's lemma, and a portfolio short one derivative and long
\(\partial f / \partial S\) shares has no \(dz\) term, so it must earn the risk-free
rate. The equation follows.

```latex
dS = \mu S \, dt + \sigma S \, dz
df = \left(\frac{\partial f}{\partial S} \mu S + \frac{\partial f}{\partial t} + \frac{1}{2} \frac{\partial^2 f}{\partial S^2} \sigma^2 S^2\right) dt + \frac{\partial f}{\partial S} \sigma S \, dz
\Pi := -f + \frac{\partial f}{\partial S} S
d\Pi = r \Pi \, dt
\frac{\partial f}{\partial t} + r S \frac{\partial f}{\partial S} + \frac{1}{2} \sigma^2 S^2 \frac{\partial^2 f}{\partial S^2} = r f
```

```python
S, t, mu, sigma, r, dt, dz = symbols("S t mu sigma r dt dz")
f = Function("f")(S, t)
dS = mu * S * dt + sigma * S * dz
df = (diff(f, S) * mu * S + diff(f, t) + diff(f, S, 2) * sigma**2 * S**2 / 2) * dt + diff(f, S) * sigma * S * dz
dPi = expand(-df + diff(f, S) * dS)  # the shares are held fixed over dt
assert dPi.coeff(dz) == 0  # no risk
equation = dPi.coeff(dt) - r * (-f + diff(f, S) * S)  # d Pi = r Pi dt
pde = diff(f, t) + r * S * diff(f, S) + sigma**2 * S**2 * diff(f, S, 2) / 2 - r * f
assert simplify(equation + pde) == 0
expected = True
```
