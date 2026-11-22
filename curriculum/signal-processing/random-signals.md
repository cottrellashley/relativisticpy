# Correlation and spectra

The autocorrelation of a sequence, written as Proakis and Manolakis write sequences, with
parentheses, \(x(n)\); the energy spectrum as the transform of the autocorrelation; and
the power spectrum of a random process, which no design of the language covers yet.

## The autocorrelation of an exponential sequence

**Level 3** · `sig.random.autocorrelation` · Proakis and Manolakis (2007), chapter 2

\(r_{xx}(l) = \sum_n x(n) x(n - l)\) for \(x(n) = a^n u(n)\) with \(0 < a < 1\): the steps cut
the sum to \(n \geq \max(0, l)\), and at every lag \(l\) it is
\(\frac{a^{|l|}}{1 - a^2}\), even in \(l\) as every autocorrelation is.

```latex
0 < a < 1
l \in \mathbb{Z}
x(n) := a^n u(n)
r_{xx}(l) := \sum_{n=-\infty}^{\infty} x(n) x(n - l)
r_{xx}(l)
```

```python
a = symbols("a", positive=True)
l = symbols("l", integer=True)
n = symbols("n", integer=True)
expected = a ** Abs(l) / (1 - a**2)
value = Rational(3, 5)
for lag in (0, 1, 3, -2):  # the sum from n = max(0, lag), at a few lags
    direct = summation(value**n * value ** (n - lag), (n, max(0, lag), oo))
    assert direct == expected.subs({a: value, l: lag})
signal = 0.6 ** np.arange(200)
for lag in (0, 2, 5):
    assert abs(np.dot(signal[lag:], signal[: len(signal) - lag]) - 0.6**lag / (1 - 0.36)) < 1e-12
```

## The energy spectrum, two ways

**Level 3** · `sig.random.energy-spectrum` · Proakis and Manolakis (2007), chapter 4

The transform of the autocorrelation is the squared magnitude of the transform:
\(\sum_l r_{xx}(l) e^{-j\omega l} = |X(\omega)|^2 = \frac{1}{1 - 2a\cos\omega + a^2}\), for the
same sequence and every real \(\omega\).

```latex
0 < a < 1
\omega \in \mathbb{R}
x(n) := a^n u(n)
r_{xx}(l) := \sum_{n=-\infty}^{\infty} x(n) x(n - l)
X(\omega) := \sum_{n=-\infty}^{\infty} x(n) e^{-j\omega n}
\sum_{l=-\infty}^{\infty} r_{xx}(l) e^{-j\omega l} = |X(\omega)|^2
```

```python
a = symbols("a", positive=True)
omega = symbols("omega", real=True)
X = 1 / (1 - a * exp(-I * omega))
squared = simplify(expand_complex(X * X.conjugate()).rewrite(cos))
assert simplify(squared - 1 / (1 - 2 * a * cos(omega) + a**2)) == 0
for value, frequency in ((0.6, 1.1), (0.3, 2.9)):
    lags = np.arange(-300, 301)
    transformed = np.sum(value ** np.abs(lags) / (1 - value**2) * np.exp(-1j * frequency * lags))
    assert abs(transformed - 1 / (1 - 2 * value * np.cos(frequency) + value**2)) < 1e-12
expected = True
```

## The power spectrum of an autoregressive process

**Level 5** · `sig.random.ar1-spectrum` · standard

A first-order autoregressive process driven by white Gaussian noise of variance
\(\sigma^2\), stationary for \(|a| < 1\). Its power spectrum is the transform of its
autocorrelation \(\mathbb{E}[x[n] x[n + k]] = \frac{\sigma^2 a^{|k|}}{1 - a^2}\), which is
\(\frac{\sigma^2}{1 - 2a\cos\omega + a^2}\). Random processes, their expectations and their
stationarity are beyond any design of the language yet.

```latex
|a| < 1
\omega \in \mathbb{R}
x[n] = a\, x[n-1] + w[n], \quad w[n] \sim \mathcal{N}(0, \sigma^2) \text{ independent}
P_{xx}(\omega) := \sum_{k=-\infty}^{\infty} \mathbb{E}\left[x[n]\, x[n+k]\right] e^{-j\omega k}
P_{xx}(\omega)
```

```python
from scipy.signal import lfilter

a, sigma, omega = symbols("a sigma omega", real=True)
expected = sigma**2 / (1 - 2 * a * cos(omega) + a**2)
# the stationary autocovariance solves gamma(0) = a^2 gamma(0) + sigma^2, gamma(k) = a gamma(k - 1)
for value, frequency in ((0.6, 1.1), (-0.4, 0.3)):
    lags = np.arange(-300, 301)
    spectrum = np.sum(value ** np.abs(lags) / (1 - value**2) * np.exp(-1j * frequency * lags))
    assert abs(spectrum - float(expected.subs({a: value, sigma: 1, omega: frequency}))) < 1e-12
noise = np.random.default_rng(0).normal(0, 1, 400_000)
process = lfilter([1], [1, -0.6], noise)[1000:]
assert abs(np.mean(process[1:] * process[:-1]) - 0.6 / (1 - 0.36)) < 0.03  # gamma(1)
```
