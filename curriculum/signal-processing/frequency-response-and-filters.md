# Frequency responses and filters

The frequency response of a system, written as engineers write it: \(j\) for the
imaginary unit, \(H(e^{j\omega})\) for the response of a discrete-time system, gains in
decibels. Nulls of a moving average, the bilinear transform, and the group delay of a
linear-phase filter.

## The imaginary unit of engineers

**Level 1** · `sig.notation.imaginary-unit` · standard

Engineers write \(j\) for \(\sqrt{-1}\), since \(i\) is a current. The gain of a first-order
low-pass filter at its corner frequency is \(\left|\frac{1}{1 + j}\right| = \frac{1}{\sqrt{2}}\).

```latex
\left| \frac{1}{1 + j} \right|
```

```python
expected = Abs(1 / (1 + I))
assert expected == sqrt(2) / 2
assert abs(abs(1 / (1 + 1j)) - 2**-0.5) < 1e-15
```

## The half-power point in decibels

**Level 1** · `sig.filters.half-power` · standard

Half the power is a gain of \(\frac{1}{\sqrt{2}}\), which is \(20 \log_{10} \frac{1}{\sqrt{2}} =
-3.01\) decibels: the "3 dB" of a cut-off frequency.

```latex
20 \log_{10} \frac{1}{\sqrt{2}}
```

```python
expected = 20 * log(1 / sqrt(2), 10)
assert simplify(expected + 10 * log(2) / log(10)) == 0
assert abs(float(expected) - 20 * np.log10(2**-0.5)) < 1e-12
assert abs(float(expected) + 3.0103) < 5e-5
```

## A null of the moving average

**Level 2** · `sig.frequency.moving-average-null` · Oppenheim and Schafer (2010), chapter 2

The frequency response of a three-point moving average,
\(\frac{1}{3}\sum_{n=0}^{2} e^{-j\omega n}\), vanishes at \(\omega = \frac{2\pi}{3}\): the three
unit vectors cancel.

```latex
\omega := \frac{2\pi}{3}
\left| \frac{1}{3} \sum_{n=0}^{2} e^{-j \omega n} \right|
```

```python
from scipy.signal import freqz

omega = 2 * pi / 3
expected = Abs(expand_complex(sum(exp(-I * omega * n) for n in range(3)))) / 3
assert expected == 0
_, response = freqz([1 / 3, 1 / 3, 1 / 3], [1], worN=[2 * np.pi / 3])
assert abs(response[0]) < 1e-15
```

## The bilinear transform

**Level 2** · `sig.filters.bilinear` · Oppenheim and Schafer (2010), chapter 7

A digital filter from the analog prototype \(H_c(s) = \frac{1}{s + 1}\), by substituting
\(s = \frac{2}{T}\frac{1 - z^{-1}}{1 + z^{-1}}\) with \(T = 2\): \(H(z) = \frac{1 + z^{-1}}{2}\).

```latex
H_c(s) := \frac{1}{s + 1}
T := 2
H_c\left(\frac{2}{T} \frac{1 - z^{-1}}{1 + z^{-1}}\right)
```

```python
from scipy.signal import bilinear

z = symbols("z")
expected = (1 + 1 / z) / 2
s = (2 / Integer(2)) * (1 - 1 / z) / (1 + 1 / z)
assert simplify(1 / (s + 1) - expected) == 0
b, a = bilinear([1], [1, 1], fs=0.5)  # fs = 1/T
assert np.allclose(b, [0.5, 0.5]) and np.allclose(a, [1, 0])
```

## The frequency response of an exponential sequence

**Level 3** · `sig.frequency.dtft` · Oppenheim and Schafer (2010), chapter 2

\(H(e^{j\omega}) = \sum_n h[n] e^{-j\omega n}\), with the response named, as the book names
it, by \(e^{j\omega}\): for \(h[n] = a^n u[n]\) with \(|a| < 1\) the series converges to
\(\frac{1}{1 - a e^{-j\omega}}\).

```latex
|a| < 1
h[n] := a^n u[n]
H(e^{j\omega}) := \sum_{n=-\infty}^{\infty} h[n] e^{-j\omega n}
H(e^{j\omega})
```

```python
from scipy.signal import freqz

a, omega = symbols("a omega")
expected = 1 / (1 - a * exp(-I * omega))
_, response = freqz([1], [1, -0.6], worN=[1.1])
assert (
    abs(response[0] - complex(expected.subs({a: Rational(3, 5), omega: Rational(11, 10)}))) < 1e-12
)
partial = sum(0.6**n * np.exp(-1.1j * n) for n in range(200))
assert abs(partial - response[0]) < 1e-12
```

## The group delay of a linear-phase filter

**Level 4** · `sig.frequency.group-delay` · Oppenheim and Schafer (2010), chapter 5

The group delay \(-\frac{d}{d\omega}\arg H(e^{j\omega})\) of the symmetric filter
\(1 + 2e^{-j\omega} + e^{-2j\omega} = e^{-j\omega}(2 + 2\cos\omega)\): a delay of one sample at
every frequency between \(-\pi\) and \(\pi\).

```latex
-\pi < \omega < \pi
H(e^{j\omega}) := 1 + 2e^{-j\omega} + e^{-2j\omega}
-\frac{d}{d\omega} \arg H(e^{j\omega})
```

```python
from scipy.signal import group_delay

omega = symbols("omega", real=True)
H = 1 + 2 * exp(-I * omega) + exp(-2 * I * omega)
assert simplify((H - exp(-I * omega) * (2 + 2 * cos(omega))).rewrite(exp)) == 0  # arg H = -omega
expected = 1
_, delay = group_delay(([1, 2, 1], [1]), w=[0.2, 1.0, 2.5])
assert np.allclose(delay, expected)
```
