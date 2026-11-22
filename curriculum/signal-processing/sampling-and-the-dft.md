# Sampling, windows and the DFT

Aliasing, the sinc function as signal processing defines it, the coherent gain of a
window, and the discrete Fourier transform \(X[k]\) of a sequence \(x[n]\): a cosine's
spectrum, Parseval's relation, and circular convolution.

## The coherent gain of the Hann window

**Level 1** · `sig.windows.coherent-gain` · Harris (1978)

The Hann window \(w[n] = \sin^2(\pi n/N)\) of length 16, averaged: its coherent gain, the
factor by which it scales the amplitude of a tone, is \(\frac{1}{2}\).

```latex
\frac{1}{16} \sum_{n=0}^{15} \sin^2\left(\frac{\pi n}{16}\right)
```

```python
from scipy.signal import get_window

expected = nsimplify(sum(sin(pi * n / 16) ** 2 for n in range(16)) / 16)
assert simplify(expected - Rational(1, 2)) == 0
assert abs(np.mean(get_window("hann", 16)) - 0.5) < 1e-15  # SciPy's periodic Hann window
```

## The sinc function

**Level 1** · `sig.notation.sinc` · Bracewell (2000), chapter 4

\(\operatorname{sinc} x = \frac{\sin \pi x}{\pi x}\), as Bracewell, signal processing and NumPy
define it, with the \(\pi\) inside: \(\operatorname{sinc}\frac{1}{2} = \frac{2}{\pi}\), and the
zeros fall on the nonzero integers.

```latex
\operatorname{sinc}\left(\frac{1}{2}\right)
```

```python
x = symbols("x")
expected = (sin(pi * x) / (pi * x)).subs(x, Rational(1, 2))
assert expected == 2 / pi
assert abs(np.sinc(0.5) - float(expected)) < 1e-15
```

## Aliasing

**Level 2** · `sig.sampling.alias` · Oppenheim and Schafer (2010), chapter 4

Sampled ten times a second, a cosine of 7 hertz and one of 3 hertz give the same samples:
at every integer \(n\), \(\cos(2\pi \cdot 0.7 n) = \cos(2\pi \cdot 0.3 n)\), since
\(0.7 = 1 - 0.3\). The claim holds only because \(n\) is an integer.

```latex
n \in \mathbb{Z}
\cos\left(2\pi \cdot \frac{7}{10} n\right) = \cos\left(2\pi \cdot \frac{3}{10} n\right)
```

```python
n, m = symbols("n m", integer=True)
y = symbols("y", real=True)
assert expand(cos(2 * pi * m - y), trig=True) == cos(y)  # with m = n, y = 2 pi (3/10) n
difference = cos(2 * pi * Rational(7, 10) * n) - cos(2 * pi * Rational(3, 10) * n)
assert all(abs(N(difference.subs(n, m))) < 1e-12 for m in range(-20, 21))
assert abs(N(difference.subs(n, Rational(1, 2)))) > 0.5  # not between the samples
expected = True
```

## The DFT of a cosine

**Level 2** · `sig.dft.cosine` · Oppenheim and Schafer (2010), chapter 8

\(X[k] = \sum_{n=0}^{N-1} x[n] e^{-j2\pi kn/N}\) for one period of a cosine in eight samples:
all its energy is in the bins \(k = 1\) and \(k = 7\), each \(N/2 = 4\).

```latex
N := 8
x[n] := \cos\left(\frac{2\pi n}{N}\right)
X[k] := \sum_{n=0}^{N-1} x[n] e^{-j 2\pi k n / N}
X[1]
```

```python
samples = np.cos(2 * np.pi * np.arange(8) / 8)
spectrum = np.fft.fft(samples)
assert np.allclose(spectrum, [0, 4, 0, 0, 0, 0, 0, 4])
expected = 4
assert nsimplify(expand_complex(sum(cos(pi * m / 4) * exp(-I * pi * m / 4) for m in range(8)))) == 4
```

## Parseval's relation for the DFT

**Level 3** · `sig.dft.parseval` · Oppenheim and Schafer (2010), chapter 8

The energy of a sequence of length \(N\) is the energy of its DFT divided by \(N\):
\(\sum |x[n]|^2 = \frac{1}{N}\sum |X[k]|^2\), here 6 on both sides.

```latex
N := 4
x[n] := \delta[n] + 2\delta[n-1] - \delta[n-3]
X[k] := \sum_{n=0}^{N-1} x[n] e^{-j 2\pi k n / N}
\sum_{n=0}^{N-1} |x[n]|^2 = \frac{1}{N} \sum_{k=0}^{N-1} |X[k]|^2
```

```python
samples = [1, 2, 0, -1]
spectrum = [sum(samples[n] * exp(-I * 2 * pi * k * n / 4) for n in range(4)) for k in range(4)]
energy = sum(value**2 for value in samples)
assert energy == 6
assert simplify(sum(Abs(X) ** 2 for X in spectrum) / 4 - energy) == 0
assert abs(np.sum(np.abs(np.fft.fft(samples)) ** 2) / 4 - 6) < 1e-12
expected = True
```

## Circular convolution

**Level 4** · `sig.dft.circular-convolution` · Oppenheim and Schafer (2010), chapter 8

The \(N\)-point circular convolution, written \(x \circledast h\), wraps the indices
modulo \(N\). With \(N = 4\), the first value collects \(x[0]h[0] + x[3]h[1] = 5\), where
the linear convolution has only 1. It is the inverse DFT of the product of the DFTs.

```latex
N := 4
x[n] := \delta[n] + 2\delta[n-1] + 3\delta[n-2] + 4\delta[n-3]
h[n] := \delta[n] + \delta[n-1]
(x \circledast h)[0]
```

```python
x, h = [1, 2, 3, 4], [1, 1, 0, 0]
circular = [sum(x[m] * h[(n - m) % 4] for m in range(4)) for n in range(4)]
assert circular == [5, 3, 5, 7]
assert np.allclose(np.fft.ifft(np.fft.fft(x) * np.fft.fft(h)).real, circular)
assert np.convolve(x, [1, 1])[0] == 1  # the linear convolution does not wrap
expected = circular[0]
```
