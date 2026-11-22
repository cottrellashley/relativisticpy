# Bessel functions

Bessel functions of the first kind, their differential equation, their elementary cases and
their zeros. DLMF is the NIST *Digital Library of Mathematical Functions*, chapter 10;
Abramowitz and Stegun (1964) is the *Handbook of Mathematical Functions*, chapter 9.

## A value of J_0

**Level 1** · `special.bessel.value` · Abramowitz and Stegun (1964), Table 9.1

\(J_0\), with its order as a subscript, is the Bessel function of the first kind.

```latex
J_0(1)
```

```python
from scipy.special import j0

expected = besselj(0, 1)
assert abs(float(expected) - 0.7651976866) < 1e-10  # the table's value
assert abs(float(expected) - j0(1.0)) < 1e-15
```

## Bessel's equation

**Level 2** · `special.bessel.equation` · DLMF 10.2.1

\(J_1\) solves Bessel's equation of order 1, written in Leibniz's notation as DLMF writes
it.

```latex
w := J_1(z)
z^2 \frac{d^2 w}{dz^2} + z \frac{dw}{dz} + (z^2 - 1) w = 0
```

```python
z = symbols("z")
w = besselj(1, z)
assert simplify(z**2 * diff(w, z, 2) + z * diff(w, z) + (z**2 - 1) * w) == 0
expected = True
```

## A Bessel function of half-integer order

**Level 2** · `special.bessel.half-order` · DLMF 10.16.1

\(J_{1/2}\) is elementary. Its order is a fraction in the subscript.

```latex
J_{1/2}(x) = \sqrt{\frac{2}{\pi x}} \sin x
```

```python
x = symbols("x", positive=True)
assert simplify(expand_func(besselj(Rational(1, 2), x)) - sqrt(2 / (pi * x)) * sin(x)) == 0
expected = True
```

## The zeros of J_0

**Level 4** · `special.bessel.zeros` · DLMF §10.21; Abramowitz and Stegun (1964), Table 9.5

\(j_{0,k}\), DLMF's notation for the \(k\)-th positive zero of \(J_0\). The ratio of the
first two is the ratio of the frequencies of the first two radially symmetric modes of a
drum.

```latex
\frac{j_{0,2}}{j_{0,1}}
```

```python
from scipy.special import jn_zeros

first, second = jn_zeros(0, 2)
assert abs(first - 2.4048255577) < 1e-9 and abs(second - 5.5200781103) < 1e-9  # the table
expected = float(second / first)
```
