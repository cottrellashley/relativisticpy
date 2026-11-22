# Orthogonal polynomials and spherical harmonics

Legendre and Hermite polynomials, their orthogonality and generating function, and the
spherical harmonics built from them. DLMF is the NIST *Digital Library of Mathematical
Functions*, chapter 18; Arfken (2013) is *Mathematical Methods for Physicists*, 7th edition;
Griffiths (2005) is *Introduction to Quantum Mechanics*, 2nd edition.

## A Legendre polynomial

**Level 1** · `special.legendre.p3` · Arfken (2013), §15.1

\(P_3\), with its degree as a subscript, is the Legendre polynomial of degree 3.

```latex
P_3(x)
```

```python
x = symbols("x")
expected = legendre(3, x)
assert expand(expected - (5 * x**3 - 3 * x) / 2) == 0
```

## Rodrigues' formula

**Level 2** · `special.legendre.rodrigues` · DLMF 18.5.5 and Table 18.5.1

\(P_n(x) = \frac{1}{2^n n!} \frac{d^n}{dx^n} (x^2 - 1)^n\) at \(n = 4\), with the numbers
written in: a fourth derivative, computed.

```latex
\frac{1}{2^4 \cdot 4!} \frac{d^4}{dx^4} (x^2 - 1)^4
```

```python
x = symbols("x")
expected = legendre(4, x)
assert expand(diff((x**2 - 1) ** 4, x, 4) / (2**4 * factorial(4)) - expected) == 0
```

## The orthogonality of Hermite polynomials

**Level 2** · `special.hermite.orthogonality` · DLMF Table 18.3.1

The norm of \(H_3\) under the weight \(e^{-x^2}\), which is \(\sqrt{\pi}\, 2^n n!\) in the
physicists' convention.

```latex
\int_{-\infty}^{\infty} H_3(x)^2 e^{-x^2} \, dx
```

```python
x = symbols("x")
expected = integrate(hermite(3, x) ** 2 * exp(-(x**2)), (x, -oo, oo))
assert expected == sqrt(pi) * 2**3 * factorial(3)
```

## The generating function of Legendre polynomials

**Level 3** · `special.legendre.generating-function` · DLMF §18.12

An identity between a function and a power series whose coefficients are Legendre
polynomials of symbolic degree \(n\).

```latex
|t| < 1
\frac{1}{\sqrt{1 - 2xt + t^2}} = \sum_{n=0}^{\infty} P_n(x) \, t^n
```

```python
x, t = symbols("x t")
expansion = series(1 / sqrt(1 - 2 * x * t + t**2), t, 0, 10).removeO()
assert all(expand(expansion.coeff(t, k) - legendre(k, x)) == 0 for k in range(10))
expected = True
```

## A spherical harmonic

**Level 1** · `special.harmonics.y10` · Griffiths (2005), Table 4.3

\(Y_l^m\), with its degree below and its order above: the superscript is a label, not a
power.

```latex
Y_1^0(\theta, \phi)
```

```python
theta, phi = symbols("theta phi")
expected = Ynm(1, 0, theta, phi).expand(func=True)
assert expected == sqrt(3 / (4 * pi)) * cos(theta)
```

## Unsöld's theorem

**Level 2** · `special.harmonics.unsold` · standard

The sum of \(|Y_l^m|^2\) over the orders \(m\) of one degree is constant over the sphere,
\((2l + 1)/4\pi\): the addition theorem at coincident points, here for \(l = 2\).

```latex
\theta, \phi \in \mathbb{R}
\sum_{m=-2}^{2} \left| Y_2^m(\theta, \phi) \right|^2 = \frac{5}{4\pi}
```

```python
theta, phi = symbols("theta phi", real=True)
total = sum(Abs(Ynm(2, m, theta, phi).expand(func=True)) ** 2 for m in range(-2, 3))
assert simplify(total) == 5 / (4 * pi)
expected = True
```
