# Residues and contour integrals

Residues from Laurent series, contour integrals by the residue theorem, and real integrals
done by contours, with a principal value. Ahlfors (1979) is *Complex Analysis*, 3rd edition.

## A residue at an essential singularity

**Level 2** · `complex.residues.essential-singularity` · Ahlfors (1979), ch. 4, §5.1

The residue of \(z^2 e^{1/z}\) at 0 is the coefficient of \(1/z\) in its Laurent series,
\(\sum_n z^{2-n}/n!\). There is no pole whose order would give it: only the series does.

```latex
\operatorname{Res}_{z=0} z^2 e^{1/z}
```

```python
w = symbols("w")  # w = 1/z, and 1/z comes from the term n = 3 of the series of e^w
expected = series(exp(w), w, 0, 4).removeO().coeff(w, 3)
assert expected == Rational(1, 6)
# (1 / 2 pi i) times the integral around |z| = 1, numerically; sympy.residue gives 0 here
s = np.linspace(0, 2 * np.pi, 2001)[:-1]
z = np.exp(1j * s)
assert abs(np.mean(z**2 * np.exp(1 / z) * z) - 1 / 6) < 1e-12
```

## The residue theorem

**Level 3** · `complex.residues.residue-theorem` · Ahlfors (1979), ch. 4, §5.1

A contour integral around the circle \(|z| = 2\): only the pole at 0 lies inside, so the
integral is \(2\pi i\) times the residue there, \(-1/3\).

```latex
\oint_{|z| = 2} \frac{e^z}{z (z - 3)} \, dz
```

```python
z = symbols("z")
expected = 2 * pi * I * residue(exp(z) / (z * (z - 3)), z, 0)  # the pole at 3 is outside
assert expected == -2 * pi * I / 3
s = np.linspace(0, 2 * np.pi, 4001)[:-1]
c = 2 * np.exp(1j * s)
assert abs(2 * np.pi * np.mean(np.exp(c) / (c * (c - 3)) * 1j * c) - complex(expected)) < 1e-12
```

## A real integral by residues

**Level 3** · `complex.residues.real-integral` · Ahlfors (1979), ch. 4, §5.3

An integral over the real line equals \(2\pi i\) times the residue of the integrand in the
upper half-plane, here at a double pole: the step by which books close the contour.

```latex
\int_{-\infty}^{\infty} \frac{dx}{(x^2 + 1)^2} = 2\pi i \operatorname{Res}_{z=i} \frac{1}{(z^2 + 1)^2}
```

```python
x = symbols("x", real=True)
z = symbols("z")
assert integrate(1 / (x**2 + 1)**2, (x, -oo, oo)) == pi / 2
assert 2 * pi * I * residue(1 / (z**2 + 1)**2, z, I) == pi / 2
expected = True
```

## A principal value

**Level 4** · `complex.residues.principal-value` · Ahlfors (1979), ch. 4, §5.3

The Cauchy principal value of an integral through a simple pole on the real axis, written
\(\mathrm{p.v.}\int\). Since \(\cos x / x\) is odd, only \(i \int \sin x / x\,dx = i\pi\)
remains.

```latex
\mathrm{p.v.} \int_{-\infty}^{\infty} \frac{e^{ix}}{x} \, dx
```

```python
x = symbols("x", real=True)
# cos(x)/x is odd, so its principal value over the line is 0
expected = I * integrate(sin(x) / x, (x, -oo, oo))
assert expected == I * pi
```
