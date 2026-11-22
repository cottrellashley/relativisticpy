# Conformal maps, branches and the argument principle

Möbius and Joukowski maps, the principal branch of the logarithm, and counting zeros with
the argument principle. Ahlfors (1979) is *Complex Analysis*, 3rd edition.

## The Cayley transform

**Level 3** · `complex.maps.cayley` · standard

The Möbius map \(w = (z - i)/(z + i)\) sends the real axis onto the unit circle. The
program declares \(x\) real, as a book does, then states \(|w| = 1\).

```latex
x \in \mathbb{R}
\left| \frac{x - i}{x + i} \right| = 1
```

```python
x = symbols("x", real=True)
assert simplify(Abs((x - I) / (x + I))) == 1
expected = True
```

## The principal logarithm

**Level 4** · `complex.maps.principal-log` · standard

\(\operatorname{Log}\), with a capital, is the principal branch of the logarithm: its cut
runs along the negative real axis, and \(-\pi < \operatorname{Im} \operatorname{Log} z \le
\pi\).

```latex
\operatorname{Log}(-1 - i)
```

```python
expected = log(-1 - I)  # SymPy's log is the principal branch
assert expected == log(sqrt(2)) - 3 * pi * I / 4
```

## The argument principle

**Level 4** · `complex.maps.argument-principle` · Ahlfors (1979), ch. 4, §5.2

The number of zeros of \(f\) inside the unit circle, as the integral of \(f'/f\) around it.
The zeros of this quintic are not expressible in radicals (its Galois group is \(S_5\));
Rouché's theorem gives the same count, since \(|3z| = 3 > |z^5 + 1|\) on the circle.

```latex
f(z) := z^5 + 3z + 1
\frac{1}{2\pi i} \oint_{|z| = 1} \frac{f'(z)}{f(z)} \, dz
```

```python
roots = np.roots([1, 0, 0, 0, 3, 1])
expected = int(sum(abs(roots) < 1))
assert expected == 1
s = np.linspace(0, 2 * np.pi, 4001)[:-1]
z = np.exp(1j * s)
assert abs(np.mean((5 * z**4 + 3) / (z**5 + 3 * z + 1) * z) - 1) < 1e-9
```

## The Joukowski map

**Level 5** · `complex.maps.joukowski` · Ahlfors (1979), ch. 3, §4.2

The image of the unit circle under \(w = \frac{1}{2}(z + 1/z)\) is the segment \([-1, 1]\),
covered twice: an equation between sets, one of them the image of a curve.

```latex
\left\{ \frac{1}{2}\left(z + \frac{1}{z}\right) : |z| = 1 \right\} = [-1, 1]
```

```python
from sympy.calculus.util import function_range

t = symbols("t", real=True)
w = (exp(I * t) + exp(-I * t)) / 2  # z = e^{it} goes once round the circle
assert simplify(w.rewrite(cos) - cos(t)) == 0
assert function_range(cos(t), t, Interval(0, 2 * pi)) == Interval(-1, 1)
expected = True
```
