# Laplace transforms

The Laplace transform in operator notation, \(\mathcal{L}\{f(t)\} = \int_0^\infty
e^{-st} f(t)\, dt\), and the theorems that make it solve initial value problems, in the
notation of Boyce and DiPrima, whose \(u_c(t)\) is the unit step at \(t = c\).

## The transform of a power

**Level 1** · `fa.laplace.power` · Boyce and DiPrima (2012), section 6.1

\(\mathcal{L}\{t^n\} = n!/s^{n+1}\), here for \(n = 2\), with the transform's variable
\(s\) left implicit, as the book leaves it.

```latex
\mathcal{L}\{t^2\}
```

```python
t, s = symbols("t s")
expected = laplace_transform(t**2, t, s, noconds=True)
assert expected == factorial(2) / s**3
```

## The transform of a derivative

**Level 2** · `fa.laplace.derivative` · Boyce and DiPrima (2012), section 6.2

\(\mathcal{L}\{f'(t)\} = s\mathcal{L}\{f(t)\} - f(0)\), the theorem that turns a
differential equation into an algebraic one, checked for \(f(t) = \cos at\).

```latex
f(t) := \cos at
\mathcal{L}\{f'(t)\} = s \mathcal{L}\{f(t)\} - f(0)
```

```python
t, s, a = symbols("t s a")
f = cos(a * t)
left = laplace_transform(diff(f, t), t, s, noconds=True)
right = s * laplace_transform(f, t, s, noconds=True) - f.subs(t, 0)
assert simplify(left - right) == 0 and simplify(left + a**2 / (s**2 + a**2)) == 0
expected = True
```

## The convolution theorem

**Level 3** · `fa.laplace.convolution` · Boyce and DiPrima (2012), section 6.6

The transform of the convolution \(\int_0^t (t - \tau) \sin\tau\, d\tau\) of \(t\) and
\(\sin t\) is the product of their transforms, \(1/s^2(s^2 + 1)\).

```latex
h(t) := \int_0^t (t - \tau) \sin \tau \, d\tau
\mathcal{L}\{h(t)\} = \mathcal{L}\{t\} \, \mathcal{L}\{\sin t\}
```

```python
t, s, tau = symbols("t s tau")
h = integrate((t - tau) * sin(tau), (tau, 0, t))
assert simplify(h - (t - sin(t))) == 0
left = laplace_transform(h, t, s, noconds=True)
right = laplace_transform(t, t, s, noconds=True) * laplace_transform(sin(t), t, s, noconds=True)
assert simplify(left - right) == 0 and simplify(left - 1 / (s**2 * (s**2 + 1))) == 0
expected = True
```

## A step and a shift

**Level 4** · `fa.laplace.step` · Boyce and DiPrima (2012), section 6.3

A function switched on at \(t = \pi/4\) with the book's unit step \(u_c(t)\): the shift
theorem \(\mathcal{L}\{u_c(t) f(t - c)\} = e^{-cs} \mathcal{L}\{f(t)\}\) gives
\((1 + s e^{-\pi s/4})/(s^2 + 1)\).

```latex
\mathcal{L}\{\sin t + u_{\pi/4}(t) \cos(t - \pi/4)\}
```

```python
t, s = symbols("t s")
function = sin(t) + Heaviside(t - pi / 4) * cos(t - pi / 4)
expected = laplace_transform(function, t, s, noconds=True)
assert simplify(expected - (1 + s * exp(-pi * s / 4)) / (s**2 + 1)) == 0
```
