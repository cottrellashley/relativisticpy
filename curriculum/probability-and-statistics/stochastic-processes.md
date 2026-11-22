# Markov chains and stochastic calculus

Markov chains by their transition matrices, with absorption and stationary
distributions; then Brownian motion, Itô's lemma and the Itô integral, which no design
of the language covers yet.

## Two days of weather in the Land of Oz

**Level 2** · `prob.markov.two-step` · Grinstead and Snell (1997), section 11.1

The two-step transition probabilities of a chain of rain, nice and snowy days are the
square of its transition matrix.

```latex
P := \begin{pmatrix} 1/2 & 1/4 & 1/4 \\ 1/2 & 0 & 1/2 \\ 1/4 & 1/4 & 1/2 \end{pmatrix}
P^2
```

```python
P = Matrix([[2, 1, 1], [2, 0, 2], [1, 1, 2]]) / 4
expected = P**2
assert expected[0, :] == Matrix([[7, 3, 6]]) / 16  # the book's .438, .188, .375
```

## The drunkard's walk

**Level 3** · `prob.markov.absorbing` · Grinstead and Snell (1997), section 11.2

A walk on the corners 0 to 4 of a street, absorbed at either end. From the transitions
\(Q\) among the transient corners 1, 2 and 3, the fundamental matrix
\(N = (I - Q)^{-1}\) and the expected number of steps before absorption, \(N c\).

```latex
Q := \begin{pmatrix} 0 & 1/2 & 0 \\ 1/2 & 0 & 1/2 \\ 0 & 1/2 & 0 \end{pmatrix}
N := (I - Q)^{-1}
c := \begin{pmatrix} 1 \\ 1 \\ 1 \end{pmatrix}
N c
```

```python
Q = Matrix([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / 2
expected = (eye(3) - Q).inv() * Matrix([1, 1, 1])
assert expected == Matrix([3, 4, 3])
```

## The stationary distribution in index notation

**Level 4** · `prob.markov.stationary-index` · Grinstead and Snell (1997), section 11.3

The distribution \(\pi\) that the Land of Oz chain leaves unchanged, \(\pi_j = \pi_i
P^i{}_j\), normalised: the long-run share of rain, nice and snowy days,
\((\frac{2}{5}, \frac{1}{5}, \frac{2}{5})\).

```latex
\mathcal{S} := \mathbb{R}^{3}
i, j \in \mathcal{S}
P^{i}{}_{j} := [[1/2, 1/4, 1/4], [1/2, 0, 1/2], [1/4, 1/4, 1/2]]
\pi_j = \pi_i P^{i}{}_{j}
\pi_i 1^i = 1
\pi_j
```

```python
P = Matrix([[2, 1, 1], [2, 0, 2], [1, 1, 2]]) / 4
w = symbols("w1:4")
solution = solve([*(Matrix([w]) * P - Matrix([w])), sum(w) - 1], w)
expected = [solution[v] for v in w]
assert expected == [Rational(2, 5), Rational(1, 5), Rational(2, 5)]
assert np.allclose(np.linalg.matrix_power(np.array(P.tolist(), float), 60), [[0.4, 0.2, 0.4]] * 3)
```

## Itô's lemma for the logarithm of a stock price

**Level 5** · `prob.stochastic.ito-log` · Hull (2018), chapter 14

A stock price follows geometric Brownian motion. By Itô's lemma, \(\ln S\) has the drift
\(\mu - \sigma^2/2\), not \(\mu\): \((dz)^2 = dt\) adds a term that ordinary calculus
does not have.

```latex
dS = \mu S \, dt + \sigma S \, dz
G := \ln S
dG = \left(\mu - \frac{\sigma^2}{2}\right) dt + \sigma \, dz
```

```python
S, mu, sigma = symbols("S mu sigma", positive=True)
G = log(S)
# Ito's lemma: dG = (G_S mu S + G_t + G_SS sigma^2 S^2 / 2) dt + G_S sigma S dz
drift = diff(G, S) * mu * S + Rational(1, 2) * diff(G, S, 2) * sigma**2 * S**2
assert simplify(drift - (mu - sigma**2 / 2)) == 0
assert simplify(diff(G, S) * sigma * S - sigma) == 0
expected = True
```

## The Itô integral of Brownian motion

**Level 5** · `prob.stochastic.ito-integral` · Øksendal (2003), section 3.1

\(\int_0^t B_s \, dB_s = \frac{1}{2} B_t^2 - \frac{1}{2} t\): an integral against
Brownian motion, whose \(-\frac{1}{2} t\) ordinary calculus would miss.

```latex
\int_0^t B_s \, dB_s = \frac{1}{2} B_t^2 - \frac{1}{2} t
```

```python
# Ito's lemma for f(B) = B^2 / 2: d(B^2 / 2) = B dB + (1/2) dt, integrated from 0 to t.
B = symbols("B")
f = B**2 / 2
assert diff(f, B) == B and diff(f, B, 2) / 2 == Rational(1, 2)
# A numerical check on one simulated path, with the Ito (left-point) sums.
rng = np.random.default_rng(1)
steps = rng.normal(0, np.sqrt(1 / 200_000), 200_000)
path = np.concatenate([[0.0], np.cumsum(steps)])
assert abs(np.sum(path[:-1] * steps) - (path[-1] ** 2 / 2 - 1 / 2)) < 0.01
expected = True
```

## The Itô isometry

**Level 5** · `prob.stochastic.ito-isometry` · Øksendal (2003), section 3.1

The variance of an Itô integral is an ordinary integral:
\(\mathbb{E}[(\int_0^t B_s \, dB_s)^2] = \int_0^t \mathbb{E}[B_s^2] \, ds =
\frac{1}{2} t^2\).

```latex
\mathbb{E}\left[\left(\int_0^t B_s \, dB_s\right)^2\right]
```

```python
s, t = symbols("s t", positive=True)
expected = integrate(s, (s, 0, t))  # the Ito isometry, with E[B_s^2] = s
# The same from the integral's value, (B_t^2 - t) / 2, and E[B_t^4] = 3 t^2.
assert simplify((3 * t**2 - 2 * t * t + t**2) / 4 - expected) == 0
assert expected == t**2 / 2
```

## The stationary variance of the Ornstein–Uhlenbeck process

**Level 5** · `prob.stochastic.ou-variance` · Øksendal (2003), section 5.1

A stochastic differential equation, and the variance of its solution as time goes on:
\(\sigma^2 / 2\alpha\).

```latex
dX_t = -\alpha X_t \, dt + \sigma \, dB_t
\lim_{t \to \infty} \operatorname{Var}(X_t)
```

```python
s, t, alpha, sigma = symbols("s t alpha sigma", positive=True)
# X_t = e^{-alpha t} X_0 + sigma int_0^t e^{-alpha (t - s)} dB_s; its variance by the Ito isometry.
variance = sigma**2 * integrate(exp(-2 * alpha * (t - s)), (s, 0, t))
expected = limit(variance, t, oo)
assert expected == sigma**2 / (2 * alpha)
```
