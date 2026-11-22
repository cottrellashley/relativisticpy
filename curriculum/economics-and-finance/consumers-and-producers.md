# Consumers and producers

Demand and its elasticities, the marginal rate of substitution, Cobb–Douglas demand by
maximising utility under a budget, Roy's identity and the Slutsky equation, returns to
scale, and, at the frontier, saving by dynamic programming.

## An arc elasticity

**Level 1** · `econ.demand.arc-elasticity` · standard

The price rises from 10 to 12 and the quantity falls from 100 to 80: the midpoint
elasticity, each change divided by the average of its two values, is
\(\frac{-20/90}{2/11} = -\frac{11}{9}\).

```latex
\frac{(80 - 100)/\frac{80 + 100}{2}}{(12 - 10)/\frac{12 + 10}{2}}
```

```python
expected = (Rational(80 - 100) / Rational(80 + 100, 2)) / (Rational(12 - 10) / Rational(12 + 10, 2))
assert expected == Rational(-11, 9)
```

## A point elasticity

**Level 2** · `econ.demand.point-elasticity` · Varian (2010), chapter 15

The price elasticity \(\varepsilon = \frac{dQ}{dP}\frac{P}{Q}\) of the linear demand
\(Q = 100 - 2P\), at \(P = 20\): \(-\frac{2}{3}\), inelastic.

```latex
Q(P) := 100 - 2P
\varepsilon(P) := \frac{dQ}{dP} \frac{P}{Q(P)}
\varepsilon(20)
```

```python
P = symbols("P")
Q = 100 - 2 * P
expected = (diff(Q, P) * P / Q).subs(P, 20)
assert expected == Rational(-2, 3)
```

## The marginal rate of substitution

**Level 2** · `econ.consumer.mrs` · Varian (2010), chapter 4

For Cobb–Douglas utility \(u = x^a y^{1-a}\), the ratio of the marginal utilities,
\(\frac{\partial u/\partial x}{\partial u/\partial y} = \frac{a}{1 - a}\frac{y}{x}\).

```latex
u(x, y) := x^{a} y^{1 - a}
\frac{\partial u / \partial x}{\partial u / \partial y}
```

```python
x, y, a = symbols("x y a", positive=True)
u = x**a * y ** (1 - a)
expected = a * y / ((1 - a) * x)
assert simplify(diff(u, x) / diff(u, y) - expected) == 0
```

## Constant returns to scale

**Level 2** · `econ.production.returns-to-scale` · Varian (2010), chapter 18

The Cobb–Douglas production function \(F(K, L) = AK^\alpha L^{1-\alpha}\) is homogeneous of
degree one: scaling both inputs by \(\lambda > 0\) scales the output by \(\lambda\). A claim
that needs the inputs and the factor positive, as the program states.

```latex
\lambda, K, L > 0
F(K, L) := A K^{\alpha} L^{1 - \alpha}
F(\lambda K, \lambda L) = \lambda F(K, L)
```

```python
A, K, L, lam, alpha = symbols("A K L lambda alpha", positive=True)
F = lambda k, l: A * k**alpha * l ** (1 - alpha)
assert simplify(powsimp(expand_power_base(F(lam * K, lam * L) - lam * F(K, L)))) == 0
expected = True
```

## Cobb–Douglas demand

**Level 3** · `econ.consumer.cobb-douglas` · Varian (2010), chapter 5

The bundle that maximises \(x^a y^{1-a}\) on the budget line \(p_x x + p_y y = m\): the
consumer spends the share \(a\) of income on \(x\), so \(x = \frac{am}{p_x}\) and
\(y = \frac{(1 - a)m}{p_y}\).

```latex
0 < a < 1
p_x, p_y, m > 0
\arg\max_{x, y} \; x^{a} y^{1 - a} \quad \text{subject to} \quad p_x x + p_y y = m
```

```python
from scipy.optimize import minimize

a, px, py, m, x, y, lam = symbols("a p_x p_y m x y lambda", positive=True)
lagrangian = a * log(x) + (1 - a) * log(y) - lam * (px * x + py * y - m)  # log u, same maximiser
found = solve([diff(lagrangian, v) for v in (x, y, lam)], [x, y, lam], dict=True)
expected = [a * m / px, (1 - a) * m / py]
assert [found[0][x], found[0][y]] == expected
best = minimize(
    lambda v: -(v[0] ** 0.3) * v[1] ** 0.7,
    [1, 1],
    constraints={"type": "eq", "fun": lambda v: 2 * v[0] + 5 * v[1] - 100},
    bounds=[(1e-9, None)] * 2,
    tol=1e-12,
)
assert np.allclose(best.x, [0.3 * 100 / 2, 0.7 * 100 / 5], atol=1e-4)
```

## Roy's identity

**Level 3** · `econ.consumer.roy` · Mas-Colell, Whinston and Green (1995), chapter 3

The indirect utility of Cobb–Douglas preferences, \(v(p_x, p_y, m)\), the most utility
income \(m\) buys, gives back the demand by Roy's identity,
\(x = -\frac{\partial v/\partial p_x}{\partial v/\partial m} = \frac{am}{p_x}\).

```latex
0 < a < 1
p_x, p_y, m > 0
v(p_x, p_y, m) := \left(\frac{a m}{p_x}\right)^{a} \left(\frac{(1 - a) m}{p_y}\right)^{1 - a}
-\frac{\partial v / \partial p_x}{\partial v / \partial m}
```

```python
a, px, py, m = symbols("a p_x p_y m", positive=True)
v = (a * m / px) ** a * ((1 - a) * m / py) ** (1 - a)
expected = a * m / px
assert simplify(-diff(v, px) / diff(v, m) - expected) == 0
```

## The Slutsky equation

**Level 4** · `econ.consumer.slutsky` · Mas-Colell, Whinston and Green (1995), chapter 3

The response of demand to its own price splits into a substitution effect, the slope of
the compensated demand \(h_1 = \partial e/\partial p_1\) at the utility \(v\) the consumer
reaches, and an income effect. For Cobb–Douglas preferences, with the expenditure
function \(e\), the claim holds identically.

```latex
0 < a < 1
p_1, p_2, m, u > 0
x_1(p_1, p_2, m) := \frac{a m}{p_1}
v(p_1, p_2, m) := \left(\frac{a m}{p_1}\right)^{a} \left(\frac{(1 - a) m}{p_2}\right)^{1 - a}
e(p_1, p_2, u) := u \left(\frac{p_1}{a}\right)^{a} \left(\frac{p_2}{1 - a}\right)^{1 - a}
h_1(p_1, p_2, u) := \frac{\partial e}{\partial p_1}
\frac{\partial x_1}{\partial p_1} = \left. \frac{\partial h_1}{\partial p_1} \right|_{u = v(p_1, p_2, m)} - x_1 \frac{\partial x_1}{\partial m}
```

```python
a, p1, p2, m, u = symbols("a p_1 p_2 m u", positive=True)
b = symbols("b", positive=True)  # b = 1 - a, positive
x1 = a * m / p1
v = (a * m / p1) ** a * (b * m / p2) ** b
e = u * (p1 / a) ** a * (p2 / b) ** b
h1 = diff(e, p1)
difference = diff(x1, p1) - (diff(h1, p1).subs(u, v) - x1 * diff(x1, m))
assert simplify(expand_power_base(difference, force=True).subs(b, 1 - a)) == 0
assert (
    simplify(powsimp(expand_power_base(e.subs(u, v), force=True), force=True).subs(b, 1 - a)) == m
)
expected = True
```

## Saving by dynamic programming

**Level 5** · `econ.dynamic.cake-eating` · standard

A cake of size \(k\) is eaten over infinitely many periods, with utility \(\ln c\) each
period discounted by \(\beta\). The value function solves Bellman's equation, and the best
policy eats the share \(1 - \beta\) of what is left: \(c^*(k) = (1 - \beta)k\). An equation for an
unknown function, with a maximum inside it, is beyond any design of the language yet.

```latex
0 < \beta < 1
k > 0
V(k) = \max_{0 < c < k} \left[\ln c + \beta V(k - c)\right]
\arg\max_{0 < c < k} \left[\ln c + \beta V(k - c)\right]
```

```python
from scipy.optimize import minimize_scalar

beta, k, c, A = symbols("beta k c A", positive=True)
V = lambda w: A + log(w) / (1 - beta)  # guess, and verify
policy = solve(diff(log(c) + beta * V(k - c), c), c)
assert policy == [k * (1 - beta)]
residual = V(k) - (log(policy[0]) + beta * V(k - policy[0]))
constant = solve(expand_log(residual, force=True), A)[0]  # the guess holds, for this A
assert simplify(expand_log(residual.subs(A, constant), force=True)) == 0
expected = (1 - beta) * k
discount, level = 0.9, float(constant.subs(beta, Rational(9, 10)))
value = lambda w: level + np.log(w) / (1 - discount)
for cake in (0.5, 2.0):  # the maximum on the right is the value on the left, at c = (1 - beta) k
    found = minimize_scalar(
        lambda eat, cake=cake: -(np.log(eat) + discount * value(cake - eat)),
        bounds=(1e-9, cake - 1e-9),
        method="bounded",
        options={"xatol": 1e-12},
    )
    assert abs(found.x - 0.1 * cake) < 1e-6 and abs(-found.fun - value(cake)) < 1e-9
```
