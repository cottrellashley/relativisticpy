# Games

Two-player games given by their payoff matrices: a mixed-strategy equilibrium found by
indifference, the value of a zero-sum game by maximin, and the set of all Nash
equilibria of the Battle of the Sexes, the game Osborne calls Bach or Stravinsky.

## A mixed equilibrium of the Battle of the Sexes

**Level 3** · `econ.games.mixed-equilibrium` · Osborne (2004), chapter 4

Player 1 prefers the first concert (payoffs 2 and 1), player 2 the second (1 and 2), and
both prefer company. In the mixed equilibrium player 1 goes to the first with the
probability \(p\) that leaves player 2 indifferent: \(\frac{\partial u_2}{\partial q} = 0\) at
\(p = \frac{2}{3}\).

```latex
B := \begin{pmatrix} 1 & 0 \\ 0 & 2 \end{pmatrix}
u_2(p, q) := \begin{pmatrix} p & 1 - p \end{pmatrix} B \begin{pmatrix} q \\ 1 - q \end{pmatrix}
solve(\frac{\partial u_2}{\partial q} = 0, p)
```

```python
p, q = symbols("p q")
u2 = (Matrix([[p, 1 - p]]) * Matrix([[1, 0], [0, 2]]) * Matrix([q, 1 - q]))[0]
expected = solve(diff(u2, q), p)
assert expected == [Rational(2, 3)]
assert u2.subs(p, Rational(2, 3)).diff(q) == 0  # player 2 is indifferent
```

## The value of a zero-sum game

**Level 4** · `econ.games.minimax` · Osborne (2004), chapter 11

The row player maximises the least expected payoff the column player can hold them to,
\(\max_p \min_q\) of the bilinear form \(p^T A q\) over mixed strategies. For this game the
value is \(\frac{1}{7}\), with the row player choosing the first row with probability
\(\frac{3}{7}\).

```latex
A := \begin{pmatrix} 3 & -1 \\ -2 & 1 \end{pmatrix}
\max_{0 \leq p \leq 1} \min_{0 \leq q \leq 1} \begin{pmatrix} p & 1 - p \end{pmatrix} A \begin{pmatrix} q \\ 1 - q \end{pmatrix}
```

```python
from scipy.optimize import linprog

p, q = symbols("p q")
payoff = expand((Matrix([[p, 1 - p]]) * Matrix([[3, -1], [-2, 1]]) * Matrix([q, 1 - q]))[0])
# Linear in q, so the inner minimum is at q = 0 or 1; the outer maximum where they cross.
crossing = solve(payoff.subs(q, 0) - payoff.subs(q, 1), p)[0]
assert crossing == Rational(3, 7)
expected = payoff.subs({p: crossing, q: 0})
assert expected == Rational(1, 7)
# The same value as a linear program: maximise v subject to p^T A >= v in each column.
found = linprog(
    [0, 0, -1],
    A_ub=[[-3, 2, 1], [1, -1, 1]],
    b_ub=[0, 0],
    A_eq=[[1, 1, 0]],
    b_eq=[1],
    bounds=[(0, None), (0, None), (None, None)],
)
assert np.allclose(found.x, [3 / 7, 4 / 7, 1 / 7])
```

## All the equilibria of the Battle of the Sexes

**Level 5** · `econ.games.equilibria` · Osborne (2004), chapter 4

The Nash equilibria as a set: the pairs of mixed strategies, each a best response to the
other. There are three, the two pure ones and the mixed one,
\(\{(1, 1), (0, 0), (\frac{2}{3}, \frac{1}{3})\}\), with \(p\) and \(q\) the probabilities of the
first concert. A set of fixed points of best-response correspondences is beyond any
design of the language yet.

```latex
A := \begin{pmatrix} 2 & 0 \\ 0 & 1 \end{pmatrix}
B := \begin{pmatrix} 1 & 0 \\ 0 & 2 \end{pmatrix}
u_1(p, q) := \begin{pmatrix} p & 1 - p \end{pmatrix} A \begin{pmatrix} q \\ 1 - q \end{pmatrix}
u_2(p, q) := \begin{pmatrix} p & 1 - p \end{pmatrix} B \begin{pmatrix} q \\ 1 - q \end{pmatrix}
\left\{ (p^*, q^*) \in [0, 1]^2 : p^* \in \arg\max_{p} u_1(p, q^*), \; q^* \in \arg\max_{q} u_2(p^*, q) \right\}
```

```python
p, q = symbols("p q")
u1 = expand((Matrix([[p, 1 - p]]) * Matrix([[2, 0], [0, 1]]) * Matrix([q, 1 - q]))[0])
u2 = expand((Matrix([[p, 1 - p]]) * Matrix([[1, 0], [0, 2]]) * Matrix([q, 1 - q]))[0])
assert diff(u1, p) == 3 * q - 1 and diff(u2, q) == 3 * p - 2  # the best responses switch there


def best(slope):  # the best responses of a payoff linear in one's own probability
    return {1} if slope > 0 else {0} if slope < 0 else "all"


grid = [Rational(i, 6) for i in range(7)]
equilibria = set()
for P in grid:
    for Q in grid:
        responds_1 = best(diff(u1, p).subs(q, Q))
        responds_2 = best(diff(u2, q).subs(p, P))
        if (responds_1 == "all" or P in responds_1) and (responds_2 == "all" or Q in responds_2):
            equilibria.add((P, Q))
assert equilibria == {(1, 1), (0, 0), (Rational(2, 3), Rational(1, 3))}
expected = FiniteSet(Tuple(1, 1), Tuple(0, 0), Tuple(Rational(2, 3), Rational(1, 3)))
```
