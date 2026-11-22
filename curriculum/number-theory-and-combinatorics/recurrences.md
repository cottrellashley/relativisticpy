# Recurrences

Sequences defined by a recurrence and its initial values, as a page of Concrete
Mathematics defines them, and the closed forms that solve them: the Tower of Hanoi, and a
linear recurrence of the second order. Graham, Knuth and Patashnik (1994) is *Concrete
Mathematics*, 2nd edition.

## The Tower of Hanoi

**Level 3** · `nt.recurrences.hanoi` · Graham, Knuth and Patashnik (1994), section 1.1

The least number of moves that transfers a tower of \(n\) discs, given by its recurrence:
the closed form is \(T_n = 2^n - 1\).

```latex
T_0 = 0
T_n = 2 T_{n-1} + 1, \quad n > 0
T_n
```

```python
n = symbols("n", integer=True, nonnegative=True)
T = Function("T")
expected = rsolve(T(n) - 2 * T(n - 1) - 1, T(n), {T(0): 0})
assert expected == 2**n - 1
moves = [0]
for _ in range(20):
    moves.append(2 * moves[-1] + 1)
assert all(moves[m] == 2**m - 1 for m in range(21))
```

## A linear recurrence of the second order

**Level 3** · `nt.recurrences.second-order` · standard

Two initial values and a recurrence with constant coefficients, solved in closed form:
the roots of its characteristic equation are 2 and 3, and \(a_n = 3^n - 2^n\).

```latex
a_0 = 0
a_1 = 1
a_n = 5 a_{n-1} - 6 a_{n-2}, \quad n \geq 2
a_n
```

```python
n, r = symbols("n r")
a = Function("a")
expected = rsolve(a(n) - 5 * a(n - 1) + 6 * a(n - 2), a(n), {a(0): 0, a(1): 1})
assert expected == 3**n - 2**n
assert solve(r**2 - 5 * r + 6, r) == [2, 3]  # the characteristic roots
values = [0, 1]
for m in range(2, 20):
    values.append(5 * values[-1] - 6 * values[-2])
assert all(values[m] == 3**m - 2**m for m in range(20))
```
