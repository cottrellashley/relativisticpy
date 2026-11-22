# Finite groups and their characters

Representations of finite groups through their characters: the orthogonality of the
characters of a cyclic group, the multiplicity of an irreducible representation as an
inner product of characters, counting orbits with Burnside's lemma, and the sum of the
squares of the degrees.

## The characters of a cyclic group are orthogonal

**Level 2** · `group.finite.cyclic-orthogonality` · Serre (1977), section 2.3

The irreducible characters of the cyclic group of order 5 are
\(\chi_k(m) = e^{2\pi i k m / 5}\). Two distinct ones are orthogonal,
\(\frac15 \sum_m \chi_1(m) \overline{\chi_4(m)} = 0\): an overline is the complex
conjugate.

```latex
\frac{1}{5} \sum_{m=0}^{4} e^{2\pi i m/5} \, \overline{e^{8\pi i m/5}}
```

```python
terms = [exp(2 * pi * I * m / 5) * conjugate(exp(8 * pi * I * m / 5)) for m in range(5)]
mean = Rational(1, 5) * sum(terms)  # the mean of exp(-6 pi i m / 5): fifth roots of unity
assert abs(complex(mean.evalf(30))) < 1e-25
expected = 0
# Without the conjugate, chi_1 chi_4 = 1 and the mean would be 1, not 0.
assert (
    Rational(1, 5) * sum(exp(2 * pi * I * m / 5) * exp(8 * pi * I * m / 5) for m in range(5)) == 1
)
gram = np.array(
    [[np.mean(np.exp(2j * np.pi * (k - l) * np.arange(5) / 5)) for l in range(5)] for k in range(5)]
)
assert np.allclose(gram, np.eye(5))  # every pair of characters
```

## Necklaces

**Level 2** · `group.finite.necklaces` · standard

Burnside's lemma: the number of orbits is the average number of fixed points. A rotation
of a necklace of six beads by \(k\) places fixes the \(2^{\gcd(k, 6)}\) colourings that
repeat after \(\gcd(k, 6)\) beads, so there are 14 necklaces of six beads in two colours,
counted up to rotation (13 if they may also be turned over).

```latex
\frac{1}{6} \sum_{k=0}^{5} 2^{\gcd(k, 6)}
```

```python
from itertools import product

expected = Rational(sum(2 ** igcd(k, 6) for k in range(6)), 6)
assert expected == 14
orbits, seen = 0, set()  # and by listing the orbits of the 64 colourings
for word in product((0, 1), repeat=6):
    if word not in seen:
        orbits += 1
        seen |= {word[k:] + word[:k] for k in range(6)}
assert orbits == 14
```

## The permutation representation of S3

**Level 2** · `group.finite.permutation-representation` · Serre (1977), section 2.3

The multiplicity of an irreducible representation is an inner product of characters,
\(\frac{1}{|G|} \sum_k h_k \chi_k \psi_k\), summed over the classes of \(S_3\): the
identity, the three transpositions and the two 3-cycles, of sizes \(h_k\). \(S_3\)
permuting three coordinates has the character \(\chi\), its numbers of fixed points; its
two-dimensional irreducible representation has \(\psi\). The characters are real, and the
standard representation occurs once.

```latex
h := [1, 3, 2]
\chi := [3, 1, 0]
\psi := [2, 0, -1]
\frac{1}{6} \sum_{k=1}^{3} h_k \chi_k \psi_k
```

```python
from sympy.combinatorics import SymmetricGroup

G = SymmetricGroup(3)
fixed = lambda p: sum(1 for i in range(3) if p(i) == i)
# the standard representation is the permutation representation less the trivial one
expected = Rational(sum(fixed(p) * (fixed(p) - 1) for p in G.elements), G.order())
assert expected == 1
h, chi, psi = [1, 3, 2], [3, 1, 0], [2, 0, -1]
assert sorted(len(c) for c in G.conjugacy_classes()) == sorted(h)
assert Rational(sum(a * b * c for a, b, c in zip(h, chi, psi, strict=True)), 6) == expected
assert Rational(sum(a * c * c for a, c in zip(h, psi, strict=True)), 6) == 1  # psi is irreducible
```

## The degrees of the representations of S4

**Level 5** · `group.finite.degrees` · Serre (1977), section 2.4

A sum over the irreducible characters of a group named by its symbol, \(S_4\), and the
order of the group: \(\sum_{\chi} \chi(1)^2 = |G|\). The five irreducible
representations of \(S_4\), one for each conjugacy class, have degrees 1, 1, 2, 3 and 3.
No design of the language has groups or their representations as values.

```latex
\sum_{\chi \in \operatorname{Irr}(S_4)} \chi(1)^2 = |S_4|
```

```python
from sympy.combinatorics import SymmetricGroup
from sympy.utilities.iterables import partitions


def degree(shape):  # the hook length formula
    columns = [sum(1 for row in shape if row > j) for j in range(shape[0])]
    hooks = prod(
        (row - j - 1) + (columns[j] - i - 1) + 1 for i, row in enumerate(shape) for j in range(row)
    )
    return factorial(sum(shape)) / hooks


shapes = [sorted([k for k, m in p.items() for _ in range(m)], reverse=True) for p in partitions(4)]
degrees = sorted(degree(shape) for shape in shapes)
assert degrees == [1, 1, 2, 3, 3]
G = SymmetricGroup(4)
assert len(G.conjugacy_classes()) == len(degrees) == 5  # one irreducible for each class
assert sum(d**2 for d in degrees) == G.order() == 24
expected = True
```
