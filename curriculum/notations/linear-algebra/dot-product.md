# Dot products

The dot product of two real vectors, \(\mathbf{u} \cdot \mathbf{v} = \sum_i u_i v_i\),
written with a dot between letters in bold or with arrows, as an inner product, as a row
times a column, as a sum, between components in angle brackets, and as each language of
computing writes it. Vectors of different lengths have none, and must be refused. Stewart
(2015) is *Calculus: Early Transcendentals*, 8th edition; Halliday, Resnick and Walker
(2014) is *Fundamentals of Physics*, 10th edition; Axler (2015) is *Linear Algebra Done
Right*, 3rd edition; Strang (2016) is *Introduction to Linear Algebra*, 5th edition;
Goodfellow, Bengio and Courville (2016) is *Deep Learning*; the physics package is the LaTeX
package of that name (de la Barrera, 2012); the Unicode Standard is version 15.0.

`«u»` and `«v»` are the components, separated by commas; `«ucol»` and `«vcol»` the same
components as the body of a column; `«n»` the length.

## Names

```python
a, b, c, x, y, z = symbols("a b c x y z")
```

## Operands

| operand | written in | `u` | `v` | `ucol` | `vcol` | `n` | separates |
|---|---|---|---|---|---|---|---|
| `three` | latex | `1, 2, 3` | `4, -5, 6` | `1 \\ 2 \\ 3` | `4 \\ -5 \\ 6` | `3` | a number, from the product of each component, and the cross product |
| | text | `1, 2, 3` | `4, -5, 6` | | | | |
| | unicode | `1, 2, 3` | `4, −5, 6` | | | | |
| | python | `[1, 2, 3]` | `[4, -5, 6]` | | | | |
| `two` | latex | `2, 4` | `3, -1` | `2 \\ 4` | `3 \\ -1` | `2` | vectors of the plane |
| | text | `2, 4` | `3, -1` | | | | |
| | unicode | `2, 4` | `3, −1` | | | | |
| | python | `[2, 4]` | `[3, -1]` | | | | |
| `four` | latex | `1, 0, -1, 2` | `3, 1, 2, 1` | `1 \\ 0 \\ -1 \\ 2` | `3 \\ 1 \\ 2 \\ 1` | `4` | vectors of four components |
| | text | `1, 0, -1, 2` | `3, 1, 2, 1` | | | | |
| | unicode | `1, 0, −1, 2` | `3, 1, 2, 1` | | | | |
| | python | `[1, 0, -1, 2]` | `[3, 1, 2, 1]` | | | | |
| `orthogonal` | latex | `1, 2, 0` | `-2, 1, 5` | `1 \\ 2 \\ 0` | `-2 \\ 1 \\ 5` | `3` | a dot product of 0, which exists |
| | text | `1, 2, 0` | `-2, 1, 5` | | | | |
| | unicode | `1, 2, 0` | `−2, 1, 5` | | | | |
| | python | `[1, 2, 0]` | `[-2, 1, 5]` | | | | |
| `symbols` | latex, text, unicode | `a, b, c` | `x, y, z` | | | | components that are letters: ax + by + cz |
| | latex | | | `a \\ b \\ c` | `x \\ y \\ z` | `3` | |
| | python | `[a, b, c]` | `[x, y, z]` | | | | |
| `mismatched` | latex | `1, 2` | `1, 2, 3` | `1 \\ 2` | `1 \\ 2 \\ 3` | | vectors of different lengths, which have no dot product |
| | text, unicode | `1, 2` | `1, 2, 3` | | | | |
| | python | `[1, 2]` | `[1, 2, 3]` | | | | |

## Oracle

```python
left, right = Matrix(u), Matrix(v)
if len(left) != len(right):
    error = "RelativisticPyError"
else:
    expected = left.dot(right)
```

## Second route

A row times a column, and the sum of the products of the components.

```python
if len(left) != len(right):
    assert error == "RelativisticPyError"
else:
    assert expand((left.T * right)[0, 0] - expected) == 0
    assert expand(sum(p * q for p, q in zip(u, v)) - expected) == 0
```

## Forms

### A dot between bold letters

**Form** · `bold-dot` · Stewart (2015), §12.3, The Dot Product: `\mathbf{a} \cdot \mathbf{b}`

```latex
\mathbf{u} := («u»)
\mathbf{v} := («v»)
\mathbf{u} \cdot \mathbf{v}
```

### A dot between arrows

**Form** · `arrow-dot` · Halliday, Resnick and Walker (2014), chapter 3, Vectors: `\vec{a} \cdot \vec{b}`

```latex
\vec{u} := («u»)
\vec{v} := («v»)
\vec{u} \cdot \vec{v}
```

### An inner product

**Form** · `inner-product` · Axler (2015), §6.A, Inner Products and Norms: `\langle u, v \rangle`, of real vectors, the dot product

```latex
u := («u»)
v := («v»)
\langle u, v \rangle
```

### A row times a column

**Form** · `transpose-product` · Strang (2016), §2.7, Transposes and Permutations: the dot product of columns as `x^T y`

```latex
u := \begin{bmatrix} «ucol» \end{bmatrix}
v := \begin{bmatrix} «vcol» \end{bmatrix}
u^T v
```

### A row times a column, with a raised tack

**Form** · `top-product` · Goodfellow, Bengio and Courville (2016), §2.2, Multiplying Matrices and Vectors: the dot product `\mathbf{x}^\top \mathbf{y}`

```latex
\mathbf{u} := \begin{bmatrix} «ucol» \end{bmatrix}
\mathbf{v} := \begin{bmatrix} «vcol» \end{bmatrix}
\mathbf{u}^\top \mathbf{v}
```

### The sum of the products of the components

**Form** · `sum-of-products` · Strang (2016), §1.2, Lengths and Dot Products: `v \cdot w = v_1 w_1 + v_2 w_2`, written as a sum

```latex
u := («u»)
v := («v»)
\sum_{i=1}^{«n»} u_i v_i
```

### Components in angle brackets

**Form** · `angle-components` · Stewart (2015), §12.3, The Dot Product: `\langle 2, 4 \rangle \cdot \langle 3, -1 \rangle`, of vectors written by their components

```latex
\langle «u» \rangle \cdot \langle «v» \rangle
```

### The physics package's dot

**Form** · `physics-vdot` · the physics package, Vector notation: `\vdot`, the dot between vectors

```latex
\mathbf{u} := («u»)
\mathbf{v} := («v»)
\mathbf{u} \vdot \mathbf{v}
```

### A call of dot

**Form** · `dot-call` · MATLAB's `dot(u, v)`

```text
u := [«u»]
v := [«v»]
dot(u, v)
```

### The Wolfram Language's dot

**Form** · `wolfram-dot` · the Wolfram Language's `Dot`: `u.v` of two vectors is their dot product

```text
u := [«u»]
v := [«v»]
u.v
```

### The middle dot in Unicode

**Form** · `unicode-middle-dot` · the Unicode Standard, U+00B7 MIDDLE DOT, as typed for a product

```unicode
u := («u»)
v := («v»)
u·v
```

### The dot operator in Unicode

**Form** · `unicode-dot-operator` · the Unicode Standard, U+22C5 DOT OPERATOR, the dot of `\cdot`, as typed

```unicode
u := («u»)
v := («v»)
u⋅v
```

### Angle brackets in Unicode

**Form** · `unicode-angle` · the Unicode Standard, U+27E8 and U+27E9, MATHEMATICAL LEFT and RIGHT ANGLE BRACKET: Axler's `⟨u, v⟩`, as typed

```unicode
u := («u»)
v := («v»)
⟨u, v⟩
```
