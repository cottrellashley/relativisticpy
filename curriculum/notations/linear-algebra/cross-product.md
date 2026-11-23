# Cross products

The cross product of two vectors of three dimensions, \(\mathbf{u} \times \mathbf{v}\),
written with a cross between letters in bold or with arrows, as a determinant of unit
vectors and components, and as each language of computing writes it. Vectors of two or four
components have none, and must be refused. Stewart (2015) is *Calculus: Early
Transcendentals*, 8th edition; Halliday, Resnick and Walker (2014) is *Fundamentals of
Physics*, 10th edition; Griffiths (2013) is *Introduction to Electrodynamics*, 4th edition;
the Unicode Standard is version 15.0.

`«u»` and `«v»` are the components, separated by commas; `«urow»` and `«vrow»` the same
components as a row of a matrix environment.

## Names

```python
a, b, c, x, y, z = symbols("a b c x y z")
```

## Operands

| operand | written in | `u` | `v` | `urow` | `vrow` | separates |
|---|---|---|---|---|---|---|
| `numbers` | latex, text, unicode | `1, 2, 3` | `4, 5, 6` | | | the order of the factors, v × u, and the product of each component |
| | latex | | | `1 & 2 & 3` | `4 & 5 & 6` | |
| | python | `[1, 2, 3]` | `[4, 5, 6]` | | | |
| `unit` | latex, text, unicode | `1, 0, 0` | `0, 1, 0` | | | the unit vectors: i × j is k |
| | latex | | | `1 & 0 & 0` | `0 & 1 & 0` | |
| | python | `[1, 0, 0]` | `[0, 1, 0]` | | | |
| `negative` | latex, text | `2, -1, 0` | `0, 3, -2` | | | components with a sign |
| | latex | | | `2 & -1 & 0` | `0 & 3 & -2` | |
| | unicode | `2, −1, 0` | `0, 3, −2` | | | |
| | python | `[2, -1, 0]` | `[0, 3, -2]` | | | |
| `symbols` | latex, text, unicode | `a, b, c` | `x, y, z` | | | components that are letters |
| | latex | | | `a & b & c` | `x & y & z` | |
| | python | `[a, b, c]` | `[x, y, z]` | | | |
| `parallel` | latex, text, unicode | `1, 2, 3` | `2, 4, 6` | | | parallel vectors, whose cross product is the zero vector |
| | latex | | | `1 & 2 & 3` | `2 & 4 & 6` | |
| | python | `[1, 2, 3]` | `[2, 4, 6]` | | | |
| `planar` | latex, text, unicode | `1, 2` | `3, 4` | | | vectors of the plane, which have no cross product |
| | python | `[1, 2]` | `[3, 4]` | | | |
| `four` | latex, text, unicode | `1, 0, 0, 0` | `0, 1, 0, 0` | | | vectors of four components, which have none either |
| | python | `[1, 0, 0, 0]` | `[0, 1, 0, 0]` | | | |

## Oracle

```python
left, right = Matrix(u), Matrix(v)
if len(left) != 3 or len(right) != 3:
    error = "RelativisticPyError"
else:
    expected = left.cross(right)
```

## Second route

Each component by the Levi-Civita symbol, \((\mathbf{u} \times \mathbf{v})_i =
\epsilon_{ijk} u_j v_k\); the cross product is orthogonal to both factors.

```python
if len(left) != 3 or len(right) != 3:
    assert error == "RelativisticPyError"
else:
    by_symbol = Matrix(
        [
            sum(LeviCivita(i, j, k) * u[j] * v[k] for j in range(3) for k in range(3))
            for i in range(3)
        ]
    )
    assert (by_symbol - expected).applyfunc(expand) == zeros(3, 1)
    assert expand(expected.dot(left)) == 0
    assert expand(expected.dot(right)) == 0
```

## Forms

### A cross between bold letters

**Form** · `bold-times` · Stewart (2015), §12.4, The Cross Product: `\mathbf{a} \times \mathbf{b}`

```latex
\mathbf{u} := («u»)
\mathbf{v} := («v»)
\mathbf{u} \times \mathbf{v}
```

### A cross between arrows

**Form** · `arrow-times` · Halliday, Resnick and Walker (2014), chapter 3, Vectors: `\vec{a} \times \vec{b}`

```latex
\vec{u} := («u»)
\vec{v} := («v»)
\vec{u} \times \vec{v}
```

### A determinant of i, j and k

**Form** · `determinant-ijk` · Stewart (2015), §12.4, The Cross Product: `\begin{vmatrix} \mathbf{i} & \mathbf{j} & \mathbf{k} \\ a_1 & a_2 & a_3 \\ b_1 & b_2 & b_3 \end{vmatrix}`

```latex
\begin{vmatrix} \mathbf{i} & \mathbf{j} & \mathbf{k} \\ «urow» \\ «vrow» \end{vmatrix}
```

### A determinant of unit vectors with hats

**Form** · `determinant-hats` · Griffiths (2013), §1.1.2, Vector Algebra: Component Form: the cross product as a determinant whose first row is `\hat{\mathbf{x}}`, `\hat{\mathbf{y}}` and `\hat{\mathbf{z}}`

```latex
\begin{vmatrix} \hat{\mathbf{x}} & \hat{\mathbf{y}} & \hat{\mathbf{z}} \\ «urow» \\ «vrow» \end{vmatrix}
```

### A call of cross

**Form** · `cross-call` · MATLAB's `cross(u, v)`, as NumPy's `cross`

```text
u := [«u»]
v := [«v»]
cross(u, v)
```

### The multiplication sign in Unicode

**Form** · `unicode-times` · the Unicode Standard, U+00D7 MULTIPLICATION SIGN; docs/reference.md, Unicode: × is `\times`

```unicode
u := («u»)
v := («v»)
u × v
```

### The cross product sign in Unicode

**Form** · `unicode-cross-product` · the Unicode Standard, U+2A2F VECTOR OR CROSS PRODUCT

```unicode
u := («u»)
v := («v»)
u ⨯ v
```
