# Lengths of vectors

The length of a vector, its Euclidean norm \(\|\mathbf{v}\| = \sqrt{\sum_i |v_i|^2}\),
written between double bars typed four ways, between single bars, with the subscript 2, as
the root of a dot product, and as each language of computing writes it. Strang (2016) is
*Introduction to Linear Algebra*, 5th edition; Stewart (2015) is *Calculus: Early
Transcendentals*, 8th edition; Halliday, Resnick and Walker (2014) is *Fundamentals of
Physics*, 10th edition; Golub and Van Loan (2013) is *Matrix Computations*, 4th edition;
Lamport (1994) is *LaTeX: A Document Preparation System*, 2nd edition; the amsmath package
is described in its *User's Guide* (American Mathematical Society, 2020); the physics
package is the LaTeX package of that name (de la Barrera, 2012); AsciiMath is the plain
text notation of asciimath.org; the Unicode Standard is version 15.0.

`«v»` is the components, separated by commas. Of a complex vector the length takes each
component's modulus; the books that write single bars, or the root of a dot product,
write them of real vectors.

## Operands

| operand | written in | `v` | separates |
|---|---|---|---|
| `three` | latex, text, unicode | `1, 2, 2` | a length of 3, from the sum of the components, 5, and of their squares, 9 |
| | python | `[1, 2, 2]` | |
| `negative` | latex, text | `3, -4` | a length of 5, from the sum of the moduli, 7, and the greatest, 4 |
| | unicode | `3, −4` | |
| | python | `[3, -4]` | |
| `four` | latex, text, unicode | `1, 1, 1, 1` | a vector of four components |
| | python | `[1, 1, 1, 1]` | |
| `irrational` | latex, text, unicode | `1, 1` | a length that is a root, √2 |
| | python | `[1, 1]` | |
| `complex` | latex, text, unicode | `1, i` | a length that takes moduli, √2, from the root of 1 + i², 0 |
| | python | `[1, I]` | |
| `zero` | latex, text, unicode | `0, 0, 0` | the zero vector, whose length is 0 |
| | python | `[0, 0, 0]` | |
| `unit` | latex | `\frac{3}{5}, \frac{4}{5}` | a unit vector, of components that are fractions |
| | text, unicode | `3/5, 4/5` | |
| | python | `[Rational(3, 5), Rational(4, 5)]` | |

## Oracle

```python
expected = Matrix(v).norm()
```

## Second route

The root of the vector's conjugate transpose times the vector.

```python
column = Matrix(v)
assert simplify(sqrt((column.H * column)[0, 0]) - expected) == 0
```

## Forms

### Double bars

**Form** · `double-bars` · Strang (2016), §1.2, Lengths and Dot Products: `\|v\|`

```latex
\mathbf{v} := («v»)
\|\mathbf{v}\|
```

### Double bars, opening and closing

**Form** · `lVert` · the amsmath package, Delimiters: `\lVert` and `\rVert`

```latex
\mathbf{v} := («v»)
\lVert \mathbf{v} \rVert
```

### Double bars by name

**Form** · `Vert` · Lamport (1994), §3.3.4, Delimiters: `\Vert`, the double bar

```latex
\mathbf{v} := («v»)
\Vert \mathbf{v} \Vert
```

### Double bars that grow

**Form** · `left-right` · Lamport (1994), §3.3.4, Delimiters: `\left\|` and `\right\|`

```latex
\mathbf{v} := («v»)
\left\| \mathbf{v} \right\|
```

### The subscript 2

**Form** · `subscript-2` · Golub and Van Loan (2013), §2.2, Vector Norms: `\|x\|_2`

```latex
x := («v»)
\|x\|_2
```

### Single bars

**Form** · `single-bars` · Stewart (2015), §12.2, Vectors: the length `|\mathbf{a}|`

```latex
\mathbf{a} := («v»)
|\mathbf{a}|
```

**Not of** · `complex` · the book's vectors are real

### Single bars round an arrow

**Form** · `arrow-bars` · Halliday, Resnick and Walker (2014), chapter 3, Vectors: the magnitude `|\vec{a}|`

```latex
\vec{a} := («v»)
|\vec{a}|
```

**Not of** · `complex` · the book's vectors are real

### The root of a dot product

**Form** · `square-root` · Strang (2016), §1.2, Lengths and Dot Products: `\|v\| = \sqrt{v \cdot v}`

```latex
\mathbf{v} := («v»)
\sqrt{\mathbf{v} \cdot \mathbf{v}}
```

**Not of** · `complex` · the book's vectors are real

### The physics package's norm

**Form** · `physics-norm` · the physics package, Vector notation: `\norm{a}`

```latex
\mathbf{v} := («v»)
\norm{\mathbf{v}}
```

**Refused** · `ParseError` · docs/reference.md, Not supported: `\norm` is an error that says to write `\| x \|`

### A call of norm

**Form** · `norm-call` · MATLAB's `norm(v)`; docs/reference.md, Functions, Matrices: `norm`

```text
v := [«v»]
norm(v)
```

### Double bars in plain text

**Form** · `text-double-bars` · AsciiMath: `||v||`

```text
v := [«v»]
||v||
```

### Single bars in plain text

**Form** · `text-bars` · Stewart's `|a|`, typed in plain text

```text
v := [«v»]
|v|
```

**Not of** · `complex` · the book's vectors are real

### Double bars in Unicode

**Form** · `unicode-double-bars` · the Unicode Standard, U+2016 DOUBLE VERTICAL LINE; docs/reference.md, Unicode: ‖ is LaTeX's

```unicode
v := («v»)
‖v‖
```

### The subscript 2 in Unicode

**Form** · `unicode-subscript-2` · the Unicode Standard, U+2016 DOUBLE VERTICAL LINE and U+2082 SUBSCRIPT TWO: `‖x‖₂`

```unicode
x := («v»)
‖x‖₂
```

### Single bars in Unicode

**Form** · `unicode-bars` · as typed, with Unicode's minus sign, U+2212

```unicode
v := («v»)
|v|
```

**Not of** · `complex` · the book's vectors are real
