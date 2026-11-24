# Vectors

> **For:** anyone who computes with vectors and fields. **You will learn:** the products
> of vectors, `\nabla` in any coordinates, and integrals along curves and over regions.
> **Before this:** [The language](language.md).

```python
>>> from relativisticpy import Workbook

```

## Vectors and their products

A list, a tuple or a column of numbers is a vector, and is read as one by what it meets:
`\cdot` between vectors is their dot product and `\times` their cross product, while
between numbers both multiply; `|\mathbf{v}|` and `\|\mathbf{v}\|` are its length, the
root of the sum of each component times its conjugate, of a row and a column alike, and
`\mathbf{p}^2` is \(\mathbf{p} \cdot \mathbf{p}\). `dot(u, v)` and `cross(u, v)`, as
programs write them, are the same two products; neither conjugates a side, as NumPy's
`dot` does not. Beside `\cdot` or `\times`, `\mathbf{1}` is the vector of ones as long as
the vector it meets. Derivatives, integrals, evaluation bars and limits of a vector are
taken component by component. Where nothing defines them, `\hat{\mathbf{x}}`,
`\hat{\imath}` and `\mathbf{i}`, `\mathbf{j}`, `\mathbf{k}` are the unit vectors of
Cartesian coordinates, and so is `\hat{x}`, unless it is the position x: in the angle
brackets of a wavefunction of x, or beside an operator that differentiates by x, as
`\hat{p} := -i\hbar \frac{d}{dx}` does, where the statement writes no hat that only a unit
vector can be, as `\hat{y}` is; a function is read as it was where it is defined.
`\hat{\mathbf{r}}` and `\mathbf{e}_\theta` are those of the coordinates in force. A
typeface does not make a name of its own yet: `\mathbf{r}` is `r` (decision 9 of
[the notation decisions](https://github.com/cottrellashley/relativisticpy/blob/main/design/notation.md)).

```python
>>> products = Workbook()
>>> products.exe(r"""
...     \mathbf{u} := (1, 2, 3)
...     \mathbf{v} := (4, 5, 6)
...     \mathbf{u} \cdot \mathbf{v}
...     \mathbf{u} \times \mathbf{v}
...     dot(\mathbf{u}, \mathbf{v})
...     cross(\mathbf{u}, \mathbf{v})
...     |\mathbf{u}|
... """)
[32, [-3, 6, -3], 32, [-3, 6, -3], sqrt(14)]

```

## Vector calculus

`\nabla` with no index is vector calculus's: `\nabla f`, `\nabla \cdot \mathbf{v}`,
`\nabla \times \mathbf{v}` and `\nabla^2 f`, or `\Delta f`, act on the factor after them,
and `(\mathbf{u} \cdot \nabla)` on what follows it. An equation between vectors is
decided component by component, and `\mathbf{0}` is the zero vector.

```python
>>> vc = Workbook()
>>> vc.exe(r"""
...     \mathbf{v} := xy\,\hat{\mathbf{x}} + 2yz\,\hat{\mathbf{y}} + 3zx\,\hat{\mathbf{z}}
...     \mathbf{v} \cdot \hat{\mathbf{y}}
...     \nabla \cdot \mathbf{v}
...     \nabla \times \mathbf{v}
...     \nabla \times (\nabla f(x, y, z)) = \mathbf{0}
...     \mathbf{r}(t) := (\cos t, \sin t, t)
...     |\mathbf{r}'(t)|
...     \int_0^{2\pi} \mathbf{r}(t) \, dt
... """)
[2*y*z, 3*x + y + 2*z, [-2*y, -3*z, -x], equation: holds in all 3 components, sqrt(2), [0, 0, 2*pi**2]]

```

## Coordinates

The operators are those of the coordinates in force, computed from their metric as index
notation gives them, with each vector's components along their unit vectors: the
gradient is \(g^{ij}\partial_j f\) and the divergence
\(\frac{1}{\sqrt{g}}\partial_i(\sqrt{g}\, v^i)\). The scale factors come from a line
element, from the Cartesian coordinates the program writes in the coordinates it
declares, as `x := r \sin\theta \cos\phi`, or, for coordinates it only declares, from
their names: spherical, cylindrical or polar, as a note says. Without coordinates, they
are x, y and z, or the variables of the function, as of `f(x, y)`.

```python
>>> cylinder = Workbook()
>>> cylinder.exe(r"""
...     ds^2 = d\rho^2 + \rho^2\, d\varphi^2 + dz^2
...     \mathbf{u} := \omega \rho\, \hat{\boldsymbol{\varphi}}
...     \nabla \times \mathbf{u}
...     \nabla^2 \rho^2
... """)
[[0, 0, 2*omega], 4]

```

## Cartesian tensors

Where the metric in force is written as the identity, as Cartesian coordinates' line
element is, an index may be repeated in any position and is summed, as in Cartesian
tensors; so too in a space declared as \(\mathbb{R}^n\). There, `\epsilon_{ijk}`, which
nothing defines, is the Levi-Civita symbol, +1 of 1, 2, 3; in spacetime, whose sign is a
convention, it is not read yet.

```python
>>> cartesian = Workbook()
>>> cartesian.exe(r"""
...     ds^2 = dx^2 + dy^2 + dz^2
...     A_{i} := [u(x, y, z), v(x, y, z), w(x, y, z)]
...     \epsilon_{ijk} \partial_j (\epsilon_{klm} \partial_l A_m) = \partial_i \partial_j A_j - \partial_j \partial_j A_i
... """)
[ε_{ijk} ∂_j (ε_{klm} ∂_l A_m) = ∂_i ∂_j A_j − ∂_j ∂_j A_i: holds in all 3 components]

```

## Gradients, Hessians and integrals

Of a function the program defines, written at a point rather than at its variables,
`\nabla f(\mathbf{a})` is its gradient there, a column, and `\nabla^2 f(\mathbf{a})` its
Hessian, as optimisation books write them, said in a note; `\Delta f` is always the
Laplacian. Integrals over a disk or a ball about the origin, written
`\iint_{x^2 + y^2 \le 1}`, and over a box, `\iiint_{[0, 1]^3}`, are taken in turn; any
other region is refused. An integral along a curve the program defines as a function of
one parameter whose value is a vector, with the parameter's range stated, as
`C(t) := (\cos t, \sin t, 0), \quad 0 \le t \le 2\pi`, is written `\oint_C` or `\int_C`:
in it, `d\mathbf{r}` is the curve's tangent times the differential of its parameter, the
curve's name is its point, and x, y and z are its components, so that
`\oint_C \mathbf{F} \cdot d\mathbf{r}` is the circulation of a field. Where SymPy finds
no closed form in five seconds, and nothing but the parameters is left, it is computed
numerically, as a note says.

```python
>>> newton = Workbook()
>>> newton.exe(r"""
...     f(x, y) := x^2 + x y + y^2 - 3x
...     \nabla^2 f(0, 0)
...     \begin{pmatrix} 0 \\ 0 \end{pmatrix} - \left[\nabla^2 f(0, 0)\right]^{-1} \nabla f(0, 0)
...     \Delta f
...     \iint_{x^2 + y^2 \le 1} e^{-(x^2 + y^2)} \, dA
...     C(t) := (\cos t, \sin t, 0), \quad 0 \le t \le 2\pi
...     \oint_C (-y\,\hat{\mathbf{x}} + x\,\hat{\mathbf{y}}) \cdot d\mathbf{r}
... """)
[[[2, 1], [1, 2]], [[2], [-1]], 4, -pi*exp(-1) + pi, 2*pi]

```

**Next:** [Random variables](probability.md), or [Tensors and geometry](tensors.md), for
index notation over any metric.
