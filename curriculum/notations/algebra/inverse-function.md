# Inverse functions

The inverse of a trigonometric or hyperbolic function, written with the power −1 on its
name and written with its own name: \(\sin^{-1} x\) and \(\arcsin x\). The power −1 on a
function's name is its inverse, never a reciprocal (docs/latex.md, Functions take what
follows them; docs/reference.md, Commands). Stewart (2015) is *Calculus: Early
Transcendentals*, 8th edition; ISO 80000-2:2019 is the standard for mathematical signs and
symbols; the amsmath package is described in its *User's Guide* (American Mathematical
Society, 2020); the Unicode Standard is version 15.0.

`«F»` is the function, `«arc»` its inverse's own name, and `«arc-name»` that name in
letters, as `\operatorname` takes it.

## Names

```python
x = symbols("x")
```

## Operands

| operand | written in | `F` | `arc` | `arc-name` | `argument` | `function` | `inverse` | separates |
|---|---|---|---|---|---|---|---|---|
| `arcsin-of-1` | latex | `\sin` | `\arcsin` | `arcsin` | `1` | | | a value |
| | text | `sin` | `asin` | `arcsin` | `1` | | | |
| | unicode | `sin` | `arcsin` | | `1` | | | |
| | python | | | | `1` | `sin` | `asin` | |
| `arccos-of-0` | latex | `\cos` | `\arccos` | `arccos` | `0` | | | |
| | text | `cos` | `acos` | `arccos` | `0` | | | |
| | unicode | `cos` | `arccos` | | `0` | | | |
| | python | | | | `0` | `cos` | `acos` | |
| `arctan-of-1` | latex | `\tan` | `\arctan` | `arctan` | `1` | | | |
| | text | `tan` | `atan` | `arctan` | `1` | | | |
| | unicode | `tan` | `arctan` | | `1` | | | |
| | python | | | | `1` | `tan` | `atan` | |
| `arcsin-of-x` | latex | `\sin` | `\arcsin` | `arcsin` | `x` | | | an inverse from a reciprocal: arcsin x, not 1/sin x |
| | text | `sin` | `asin` | `arcsin` | `x` | | | |
| | unicode | `sin` | `arcsin` | | `x` | | | |
| | python | | | | `x` | `sin` | `asin` | |
| `arctan-of-a-half` | latex | `\tan` | `\arctan` | `arctan` | `\frac{x}{2}` | | | a fraction as the argument |
| | python | | | | `x/2` | `tan` | `atan` | |
| `arsinh-of-x` | latex | `\sinh` | `\operatorname{arsinh}` | `arsinh` | `x` | | | a hyperbolic function, whose inverse ISO names arsinh |
| | text | `sinh` | `asinh` | `arcsinh` | `x` | | | |
| | python | | | | `x` | `sinh` | `asinh` | |

## Oracle

```python
expected = inverse(argument)
```

## Second route

The function of its inverse is the argument.

```python
assert simplify(function(expected) - argument) == 0
```

## Forms

### The power −1 on the name

**Form** · `minus-one` · Stewart (2015), §1.5: `\sin^{-1} x`, which is not `1/\sin x`

```latex
«F»^{-1} «argument»
```

### The power −1 on the name, the argument in brackets

**Form** · `minus-one-round` · Stewart (2015), §1.5: `\sin^{-1}(\sin x) = x`

```latex
«F»^{-1}(«argument»)
```

### The inverse's own name

**Form** · `arc` · ISO 80000-2:2019: `\arcsin x`, `\operatorname{arsinh} x`; Stewart (2015), §1.5

```latex
«arc» «argument»
```

### The inverse's own name, the argument in brackets

**Form** · `arc-round` · ISO 80000-2:2019: `\arcsin x`, with its argument in brackets

```latex
«arc»(«argument»)
```

### The inverse's name with operatorname

**Form** · `operatorname` · the amsmath package, Operator names: `\operatorname{arcsin}`

```latex
\operatorname{«arc-name»}(«argument»)
```

### A call by the short name

**Form** · `call` · docs/reference.md, Functions, Elementary functions: `asin`, `acos`, `atan`, `asinh`; SymPy's and C's names

```text
«arc»(«argument»)
```

### A call by the long name

**Form** · `call-long` · docs/reference.md, Functions, Elementary functions: `arcsin`, `arccos`, `arctan`; NumPy's and Mathematica's long names

```text
«arc-name»(«argument»)
```

### A superscript −1 in Unicode

**Form** · `unicode-minus-one` · the Unicode Standard, U+207B and U+00B9: `sin⁻¹ x`

```unicode
«F»⁻¹ «argument»
```

### A superscript −1 in Unicode, the argument in brackets

**Form** · `unicode-minus-one-round` · the Unicode Standard, U+207B and U+00B9: `sin⁻¹(x)`

```unicode
«F»⁻¹(«argument»)
```

### The inverse's own name, in Unicode

**Form** · `unicode-arc` · as typed: `arcsin x`

```unicode
«arc» «argument»
```
