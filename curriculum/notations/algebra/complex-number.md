# The imaginary unit

The square of a complex number \(a + bi\), with the imaginary unit written each way books
write it. ISO 80000-2:2019 is the standard for mathematical signs and symbols; DLMF is the
NIST *Digital Library of Mathematical Functions*; Churchill and Brown (2014) is *Complex
Variables and Applications*, 9th edition; Knuth (1984) is *The TeXbook*; Nilsson and Riedel
(2015) is *Electric Circuits*, 10th edition.

Engineers write j for the unit, and a program that does says so once
(`design/notation.md`, decision 24).

## Names

```python
x, y = symbols("x y")
```

## Operands

| operand | written in | `a` | `b` | separates |
|---|---|---|---|---|
| `3-4` | latex, text, unicode | `3` | `4` | a value: −7 + 24i |
| | python | `3` | `4` | |
| `1-2` | latex, text, unicode | `1` | `2` | |
| | python | `1` | `2` | |
| `letters` | latex, text, unicode | `x` | `y` | a letter beside the unit, which is their product |
| | python | `x` | `y` | |
| `fractions` | latex | `\frac{1}{2}` | `\frac{\sqrt{3}}{2}` | a cube root of −1, squared |
| | text | `(1/2)` | `(sqrt(3)/2)` | |
| | python | `Rational(1, 2)` | `sqrt(3) / 2` | |

## Oracle

```python
expected = expand((a + b * I) ** 2)
```

## Second route

The square by its real and imaginary parts.

```python
assert simplify(expected - (S(a) ** 2 - S(b) ** 2 + 2 * a * b * I)) == 0
```

## Forms

### The unit after its coefficient

**Form** · `i-after` · Churchill and Brown (2014), §1: `z = x + iy`, and of numbers `3 + 4i`

```latex
(«a» + «b»i)^2
```

### The unit before its coefficient

**Form** · `i-before` · Churchill and Brown (2014), §1: `z = x + iy`

```latex
(«a» + i«b»)^2
```

**Only of** · `letters`, `fractions` · before a number, the unit is written after it

### An upright unit

**Form** · `upright-i` · ISO 80000-2:2019 and DLMF, §1.9: `\mathrm{i}`, the unit set upright

```latex
(«a» + «b»\mathrm{i})^2
```

### The unit after a thin space

**Form** · `thin-space` · Knuth (1984), chapter 18: a thin space, `\,`, between factors

```latex
(«a» + «b»\,i)^2
```

### The unit after a centred dot

**Form** · `cdot` · ISO 80000-2:2019, multiplication: `a \cdot b`

```latex
(«a» + «b» \cdot i)^2
```

### The root of −1

**Form** · `root` · Churchill and Brown (2014), §1: i is a square root of −1

```latex
(«a» + «b»\sqrt{-1})^2
```

### The engineers' unit, declared

**Form** · `j-declared` · design/notation.md, decision 24: `j := \sqrt{-1}`, as engineering books say in their conventions

```latex
j := \sqrt{-1}
(«a» + «b»j)^2
```

### The engineers' unit before its coefficient

**Form** · `j-before` · Nilsson and Riedel (2015), chapter 9: a complex number written `a + jb`; design/notation.md, decision 24

```latex
j := \sqrt{-1}
(«a» + j«b»)^2
```

### The unit in plain text

**Form** · `text-i` · docs/language.md, Numbers and arithmetic: `i` is the constant

```text
(«a» + «b»*i)^2
```

### The unit side by side, in plain text

**Form** · `text-i-juxtaposed` · MATLAB's: `3 + 4i`

```text
(«a» + «b»i)^2
```

**Only of** · `3-4`, `1-2` · a number beside the unit

### A capital I

**Form** · `text-capital-i` · SymPy's and Mathematica's name for the unit: `3 + 4*I`

```text
(«a» + «b»*I)^2
```

### The unit in Unicode

**Form** · `unicode-i` · as typed, with a superscript two

```unicode
(«a» + «b»i)²
```
