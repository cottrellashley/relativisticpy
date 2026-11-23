# Products

Two quantities multiplied, written side by side, with a dot, with a cross and with a
star. Knuth (1984) is *The TeXbook*; the SI Brochure is *The International System of
Units*, 9th edition (BIPM, 2019); ISO 80000-2:2019 is the standard for mathematical signs
and symbols; Stewart (2015) is *Calculus: Early Transcendentals*, 8th edition; the
Unicode Standard is version 15.0.

`«(left)»` and `«(right)»` are the factors as a product written with no sign between them writes
them: a sum in brackets.

## Names

```python
x, a, b, alpha, beta, x_1, x_2, rate, t = symbols("x a b alpha beta x_1 x_2 rate t")
```

## Operands

| operand | written in | `left` | `right` | `(left)` | `(right)` | separates |
|---|---|---|---|---|---|---|
| `number-symbol` | latex, text, unicode | `2` | `x` | `2` | `x` | |
| | python | `2` | `x` | | | |
| `letters` | latex, text, unicode | `a` | `b` | `a` | `b` | two letters side by side, which are two names in mathematics |
| | python | `a` | `b` | | | |
| `numbers` | latex, text, unicode | `3` | `4` | `3` | `4` | numbers, which are never multiplied by writing them side by side |
| | python | `3` | `4` | | | |
| `greek` | latex | `\alpha` | `\beta` | `\alpha` | `\beta` | two commands side by side |
| | text | `alpha` | `beta` | `alpha` | `beta` | |
| | unicode | `α` | `β` | `α` | `β` | |
| | python | `alpha` | `beta` | | | |
| `subscripts` | latex, text | `x_1` | `x_2` | `x_1` | `x_2` | a subscript that ends before the next factor |
| | unicode | `x₁` | `x₂` | `x₁` | `x₂` | |
| | python | `x_1` | `x_2` | | | |
| `symbol-bracket` | latex, text, unicode | `x` | `x + 1` | `x` | `(x + 1)` | a name before a bracket: a product, where the name is a variable in it |
| | python | `x` | `x + 1` | | | |
| `number-bracket` | latex, text, unicode | `2` | `x + 1` | `2` | `(x + 1)` | |
| | python | `2` | `x + 1` | | | |
| `brackets` | latex, text | `x + 1` | `x - 1` | `(x + 1)` | `(x - 1)` | |
| | unicode | `x + 1` | `x − 1` | `(x + 1)` | `(x − 1)` | |
| | python | `x + 1` | `x - 1` | | | |
| `symbol-function` | latex | `x` | `\sin x` | `x` | `\sin x` | a factor before a function |
| | text | `x` | `sin(x)` | `x` | `sin(x)` | |
| | unicode | `x` | `sin x` | `x` | `sin x` | |
| | python | `x` | `sin(x)` | | | |
| `long-name` | latex | `\mathrm{rate}` | `t` | `\mathrm{rate}` | `t` | a name of several letters |
| | text | `rate` | `t` | `rate` | `t` | |
| | python | `rate` | `t` | | | |

## Oracle

```python
expected = S(left) * S(right)
```

## Second route

The product divided by one factor is the other.

```python
assert simplify(expected / S(right) - S(left)) == 0
```

## Forms

### Side by side

**Form** · `juxtaposed` · the SI Brochure (2019), §5.4.6: `ab`

```latex
«(left)»«(right)»
```

**Not of** · `numbers` · numbers side by side are one number

### Side by side, with a space

**Form** · `spaced` · the SI Brochure (2019), §5.4.6: `a b`

```latex
«(left)» «(right)»
```

**Not of** · `numbers` · numbers side by side are one number

### Side by side, with a thin space

**Form** · `thin-space` · Knuth (1984), chapter 18: a thin space, `\,`, between factors that would run together

```latex
«(left)»\,«(right)»
```

**Not of** · `numbers` · numbers with a thin space between them are one number, in threes

### A centred dot

**Form** · `cdot` · the SI Brochure (2019), §5.4.6: `a \cdot b`, of quantities; ISO 80000-2:2019, multiplication; Stewart (2015), §11.10, of numbers: `n! = 1 \cdot 2 \cdot 3 \cdot \cdots \cdot n`

```latex
«(left)» \cdot «(right)»
```

### A cross

**Form** · `times` · the SI Brochure (2019), §5.4.6: `a \times b`; ISO 80000-2:2019, multiplication

```latex
«(left)» \times «(right)»
```

### A star, in LaTeX

**Form** · `latex-star` · design/notation.md, decision 27: between anything but two signals, `*` is multiplication

```latex
«(left)» * «(right)»
```

### Each factor in brackets

**Form** · `both-bracketed` · as school arithmetic writes a product of two numbers: `(3)(4) = 12`

```latex
(«left»)(«right»)
```

### A star

**Form** · `star` · the star of plain text, as every programming language writes multiplication

```text
«(left)»*«(right)»
```

### A star with spaces

**Form** · `star-spaced` · the star of plain text, with spaces round it

```text
«(left)» * «(right)»
```

### Side by side, in plain text

**Form** · `text-spaced` · docs/reference.md, Names and numbers: a space between factors multiplies

```text
«(left)» «(right)»
```

**Not of** · `numbers` · numbers side by side are one number

### A middle dot

**Form** · `unicode-middle-dot` · the Unicode Standard, U+00B7 MIDDLE DOT, the half-high dot of the SI Brochure, §5.4.6

```unicode
«(left)»·«(right)»
```

### A dot operator

**Form** · `unicode-dot-operator` · the Unicode Standard, U+22C5 DOT OPERATOR

```unicode
«(left)» ⋅ «(right)»
```

### A multiplication sign

**Form** · `unicode-times` · the Unicode Standard, U+00D7 MULTIPLICATION SIGN

```unicode
«(left)» × «(right)»
```

### Side by side, in Unicode

**Form** · `unicode-spaced` · as typed, with Greek letters and subscripts as characters

```unicode
«(left)» «(right)»
```

**Not of** · `numbers` · numbers side by side are one number
