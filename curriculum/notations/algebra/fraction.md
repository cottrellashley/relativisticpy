# Fractions

One quantity divided by another, \(\frac{a}{b}\), as a built-up fraction, with a solidus and
with a division sign. Lamport (1994) is *LaTeX: A Document Preparation System*, 2nd
edition; Knuth (1984) is *The TeXbook*; the amsmath package is described in its *User's
Guide* (American Mathematical Society, 2020); ISO 80000-2:2019 is the standard for
mathematical signs and symbols; the SI Brochure is *The International System of Units*,
9th edition (BIPM, 2019); the Unicode Standard is version 15.0.

`«(top)»` and `«(bottom)»` are the numerator and the denominator as a line of text writes them
beside a solidus: in brackets where they are sums or products.

## Names

```python
x, a, b, c, r, r_s = symbols("x a b c r r_s")
```

## Operands

| operand | written in | `top` | `bottom` | `(top)` | `(bottom)` | separates |
|---|---|---|---|---|---|---|
| `numbers` | latex, text, unicode | `3` | `4` | `3` | `4` | exact division from a decimal |
| | python | `3` | `4` | | | |
| `symbols` | latex, text, unicode | `a` | `b` | `a` | `b` | |
| | python | `a` | `b` | | | |
| `sums` | latex, text | `x + 1` | `x - 1` | `(x + 1)` | `(x - 1)` | a fraction that takes its whole numerator and denominator |
| | unicode | `x + 1` | `x − 1` | `(x + 1)` | `(x − 1)` | |
| | python | `x + 1` | `x - 1` | | | |
| `product-below` | latex | `a` | `bc` | `a` | `(bc)` | a denominator of two letters, which is their product |
| | text | `a` | `b*c` | `a` | `(b*c)` | |
| | unicode | `a` | `b c` | `a` | `(b c)` | |
| | python | `a` | `b*c` | | | |
| `negative` | latex, text | `-3` | `4` | `-3` | `4` | a sign in the numerator |
| | unicode | `−3` | `4` | `−3` | `4` | |
| | python | `-3` | `4` | | | |
| `nested` | latex | `1` | `1 + \frac{1}{x}` | `1` | `(1 + 1/x)` | a fraction within a fraction |
| | text, unicode | `1` | `1 + 1/x` | `1` | `(1 + 1/x)` | |
| | python | `1` | `1 + 1/x` | | | |
| `subscript` | latex | `r_s` | `2r` | `r_s` | `(2r)` | a name with a subscript above a product |
| | text | `r_s` | `2*r` | `r_s` | `(2*r)` | |
| | unicode | `rₛ` | `2r` | `rₛ` | `(2r)` | |
| | python | `r_s` | `2*r` | | | |
| `half` | latex, text, unicode | `1` | `2` | `1` | `2` | a fraction of two digits, which TeX takes without braces |
| | python | `1` | `2` | | | |

## Oracle

```python
expected = S(top) / S(bottom)
```

## Second route

The fraction times its denominator is its numerator.

```python
assert simplify(expected * S(bottom) - S(top)) == 0
```

## Forms

### LaTeX's fraction

**Form** · `frac` · Lamport (1994), §3.3.1, Fractions: `\frac{numerator}{denominator}`

```latex
\frac{«top»}{«bottom»}
```

### A fraction in display style

**Form** · `dfrac` · the amsmath package, Fractions and related constructions: `\dfrac`

```latex
\dfrac{«top»}{«bottom»}
```

### A fraction in text style

**Form** · `tfrac` · the amsmath package, Fractions and related constructions: `\tfrac`

```latex
\tfrac{«top»}{«bottom»}
```

### A continued fraction's fraction

**Form** · `cfrac` · the amsmath package, Continued fractions: `\cfrac`

```latex
\cfrac{«top»}{«bottom»}
```

### Plain TeX's fraction

**Form** · `over` · Knuth (1984), chapter 17: `{a \over b}`

```latex
{«top» \over «bottom»}
```

### A fraction without braces

**Form** · `frac-digits` · Knuth (1984), chapter 20: an argument without braces is one token, so `\frac12` is one half

```latex
\frac«top»«bottom»
```

**Only of** · `half` · each part is one digit; before a letter, `\frac` needs a space

### A solidus

**Form** · `solidus` · ISO 80000-2:2019, division: `a/b`

```latex
«(top)»/«(bottom)»
```

### A division sign

**Form** · `div` · LaTeX's `\div`, the sign of school arithmetic: `3 \div 4`

```latex
«(top)» \div «(bottom)»
```

### A negative power

**Form** · `negative-power` · the SI Brochure (2019), §5.4.6: `a/b`, `\frac{a}{b}` or `a b^{-1}`

```latex
«(top)» «(bottom)»^{-1}
```

**Only of** · `symbols`, `sums`, `product-below`, `subscript` · written of quantities, not of numbers side by side

### A solidus in plain text

**Form** · `slash` · the solidus of plain text, as every programming language writes division

```text
«(top)»/«(bottom)»
```

### A division sign in Unicode

**Form** · `unicode-div` · the Unicode Standard, U+00F7 DIVISION SIGN

```unicode
«(top)» ÷ «(bottom)»
```

### A fraction slash in Unicode

**Form** · `unicode-fraction-slash` · the Unicode Standard, U+2044 FRACTION SLASH: `1⁄2`

```unicode
«(top)»⁄«(bottom)»
```

### A solidus in Unicode

**Form** · `unicode-solidus` · as typed, with Unicode's minus sign and subscripts

```unicode
«(top)»/«(bottom)»
```

### A vulgar fraction

**Form** · `unicode-vulgar` · the Unicode Standard, U+00BD VULGAR FRACTION ONE HALF

```unicode
½
```

**Only of** · `half` · the character is one half
