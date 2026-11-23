# Products

The product of the terms of a sequence, finite, \(\prod_{k=1}^{n} a_k\), or infinite,
\(\prod_{n=1}^{\infty} a_n\), the limit of its partial products. DLMF is the NIST *Digital
Library of Mathematical Functions*; Knuth (1984) is *The TeXbook*; Rudin (1987) is *Real and
Complex Analysis*, 3rd edition; AsciiMath is the plain text notation of asciimath.org;
UnicodeMath is the linear format of Unicode Technical Note #28; the Unicode Standard is
version 15.0.

`«f»` is the factor, `«var»` the index, and `«lower»` and `«upper»` its limits.

## Names

```python
k, n, x = symbols("k n x")
big_m = symbols("M")
```

## Operands

| operand | written in | `f` | `var` | `lower` | `upper` | separates |
|---|---|---|---|---|---|---|
| `factorial` | latex, text, unicode | `k` | `k` | `1` | `n` | the product of the index, to a name: n! |
| | python | `k` | `k` | `1` | `n` | |
| `odd` | latex | `(2k - 1)` | `k` | `1` | `5` | a bracket, and a finite product of numbers |
| | text | `(2*k - 1)` | `k` | `1` | `5` | |
| | unicode | `(2k − 1)` | `k` | `1` | `5` | |
| | python | `2*k - 1` | `k` | `1` | `5` | |
| `constant` | latex, text, unicode | `2` | `k` | `1` | `n` | a factor that does not hold the index: a power |
| | python | `2` | `k` | `1` | `n` | |
| `telescoping` | latex | `\frac{k}{k + 1}` | `k` | `1` | `n` | a quotient whose factors cancel, to 1/(n + 1) |
| | text, unicode | `k/(k + 1)` | `k` | `1` | `n` | |
| | python | `k/(k + 1)` | `k` | `1` | `n` | |
| `power` | latex, text | `x^k` | `k` | `1` | `n` | a parameter raised to the index: a product over k alone |
| | unicode | `xᵏ` | `k` | `1` | `n` | |
| | python | `x**k` | `k` | `1` | `n` | |
| `reciprocal-squares` | latex | `\left(1 - \frac{1}{n^2}\right)` | `n` | `2` | `\infty` | an infinite product from 2, with a rational value |
| | text | `(1 - 1/n^2)` | `n` | `2` | `oo` | |
| | unicode | `(1 − 1/n²)` | `n` | `2` | `∞` | |
| | python | `1 - 1/n**2` | `n` | `2` | `oo` | |
| `wallis` | latex | `\frac{4n^2}{4n^2 - 1}` | `n` | `1` | `\infty` | Wallis's product, of rational factors, whose value is π/2 |
| | text | `4*n^2/(4*n^2 - 1)` | `n` | `1` | `oo` | |
| | unicode | `4n²/(4n² − 1)` | `n` | `1` | `∞` | |
| | python | `4*n**2/(4*n**2 - 1)` | `n` | `1` | `oo` | |

## Oracle

An infinite product is the limit of its partial products.

```python
f = S(f)
if S(upper).is_infinite:
    expected = limit(product(f, (var, lower, big_m)), big_m, oo)
else:
    expected = product(f, (var, lower, upper))
```

## Second route

Factor by factor: a finite product multiplied out, at n = 6 where its limit is n; an
infinite one, numerically, to a hundred thousand factors.

```python
if S(upper).is_infinite:
    factor = lambdify(var, f)
    partial = 1.0
    for j in range(int(lower), 10**5):
        partial *= factor(j)
    assert abs(partial - float(expected)) < 10**-4
else:
    at = {n: 6, x: Rational(7, 10)}
    top = int(S(upper).subs(at))
    multiplied = Mul(*[f.subs(at).subs(var, j) for j in range(int(lower), top + 1)])
    assert simplify(expected.subs(at) - multiplied) == 0
```

## Forms

### The product

**Form** · `prod` · DLMF, §5.8, Infinite Products: `\prod_{k=1}^{\infty}`

```latex
\prod_{«var»=«lower»}^{«upper»} «f»
```

### An upper limit without braces

**Form** · `one-token-limits` · Knuth (1984), chapter 16: a superscript without braces is one character or command, as in `\prod_{k=1}^n`

```latex
\prod_{«var»=«lower»}^«upper» «f»
```

### The limits below and above in a line of text

**Form** · `limits` · Knuth (1984), chapter 17: `\limits` sets an operator's limits below and above it

```latex
\prod\limits_{«var»=«lower»}^{«upper»} «f»
```

### In display style

**Form** · `displaystyle` · Knuth (1984), chapter 17: `\displaystyle`, as a line of text sets a product as a display does

```latex
\displaystyle\prod_{«var»=«lower»}^{«upper»} «f»
```

### Of factors the program defines

**Form** · `factors` · Rudin (1987), chapter 15, Infinite products: `\prod_{n=1}^{\infty} (1 + u_n)`, of factors named first

```latex
a_{«var»} := «f»
\prod_{«var»=«lower»}^{«upper»} a_{«var»}
```

**Not of** · `constant` · a factor without the index is not named by it

### A call of product

**Form** · `product` · SymPy's `product(k, (k, 1, n))`

```text
product(«f», («var», «lower», «upper»))
```

### The product in plain text

**Form** · `asciimath` · AsciiMath: `prod`, the product sign, with its limits as `sum_(i=1)^n` writes them

```text
prod_(«var»=«lower»)^«upper» «f»
```

### The sign in Unicode

**Form** · `unicode-sign` · docs/reference.md, Unicode: ∏, U+220F N-ARY PRODUCT, is `\prod`

```unicode
∏_{«var»=«lower»}^{«upper»} «f»
```

### The product in UnicodeMath

**Form** · `unicodemath` · UnicodeMath, Unicode Technical Note #28: `∏_(n=1)^∞`, the limits in round brackets

```unicode
∏_(«var»=«lower»)^«upper» «f»
```
