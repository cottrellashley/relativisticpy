### Example 1: Polynomial Function
Define a polynomial function and evaluate it at a specific point.

```python
f(x) := 3*x**3 - 2*x**2 + x - 5
f(2)
```

### Example 2: Exponential and Logarithmic Function
Define an exponential function and a logarithmic function, then evaluate them.

```python
g(x) := e**x - ln(x)
g(1)
```

### Example 3: Trigonometric Function
Define a trigonometric function combining sine and cosine, then evaluate it at a given angle (in radians).

```python
t(x) := sin(x)**2 + cos(x)**2
t(pi/4)
```

### Example 4: Absolute Value and Piecewise Function
Define a function that uses absolute value and evaluate it. Also, show a piecewise function definition and evaluation.

```python
abs_func(x) := |x - 3|
abs_func(-1)

piecewise(x) := { x**2, x > 0; -x, x <= 0 }
piecewise(-2)
piecewise(2)
```

### Example 5: Composition of Functions
Define two functions and then create a composition of those functions.

```python
f(x) := x**2 + 2*x - 1
g(x) := 2*sin(x)
h(x) := f(g(x))
h(pi/2)
```

### Example 6: Parametric Functions
Define functions that depend on a parameter and evaluate them for specific parameter values.

```python
p(x, a) := a*x**2 + a**2*x + 1
p(2, 3)
```

### Example 7: Rational Function
Define a rational function (a ratio of two polynomials) and evaluate it.

```python
r(x) := (x**2 - 4) / (x - 2)
r(3)
```

### Example 8: Factorial and Combinatorial Functions
Define a function involving factorials and evaluate it. Note: This depends on whether your solver supports factorial operations.

```python
factorial_func(n) := n!
factorial_func(5)

comb(n, k) := n! / (k! * (n - k)!)
comb(5, 2)
```

You can test these examples in your mathematical symbolic solver to understand how different types of functions are defined, manipulated, and evaluated.