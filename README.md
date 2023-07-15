
# RelativisticPy
![RelativisticPy](./assets/logo.png)

[PyPi](https://pypi.org/project/relativisticpy/)

This Python package is designed to assist in performing mathematical operations, particularly in the field of General Relativity. It includes a variety of tools for working with symbolic expressions, including a workflow module that allows users to create linear mathematical workflows and solve tensor expressions.

| Directory            | Description |
|----------------------|-------------|
| `core`               | Defined data structures (single-index, multi-index, tensor) 
coupled with index rules,such that tensors can be multiplied together as defined by einstein summation convention.|
| `data_objects`       | Data        |
| `workbook`           | Data        |
| `general_relativity` | Data        |
| `descerializers`     | Data        |

## Philosophy

## Features

Some of the key features of this package include:

Symbolic expression manipulation: Perform symbolic calculations and manipulations with ease.
Workflow module: Create linear mathematical workflows that allow you to assign variables and use them later on.

## General Relativity functionality: 

- Solve tensor expressions in General Relativity, such as "G_{mu}{nu} * R^{nu}{a}{b}{c}".
- Mathematical operations: Perform mathematical operations such as differentiation, integration, and simplification of expressions.

## Dependencies

- Sympy: This package has a large dependency in Sympy to perform all it's symbolic calculations and functions.
- JsonMathPy: This package depends on the [JsonMathPy](https://pypi.org/project/jsonmathpy/) package ([GitHub](https://github.com/cottrellashley/jsonmathpy)), which is a Python package that can take a string as input and parse it into a Python dictionary (and, optionally, a JSON math file). The package can parse and evaluate expressions using the operations and objects injected by the user.

## Installation

To install the package and its dependencies, simply use pip:

```
pip install relativisticpy
```

Once installed, the package can be used in your Python code by importing it:

```
import relativisticpy as rel
```

From there, you can use the various modules and functions provided by the package. For example, to perform symbolic calculations:

```
w = rel.Workbook()
```

Which you can now use `w.expr()` to enter new expressions.

# Feature Summary

- **Variable Assignment:** Assign expressions to variables using equals sign, to use later on in your workflow.
- **Integration:** Integrate expressions with respect to specified variables.
- **Differentiation:** Differentiate expressions with respect to specified variables.
- **Solve:** Solve a algebraic equation, or set of algebraic equations with repect to set of variables.
- **Simplify:** Simplify a specified mathematical expression.
- **Expand:** Expand expression raised to some power.
- **Substitute:** Substitute a variable within an expression (if not already assigned.)
- **Sum:** Find the descrete sum of an expression.
- **Series:** Find the taylor series of some function/expr.
- **Factor:** Factor an expression if possible.
- **Limit:** Find the limit of an expression if possible.
- **General Relativity Tensor Operations:**
    - Define Metric and Basis using the assignment operation.
    - Write Tensor Expressions using indices to represent the summation of the components (Einstein summation convention.)
    - Get and Assign Specific Components from Tensors: Choice to assign indices to numbers, to return subset of components. And also assign those to other variables.
)
## Examples

### 1. Variable Assignment
The variable assignment function allows you to assign a value or expression to a variable and use it in subsequent calculations.


```
y = x**2 + 5*x + 9
```

We can now use this variable *y* at any point later in the notebook and use it as a replacement for the expression above.

```
integrate( y , x )
```

### 2. Integration
The integrate function allows you to perform integration of an expression with respect to the specified variables.

```
integrate( x**4 + x - 8 , x )
```

### 3. Differentiation
The diff function allows you to perform differentiation of an expression with respect to the specified variables.

```
diff(sin(x) + exp(x**2) + x**2, x)
```

### 4. Solve
The solve function allows you to solve an equation or system of equations.

```
solve(x**2 + 4*x + 4, x)
```

### 5. Simplify
The simplify function allows you to simplify a given expression by applying algebraic transformations.

```
simplify( x**2 + 4*x + 4 )
```

### 6. Expand
The expand function allows you to expand a given expression by multiplying out parentheses and combining like terms.

```
expand( (x + 5)**3 )
```

### 7. Series
The series function allows you to compute the power series expansion of a given expression.

```
series(sin(x), x, 1, 5)
```

### 8. Factor
The factor function allows you to factor a given expression by finding its prime factors.

```
factor( x**2 + 4*x + 4 )
```


### 8. Limit
The limit function allows you to compute the limit of a given expression as a variable approaches a specified value.

```
limit(1/x**2, x, 10)
```

# General Relativity Tensor Operations

The General Relativity Tensor Operations allow you to perform tensor calculations on a defined metric and basis.

### 1. Define Metric and Basis

The define metric and basis function allows you to define a metric and basis for use in tensor calculations.


```
Metric([[g00, ....], [g10, ....], ... , [..., gNN]], [var1, ...., varN])
```

Now you have defined the metric components as `[[g00, ....], [g10, ....], ... , [..., gNN]]` with the basis coordinates as `[var1, ...., varN]` and we can now write Tensor Expression!

The write tensor expression function allows you to write a tensor expression using the defined metric and basis.


```
G_{mu}_{nu}
```

This will output the components you have entered:

```
[[g00, ....], [g10, ....], ... , [..., gNN]]
```

Get and Assign Specific Components from Tensors

The get and assign specific components from tensors function allows you to get or assign a specific component of a tensor using the defined metric and basis.

```
Input: G_{mu:1}_{nu:1}

Output: The (1,1) component of the metric components
```

This also works for any tensor and their assignment of individual components:

```
R_{nu:1}_{b:2}_{mu}_{d}
R_{nu:1}_{b}_{mu}_{d}
R_{nu:1}_{b:2}_{mu:1}_{d:2}
```

Assign the (1, 1) component of a tensor to a variable `Comp`:

```
Comp = G_{mu:1}_{nu:1}
```

The tensor expressions function allows you to perform tensor expressions using the defined metric and basis.

```
G^{mu}^{nu}*R_{nu}_{b}_{c}_{d}
R^{a}_{b}_{a}_{d}
G^{mu}^{nu}*R_{nu}_{b}_{mu}_{d}
```

This app is a calculator for various relativistic equations. To use it, simply enter the equation you would like to solve or manipulate, and select the function you want to perform.

### Equations

Equations can be entered using standard Python syntax. For example, to solve the equation `x**2 - 3*x + 2 = 0`, you would enter `x**2 - 3*x + 2` in the equation field.

### Furture Features

- substitute( var_from, var_to , expression)
- square root
- factor ---> factor an expression
- linsolve ---> solve a set of linear equations
- Eq --> define an equation

### Furture Calculator Features 

- substitute( var_from, var_to , expression)
- square root
- factor ---> factor an expression
- linsolve ---> solve a set of linear equations
- Eq --> define an equation
- sum ----> descrete summation of a function

### Future User interface Features

Each cell has following options:
- Latex --> Outputs latex
- Mardown ---> Outputs Markdown
- RelativisticPy ----> Outputs / Calculates eqautions
- Open AI ----> Sends API calls to AI

At the very top of page have a button:
-------> Export Notebook <--------

This will take all the input cells and outut a document pdf etc with just the concatination of the outputs. 
Before it creates the file, it will ask you which output cells you wish to output!!

# License

This package is licensed. See the LICENSE file for more information.

# Contributing

If you would like to contribute to the development of this package, please feel free to fork the repository and submit a pull request. All contributions are welcome!
