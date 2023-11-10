# RelativisticPy Package README Improvement

## Upcoming Features: 
1. Implementation of the Einstein Tensor.
2. Integration of a setter function for defining tensor expressions such as T_{a}_{b} = [custom tensor expression with matching indices].
3. Integration of an intelligent well defined rules for a dynamic derivative, which uses defined coordinates or user defined.
4. Metric defined via line element: this will require big edit to the parser grammar => multiple different grammars.
5. Clean up Index and Indices properties handling/lifecycle -> (basis, shape, dimention, values, etc...)



## Quick Overview - (Compute Black Hole Solutions in Less Than 10 Lines)
### NOTE: Package is in Pre-release Stage

### Setting Up Your Environment
To prepare for using the RelativisticPy package, install it via pip:
```
pip install relativisticpy
```

### Working with the `Schild_solution.txt` Script:
This file outlines a step-by-step approach to solving a spherically symmetric metric in General Relativity.

``` 
# Step 1: Establish foundational elements
# Declare symbols for the Metric and Ricci tensors
MetricSymbol := G 
RicciSymbol := Ric 
Coordinates := [t, r, theta, phi] 

# Step 2: Define the spherically symmetric metric tensor components
G_{mu}_{nu} := [[-A(r),0,0,0], [0,B(r),0,0], [0,0,r**2,0], [0,0,0,r**2*sin(theta)**2]] 

# Step 3: Access Ricci tensor components derived from the metric
eq0, eq1, eq2 = Ric_{mu:0}_{nu:0}, Ric_{mu:1}_{nu:1}, Ric_{mu:2}_{nu:2} 

# Step 4: Formulate a simplified Ricci component equation
eq5 = (eq0*B(r) + eq1*A(r))*(r*B(r)) 

# Step 5: Solve the differential equation for B(r)
solutionB = dsolve(eq5, B(r)) # Solution: B(r) = C1/A(r)

# Step 6: Refine the solution by substituting B(r) into the remaining equation
eq6 = simplify(subs(eq2, B(r), C1/A(r)))

# Step 7: Determine the functions A(r) and B(r)
A = dsolve(eq6, A(r)) 
B = 1/A

# Optional: Export the solution in LaTeX format
latex([[-A,0,0,0], [0,B,0,0], [0,0,r**2,0], [0,0,0,r**2*sin(theta)**2]]) 

```

### Utilizing the `main.py` Script:
Execute the workflow script within a Python environment using the provided `Workbook` module.

```
from relativisticpy import Workbook

def main():
    wb = Workbook()
    wb.execute('<file-path>/Schild_solution.txt')

if __name__ == '__main__':
    main()
```

The RelativisticPy package serves as a sophisticated toolkit for mathematicians and physicists, streamlining complex calculations in General Relativity. It offers an extensive suite of functionalities for manipulating symbolic expressions, complemented by a workflow module to execute linear mathematical procedures and resolve intricate tensor equations seamlessly.

# Features

Some of the key features of this package include:

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

## General Relativity functionality: 

- Solve tensor expressions in General Relativity, such as "G_{mu}{nu} * R^{nu}{a}{b}{c}".
- Mathematical operations: Perform mathematical operations such as differentiation, integration, and simplification of expressions.

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

## Package Philosophy

1. Ease of use. The main reason for this package's existance is to make Einstein tensor equations look exactly like they do in mathematical papers, but with a computational engine which can actually compute those symbolic tensor equations.

2. Keep it very lightweight. The packge is meant to have all the main functionality of a mathematical symbolic compotational tool, but yet not big and cluncky. We leave it to Sympy to do all the symbolic calculation work for us.

To put it simply, keep it simple for devs and users.

## Directory Breakdown

| Directory            | Description |
|----------------------|-------------|
| `core`               | Core Module. Contains logic for einstein summation convention of multi-indexed array like objects and descerialisation logic of tensors. |
| `gr`                 | Defines all the main General Relativity Tensors, such as initialization logic and interation logic. Inherits core module logic (which provides the logic of einsum tensor manipulations.) |
| `providers`           | Contains all the external package dependencies of relativisticpy, which currently are: Sympy. Any module within RelativisticPy only imports relativisticpy.provider module, they should now about what is implementing the methods, whether it is Sympy or another symbolic module we choose to swapt Sympy with in future. |
| `workbook` | This modules is the module which 'brings it all together' if you will. It here to allow non-python users to use the RelativisticPy package. It handles all the parsing from strings, the object initializations, memory storage and workflow for the end-user.        |
| `em` | Future module -> Electromagnetism   |
| `ft` | Future module -> Field Theory       |

### Package Tree-Structure

```
core (!Rough downward dependency!)
|
|-- Indices
|   |
|   |-- Tuple[ Idx ]
|
|-- Einstein Summation Convention Decorator
|
|-- MultiIndexObject
|
|-- Metric
|
|------ gr
|       |
|       |-- GeometricObject -> Metric Dependent
|       |   |
|       |   |-- Connection
|       |   |
|       |   |-- Riemann
|       |   |
|       |   |-- Ricci
|       |   |
|       |   |-- CovariantDerivative
|       |
|       |-- PhysicalObject -> User/State Defined
|           |
|           |-- StressEnergyTensor
|           |
|           |-- ElectromagneticTensor
|    
|------ em (TODO: Electromagnetism Module)
|
|------ ft (TODO: Field Theory Module)
|
|-- Descerializer
    |
    |-- String -> Any Tensor Objects Above
```




## Project Dependencies

- Sympy: This package has a large dependency in Sympy to perform all it's symbolic calculations and functions.

## Use Examples

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
