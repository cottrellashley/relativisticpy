
# RelativisticPy
<p align="center">
  <img src="./assets/logo.gif" alt="RelativisticPy" width="300">
</p>

RelativisticPy - Genaral Relativity for physicists in a hurry. Simple equation looking User Interface. Write and Go! 
It is not the most complex, conprehensive or fast symbolic calculator, but it is a symbolic calculator.

[PyPi](https://pypi.org/project/relativisticpy/)
[Source](https://github.com/cottrellashley/relativisticpy)


# TL;DR - Features & Installation: 

#### `install python environment`
```
pip install relativisticpy
```

## (Black Hole Solution in < 10 lines)


#### `Schild_solution.py` file:
``` 
import relativisticpy as rel

wb = rel.Workbook()

wb.expr(
'''
Coordinates := [t, r, theta, phi]

g_{mu}_{nu} := [ 
                 [-A(r),0,0,0], 
                 [0,B(r),0,0], 
                 [0,0,r**2,0], 
                 [0,0,0,r**2*sin(theta)**2]
               ]


# Now we have defined the metric above, we can call any individual component of the Ricci tensor itself (as it is metric dependent)
eq0 = Ric_{mu:0}_{nu:0}
eq1 = Ric_{mu:1}_{nu:1}
eq2 = Ric_{mu:2}_{nu:2}

eq5 = (eq0*B(r) + eq1*A(r))*(r*B(r))

B = RHS( dsolve(eq5, B(r)) )

eq6 = simplify( subs(eq2, B(r), B) )

A = RHS( dsolve(eq6, A(r)) )

g_{mu}_{nu} := [
                 [A,0,0,0], 
                 [0,1/A,0,0], 
                 [0,0,r**2,0], 
                 [0,0,0,r**2*sin(theta)**2]
               ]

# Step 5: We prove that C_1 and C_2 equations are in terms of c, G, M by comparing with Newton at large radius

a = C^{a:1}_{b:0 c:0}
solve( a*c**2 + G*M/r**2 ) # This shows us what 

              
g_{mu}_{nu} := [
                  [-(1 - (2 * G * M) / (c**2*r)), 0, 0, 0],
                  [0, 1 / (1 - (2 * G * M) / (c**2*r)), 0, 0],
                  [0, 0, r**2, 0],
                  [0, 0, 0, r**2 * sin(theta) ** 2]
              ]
              
Gamma^{a}_{c f} := (1/2)*g^{a b}*(d_{c}*g_{b f} + d_{f}*g_{b c} - d_{b}*g_{c f})

Riemann^{a}_{m b n} := d_{b}*Gamma^{a}_{n m} + Gamma^{a}_{b l}*Gamma^{l}_{n m} - d_{n}*Gamma^{a}_{b m} - Gamma^{a}_{n l}*Gamma^{l}_{b m}

Ricci_{m n} := Riemann^{a}_{m a n}

T1^{a f h i} := g^{i d}*(g^{h c}*(g^{f b}*Riemann^{a}_{b c d}))
T2_{a f h i} := g_{a n}*Riemann^{n}_{f h i}

S =  T1^{a f h i}*T2_{a f h i}
              
tsimplify( S )
''') # Will parse string and compute the equations

```


This Python package is designed to assist in performing mathematical operations, particularly in the field of General Relativity. It includes a variety of tools for working with symbolic expressions, including a workflow module that allows users to create linear mathematical workflows and solve tensor expressions.

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

## Package Philosophy

1. Ease of use. The main reason for this package's existance is to make Einstein tensor equations look exactly like they do in mathematical papers, but with a computational engine which can actually compute those symbolic tensor equations.

2. Keep it very lightweight. The packge is meant to have all the main functionality of a mathematical symbolic compotational tool, but yet not big and cluncky. We leave it to Sympy to do all the symbolic calculation work for us.

To put it simply, keep it simple for devs and users.

## Directory Breakdown

| Directory            | Description |
|----------------------|-------------|
| `core`               | Core Module. Contains logic for einstein summation convention of multi-indexed array like objects and descerialisation logic of tensors. |
| `gr`                 | Defines all the main General Relativity Tensors, such as initialization logic and interation logic. Inherits core module logic (which provides the logic of einsum tensor manipulations.) |
| `symengine`           | Interface module for symbolic engine dependencies of relativisticpy, which currently are: Sympy. Any module within RelativisticPy only imports relativisticpy.symengine module, they should now about what is implementing the methods, whether it is Sympy or another symbolic module we choose to swapt Sympy with in future. |
| `workbook` | This modules is the module which 'brings it all together' if you will. It here to allow non-python users to use the RelativisticPy package. It handles all the parsing from strings, the object initializations, memory storage and workflow for the end-user.        |
| `parser` | String -> Lexer(String) = Tokens -> Parser(Tokens) = AST -> SemanticAnalyzer(AST) = ActionTree -> Interpreter(ActionTree, ImplementerClass) = Result.  ImplementerClass dependent on whole package i.e. uses all Tensor implementations + Sympy + Worrkbook State -> Computes the language and returns answer.   |
| `ft` | Future module -> Field Theory       |

### Package Tree-Structure

TODO

## Project Dependencies

- Sympy: This package has a large dependency in Sympy to perform all it's symbolic calculations and functions.

# License

This package is licensed. See the LICENSE file for more information.

# Contributing

If you would like to contribute to the development of this package, please feel free to fork the repository and submit a pull request. All contributions are welcome!
