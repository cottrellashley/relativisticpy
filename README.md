![RelativisticPy](./assets/logo.png)

# RelativisticPy

[PyPi](https://pypi.org/project/relativisticpy/)

This Python package is designed to assist in performing mathematical operations, particularly in the field of General Relativity. It includes a variety of tools for working with symbolic expressions, including a workflow module that allows users to create linear mathematical workflows and solve tensor expressions.

## Features

Some of the key features of this package include:

Symbolic expression manipulation: Perform symbolic calculations and manipulations with ease.
Workflow module: Create linear mathematical workflows that allow you to assign variables and use them later on.

## General Relativity functionality: 

- Solve tensor expressions in General Relativity, such as "G_{mu}{nu} * R^{nu}{a}{b}{c}".
- Mathematical operations: Perform mathematical operations such as differentiation, integration, and simplification of expressions.

## Dependencies

This package depends on the [JsonMathPy](https://pypi.org/project/jsonmathpy/) package ([GitHub](https://github.com/cottrellashley/jsonmathpy)), which is a Python package that can take a string as input and parse it into a Python dictionary (and, optionally, a JSON math file). The package can parse and evaluate expressions using the operations and objects injected by the user.

## Installation

To install the package and its dependencies, simply use pip:

```
pip install relativisticpy
```

## Usage

Once installed, the package can be used in your Python code by importing it:

python

```
import relativisticpy as rel
```

From there, you can use the various modules and functions provided by the package. For example, to perform symbolic calculations:

```
w = rel.Workbook()
w.expr('f = r**2 + 3*r +10')
w.expr('diff(f, r)')
```

Output:

```
2 𝑟 + 3
```

For more information on how to use the package, please refer to the documentation.

## License

This package is licensed under the MIT license. See the LICENSE file for more information.

# Contributing

If you would like to contribute to the development of this package, please feel free to fork the repository and submit a pull request. All contributions are welcome!
