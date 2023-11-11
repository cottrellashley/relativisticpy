# Sympy Dependencies: By importing all the sympy functionality via this file, we can controll what it used and can also swap out implementations in future.
from sympy import Symbol, Rational, diff, integrate, simplify, tensorproduct, symbols, zeros, permutedims, sin, cos, tan, cosh, tanh, sinh
from sympy import MutableDenseNDimArray as SymbolArray
# Implement `function` - `constant` - `infinity` 