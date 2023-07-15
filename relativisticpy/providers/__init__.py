"""
All external dependecies of relativisticpy are interfaces through this provider module (and so should/will any future dependencies).
"""
from sympy import Symbol, Rational, diff, integrate, simplify, tensorproduct, symbols
from sympy import MutableDenseNDimArray as SymbolArray

from relativisticpy.providers.interfaces import IMultiIndexArray, IIdx, IIndices
from relativisticpy.providers.helper_functions import transpose_list