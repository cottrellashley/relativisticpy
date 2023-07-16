"""
All external dependecies of relativisticpy are interfaces through this provider module (and so should/will any future dependencies).
"""

# Note on swapping implementations: simple methods such as zeros and Rational are much simpler to swap if needed. 
# So if needed we can just implement zeros or Rational ourselfves and swap the import here. However, many sympy objects are interelated. 
# Meaning if we are to swap the implementation of one, we must swap the implementation of the rest of the objects and mimmic the way in which, 
# sympy has implemented the interface. This is still quite some work but it is none the less limited and simpler than the alternative.

# Sympy Dependencies: By importing all the sympy functionality via this file, we can controll what it used and can also swap out implementations in future.
from sympy import Symbol, Rational, diff, integrate, simplify, tensorproduct, symbols, zeros
from sympy import MutableDenseNDimArray as SymbolArray

from relativisticpy.providers.interfaces import IMultiIndexArray, IIdx, IIndices
from relativisticpy.providers.helper_functions import transpose_list
from relativisticpy.providers.jsonmathpy.sympify import Sympify # Sympify: string => parser => sympy objects