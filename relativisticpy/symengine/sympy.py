from sympy import MutableDenseNDimArray, Pow, Rational

### THE ONLY REASON THIS EXISTS IS BECAUSE OF Mojo Language, I wanted to reduce refactoring if we 

def SymbolArray(*args, **kwargs) -> MutableDenseNDimArray: return MutableDenseNDimArray(*args, **kwargs)

def root(arg, k: int, evaluate = None): return Pow(arg, Rational(1, k), evaluate=evaluate)