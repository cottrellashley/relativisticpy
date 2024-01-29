from sympy import MutableDenseNDimArray

### THE ONLY REASON THIS EXISTS IS BECAUSE OF Mojo Language, I wanted to reduce refactoring if we 

def SymbolArray(*args, **kwargs) -> MutableDenseNDimArray: return MutableDenseNDimArray(*args, **kwargs)