from sympy import Symbol, MutableDenseNDimArray, diff, integrate

class ISymCalc:
    def __init__(self, pkg='sympy'):
        self.pkg = pkg

    def diff(self, *args, **kwargs):
        if ISymCalc.pkg == 'sympy':
            return diff(*args, **kwargs)
        else:
            raise NotImplementedError('Differentiation not implemented for this package')

    def integrate(self, *args, **kwargs):
        if ISymCalc.pkg == 'sympy':
            return integrate(*args, **kwargs)
        else:
            raise NotImplementedError('Integration not implemented for this package')

    def symbol(self, *args, **kwargs):
        if ISymCalc.pkg == 'sympy':
            return Symbol(*args, **kwargs)
        else:
            raise NotImplementedError('Symbol not implemented for this package')

    def array(self, *args, **kwargs):
        if ISymCalc.pkg == 'sympy':
            return MutableDenseNDimArray(*args, **kwargs)
        else:
            raise NotImplementedError('Array not implemented for this package')

    def zeros_array(self, *args):
        if ISymCalc.pkg == 'sympy':
            return MutableDenseNDimArray.zeros(*args)
        else:
            raise NotImplementedError('Zeros Array not implemented for this package')