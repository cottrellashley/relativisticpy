from jsonmathpy import MathJSONInterpreter
from jsonmathpy import Mathify
import sympy as smp

class Sympify:

    def operations(self):
        return {
                    # Build Objects
                    "BUILD_INT"         : self.build_int,
                    "BUILD_FLOAT"       : self.build_float,
                    "BUILD_FUNCTION"    : self.build_function,
                    "FUNCTION"          : self.function,
                    "BUILD_MINUS"       : self.minus,
                    "BUILD_VARIABLE"    : self.variable,

                    # Simple Operations
                    "ADD"               : self.add,
                    "MULTIPLY"          : self.multiply,
                    "SUBTRACTION"       : self.subtract,
                    "DIVISION"          : self.divide,
                    "POWER"             : self.power,
                    "ARRAY"             : self.array,
                    "CONSTANT"          : self.constant,
                    "EXP"               : self.exp,

                    # Simpy Operations
                    "DIFFERENTIAL"      : self.differentiate,
                    "FUNCTION"          : self.function,
                    "INTEGRAL"          : self.integrate,
                    "SIMPLIFY"          : self.simplify,
                    "ARRAY"             : self.array,
                    "SOLVE"             : self.solve,
                    "SERIES"            : self.series,
                    "LIMIT"             : self.limit,
                    "EXPAND"            : self.expand,
                    "EXP"               : self.exp,
                    "CONSTANT"          : self.constant,
                    "NUMERICAL"         : self.numerical,

                    # Trigonometry Functions
                    "SIN"               : self.sin,
                    "COS"               : self.cos,
                    "TAN"               : self.tan,
                    "ASIN"              : self.asin,
                    "ACOS"              : self.acos,
                    "ATAN"              : self.atan,

                    # Hyperbolic Functions 
                    "SINH"               : self.sinh,
                    "COSH"               : self.cosh,
                    "TANH"               : self.tanh,
                    "ASINH"              : self.asinh,
                    "ACOSH"              : self.acosh,
                    "ATANH"              : self.atanh,
                }

    def parse(self, expression: str):
        ans = MathJSONInterpreter(self.operations()).interpret(Mathify(expression)())
        return ans

    def series(self, *args):
        expr = args[0]
        if len(args[1]) == 2:
            point = MathJSONInterpreter(self.operations()).interpret(args[1][0])
            n = MathJSONInterpreter(self.operations()).interpret(args[1][1])
            return smp.series(expr, x0 = point, n = n)
        elif len(args[1]) == 1:
            point = MathJSONInterpreter(self.operations()).interpret(args[1][0])
            return smp.series(expr, x0 = point, n = 5)
        else:
            return smp.series(expr, x0 = 0, n = 5)

    def limit(self, *args):
        expr = args[0]
        x0 = MathJSONInterpreter(self.operations()).interpret(args[1][0])
        x1 = MathJSONInterpreter(self.operations()).interpret(args[1][1])
        return smp.limit(expr, x0, x1)

    def expand(self, *args):
        return smp.expand(args[0])

    def differentiate(self, *args):
        expr = args[0]
        wrt = MathJSONInterpreter(self.operations()).interpret(args[1][0])
        return smp.diff(expr, wrt)

    def integrate(self, *args):
        expr = args[0]
        wrt = MathJSONInterpreter(self.operations()).interpret(args[1][0])
        return smp.integrate(expr, wrt)

    def simplify(self, *args):
        expr = args
        return smp.simplify(expr[0])

    def solve(self, *args):
        expr = args
        return smp.solve(expr[0])

    def numerical(self, *args):
        wrt = MathJSONInterpreter(self.operations()).interpret(args[1][0])
        return smp.N(args[0], wrt)

    def add(self, *args):
        return args[0] + args[1]

    def subtract(self, *args):
        return args[0] - args[1]

    def multiply(self, *args):
        return args[0] * args[1]

    def power(self, *args):
        return args[0] ** args[1]

    def divide(self, *args):
        return args[0] / args[1]

    def build_int(self, *args):
        return int(''.join(args))

    def build_float(self, *args):
        return float(''.join(args))

    def array(self, *args):
        return smp.MutableDenseNDimArray(list(args))

    def minus(self, *args):
        a = args[0]
        return -a

    def exp(self, *args):
        return smp.exp(args[0])

    def constant(self, *args):
        if args[0] == 'pi':
            return smp.pi
        if args[0] == 'e':
            return smp.E

    def variable(self, *args):
        a = ''.join(args)
        return smp.symbols('{}'.format(a))

    def build_function(self, *args):
        a = ''.join(args)
        if a == 'ln':
            return smp.ln
        return smp.symbols('{}'.format(a), cls=smp.Function)

    def function(self, *args):
        expr = args[0]
        wrt = [MathJSONInterpreter(self.operations()).interpret(args[1][i]) for i in range(len(args[1]))]
        return expr(*wrt)
    
    def sin(self, *args):
        return smp.sin(args[0])

    def cos(self, *args):
        return smp.cos(args[0])

    def tan(self, *args):
        return smp.tan(args[0])

    def asin(self, *args):
        return smp.asin(args[0])

    def acos(self, *args):
        return smp.acos(args[0])

    def atan(self, *args):
        return smp.atan(args[0])

    def sinh(self, *args):
        return smp.sinh(args[0])

    def cosh(self, *args):
        return smp.cosh(args[0])

    def tanh(self, *args):
        return smp.tanh(args[0])

    def asinh(self, *args):
        return smp.asinh(args[0])

    def acosh(self, *args):
        return smp.acosh(args[0])

    def atanh(self, *args):
        return smp.atanh(args[0])




