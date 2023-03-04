import re
import sympy as smp

from src.relativisticpy.tensors.metric import Metric
from src.relativisticpy.tensors.metric import MetricWork
from src.relativisticpy.tensors.riemann import Riemann
from src.relativisticpy.tensors.derivative import Derivative


class Workbook:

    """ Cache for storing values accessed by string keys """
    def __init__(self):
        self.cache = {
            "Schwarzschild" : "[[-(1 - (G)/(r)),0,0,0],[0,1/(1 - (G)/(r)),0,0],[0,0,r**2,0],[0,0,0,r**2*sin(theta)**2]]",
            "SchildGeneral" : "[[-F(r),0,0,0],[0,1/(F(r)),0,0],[0,0,r**2,0],[0,0,0,r**2*sin(theta)**2]]",
            "AntiDeSitter" : "[[-(k**2*r**2 + 1),0,0,0],[0,1/(k**2*r**2 + 1),0,0],[0,0,r**2,0],[0,0,0,r**2*sin(theta)**2]]",
            "PolarCoordinates" : "[[-1,0,0,0],[0,1,0,0],[0,0,r**2,0],[0,0,0,r**2*sin(theta)**2]]",
            "Minkowski" : "[[-1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]",
            "WeylLewisPapapetrou": "[[-1,0,0,0],[0,1,0,0],[0,0,r**2,0],[0,0,0,r**2*sin(theta)**2]]",
            "ReissnerNordström" : "[[-(1 - (G)/(r) + (Q**2)/(r**2)),0,0,0],[0,1/(1 - (G)/(r) + (Q**2)/(r**2)),0,0],[0,0,r**2,0],[0,0,0,r**2*sin(theta)**2]]"
        }


    def cache_equation(self, equation : str):
        """ Add a equation to the cache """
        key, value = equation.replace(' ','').split('=')
        self.cache[key] = value

    def cache_value(self, key, value):
        """ Add a value to the cache """
        self.cache[key] = value

    def get_cached_value(self, key):
        """ Get a value from the cache by key """
        return self.cache.get(key)

    def key_exists(self, key):
        """ Get boolean determining whether key exists """
        return bool(self.get_cached_value(key))

    def clear_cache(self):
        """ Clear the cache """
        self.cache.clear()

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
        return list(args)

    def differentiate(self, *args):
        expr = args[0]
        wrt = MathJSONInterpreter(self.get_operations_with_latest_cache()).interpret(args[1][0])
        return smp.diff(expr, wrt)

    def integrate(self, *args):
        expr = args[0]
        wrt = MathJSONInterpreter(self.get_operations_with_latest_cache()).interpret(args[1][0])
        return smp.integrate(expr, wrt)

    def tensor(self, *args):
        tensor_string_representation = ''.join(args)
        tensor_name = re.match('([a-zA-Z]+)', tensor_string_representation).group()
        tesnor_indices = tensor_string_representation.replace(tensor_name, '')

        if self.key_exists('Metric') and self.key_exists('Basis'):
            self.METRIC  =  Metric(self.get_cached_value('Metric'), "_{a}_{b}", self.get_cached_value('Basis'))
        else:
            raise ValueError('No Metric has been defined')
        if tensor_name == 'G':
            return MetricWork(self.METRIC, tesnor_indices)
        elif tensor_name == 'd':
            return Derivative(self.METRIC.components, tesnor_indices, self.METRIC.basis)
        elif tensor_name == "R":
            return Riemann(self.METRIC, tesnor_indices)

    def minus(self, *args):
        a = args[0]
        return -a

    def get_operations_with_latest_cache(self):
        return {
                    "BUILD_INT"         : self.build_int,
                    "BUILD_FLOAT"       : self.build_float,
                    "ADD"               : self.add,
                    "MULTIPLY"          : self.multiply,
                    "SUBTRACTION"       : self.subtract,
                    "DIVISION"          : self.divide,
                    "POWER"             : self.power,
                    "BUILD_VARIABLE"    : self.variable,
                    "DIFFERENTIAL"      : self.differentiate,
                    "INTEGRAL"          : self.integrate,
                    "BUILD_TENSOR"      : self.tensor,
                    "BUILD_MINUS"       : self.minus
                }

    def variable(self, *args):
        a = ''.join(args)
        if self.key_exists(a):
            return MathJSONInterpreter(self.get_operations_with_latest_cache()).interpret(Mathify(self.get_cached_value(a))())
        else:
            return smp.symbols('{}'.format(a))
        
    def expr(self, expression):
        if bool(re.match('([a-zA-Z]+)(\=)', expression.replace(' ',''))):
            self.cache_equation(expression)
        else:
            return MathJSONInterpreter(self.get_operations_with_latest_cache()).interpret(Mathify(expression)())


