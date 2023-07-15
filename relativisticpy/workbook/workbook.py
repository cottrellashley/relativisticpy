# from cmath import exp
# from dataclasses import dataclass
# import re
# import sympy as smp
# from relativisticpy.shared.functions import sympify
# from relativisticpy.tensors.core.tensor import TensorObject
# from relativisticpy.tensors.core.tensor import GrTensor
# from relativisticpy.tensors.metric import Metric
# from relativisticpy.tensors.riemann import Riemann
# from relativisticpy.tensors.derivative import Derivative
# from relativisticpy.deserializers.transformation_deserializer import TransformationDeserializer

# @dataclass
# class ChacheItem:
#     name: str
#     string_obj: str
#     decesrialized_obj: any
#     mathjson_obj: dict = None


#     def __hash__(self):
#         return hash((self.name, str(self.mathjson_obj), self.string_obj, str(self.decesrialized_obj)))

#     def __eq__(self, other):
#         if isinstance(other, ChacheItem):
#             return (self.name == other.name and
#                     self.mathjson_obj == other.mathjson_obj and
#                     self.string_obj == other.string_obj and
#                     self.decesrialized_obj == other.decesrialized_obj)
#         return False

# class Workbook:

#     """ Cache for storing values accessed by string keys """
#     def __init__(self):
#         schild = ChacheItem(
#             "Schwarzschild", 
#             sympify("[[-(1 - (G)/(r)),0,0,0],[0,1/(1 - (G)/(r)),0,0],[0,0,r**2,0],[0,0,0,r**2*sin(theta)**2]]"), 
#             "[[-(1 - (G)/(r)),0,0,0],[0,1/(1 - (G)/(r)),0,0],[0,0,r**2,0],[0,0,0,r**2*sin(theta)**2]]",
#             Metric("[[-(1 - (G)/(r)),0,0,0],[0,1/(1 - (G)/(r)),0,0],[0,0,r**2,0],[0,0,0,r**2*sin(theta)**2]]",'_{mu}_{nu}', '[t, r, theta, phi]')
#             )
#         schild_general = ChacheItem(
#             "SchildGeneral", 
#             sympify("[[-F(r),0,0,0],[0,1/(F(r)),0,0],[0,0,r**2,0],[0,0,0,r**2*sin(theta)**2]]"), 
#             "[[-F(r),0,0,0],[0,1/(F(r)),0,0],[0,0,r**2,0],[0,0,0,r**2*sin(theta)**2]]",
#             Metric("[[-F(r),0,0,0],[0,1/(F(r)),0,0],[0,0,r**2,0],[0,0,0,r**2*sin(theta)**2]]",'_{mu}_{nu}', '[t, r, theta, phi]')
#             )
#         AntiDeSitter = ChacheItem(
#             "AntiDeSitter", 
#             sympify("[[-(k**2*r**2 + 1),0,0,0],[0,1/(k**2*r**2 + 1),0,0],[0,0,r**2,0],[0,0,0,r**2*sin(theta)**2]]"), 
#             "[[-(k**2*r**2 + 1),0,0,0],[0,1/(k**2*r**2 + 1),0,0],[0,0,r**2,0],[0,0,0,r**2*sin(theta)**2]]",
#             Metric("[[-(k**2*r**2 + 1),0,0,0],[0,1/(k**2*r**2 + 1),0,0],[0,0,r**2,0],[0,0,0,r**2*sin(theta)**2]]",'_{mu}_{nu}', '[t, r, theta, phi]')
#             )
#         PolarCoordinates = ChacheItem(
#             "PolarCoordinates", 
#             sympify("[[-1,0,0,0],[0,1,0,0],[0,0,r**2,0],[0,0,0,r**2*sin(theta)**2]]"), 
#             "[[-1,0,0,0],[0,1,0,0],[0,0,r**2,0],[0,0,0,r**2*sin(theta)**2]]",
#             Metric("[[-1,0,0,0],[0,1,0,0],[0,0,r**2,0],[0,0,0,r**2*sin(theta)**2]]", '_{mu}_{nu}', '[t, r, theta, phi]')
#             )
#         Minkowski = ChacheItem(
#             "Minkowski", 
#             sympify("[[-1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]"), 
#             "[[-1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]",
#             Metric("[[-1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]",'_{mu}_{nu}', '[t, r, theta, phi]')
#             )
#         Minkowski = ChacheItem(
#             "Minkowski", 
#             sympify("[[-1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]"), 
#             "[[-1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]",
#             Metric("[[-1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]", '_{mu}_{nu}', '[t, x, y, z]')
#             )
#         ReissnerNordstrom = ChacheItem(
#             "ReissnerNordstrom", 
#             sympify("[[-(1 - (G)/(r) + (Q**2)/(r**2)),0,0,0],[0,1/(1 - (G)/(r) + (Q**2)/(r**2)),0,0],[0,0,r**2,0],[0,0,0,r**2*sin(theta)**2]]"), 
#             "[[-(1 - (G)/(r) + (Q**2)/(r**2)),0,0,0],[0,1/(1 - (G)/(r) + (Q**2)/(r**2)),0,0],[0,0,r**2,0],[0,0,0,r**2*sin(theta)**2]]",
#             Metric("[[-(1 - (G)/(r) + (Q**2)/(r**2)),0,0,0],[0,1/(1 - (G)/(r) + (Q**2)/(r**2)),0,0],[0,0,r**2,0],[0,0,0,r**2*sin(theta)**2]]", '_{mu}_{nu}', '[t, r, theta, phi]')
#             )           
#         self.cache = {
#             "Schwarzschild": schild,
#             "SchildGeneral" : schild_general,
#             "AntiDeSitter" : AntiDeSitter,
#             "PolarCoordinates" : PolarCoordinates,
#             "Minkowski" : Minkowski,
#             "ReissnerNordstrom" : ReissnerNordstrom
#         }

#     def get_operations_with_latest_cache(self):
#         return {
#                     # Build Objects
#                     "BUILD_INT"         : self.build_int,
#                     "BUILD_FLOAT"       : self.build_float,
#                     "BUILD_TENSOR"      : self.tensor,
#                     "BUILD_MINUS"       : self.minus,
#                     "BUILD_VARIABLE"    : self.variable,

#                     # Simple Operations
#                     "ADD"               : self.add,
#                     "MULTIPLY"          : self.multiply,
#                     "SUBTRACTION"       : self.subtract,
#                     "DIVISION"          : self.divide,
#                     "POWER"             : self.power,
#                     "BUILD_FUNCTION"    : self.build_function,

#                     # Simpy Operations
#                     "DIFFERENTIAL"      : self.differentiate,
#                     "FUNCTION"          : self.function,
#                     "INTEGRAL"          : self.integrate,
#                     "SIMPLIFY"          : self.simplify,
#                     "ARRAY"             : self.array,
#                     "SOLVE"             : self.solve,
#                     "SERIES"            : self.series,
#                     "LIMIT"             : self.limit,
#                     "EXPAND"            : self.expand,
#                     "EXP"               : self.exp,
#                     "CONSTANT"          : self.constant,
#                     "NUMERICAL"         : self.numerical,

#                     # Trigonometry Functions
#                     "SIN"               : self.sin,
#                     "COS"               : self.cos,
#                     "TAN"               : self.tan,
#                     "ASIN"              : self.asin,
#                     "ACOS"              : self.acos,
#                     "ATAN"              : self.atan,

#                     # Hyperbolic Functions 
#                     "SINH"               : self.sinh,
#                     "COSH"               : self.cosh,
#                     "TANH"               : self.tanh,
#                     "ASINH"              : self.asinh,
#                     "ACOSH"              : self.acosh,
#                     "ATANH"              : self.atanh,
#                 }

#     def expr(self, expression):
#         if bool(re.match('([a-zA-Z]+)(\=)', expression.replace(' ',''))):
#             self.cache_equation(expression)
#         else:
#             ans = MathJSONInterpreter(self.get_operations_with_latest_cache()).interpret(Mathify(expression)())
#             if isinstance(ans, (Metric, GrTensor, Riemann, TensorObject)):
#                 return ans.get_specified_components()
#             else:
#                 return ans

#     def cache_equation(self, equation : str):
#         """ Add a equation to the cache """
#         key, value = equation.replace(' ','').split('=')
#         self.cache[key] = ChacheItem(
#             name = key,
#             decesrialized_obj = self.expr(value),
#             string_obj = value,
#             mathjson_obj = Mathify(value)()
#         )
#     def cache_value(self, key, value):
#         """ Add a value to the cache """
#         self.cache[key] = self.expr(value)

#     def get_cached_value(self, key):
#         """ Get a value from the cache by key """
#         return self.cache.get(key)

#     def key_exists(self, key):
#         """ Get boolean determining whether key exists """
#         return bool(self.get_cached_value(key))

#     def clear_cache(self, clear_item = None):
#         """ Clear the cache """

#         if clear_item == None:
#             self.cache.clear()
#         elif self.key_exists(clear_item):
#             del self.cache[clear_item]

#     def transform(self, transformation, basis):
#         transformation_obj = TransformationDeserializer(transformation=transformation, basis=basis).deserialize()
#         self.METRIC = self.METRIC.get_transformed_metric(transformation_obj)
        
#     def metric(self, components, basis):
#         self.cache['Metric'] = ChacheItem(
#             name = 'Metric',
#             string_obj = components,
#             decesrialized_obj = Metric(components, '_{mu}_{nu}', basis)
#         )
#         self.cache['Basis'] = ChacheItem(
#             name = 'Basis',
#             string_obj = basis,
#             decesrialized_obj = basis
#         )
#         self.METRIC = Metric(components, '_{mu}_{nu}', basis)
        
#     def build_function(self, *args):
#         functions = { 
#             'Sum' : (smp.Sum, 'obj'), 
#             'sum': (smp.Sum, 'doit'), 
#             'transform' : self.transform, 
#             'Metric' : self.metric
#             }
#         a = ''.join(args)

#         if bool(functions.get(a)):
#             return functions[a]
#         else:
#             return smp.symbols('{}'.format(a), cls=smp.Function)

#     def function(self, *args):
#         function_obejct = args[0]
#         parameters = [MathJSONInterpreter(self.get_operations_with_latest_cache()).interpret(args[1][i]) for i in range(len(args[1]))]
#         if isinstance(function_obejct, tuple) and function_obejct[1] == 'doit':
#             return smp.Sum(parameters[0], (parameters[1], parameters[2], parameters[3])).doit()
#         elif isinstance(function_obejct, tuple) and function_obejct[1] == 'obj':
#             return smp.Sum(parameters[0], (parameters[1], parameters[2], parameters[3]))
#         return function_obejct(*parameters)
        
#     def add(self, *args):
#         return args[0] + args[1]

#     def subtract(self, *args):
#         return args[0] - args[1]

#     def multiply(self, *args):
#         return args[0] * args[1]

#     def power(self, *args):
#         return args[0] ** args[1]

#     def divide(self, *args):
#         return args[0] / args[1]

#     def build_int(self, *args):
#         return int(''.join(args))

#     def build_float(self, *args):
#         return float(''.join(args))

#     def array(self, *args):
#         return smp.MutableDenseNDimArray(list(args))

#     def differentiate(self, *args):
#         expr = args[0]
#         wrt = MathJSONInterpreter(self.get_operations_with_latest_cache()).interpret(args[1][0])
#         return smp.diff(expr, wrt)

#     def integrate(self, *args):
#         expr = args[0]
#         wrt = MathJSONInterpreter(self.get_operations_with_latest_cache()).interpret(args[1][0])
#         return smp.integrate(expr, wrt)

#     def simplify(self, *args):
#         expr = args
#         return smp.simplify(expr[0])

#     def solve(self, *args):
#         expr = args
#         return smp.solve(expr[0])

#     def numerical(self, *args):
#         wrt = MathJSONInterpreter(self.get_operations_with_latest_cache()).interpret(args[1][0])
#         return smp.N(args[0], wrt)

#     def tensor(self, *args):
#         tensor_string_representation = ''.join(args)
#         tensor_name = re.match('([a-zA-Z]+)', tensor_string_representation).group()
#         tesnor_indices = tensor_string_representation.replace(tensor_name, '')

#         if not self.key_exists('Metric') or not self.key_exists('Basis'):
#             raise ValueError('No Metric has been defined')
        
#         if tensor_name == 'G':
#             return self.METRIC.new_indices(tesnor_indices)
#         elif tensor_name == 'd':
#             return Derivative(self.METRIC.components, tesnor_indices, self.METRIC.basis)
#         elif tensor_name == "R":
#             return Riemann(self.METRIC, tesnor_indices)

#     def minus(self, *args):
#         a = args[0]
#         return -a

#     def series(self, *args):
#         expr = args[0]
#         if len(args[1]) == 2:
#             point = MathJSONInterpreter(self.get_operations_with_latest_cache()).interpret(args[1][0])
#             n = MathJSONInterpreter(self.get_operations_with_latest_cache()).interpret(args[1][1])
#             return smp.series(expr, x0 = point, n = n)
#         elif len(args[1]) == 1:
#             point = MathJSONInterpreter(self.get_operations_with_latest_cache()).interpret(args[1][0])
#             return smp.series(expr, x0 = point, n = 5)
#         else:
#             return smp.series(expr, x0 = 0, n = 5)

#     def limit(self, *args):
#         expr = args[0]
#         x0 = MathJSONInterpreter(self.get_operations_with_latest_cache()).interpret(args[1][0])
#         x1 = MathJSONInterpreter(self.get_operations_with_latest_cache()).interpret(args[1][1])
#         return smp.limit(expr, x0, x1)

#     def expand(self, *args):
#         return smp.expand(args[0])

#     def exp(self, *args):
#         return smp.exp(args[0])

#     def constant(self, *args):
#         return { 'pi' : smp.pi, 'e' : smp.E, 'infty' : smp.oo }.get(args[0])

#     def variable(self, *args):
#         a = ''.join(args)
#         if self.key_exists(a):
#             if isinstance(self.cache.get(a).decesrialized_obj, (smp.Basic, smp.MutableDenseNDimArray, int, float)):
#                 return self.cache.get(a).decesrialized_obj
#             else:
#                 return MathJSONInterpreter(self.get_operations_with_latest_cache()).interpret(Mathify(self.get_cached_value(a))())
#         else:
#             return smp.symbols('{}'.format(a))
    
#     def sin(self, *args):
#         return smp.sin(args[0])

#     def cos(self, *args):
#         return smp.cos(args[0])

#     def tan(self, *args):
#         return smp.tan(args[0])

#     def asin(self, *args):
#         return smp.asin(args[0])

#     def acos(self, *args):
#         return smp.acos(args[0])

#     def atan(self, *args):
#         return smp.atan(args[0])

#     def sinh(self, *args):
#         return smp.sinh(args[0])

#     def cosh(self, *args):
#         return smp.cosh(args[0])

#     def tanh(self, *args):
#         return smp.tanh(args[0])

#     def asinh(self, *args):
#         return smp.asinh(args[0])

#     def acosh(self, *args):
#         return smp.acosh(args[0])

#     def atanh(self, *args):
#         return smp.atanh(args[0])




