from dataclasses import dataclass
from relativisticpy.relparser import RelParser
import sympy as smp
import re
from dataclasses import dataclass
from relativisticpy.core import Metric
from relativisticpy.core import Mathify
from relativisticpy.gr import Derivative, Riemann, Ricci

def Mathify_two(expression: str): 
    """Builds mathematical python object representing the string exporession enterred.""" 
    return RelParser(WorkbookNode(), node_configuration).exe(expression)

@dataclass
class ChacheItem:
    name: str
    string_obj: str
    decesrialized_obj: any
    mathjson_obj: dict = None

    def __hash__(self):
        return hash((self.name, str(self.mathjson_obj), self.string_obj, str(self.decesrialized_obj)))

    def __eq__(self, other):
        if isinstance(other, ChacheItem):
            return (self.name == other.name and
                    self.mathjson_obj == other.mathjson_obj and
                    self.string_obj == other.string_obj and
                    self.decesrialized_obj == other.decesrialized_obj)
        return False

@dataclass
class Node:
    node: str
    handler: str
    args: any

def gr_tensor_mapper(key): return { 'G': Metric, 'd': Derivative, 'R': (Riemann, Ricci) }[key]

class WorkbookNode:

    # def add(self, node: Node):
    #     return node.args[0] + node.args[1]

    # How should it work:
    # Metric Symbol should be defined as G but user can overide it.
    # User must ALWAYS define a coordinate system in which the tensors are in as otherwise derivatives are not known. 

    # If metric dependent tensors are written, we throw an error as metric dependent imply user must first define the metric for obvious reasons.
    # If user writes T_{a}_{c}_{s} := [... [... [... ]]]
    # We simply initiate a tensor object with key set as the object. So user can if they wish, 
    # def __new__(self): 
    #     self.cache = {}
    #     self.Metric = None
    #     self.MetricSymbol = 'G'
    #     self.Coordinates = None

    def define(self, node: Node):
        # This method instantiates a class set by user:
        setattr(self, node.args[0], node.args[1])
        if node.args[0] == 'MetricSymbol': self.cache['MetricSymbol'] = node.args[1]
        elif node.args[0] == 'Coordinates': self.cache['Basis'] = Mathify(node.args[1])
        elif re.match('([a-zA-Z]+)', node.args[0]).group() == self.cache['MetricSymbol']: 
            return Metric(
                            node.args[0].replace(node.args[0], ''),
                            Mathify(node.args[1]),
                            self.cache['Basis']
                        )

    def tensor(self, node: Node):
        expr = node.args[0]
        x0   = node.args[1]
        x1   = node.args[2]
        return smp.limit(expr, x0, x1)

    # def tensor(self, node: Node):
    #     tensor_string_representation = ''.join(node.args)
    #     tensor_name = re.match('([a-zA-Z]+)', tensor_string_representation).group()
    #     tesnor_indices = tensor_string_representation.replace(tensor_name, '')

    #     if not self.key_exists('Metric') or not self.key_exists('Basis'):
    #         raise ValueError('No Metric has been defined')

    #     if tensor_name == 'G':
    #         return self.METRIC.new_indices(tesnor_indices)
    #     elif tensor_name == 'd':
    #         return Derivative(self.METRIC.components, tesnor_indices, self.METRIC.basis)
    #     elif tensor_name == "R":
    #         return Riemann(self.METRIC, tesnor_indices)

    def sub(self, node):
        return node.args[0] - node.args[1]

    def mul(self, node):
        return node.args[0] * node.args[1]

    def div(self, node):
        return node.args[0] / node.args[1]

    def pow(self, node):
        return node.args[0] ** node.args[1]

    def float(self, node):
        return float(''.join(node.args))

    def int(self,  node):
        return int(''.join(node.args))

    def limit(self, node: Node):
        expr = node.args[0]
        x0   = node.args[1]
        x1   = node.args[2]
        return smp.limit(expr, x0, x1)

    def expand(self, node: Node):
        return smp.expand(node.args[0])

    def differentiate(self, node: Node):
        expr = node.args[0]
        wrt = node.args[1]
        return smp.diff(expr, wrt)

    def integrate(self, node: Node):
        expr = node.args[0]
        wrt = node.args[1]
        return smp.integrate(expr, wrt)

    def simplify(self, node: Node):
        expr = node.args[0]
        return smp.simplify(expr)

    def solve(self, node: Node):
        expr = node.args[0]
        return smp.solve(expr)

    # def numerical(self, *args):
    #     wrt = args[1]
    #     return smp.N(args[0], wrt)

    def array(self, node: Node):
        return smp.MutableDenseNDimArray(list(node.args))

    # def minus(self, *args):
    #     a = args[0]
    #     return -a

    def exp(self, node: Node):
        return smp.exp(node.args[0])

    # def constant(self, *args):
    #     if args[0] == 'pi':
    #         return smp.pi
    #     if args[0] == 'e':
    #         return smp.E

    def object(self, node: Node):
        a = ''.join(node.args)
        return smp.symbols('{}'.format(a))

    def function(self, node: Node):
        return smp.symbols('{}'.format(node.handler), cls=smp.Function)(*node.args)
    
    def sin(self, node: Node):
        return smp.sin(node.args[0])

    def cos(self, node: Node):
        return smp.cos(node.args[0])

    def tan(self, node: Node):
        return smp.tan(node.args[0])

    def asin(self, node: Node):
        return smp.asin(node.args[0])

    def acos(self, node: Node):
        return smp.acos(node.args[0])

    def atan(self, node: Node):
        return smp.atan(node.args[0])

    def sinh(self, node: Node):
        return smp.sinh(node.args[0])

    def cosh(self, node: Node):
        return smp.cosh(node.args[0])

    def tanh(self, node: Node):
        return smp.tanh(node.args[0])

    def asinh(self, node: Node):
        return smp.asinh(node.args[0])

    def acosh(self, node: Node):
        return smp.acosh(node.args[0])

    def atanh(self, node: Node):
        return smp.atanh(node.args[0])

node_configuration = [
            {
                'node': '+',
                'handler': "add"
            },
            {
                'node': '-',
                'handler': "sub"
            },
            {
                'node': '*',
                'handler': "mul"
            },
            {
                'node': '^',
                'handler': "pow"
            },
            {
                'node': '**',
                'handler': "pow"
            },
            {
                'node': '/',
                'handler': "div"
            },
            {
                'node': 'array',
                'handler': "array"
            },
            {
                'node': 'integer',
                'handler': "int"
            },
            {
                'node': 'float',
                'handler': "float"
            }
        ]

class Workbook:

    parser = RelParser(WorkbookNode(), node_configuration)

    def expr(self, string: str):
        return Workbook.parser.exe(string)

# class Workbook:

#     """ Cache for storing values accessed by string keys """
#     def __init__(self):
#         schild = ChacheItem(
#             "Schwarzschild", 
#             Mathify("[[-(1 - (G)/(r)),0,0,0],[0,1/(1 - (G)/(r)),0,0],[0,0,r**2,0],[0,0,0,r**2*sin(theta)**2]]"), 
#             "[[-(1 - (G)/(r)),0,0,0],[0,1/(1 - (G)/(r)),0,0],[0,0,r**2,0],[0,0,0,r**2*sin(theta)**2]]",
#             Metric("[[-(1 - (G)/(r)),0,0,0],[0,1/(1 - (G)/(r)),0,0],[0,0,r**2,0],[0,0,0,r**2*sin(theta)**2]]",'_{mu}_{nu}', '[t, r, theta, phi]')
#             )
#         schild_general = ChacheItem(
#             "SchildGeneral", 
#             Mathify("[[-F(r),0,0,0],[0,1/(F(r)),0,0],[0,0,r**2,0],[0,0,0,r**2*sin(theta)**2]]"), 
#             "[[-F(r),0,0,0],[0,1/(F(r)),0,0],[0,0,r**2,0],[0,0,0,r**2*sin(theta)**2]]",
#             Metric("[[-F(r),0,0,0],[0,1/(F(r)),0,0],[0,0,r**2,0],[0,0,0,r**2*sin(theta)**2]]",'_{mu}_{nu}', '[t, r, theta, phi]')
#             )
#         AntiDeSitter = ChacheItem(
#             "AntiDeSitter", 
#             Mathify("[[-(k**2*r**2 + 1),0,0,0],[0,1/(k**2*r**2 + 1),0,0],[0,0,r**2,0],[0,0,0,r**2*sin(theta)**2]]"), 
#             "[[-(k**2*r**2 + 1),0,0,0],[0,1/(k**2*r**2 + 1),0,0],[0,0,r**2,0],[0,0,0,r**2*sin(theta)**2]]",
#             Metric("[[-(k**2*r**2 + 1),0,0,0],[0,1/(k**2*r**2 + 1),0,0],[0,0,r**2,0],[0,0,0,r**2*sin(theta)**2]]",'_{mu}_{nu}', '[t, r, theta, phi]')
#             )
#         PolarCoordinates = ChacheItem(
#             "PolarCoordinates", 
#             Mathify("[[-1,0,0,0],[0,1,0,0],[0,0,r**2,0],[0,0,0,r**2*sin(theta)**2]]"), 
#             "[[-1,0,0,0],[0,1,0,0],[0,0,r**2,0],[0,0,0,r**2*sin(theta)**2]]",
#             Metric("[[-1,0,0,0],[0,1,0,0],[0,0,r**2,0],[0,0,0,r**2*sin(theta)**2]]", '_{mu}_{nu}', '[t, r, theta, phi]')
#             )
#         Minkowski = ChacheItem(
#             "Minkowski", 
#             Mathify("[[-1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]"), 
#             "[[-1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]",
#             Metric("[[-1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]",'_{mu}_{nu}', '[t, r, theta, phi]')
#             )
#         Minkowski = ChacheItem(
#             "Minkowski", 
#             Mathify("[[-1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]"), 
#             "[[-1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]",
#             Metric("[[-1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]", '_{mu}_{nu}', '[t, x, y, z]')
#             )
#         ReissnerNordstrom = ChacheItem(
#             "ReissnerNordstrom", 
#             Mathify("[[-(1 - (G)/(r) + (Q**2)/(r**2)),0,0,0],[0,1/(1 - (G)/(r) + (Q**2)/(r**2)),0,0],[0,0,r**2,0],[0,0,0,r**2*sin(theta)**2]]"), 
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