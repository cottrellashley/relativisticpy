from dataclasses import dataclass
from typing import List
from relativisticpy.core.metric import MetricIndices
from relativisticpy.relparser import RelParser
import sympy as smp
import re
from dataclasses import dataclass
from relativisticpy.core import Metric
from relativisticpy.core import Mathify
from relativisticpy.gr import Derivative, Riemann, Ricci
from relativisticpy.workbook.var_matchers import variable_matchers

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
    args: List['Node']

@dataclass
class TensorNode:
    key: str
    obj: any

    def __str__(self) -> str:
        return self.key

class VariableStore:
    def __init__(self):
        self.store = {}

    def set_variable(self, name, value):
        self.store[name] = value

    def get_variable(self, name):
        return self.store.get(name, None)

    def has_variable(self, name):
        return name in self.store

    def metric(self):
        pass

global_store = VariableStore()

def gr_tensor_mapper(key): return { 'G': Metric, 'd': Derivative, 'R': (Riemann, Ricci) }[key]

class WorkbookNode:

    def add(self, node: Node):
        return node.args[0] + node.args[1]

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

    def assigner(self, node: Node):
        global_store.set_variable(str(node.args[0]), node.args[1])

    def define(self, node: Node):
        # This method instantiates a class set by user:
        node_symbol = re.match('([a-zA-Z]+)', str(node.args[0])).group()
        if global_store.has_variable('MetricSymbol'): metric_symbol = global_store.get_variable('MetricSymbol')
        if str(node.args[0]) == 'MetricSymbol': global_store.set_variable('MetricSymbol', str(node.args[1]))
        elif str(node.args[0]) == 'Coordinates': global_store.set_variable(str(node.args[0]), node.args[1])
        elif node_symbol == metric_symbol:
            metric = Metric(
                            MetricIndices.from_string(str(node.args[0]).replace(metric_symbol, '')),
                            node.args[1],
                            global_store.get_variable('Coordinates')
                        )
            global_store.set_variable('Metric',  metric)
        else:
            global_store.set_variable(str(node.args[0]), node.args[1])

    # def tensor(self, node: Node):
    #     expr = node.args[0]
    #     x0   = node.args[1]
    #     x1   = node.args[2]
    #     return smp.limit(expr, x0, x1)

    def tensor(self, node: Node):
        # return TensorNode(type(global_store.get_variable('Metric')))
        tensor_string_representation = ''.join(node.args)
        tensor_name = re.match('([a-zA-Z]+)', tensor_string_representation).group()
        tesnor_indices = tensor_string_representation.replace(tensor_name, '')

        # if not self.key_exists('Metric') or not self.key_exists('Basis'):
        #     raise ValueError('No Metric has been defined')

        if tensor_name == global_store.get_variable('MetricSymbol') and global_store.has_variable('Metric'):
            return global_store.get_variable('Metric')
        elif tensor_name == 'd':
            return Derivative(global_store.get_variable('Metric'), tesnor_indices, global_store.get_variable('Coordinates'))
        elif tensor_name == "R":
            return Riemann(global_store.get_variable('Metric'), tesnor_indices)
        else:
            return tensor_string_representation

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
        if not global_store.has_variable(a):
            return smp.symbols('{}'.format(a))
        else:
            return global_store.get_variable(str(a))

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
                'node': '=',
                'handler': "assigner"
            },
            {
                'node': ':=',
                'handler': "define"
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

    parser = RelParser(WorkbookNode(), node_configuration, object_configuration=variable_matchers)

    def expr(self, string: str):
        return Workbook.parser.exe(string)

    def parse(self, string: str):
        return Workbook.parser.parse(string)

