import sympy as smp
import re
from dataclasses import dataclass

from relativisticpy.core import Metric
from relativisticpy.core import Mathify
from relativisticpy.core.mathify import MathNode
from relativisticpy.gr import Derivative, Riemann, Ricci

@dataclass
class Node:
    node: str
    handler: str
    args: any

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
    def __new__(self): 
        self.cache = {}
        self.Metric = None
        self.MetricSymbol = 'G'
        self.Coordinates = None

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

    def tensor(self, node: Node):
        tensor_string_representation = ''.join(node.args)
        tensor_name = re.match('([a-zA-Z]+)', tensor_string_representation).group()
        tesnor_indices = tensor_string_representation.replace(tensor_name, '')

        if not self.key_exists('Metric') or not self.key_exists('Basis'):
            raise ValueError('No Metric has been defined')

        if tensor_name == 'G':
            return self.METRIC.new_indices(tesnor_indices)
        elif tensor_name == 'd':
            return Derivative(self.METRIC.components, tesnor_indices, self.METRIC.basis)
        elif tensor_name == "R":
            return Riemann(self.METRIC, tesnor_indices)

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
                'handler': "adds"
            },
            {
                'node': '-',
                'handler': "sub"
            },
            {
                'node': 'exp',
                'handler': "haha"
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