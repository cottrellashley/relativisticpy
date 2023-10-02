from cmath import exp
from dataclasses import dataclass
from typing import List
from relativisticpy.core.indices import Indices
from relativisticpy.core.metric import MetricIndices
from relativisticpy.providers.regex import extract_tensor_symbol, extract_tensor_indices, tensor_index_running
from relativisticpy.relparser import RelParser
import sympy as smp
import re
from dataclasses import dataclass
from relativisticpy.core import Metric
from relativisticpy.core import Mathify
from relativisticpy.gr import Derivative, Riemann, Ricci
from relativisticpy.workbook.var_matchers import variable_matchers

class TensorKey:

    def __init__(self, tensor_string_repr: str):
        self.str_key = extract_tensor_symbol(tensor_string_repr)
        self.str_tensor = tensor_string_repr
        self.str_indices = extract_tensor_indices(tensor_string_repr)


@dataclass
class Node:
    node: str
    handler: str
    args: List['Node']

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

class WorkbookNode:

    def subs(self, node: Node):
        expr: smp.Symbol = node.args[0]
        var = node.args[1]
        sub_value = node.args[2]

        return expr.subs(var, sub_value)
    
    def latex(self, node: Node):
        return smp.latex(node.args[0])

    def assigner(self, node: Node):
        global_store.set_variable(str(node.args[0]), node.args[1])

    def define(self, node: Node):
        if isinstance(node.args[0], str):
            key = node.args[0]
        else:
            raise ValueError('Node is not a string.')

        global_store.set_variable(key, node.args[1])

    def sub(self, node):
        return node.args[0] - node.args[1]
    
    def neg(self, node):
        return -node.args[0]

    def pos(self, node):
        return +node.args[0]

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

    def numerical(self, node: Node):
        return smp.N(node.args[0], node.args[1]) if len(node.args) == 2 else smp.N(node.args[0], 15)

    def array(self, node: Node):
        return smp.MutableDenseNDimArray(list(node.args))

    def exp(self, node: Node):
        return smp.exp(node.args[0])

    def constant(self, node: Node):
        a = ''.join(node.args)
        if a == 'pi':
            return smp.pi
        elif a == 'e':
            return smp.E

    def add(self, node: Node):
        return node.args[0] + node.args[1]

    def dsolve(self, node: Node):
        if len(node.args) < 2:
            print('Incorect number of arguments.')
        
        equation = node.args[0]
        wrt = node.args[1]

        return smp.dsolve(equation, wrt).rhs

    def object(self, node: Node):
        a = ''.join(node.args)

        if a in ['pi', 'e']:
            return self.constant(node)

        elif not global_store.has_variable(a):
            return smp.symbols('{}'.format(a))

        else:
            return global_store.get_variable(str(a))

    def variable_key(self, node: Node):
        """ This is a key to be used as storage for an object as value. Return string """
        a = ''.join(node.args)
        return a

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

    def symbol_key(self, node: Node):
        a = ''.join(node.args)
        return a

    def symbol_definition(self, node: Node):
        global_store.set_variable(node.args[0],  str(node.args[1]))

    def tensor_key(self, node: Node):
        """ This is a key to be used as storage for an object as value. Return string """
        a = ''.join(node.args)
        return TensorKey(a)

    def tensor_init(self, node: Node):
        tensor_key = node.args[0]
        tensor_comps = node.args[1]
        metric_symbol = global_store.get_variable('MetricSymbol') if global_store.has_variable('MetricSymbol') else 'G'

        metric = Metric(
                        MetricIndices.from_string(tensor_key.str_indices),
                        tensor_comps,
                        global_store.get_variable('Coordinates')
                    )

        global_store.set_variable('Metric',  metric)

    def tensor(self, node: Node):

        tensor_repr = TensorKey(''.join(node.args))
        indices_obj = Indices.from_string(tensor_repr.str_indices)

        # if not self.key_exists('Metric') or not self.key_exists('Basis'):
        #     raise ValueError('No Metric has been defined')
        metric_defined = global_store.has_variable('Metric')

        if tensor_repr.str_key == global_store.get_variable('MetricSymbol') and metric_defined:
            return self.metric(tensor_repr.str_indices)
        elif tensor_repr.str_key == global_store.get_variable('DiffSymbol') and metric_defined:
            return Derivative(indices_obj)
        elif tensor_repr.str_key == global_store.get_variable('RiemannSymbol') and metric_defined:
            return self.riemann(tensor_repr.str_indices)
        elif tensor_repr.str_key == global_store.get_variable('RicciSymbol') and metric_defined:
            return self.ricci(tensor_repr.str_indices)
        else:
            return tensor_repr.str_tensor

    def metric(self, indices_str: str):
        indices = MetricIndices.from_string(indices_str)
        return global_store.get_variable('Metric')[indices] if not tensor_index_running(indices_str) else global_store.get_variable('Metric')

    def ricci(self, indices_str: str):
        indices = Indices.from_string(indices_str)

        if not global_store.has_variable('Ricci'):
            global_store.set_variable('Ricci', Ricci(Indices.from_string(indices_str if tensor_index_running(indices_str) else '_{a}_{b}'), global_store.get_variable('Metric')))

        return global_store.get_variable('Ricci')[indices] if not tensor_index_running(indices_str) else global_store.get_variable('Ricci')

    def riemann(self, indices_str: str):
        indices = Indices.from_string(indices_str)

        if not global_store.has_variable('Riemann'):
            global_store.set_variable('Riemann', Riemann(Indices.from_string(indices_str if tensor_index_running(indices_str) else '_{a}_{b}_{c}_{d}'), global_store.get_variable('Metric')))

        return global_store.get_variable('Riemann')[indices] if not tensor_index_running(indices_str) else global_store.get_variable('Riemann')


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
            },
            {
                'node': 'negative',
                'handler': "neg"
            },
            {
                'node': 'positive',
                'handler': "pos"
            },
            {
                'node': 'tensor_init',
                'handler': "tensor_init"
            },
            {
                'node': 'tensor_key',
                'handler': 'tensor_key'
            },
            {
                'node': 'variable_key',
                'handler': "variable_key"
            },
            {
                'node': 'symbol_definition',
                'handler': "symbol_definition"
            },
            {
                'node': 'symbol_key',
                'handler': "symbol_key"
            }
        ]

class Workbook:

    parser = RelParser(WorkbookNode(), node_configuration, object_configuration=variable_matchers)

    def expr(self, string: str):
        if re.search(r'\n|\r\n?', string):
            tasks = []
            lines = string.splitlines()
            for line in lines:
                if line.strip():
                    # If line starts with "#", ignore it
                    if line.startswith("#"):
                        continue
                    
                    # If line contains "#", split and take the part before "#"
                    if "#" in line:
                        line = line.split("#")[0].strip()

                    if ";" in line:
                        lines = line.split(";")
                        for l in lines:
                            tasks.append(l.strip())
                        continue
                    
                    tasks.append(line.strip())
            res = [Workbook.parser.exe(task) for task in tasks if task.strip()]
            return [r for r in res if r != None ]

        return Workbook.parser.exe(string)

    def parse(self, string: str):
        return Workbook.parser.parse(string)

    def tokens(self, string: str):
        return Workbook.parser.tokenize(string)

    def exe(self, file_path):
        with open(file_path, 'r') as file:
            tasks = []
            for line in file:
                if line.strip():
                    # If line starts with "#", ignore it
                    if line.startswith("#"):
                        continue
                    
                    # If line contains "#", split and take the part before "#"
                    if "#" in line:
                        line = line.split("#")[0].strip()

                    if ";" in line:
                        lines = line.split(";")
                        for l in lines:
                            tasks.append(l.strip())
                        continue
                    
                    tasks.append(line.strip())

            res = [Workbook.parser.exe(task) for task in tasks if task.strip()]

            return [r for r in res if r != None ]
