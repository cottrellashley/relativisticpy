from cmath import exp
from dataclasses import dataclass
from typing import List
from relativisticpy.core.indices import Indices
from relativisticpy.core.metric import MetricIndices
from relativisticpy.relparser import RelParser
import sympy as smp
import re
from dataclasses import dataclass
from relativisticpy.core import Metric
from relativisticpy.core import Mathify
from relativisticpy.gr import Derivative, Riemann, Ricci
from relativisticpy.workbook.var_matchers import variable_matchers

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

def gr_tensor_mapper(key): return { 'G': Metric, 'd': Derivative, 'R': (Riemann, Ricci) }[key]

class WorkbookNode:

    def subs(self, node: Node):
        expr = node.args[0]
        var = node.args[1]
        sub_value = node.args[2]

        return expr.subs(var, sub_value)
    
    def latex(self, node: Node):
        return smp.latex(node.args[0])

    def assigner(self, node: Node):
        global_store.set_variable(str(node.args[0]), node.args[1])

    def define(self, node: Node):
        node_symbol = re.match('([a-zA-Z]+)', str(node.args[0])).group()

        metric_symbol = global_store.get_variable('MetricSymbol') if global_store.has_variable('MetricSymbol') else 'G'

        if bool(re.search("^((\^|\_)(\{)(\}))+$", re.sub('[^\^^\_^\{^\}]',"", str(node.args[0]).replace(metric_symbol, '')).replace(" ",''))) and node_symbol == metric_symbol:
            metric = Metric(
                            MetricIndices.from_string(str(node.args[0]).replace(metric_symbol, '')),
                            node.args[1],
                            global_store.get_variable('Coordinates')
                        )
            global_store.set_variable('Metric',  metric)
        elif re.match(r'\b\w+Symbol\b', str(node.args[0])):
            global_store.set_variable(str(node.args[0]), str(node.args[1]))
        else:
            global_store.set_variable(str(node.args[0]), node.args[1])

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

    def tensor(self, node: Node):

        tensor_string_representation = ''.join(node.args)
        tensor_name = re.match('([a-zA-Z]+)', tensor_string_representation).group()
        tesnor_indices = tensor_string_representation.replace(tensor_name, '')
        indices_obj = Indices.from_string(tesnor_indices)

        # if not self.key_exists('Metric') or not self.key_exists('Basis'):
        #     raise ValueError('No Metric has been defined')
        metric_defined = global_store.has_variable('Metric')

        if tensor_name == global_store.get_variable('MetricSymbol') and metric_defined:
            return self.metric(tesnor_indices)
        elif tensor_name == global_store.get_variable('DiffSymbol') and metric_defined:
            return Derivative(indices_obj)
        elif tensor_name == global_store.get_variable('RiemannSymbol') and metric_defined:
            return self.riemann(tesnor_indices)
        elif tensor_name == global_store.get_variable('RicciSymbol') and metric_defined:
            return self.ricci(tesnor_indices)
        else:
            return tensor_string_representation

    def metric(self, indices_str: str):
        indices = MetricIndices.from_string(indices_str)
        return global_store.get_variable('Metric')[indices] if re.search(r'\d',  indices_str) else global_store.get_variable('Metric')

    def ricci(self, indices_str: str):
        indices = Indices.from_string(indices_str)

        if not global_store.has_variable('Ricci'):
            global_store.set_variable('Ricci', Ricci(Indices.from_string(indices_str if not re.search(r'\d',  indices_str) else '_{a}_{b}'), global_store.get_variable('Metric')))

        return global_store.get_variable('Ricci')[indices] if re.search(r'\d', indices_str) else global_store.get_variable('Ricci')

    def riemann(self, indices_str: str):
        indices = Indices.from_string(indices_str)

        if not global_store.has_variable('Riemann'):
            global_store.set_variable('Riemann', Riemann(Indices.from_string(indices_str if not re.search(r'\d',  indices_str) else '_{a}_{b}_{c}_{d}'), global_store.get_variable('Metric')))

        return global_store.get_variable('Riemann')[indices] if re.search(r'\d',  indices_str) else global_store.get_variable('Riemann')


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
            }
        ]

class Workbook:

    parser = RelParser(WorkbookNode(), node_configuration, object_configuration=variable_matchers)

    def expr(self, string: str):
        if re.search(r'\n|\r\n?', string):
            lines = string.splitlines()
            tasks = [Workbook.parser.exe(line) for line in lines if line.replace(" ", "") != ""]
            return [task for task in tasks if task != None]

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
                        line = line.split("#")[0].strip() + " "
                    
                    tasks.append(Workbook.parser.exe(line))
        
            return [task for task in tasks if task != None]
