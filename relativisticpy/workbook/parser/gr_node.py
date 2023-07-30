import sympy as smp
import re
from dataclasses import dataclass

from relativisticpy.providers.jsonmathpy import SympyNode
from relativisticpy.gr import Derivative, Riemann, Metric, Ricci

@dataclass
class Node:
    node: str
    handler: str
    args: any

def gr_tensor_mapper(key): return { 'G': Metric, 'd': Derivative, 'R': (Riemann, Ricci) }[key]

class GRNode(SympyNode):

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

    def metric(self, components, basis):
        self.cache['Metric'] = ChacheItem(
            name = 'Metric',
            string_obj = components,
            decesrialized_obj = Metric(components, '_{mu}_{nu}', basis)
        )
        self.cache['Basis'] = ChacheItem(
            name = 'Basis',
            string_obj = basis,
            decesrialized_obj = basis
        )
        self.METRIC = Metric(components, '_{mu}_{nu}', basis)