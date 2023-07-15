import sympy as smp
from dataclasses import dataclass
from relativisticpy.workbook.frontends.sympify.simple_node import SimpleNode

@dataclass
class Node:
    node: str
    handler: str
    args: any

class SympyNode(SimpleNode):

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

