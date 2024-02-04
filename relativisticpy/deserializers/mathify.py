import sympy as smp
from dataclasses import dataclass
from relativisticpy.parsers import RelParser
from relativisticpy.symengine import SymbolArray, diff, sin, cos, tan


def Mathify(expression: str):
    """Builds mathematical python object representing the string exporession enterred."""
    return RelParser(MathNode()).exe(expression)[0].value


@dataclass
class Node:
    node: str
    handler: str
    args: any


class MathNode:
    def neg(self, node):
        return -node.args[0]

    def pos(self, node):
        return +node.args[0]

    def add(self, node):
        return node.args[0] + node.args[1]

    def sub(self, node):
        return node.args[0] - node.args[1]

    def mul(self, node):
        return node.args[0] * node.args[1]

    def div(self, node):
        return node.args[0] / node.args[1]

    def pow(self, node):
        return node.args[0] ** node.args[1]

    def float(self, node):
        return float("".join(node.args))

    def int(self, node):
        return int("".join(node.args))

    def simplify(self, node: Node):
        return smp.simplify(node.args[0])

    def solve(self, node: Node):
        return smp.solve(node.args[0])

    def exp(self, node: Node):
        return smp.exp(node.args[0])

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

    def expand(self, node: Node):
        return smp.expand(node.args[0])

    def diff(self, node: Node):
        return smp.diff(node.args[0], node.args[1])

    def integrate(self, node: Node):
        return smp.integrate(node.args[0], node.args[1])

    def array(self, node: Node):
        return SymbolArray(list(node.args))

    def symbol(self, node: Node):
        return smp.symbols("{}".format("".join(node.args)))

    def limit(self, node: Node):
        return smp.limit(node.args[0], node.args[1], node.args[2])

    def function(self, node: Node):
        return smp.symbols("{}".format(node.handler), cls=smp.Function)(*node.args)
