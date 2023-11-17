import sympy as smp

from relativisticpy.workbook.itensors import (
    TensorDefinitionNode,
    TensorKeyNode,
    TensorNode,
    TensorDiagBuilder,
    DefinitionNode,
)
from relativisticpy.workbook.matchers import match_tensors
from relativisticpy.workbook.node import AstNode
from relativisticpy.workbook.state import WorkbookState


class RelPyAstNodeTraverser:
    def __init__(self, cache: WorkbookState):
        self.cache = cache

    # Cache Node handlers
    def assigner(self, node: AstNode):
        self.cache.set_variable(str(node.args[0]), node.args[1])

    def symbol_definition(self, node: AstNode):
        self.cache.set_variable(node.args[0], str(node.args[1]))

    def define(self, node: AstNode):
        DefinitionNode(self.cache).handle(node)

    # Basic Node handlers
    def variable_key(self, node: AstNode):
        return "".join(node.args)

    def symbol_key(self, node: AstNode):
        return "".join(node.args)

    def sub(self, node: AstNode):
        return node.args[0] - node.args[1]

    def add(self, node: AstNode):
        return node.args[0] + node.args[1]

    def neg(self, node: AstNode):
        return -node.args[0]

    def pos(self, node: AstNode):
        return +node.args[0]

    def mul(self, node: AstNode):
        return node.args[0] * node.args[1]

    def div(self, node: AstNode):
        return node.args[0] / node.args[1]

    def pow(self, node: AstNode):
        return node.args[0] ** node.args[1]

    def int(self, node: AstNode):
        return int("".join(node.args))

    def float(self, node: AstNode):
        return float("".join(node.args))

    # Sympy node handlers

    def subs(self, node: AstNode):
        expr: smp.Symbol = node.args[0]
        return expr.subs(node.args[1], node.args[2])

    def limit(self, node: AstNode):
        return smp.limit(node.args[0], node.args[1], node.args[2])

    def expand(self, node: AstNode):
        return smp.expand(node.args[0])

    def diff(self, node: AstNode):
        return smp.diff(node.args[0], node.args[1])

    def integrate(self, node: AstNode):
        return smp.integrate(node.args[0], node.args[1])

    def simplify(self, node: AstNode):
        return smp.simplify(node.args[0])

    def latex(self, node: AstNode):
        return smp.latex(node.args[0])

    def solve(self, node: AstNode):
        return smp.solve(*node.args)

    def numerical(self, node: AstNode):
        return (
            smp.N(node.args[0], node.args[1])
            if len(node.args) == 2
            else smp.N(node.args[0], 15)
        )

    def array(self, node: AstNode):
        return smp.MutableDenseNDimArray(list(node.args))

    def exp(self, node: AstNode):
        return smp.exp(node.args[0])

    def dsolve(self, node: AstNode):
        return smp.dsolve(node.args[0], node.args[1]).rhs

    # Sympy Trigs
    def sin(self, node: AstNode):
        return smp.sin(node.args[0])

    def cos(self, node: AstNode):
        return smp.cos(node.args[0])

    def tan(self, node: AstNode):
        return smp.tan(node.args[0])

    def asin(self, node: AstNode):
        return smp.asin(node.args[0])

    def acos(self, node: AstNode):
        return smp.acos(node.args[0])

    def atan(self, node: AstNode):
        return smp.atan(node.args[0])

    def sinh(self, node: AstNode):
        return smp.sinh(node.args[0])

    def cosh(self, node: AstNode):
        return smp.cosh(node.args[0])

    def tanh(self, node: AstNode):
        return smp.tanh(node.args[0])

    def asinh(self, node: AstNode):
        return smp.asinh(node.args[0])

    def acosh(self, node: AstNode):
        return smp.acosh(node.args[0])

    def atanh(self, node: AstNode):
        return smp.atanh(node.args[0])

    # Sympy constants
    def constant(self, node: AstNode):
        a = "".join(node.args)
        if a == "pi":
            return smp.pi
        elif a == "e":
            return smp.E

    # Sympy symbols / function initiators
    def function(self, node: AstNode):
        return smp.symbols("{}".format(node.handler), cls=smp.Function)(*node.args)

    def object(self, node: AstNode):
        a = "".join(node.args)

        if a in ["pi", "e"]:
            return self.constant(node)

        elif not self.cache.has_variable(a):
            return smp.symbols("{}".format(a))

        else:
            return self.cache.get_variable(str(a))

    def diag(self, node: AstNode):
        return TensorDiagBuilder().handle(node)

    # Tensor type node handlers
    def tensor_key(self, node: AstNode):
        return TensorKeyNode().handle(
            node
        )  # Handles Tensor identifyers G_{a}_{b} etc ...

    def tensor_init(self, node: AstNode):
        TensorDefinitionNode(self.cache).handle(
            node
        )  # Tensor Setter : G_{a}_{b} := [[a, b],[c,d]]

    def tensor(self, node: AstNode):
        return TensorNode(self.cache).handle(
            node
        )  # Tensor Getter : G_{a}_{b} <-- go get me the object from cache or init new

    # Node Traverser Configuration

    """ For Ast traverser to be able to match on objects and create user defined object nodes with their own node handlers. """
    variable_matchers = [
        {
            "node": "object",
            "node_key": "tensor",
            "string_matcher_callback": match_tensors,
        }
    ]

    """ For Ast traverser to know which function handles each node type. """
    node_configuration = [
        {"node": "+", "handler": "add"},
        {"node": "=", "handler": "assigner"},
        {"node": ":=", "handler": "define"},
        {"node": "-", "handler": "sub"},
        {"node": "*", "handler": "mul"},
        {"node": "^", "handler": "pow"},
        {"node": "**", "handler": "pow"},
        {"node": "/", "handler": "div"},
        {"node": "array", "handler": "array"},
        {"node": "integer", "handler": "int"},
        {"node": "float", "handler": "float"},
        {"node": "negative", "handler": "neg"},
        {"node": "positive", "handler": "pos"},
        {"node": "tensor_init", "handler": "tensor_init"},
        {"node": "tensor_key", "handler": "tensor_key"},
        {"node": "variable_key", "handler": "variable_key"},
        {"node": "symbol_definition", "handler": "symbol_definition"},
        {"node": "symbol_key", "handler": "symbol_key"},
    ]
