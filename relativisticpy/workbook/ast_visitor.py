from relativisticpy.workbook.itensors import (
    TensorDefinitionNode,
    TensorNode,
    TensorDiagBuilder,
    DefinitionNode,
    InitTensorFromComponentsNode,
    InitTensorFromExpressionNode,
)
from relativisticpy.workbook.matchers import match_tensors, match_symbol
from relativisticpy.workbook.node import AstNode
from relativisticpy.workbook.state import WorkbookState, TensorReference
from relativisticpy.utils import str_is_tensors

from relativisticpy.symengine import (
    Symbol,
    Rational,
    Function,
    Interval,
    Order,
    Sum,
    SymbolArray,
    O,
    LaplaceTransform,
    diff,
    integrate,
    simplify,
    tensorproduct,
    symbols,
    residue,
    laplace_transform,
    inverse_laplace_transform,
    inverse_mellin_transform,
    mellin_transform,
    fourier_transform,
    inverse_fourier_transform,
    sine_transform,
    inverse_sine_transform,
    cosine_transform,
    inverse_cosine_transform,
    hankel_transform,
    inverse_hankel_transform,
    zeros,
    permutedims,
    solve,
    dsolve,
    expand,
    latex,
    limit,
    Limit,
    fourier_series,
    sequence,
    series,
    euler_equations,
    sqrt,
    exp_polar,
    bell,
    bernoulli,
    binomial,
    gamma,
    conjugate,
    hyper,
    catalan,
    euler,
    factorial,
    fibonacci,
    harmonic,
    log,
    I,
    oo,
    E,
    N,
    pi,
    exp,
    sin,
    asin,
    sinh,
    asinh,
    cos,
    acos,
    cosh,
    acosh,
    tan,
    atan,
    tanh,
    atanh,
)


class RelPyAstNodeTraverser:
    def __init__(self, cache: WorkbookState):
        self.cache = cache

    # Cache Node handlers
    def variable_assignment(self, node: AstNode):
        self.cache.set_variable(str(node.args[0]), node.args[1])

    def tensor_component_assigning(self, node: AstNode):
        if ":" in str(node.args[0]):
            raise ValueError(
                "Feature: <Assigning value to tensor sub-components> is not implemented yet."
            )
        InitTensorFromComponentsNode(self.cache).handle(node)

    def tensor_expr_assignment(self, node: AstNode):
        if ":" in str(node.args[0]):
            raise ValueError(
                "Feature: <Assigning value to tensor sub-components> is not implemented yet."
            )
        InitTensorFromExpressionNode(self.cache).handle(node)

    def function_definition(self, node: AstNode):
        pass  # User can define a function which they can actually call

    def symbol_definition(self, node: AstNode):
        self.cache.set_variable(node.args[0], str(node.args[1]))

    def define(self, node: AstNode):
        DefinitionNode(self.cache).handle(node)

    def coordinate_definition(self, node: AstNode):
        key = node.args[0]
        coordinates = node.args[1]

        self.cache.set_coordinates(coordinates)
        self.cache.set_variable(key, coordinates)

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

    def subs(self, node: AstNode):
        return node.args[0].subs(node.args[1], node.args[2])

    def limit(self, node: AstNode):
        return limit(*node.args)

    def expand(self, node: AstNode):
        return expand(*node.args)

    def diff(self, node: AstNode):
        return diff(*node.args)

    def integrate(self, node: AstNode):
        return integrate(*node.args)

    def simplify(self, node: AstNode):
        return simplify(*node.args)

    def latex(self, node: AstNode):
        return latex(*node.args)

    def solve(self, node: AstNode):
        return solve(*node.args)

    def numerical(self, node: AstNode):
        return N(*node.args)

    def exp(self, node: AstNode):
        return exp(*node.args)

    def dsolve(self, node: AstNode):
        return dsolve(*node.args)

    # Sympy Trigs
    def sin(self, node: AstNode):
        return sin(*node.args)

    def cos(self, node: AstNode):
        return cos(*node.args)

    def tan(self, node: AstNode):
        return tan(*node.args)

    def asin(self, node: AstNode):
        return asin(*node.args)

    def acos(self, node: AstNode):
        return acos(*node.args)

    def atan(self, node: AstNode):
        return atan(*node.args)

    def sinh(self, node: AstNode):
        return sinh(*node.args)

    def cosh(self, node: AstNode):
        return cosh(*node.args)

    def tanh(self, node: AstNode):
        return tanh(*node.args)

    def asinh(self, node: AstNode):
        return asinh(*node.args)

    def acosh(self, node: AstNode):
        return acosh(*node.args)

    def atanh(self, node: AstNode):
        return atanh(*node.args)

    def array(self, node: AstNode):
        return SymbolArray(list(node.args))

    def simplify_tensor(self, node: AstNode):
        tensor = node.args[0]
        tensor.components = simplify(tensor.components)
        return tensor

    # Sympy constants
    def constant(self, node: AstNode):
        a = "".join(node.args)
        if a == "pi":
            return pi
        elif a == "e":
            return E

    # Sympy symbols / function initiators
    def function(self, node: AstNode):
        func_id = node.identifier
        func = symbols("{}".format(func_id), cls=Function)(*node.args)
        return func

    def symbol(self, node: AstNode):
        a = "".join(node.args)

        if a in ["pi", "e"]:
            return self.constant(node)

        elif a == self.cache.metric_symbol:
            self.cache.set_metric_scalar()
            return self.cache.metric_scalar

        elif a == self.cache.ricci_symbol:
            self.cache.set_ricci_scalar()
            return self.cache.ricci_scalar

        elif not self.cache.has_variable(a):
            return symbols("{}".format(a))

        else:
            return self.cache.get_variable(str(a))

    def diag(self, node: AstNode):
        return TensorDiagBuilder().handle(node)

    def metric_tensor_definition(self, node: AstNode):
        TensorDefinitionNode(self.cache).handle(
            node
        )  # Tensor Setter : G_{a}_{b} := [[a, b],[c,d]]

    def tensor(self, node: AstNode):
        return TensorNode(self.cache).handle(
            node
        )  # Tensor Getter : G_{a}_{b} <-- go get me the object from cache or init new
    
    def tensor_definition(self, node: AstNode):
        test = node
        return TensorDefinitionNode(self.cache).handle(
            node
        ) 

    def print_(self, node: AstNode):
        return node.args