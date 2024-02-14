from relativisticpy.core import EinsteinArray
from relativisticpy.workbook.itensors import (
    TensorHandler,
    TensorAssignmentHandler,
    DefinitionNode,
    InitTensorFromComponentsNode,
)
from relativisticpy.workbook.matchers import match_tensors, match_symbol
from relativisticpy.workbook.state import WorkbookState, TensorReference
from relativisticpy.utils import str_is_tensors
from relativisticpy.parsers.types.gr_nodes import TensorNode
from relativisticpy.parsers.types.base import UnaryNode, BinaryNode, AstNode

from relativisticpy.symengine import (
    Symbol,
    Basic,
    Rational,
    Function,
    Interval,
    Order,
    Sum,
    Product,
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
    root,
    exp_polar,
    bell,
    bernoulli,
    Pow,
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

# @node_handler_implementor(parser=RelPyParser)


class RelPyAstNodeTraverser:
    def __init__(self, cache: WorkbookState):
        self.cache = cache

    # Cache Node handlers
    def definition(self, node: AstNode):
        self.cache.set_variable(str(node.args[0]), node.args[1])

    def tensor_assignment(self, node: TensorNode):
        tref = TensorReference(node)
        if node.sub_components_called:
            raise ValueError(
                "Feature: <Assigning value to tensor sub-components> is not implemented yet."
            )
        if node.identifier == self.cache.metric_symbol:
            tref.is_metric = True
        TensorAssignmentHandler(self.cache).handle(tref)

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

    def lim(self, node: AstNode):
        return limit(*node.args)
    
    def sqrt(self, node: AstNode):
        if isinstance(node.args[0], Basic):
            return Pow(node.args[0], Rational(1, 2))
        else:
            return node.args[0] ** 0.5

    def expand(self, node: AstNode):
        return expand(*node.args)
    
    def func_derivative(self, node: AstNode):
        if isinstance(node.args[1], Basic):
            wrt = list(node.args[1].free_symbols)[0] if not [] == list(node.args[1].free_symbols) else None
            return diff(node.args[0], wrt, node.args[2])

    def diff(self, node: AstNode):
        if isinstance(node.args[0], EinsteinArray):
            old_tensor: EinsteinArray = node.args[0]
            if len(node.args) == 3:
                new_tensor = EinsteinArray(
                    old_tensor.indices,
                    diff(old_tensor.components, node.args[1], node.args[2]),
                    old_tensor.basis,
                )
            else:
                new_tensor = EinsteinArray(
                    old_tensor.indices,
                    diff(old_tensor.components, node.args[1]),
                    old_tensor.basis,
                )
            return new_tensor
        ans = diff(*node.args)
        return ans

    def integrate(self, node: AstNode):
        return integrate(*node.args)

    def simplify(self, node: AstNode):
        return simplify(*node.args)

    def latex(self, node: AstNode):
        return latex(*node.args)

    def solve(self, node: AstNode):
        a = node.args
        res = solve(*a)
        return res

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

    def tsimplify(self, node: UnaryNode):
        tensor = node.args[0]
        tensor.components = simplify(
            list(tensor.components)[0]
        )  # ??????? Why do we need to call list ? can we standardise ?
        return tensor

    # Sympy constants
    def constant(self, node: AstNode):
        a = "".join(node.args)
        if a == "pi":
            return pi
        elif a == "e":
            return E
        elif a in ["oo", "infty"]:
            return oo

    # Sympy symbols / function initiators
    def function(self, node: AstNode):
        func_id = node.identifier
        func = symbols("{}".format(func_id), cls=Function)(*node.args)
        return func

    def symbol(self, node: AstNode):
        a = "".join(node.args)

        if a == self.cache.metric_symbol:
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
        # Determine n from the length of diag_values
        n = len(node.args)

        # Create an NxN MutableDenseNDimArray with zeros
        ndarray = SymbolArray.zeros(n, n)

        # Set the diagonal values
        for i in range(n):
            ndarray[i, i] = node.args[i]

        return ndarray

    def tensor(self, node: AstNode):
        tref = TensorReference(node)
        if tref.id == self.cache.metric_symbol:
            tref.is_metric = True
        return TensorHandler(self.cache).handle(
            tref
        )  # Tensor Getter : G_{a}_{b} <-- go get me the object from cache or init new

    def print_(self, node: AstNode):
        return node.args

    def RHS(self, node: AstNode):
        return node.args[0].rhs

    def LHS(self, node: AstNode):
        return node.args[0].lhs

    def sum(self, node: AstNode):
        return Sum(node.args[0], (node.args[1], node.args[2], node.args[3]))

    def dosum(self, node: AstNode):
        return Sum(node.args[0], (node.args[1], node.args[2], node.args[3])).doit()

    def prod(self, node: AstNode):
        return Product(node.args[0], (node.args[1], node.args[2], node.args[3]))

    def doprod(self, node: AstNode):
        return Product(node.args[0], (node.args[1], node.args[2], node.args[3])).doit()
