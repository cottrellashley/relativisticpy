from dataclasses import dataclass
from typing import List

from relativisticpy.algebras import Indices, Idx, Tensor
from relativisticpy.diffgeom import Metric, MetricIndices
from relativisticpy.diffgeom import RicciScalar, MetricScalar, Ricci, Riemann, LeviCivitaConnection, Derivative, CovDerivative 

from relativisticpy.gr.einstein import EinsteinTensor

from relativisticpy.interpreter.protocols import Implementer
from relativisticpy.interpreter import ScopedState

from relativisticpy.symengine import (
    Symbol,
    Basic,
    Rational,
    Function,
    Sum,
    Product,
    SymbolArray,
    diff,
    integrate,
    simplify,
    symbols,
    solve,
    dsolve,
    expand,
    latex,
    limit,
    Pow,
    oo,
    E,
    Abs,
    N,
    I,
    pi,
    Eq,
    ln,
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
    factorial,
)

class RelPyError:
    pass

@dataclass
class AstNode:
    node: str
    handler: str
    args: List["AstNode"]

# @node_handler_implementor(parser=RelPyParser)

# @implements(Implementer) <-- This could do a iteration check that all methods from GrNodeTraverser are implemented by RelPyAstNodeTraverser
class RelPyAstNodeTraverser(Implementer):

    @property
    def state(self) -> ScopedState:
        return self._state
    
    @state.setter
    def state(self, state: ScopedState) -> None:
        self._state = state

    def clear(self, node: AstNode): self.state.reset()
    
    # Python lib implementations
    def sub(self, node: AstNode): return node.args[0] - node.args[1]
    def add(self, node: AstNode): return node.args[0] + node.args[1]
    def neg(self, node: AstNode): return -node.args[0]
    def pos(self, node: AstNode): return +node.args[0]
    def mul(self, node: AstNode): return node.args[0] * node.args[1]
    def div(self, node: AstNode): return node.args[0] / node.args[1]
    def pow(self, node: AstNode): return node.args[0] ** node.args[1]
    def int(self, node: AstNode): return int("".join(node.args))
    def float(self, node: AstNode): return float("".join(node.args))

    # Symengine implementations
    def subs(self, node: AstNode): return node.args[0].subs(node.args[1], node.args[2])
    def lim(self, node: AstNode): return limit(*node.args)
    def sqrt(self, node: AstNode): return Pow(node.args[0], Rational(1, 2)) if isinstance(node.args[0], Basic) else node.args[0] ** 0.5
    def expand(self, node: AstNode): return expand(*node.args)
    def integrate(self, node: AstNode): return integrate(*node.args)
    def simplify(self, node: AstNode): return simplify(*node.args)
    def latex(self, node: AstNode): return latex(*node.args)
    def solve(self, node: AstNode): return solve(*node.args)
    def numerical(self, node: AstNode): return N(*node.args)
    def equation(self, node: AstNode): return Eq(*node.args)
    def exp(self, node: AstNode): return exp(*node.args)
    def ln(self, node: AstNode): return ln(*node.args)
    def dsolve(self, node: AstNode): return dsolve(*node.args)
    def symbol(self, node: AstNode): return symbols("{}".format(node.var_key))
    def symbol_str(self, *arg, **kwargs): return Symbol(*arg, **kwargs)
    def print_(self, node: AstNode): return node.args
    def RHS(self, node: AstNode): return node.args[0].rhs
    def LHS(self, node: AstNode): return node.args[0].lhs
    def sum(self, node: AstNode): return Sum(node.args[0], (node.args[1], node.args[2], node.args[3]))
    def dosum(self, node: AstNode): return Sum(node.args[0], (node.args[1], node.args[2], node.args[3])).doit()
    def prod(self, node: AstNode): return Product(node.args[0], (node.args[1], node.args[2], node.args[3]))
    def doprod(self, node: AstNode): return Product(node.args[0], (node.args[1], node.args[2], node.args[3])).doit()
    def array(self, node: AstNode): return SymbolArray(list(node.args))
    def sin(self, node: AstNode): return sin(*node.args)
    def cos(self, node: AstNode): return cos(*node.args)
    def tan(self, node: AstNode): return tan(*node.args)
    def asin(self, node: AstNode): return asin(*node.args)
    def acos(self, node: AstNode): return acos(*node.args)
    def atan(self, node: AstNode): return atan(*node.args)
    def sinh(self, node: AstNode): return sinh(*node.args)
    def cosh(self, node: AstNode): return cosh(*node.args)
    def tanh(self, node: AstNode): return tanh(*node.args)
    def asinh(self, node: AstNode): return asinh(*node.args)
    def acosh(self, node: AstNode): return acosh(*node.args)
    def atanh(self, node: AstNode): return atanh(*node.args)
    def factorial(self, node: AstNode): return factorial(*node.args)
    def absolute(self, node: AstNode): return Abs(*node.args)

    def func_derivative(self, node: AstNode):
        if isinstance(node.args[1], Basic):
            wrt = list(node.args[1].free_symbols)[0] if not [] == list(node.args[1].free_symbols) else None
            return diff(node.args[0], wrt, node.args[2])

    def diff(self, node: AstNode):
        if isinstance(node.args[0], Tensor):
            old_tensor: Tensor = node.args[0]
            if len(node.args) == 3:
                new_tensor = Tensor(
                    old_tensor.indices,
                    diff(old_tensor.components, node.args[1], node.args[2]),
                    old_tensor.basis,
                )
            else:
                new_tensor = Tensor(
                    old_tensor.indices,
                    diff(old_tensor.components, node.args[1]),
                    old_tensor.basis,
                )
            return new_tensor
        ans = diff(*node.args)
        return ans

    def tsimplify(self, node: AstNode):
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
        elif a == "i":
            return I
        elif a in ["oo", "infty"]:
            return oo
        
    def symbolfunc(self, node: AstNode):
        if not self.state.current_scope.check_variable(node.identifier):
            func_id = node.identifier
            func = symbols("{}".format(func_id), cls=Function)(*node.args)
            return func
        else:
            return Function(self.state.current_scope.variables[node.identifier])(*node.args)

    def diag(self, node: AstNode):
        # Determine n from the length of diag_values
        n = len(node.args)

        # Create an NxN MutableDenseNDimArray with zeros
        ndarray = SymbolArray.zeros(n, n)

        # Set the diagonal values
        for i in range(n):
            ndarray[i, i] = node.args[i]

        return ndarray

    # TENSOR IMPLEMENTATIONS

    def metric_dependent_types(self, tensor_key: str):
        types_map = {
            self.state.get_variable("MetricSymbol"): Metric,
            self.state.get_variable("RicciSymbol"): Ricci,
            self.state.get_variable("EinsteinTensorSymbol"): EinsteinTensor,
            self.state.get_variable("ConnectionSymbol"): LeviCivitaConnection,
            self.state.get_variable("RiemannSymbol"): Riemann,
            self.state.get_variable("CovariantDerivativeSymbol"): CovDerivative
        }
        return types_map[tensor_key] if tensor_key in types_map else None

    def init_indices(self, node: AstNode):
        if not self.state.get_variable("MetricSymbol") == node.identifier:
            indices = node.indices.indices
            return Indices(*[Idx(symbol=idx.identifier, values=idx.values) if idx.covariant else -Idx(symbol=idx.identifier, values=idx.values) for idx in indices])
        indices = node.indices.indices
        return MetricIndices(*[Idx(symbol=idx.identifier, values=idx.values) if idx.covariant else -Idx(symbol=idx.identifier, values=idx.values) for idx in indices])

    def init_metric_tensor(self, indices: Indices, components: SymbolArray, basis: SymbolArray) -> Metric:
        "Based on the state of the Tensor node and the sate - we will initialize the indices of a tensor."
        return Metric(indices, components, basis)

    def init_einstein_array(self, indices: Indices, components: SymbolArray, basis: SymbolArray) -> Tensor:
        "Based on the state of the Tensor node and the sate - we will initialize the indices of a tensor."
        return Tensor(indices, components, basis)

    def init_ricci_scalar(self, node: AstNode) -> RicciScalar:
        "Based on the state of the Tensor node and the sate - we will initialize the indices of a tensor."
        metric = self.state.metric_tensor
        basis = self.state.get_variable("Coordinates")
        return RicciScalar(metric, basis)

    def init_metric_scalar(self, node: AstNode) -> MetricScalar:
        "Based on the state of the Tensor node and the sate - we will initialize the indices of a tensor."
        metric = self.state.metric_tensor
        basis = self.state.get_variable("Coordinates")
        return MetricScalar(metric, basis)
    
    def init_tensor_derivative(self, node: AstNode) -> Derivative:
        "Based on the state of the Tensor node and the sate - we will initialize the indices of a tensor."
        basis = self.state.get_variable("Coordinates")
        return Derivative(self.init_indices(node), basis)