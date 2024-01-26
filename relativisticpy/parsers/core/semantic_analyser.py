######################  WHAT DOES THE SEMANTIC ANALYZER DO (FOR RELATIVISTICPY SPECIFICALLY)
# We know Sematic Analyzers are complex and is there to check whether the AST form a sensible set of instructions in the programming language.
# But we are being quite flexible with the word in this package and adding some of the meaning of what a Semantic Analyzer does:
#
#
#    Since this package is built without it's own symbolic/numeric/plotting tools/engines
#    We get into the issue where user wants this layer abstracted out - they simpley want to write equations and see the result.
#
#    We run into the issue where different tools we have different ways of interacting with each other:
#    Here is a list of package current/future interfaces we currently/will have:
#
#   SymPy <-> Numpy/Scipy
#   Scipy <-> MatplotLib
#   Numpy <-> MatplotLib
#   SymPy <-> MatplotLib
#   SymPy <-> RelativisticPy
#   Numpy/Scipy <-> RelativisticPy
#   MatplotLib <-> RelativisticPy
#
#   A generic Abstract Syntax Tree does not know of this layer of context. It simply parses the tokens into a Tree.
#
#   It is the job of the Semantic Analyzer (SA) to:
#       1. Check the AST and see if it makes grammatical sense. Throwing an inforemative error if it does not.
#       2. Add a layer of context to the AST so that when we implement the executor of the tree, it needs only to worry about implementation.
#       3. Add context as to what type the user wants from the AST as a whole. Does he AST return a graph? equation? tensor? if we know we can help optimize.
#
#   Since TECHNICALLY this whole parser module should NOT know the implementation of the methods and even less what tools (packages) are being used to implememt
#   the computations, we shall keep the langauge and naming abstract in such a way to keep the parser at a lower layer:
#   Language:
#
#       Types: Tensor - Symbolic Expression - Numerical Expression / Object - Visual Object
#

from relativisticpy.parsers.core.operator_lookup_tables import (
    powOperatorTypes,
    mulOperatorTypes,
    plusOperatorTypes,
    minusOperatorTypes,
    simplifyOperatorTypes,
    divOperatorTypes,
    assigningTypes,
    defineTypes,
    negOperatorTypes,
    posOperatorTypes
)
from relativisticpy.parsers.shared.constants import NodeKeys
from relativisticpy.parsers.shared.interfaces.semantic_analyzer import ISemanticAnalyzer
from relativisticpy.parsers.shared.models.semantic_analyzer_node import SANode


class SemanticAnalyzer(ISemanticAnalyzer):
    def __init__(self):
        self.symbol_table = set()

    def analyse_tree(self, mathjson):
        if isinstance(mathjson, dict):
            node_type = mathjson[NodeKeys.Node.value]
            handler = mathjson[NodeKeys.Handler.value]
            arguments = mathjson[NodeKeys.Arguments.value]
            node_methods = dir(self)

            if (
                (node_type in ["object", "function"])
                and (handler not in node_methods)
                and (node_type in node_methods)
            ):
                return getattr(self, node_type)(
                    AstNode(
                        node_type,
                        handler,
                        [*[self.analyse_tree(arg) for arg in arguments]],
                    )
                )
            elif (
                handler in node_methods
            ):  # "developer defined" implemented in ast_traverser
                handler_return_value = getattr(self, handler)(
                    AstNode(
                        node_type,
                        handler,
                        [*[self.analyse_tree(arg) for arg in arguments]],
                    )
                )
                return handler_return_value
            else:
                raise Exception(
                    f"Method '{handler}' has not been inplemented by {self}. Please implement '{handler}' within {self.__class__} and declaire it in NodeConfiguration parameter."
                )
        else:
            return mathjson

    # Cache Node handlers
    def assigner(self, node: SANode):
        lhs_node_type = node.args[0].callback_return_type
        rhs_node_type = node.args[1].callback_return_type
        assigning_types = assigningTypes[lhs_node_type][rhs_node_type]
        if assigning_types == 'tensor_expr_assignment':
            node.args[0] = SANode(
                                node="tensor_identifyer",
                                callback="tensor_identifyer",
                                callback_return_type="tensor",
                                args=node.args[0].args,
                        )
            return SANode(
                node="=",
                callback=assigning_types,
                callback_return_type="none",
                args=node.args,
        )
        elif assigning_types == 'tensor_component_assigning':
            node.args[0] = SANode(
                                node="tensor_identifyer",
                                callback="tensor_identifyer",
                                callback_return_type="tensor",
                                args=node.args[0].args,
                        )
            return SANode(
                node="=",
                callback=assigning_types,
                callback_return_type="none",
                args=node.args,
        )
        elif assigning_types == 'variable_assignment':
            node.args[0] = SANode(
                                node="variable_identifyer",
                                callback="symbol_key",
                                callback_return_type="symbol",
                                args=node.args[0].args,
                        )
            return SANode(
                node="=",
                callback=assigning_types,
                callback_return_type="none",
                args=node.args,
        )
        else:
            return SANode(
                node="=",
                callback=assigning_types,
                callback_return_type="none",
                args=node.args,
            )

    def define(self, node: SANode):
        lhs_node_type = node.args[0].callback_return_type
        rhs_node_type = node.args[1].callback_return_type
        node_define_type = defineTypes[lhs_node_type][rhs_node_type]
        if node_define_type == 'metric_tensor_definition':
            node.args[0] = SANode(
                                node="tensor_identifyer",
                                callback="tensor_identifyer",
                                callback_return_type="tensor",
                                args=node.args[0].args,
                        )
        if node_define_type == 'coordinate_definition':
            node.args[0] = SANode(
                                node=node.args[0].node,
                                callback='definition_identifyer',
                                callback_return_type="symbol",
                                args=node.args[0].args,
                        )
            return SANode(
                node=":=",
                callback=node_define_type,
                callback_return_type="none",
                args=node.args,
        )
        return SANode(
            node=":=",
            callback=node_define_type,
            callback_return_type="none",
            args=node.args,
        )

    def sub(self, node: SANode):
        lhs_node_type = node.args[0].callback_return_type
        rhs_node_type = node.args[1].callback_return_type
        resulting_node_type = minusOperatorTypes[lhs_node_type][rhs_node_type]
        return SANode(
            node="-",
            callback="sub",
            callback_return_type=resulting_node_type,
            args=node.args,
        )

    def add(self, node: SANode):
        lhs_node_type = node.args[0].callback_return_type
        rhs_node_type = node.args[1].callback_return_type
        resulting_node_type = plusOperatorTypes[lhs_node_type][rhs_node_type]
        return SANode(
            node="+",
            callback="add",
            callback_return_type=resulting_node_type,
            args=node.args,
        )

    def mul(self, node: SANode):
        lhs_node_type = node.args[0].callback_return_type
        rhs_node_type = node.args[1].callback_return_type
        resulting_node_type = mulOperatorTypes[lhs_node_type][rhs_node_type]
        return SANode(
            node="*",
            callback="mul",
            callback_return_type=resulting_node_type,
            args=node.args,
        )

    def div(self, node: SANode):
        lhs_node_type = node.args[0].callback_return_type
        rhs_node_type = node.args[1].callback_return_type
        resulting_node_type = divOperatorTypes[lhs_node_type][rhs_node_type]
        return SANode(
            node="/",
            callback="div",
            callback_return_type=resulting_node_type,
            args=node.args,
        )

    def pow(self, node: SANode):
        lhs_node_type = node.args[0].callback_return_type
        rhs_node_type = node.args[1].callback_return_type
        resulting_node_type = powOperatorTypes[lhs_node_type][rhs_node_type]
        return SANode(
            node="**",
            callback="pow",
            callback_return_type=resulting_node_type,
            args=node.args,
        )

    def neg(self, node: SANode):
        node_type = node.args[0].callback_return_type
        resulting_node_type = negOperatorTypes[node_type]
        return SANode(
            node="negative",
            callback="neg",
            callback_return_type=resulting_node_type,
            args=node.args,
        )

    def pos(self, node: SANode):
        node_type = node.args[0].callback_return_type
        resulting_node_type = posOperatorTypes[node_type]
        return SANode(
            node="positive",
            callback="pos",
            callback_return_type=resulting_node_type,
            args=node.args,
        )

    ###### THE CALLBACK OF THE NODE HANDLERS BELLOW ARE CREATION => WE KNOW WHAT THE RETURN TYPE IS AS WE ARE CREATING THE OBJECTS

    def function(self, node: SANode):
        # We should actually be checking that the arguments are correct!
        return SANode(
            node=node.node,
            callback=node.handler,
            callback_return_type="function",
            args=node.args,
        )

    def int(self, node: SANode):
        return SANode(
            node=node.node,
            callback=node.handler,
            callback_return_type="int",
            args=node.args,
        )

    def float(self, node: SANode):
        return SANode(
            node=node.node,
            callback=node.handler,
            callback_return_type="float",
            args=node.args,
        )

    def tensor(self, node: SANode):
        return SANode(
            node=node.node,
            callback=node.handler,
            callback_return_type="tensor",
            args=node.args,
        )

    def symbol(self, node: SANode):
        return SANode(
            node=node.node,
            callback=node.handler,
            callback_return_type="symbol",
            args=node.args,
        )

    def sym_expr(self, node: SANode):
        return SANode(
            node=node.node,
            callback=node.handler,
            callback_return_type="sym_expr",
            args=node.args,
        )

    def array(self, node: SANode):
        return SANode(
            node=node.node,
            callback=node.handler,
            callback_return_type="array",
            args=node.args,
        )

    ##### BUILT IN FUNCTIONS TO RELATIVISTICPY (will later be implemented by sympy but we are just checking the grammar and Semantics are correct)
    def simplify(self, node: SANode):
        arg_node_type = node.args[0].callback_return_type
        resulting_node_type = simplifyOperatorTypes[arg_node_type]
        if resulting_node_type == "tensor":
            return SANode(
                node="function",
                callback="simplify_tensor",
                callback_return_type=resulting_node_type,
                args=node.args,
            )
        return SANode(
            node="function",
            callback="simplify",
            callback_return_type=resulting_node_type,
            args=node.args,
        )

    def limit(self, node: SANode):
        # return types is always either = (infinity, int, float, symbol, sym_expr) which depends on the value of argument
        # The most general these types being sym_expr
        return SANode(
            node=node.node,
            callback=node.handler,
            callback_return_type="sym_expr",
            args=node.args,
        )

    def expand(self, node: SANode):
        # Expanding an sym_expr will always return another sym_expr
        return SANode(
            node=node.node,
            callback=node.handler,
            callback_return_type="sym_expr",
            args=node.args,
        )

    def diff(self, node: SANode):
        return SANode(
            node=node.node,
            callback=node.handler,
            callback_return_type="sym_expr",
            args=node.args,
        )

    def integrate(self, node: SANode):
        return SANode(
            node=node.node,
            callback=node.handler,
            callback_return_type="sym_expr",
            args=node.args,
        )

    def latex(self, node: SANode):
        return SANode(
            node=node.node,
            callback=node.handler,
            callback_return_type="sym_expr",
            args=node.args,
        )

    def solve(self, node: SANode):
        # actually returns an equation which is a compound type, composing of 'sym|function = sym_expr|sym'
        return SANode(
            node=node.node,
            callback=node.handler,
            callback_return_type="sym_expr",
            args=node.args,
        )

    def numerical(self, node: SANode):
        return SANode(
            node=node.node,
            callback=node.handler,
            callback_return_type="sym_expr",
            args=node.args,
        )

    def exp(self, node: SANode):
        return SANode(
            node=node.node,
            callback=node.handler,
            callback_return_type="sym_expr",
            args=node.args,
        )

    def dsolve(self, node: SANode):
        # actually returns an equation which is a compound type, composing of 'sym|function = sym_expr|sym'
        return SANode(
            node=node.node,
            callback=node.handler,
            callback_return_type="sym_expr",
            args=node.args,
        )

    def sin(self, node: SANode):
        return SANode(
            node=node.node,
            callback=node.handler,
            callback_return_type="sym_expr",
            args=node.args,
        )

    def cos(self, node: SANode):
        return SANode(
            node=node.node,
            callback=node.handler,
            callback_return_type="sym_expr",
            args=node.args,
        )

    def tan(self, node: SANode):
        return SANode(
            node=node.node,
            callback=node.handler,
            callback_return_type="sym_expr",
            args=node.args,
        )

    def asin(self, node: SANode):
        return SANode(
            node=node.node,
            callback=node.handler,
            callback_return_type="sym_expr",
            args=node.args,
        )

    def acos(self, node: SANode):
        return SANode(
            node=node.node,
            callback=node.handler,
            callback_return_type="sym_expr",
            args=node.args,
        )

    def atan(self, node: SANode):
        return SANode(
            node=node.node,
            callback=node.handler,
            callback_return_type="sym_expr",
            args=node.args,
        )

    def sinh(self, node: SANode):
        return SANode(
            node=node.node,
            callback=node.handler,
            callback_return_type="sym_expr",
            args=node.args,
        )

    def cosh(self, node: SANode):
        return SANode(
            node=node.node,
            callback=node.handler,
            callback_return_type="sym_expr",
            args=node.args,
        )

    def tanh(self, node: SANode):
        return SANode(
            node=node.node,
            callback=node.handler,
            callback_return_type="sym_expr",
            args=node.args,
        )

    def asinh(self, node: SANode):
        return SANode(
            node=node.node,
            callback=node.handler,
            callback_return_type="sym_expr",
            args=node.args,
        )

    def acosh(self, node: SANode):
        return SANode(
            node=node.node,
            callback=node.handler,
            callback_return_type="sym_expr",
            args=node.args,
        )

    def atanh(self, node: SANode):
        return SANode(
            node=node.node,
            callback=node.handler,
            callback_return_type="sym_expr",
            args=node.args,
        )
