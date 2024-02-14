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

from dataclasses import dataclass
from typing import List, Union

from relativisticpy.parsers.parsers.base import ParserResult

from .operator_lookup_tables import (
    powOperatorTypes,
    mulOperatorTypes,
    plusOperatorTypes,
    minusOperatorTypes,
    simplifyOperatorTypes,
    diffOperatorTypes,
    divOperatorTypes,
    assigningTypes,
    ID_definitionsLookup,
    negOperatorTypes,
    posOperatorTypes,
    integrateOperatorTypes,
    trigFunctionFunctionType
)
from relativisticpy.parsers.shared.constants import NodeKeys
from relativisticpy.parsers.shared.errors import Error, IllegalAssignmentError, IllegalSyntaxError
from relativisticpy.parsers.types.base import AstNode, BinaryNode, UnaryNode, IntNode, FloatNode, ArrayNode, NotNode, PosNode, NegNode, PrintNode, SymbolNode, Infinitesimal
from relativisticpy.parsers.types.gr_nodes import TensorNode, Function, Definition

@dataclass
class ActionTree:
    ast: AstNode
    return_type: str
    returns_object: bool


@dataclass
class GrScriptTree:
    action_trees: List[ActionTree]
    contains_error: bool
    display_error_str: List[str]


class SemanticAnalyzer:
    """
    Responsibilities:
        1. If there is an Error node: take it, store it, show it.
        2. If there is a function definition, and the function is called later on, tag the function as a callable.
        3. If there is a tensor expression, track the index rules and if it does not match, place an Error, show it.
        4. If there is a type issue, build an error, and return the error.
        5. If there is an assignment, check that the type being stored/computed is valid for the identifier which it is assigned to. Error check.
    """

    def __init__(self):
        self.function_table = set()

    def analyse(self, parser_result: ParserResult) -> GrScriptTree:
        self.raw_code = parser_result.code
        all_nodes = parser_result.ast_tree
        # Error types of the class reset each time we call call exe method.
        self.assignment_error = (
            None  # T_{a} = a -------> Error on incompatible assignment types.
        )
        self.invalid_arguments_error = None  # f(x, x**2) = x**4 -----> this is wrong as we cannot accept expr in func definition args.
        self.error_from_ast_node = None
        self.invalid_tensor_expression_error = None

        # init GrScriptTree attributes so we set them along the way.
        self.action_trees = []
        self.contains_error = False
        self.display_error_str = ""

        # Build context from tree
        for ast_node in all_nodes:
            self.__analyse_tree(ast_node)
            self.action_trees.append(
                ActionTree(
                    ast_node,
                    ast_node.data_type if isinstance(ast_node, AstNode) else None,
                    ast_node.data_type
                    in (
                        "tensor",
                        "int",
                        "float",
                        "array",
                        "symbol",
                        "sym_expr",
                        "function",
                    ) if isinstance(ast_node, AstNode) else False,
                )
            )
        
        return GrScriptTree(
            self.action_trees, 
            self.contains_error or (self.invalid_arguments_error != None or self.error_from_ast_node != None or self.invalid_tensor_expression_error != None),
            self.display_error_str
            )

    def __analyse_tree(self, node: Union[AstNode, Error]):

        # Handle error nodes
        if isinstance(node, (Error, IllegalSyntaxError)):
            self.error_from_ast_node = node
            self.contains_error = True
            self.display_error_str = node.as_string()
            return node

        # Check and process child nodes if they exist
        if hasattr(node, 'is_leaf'):
            if not node.is_leaf:
                node.execute_node(self.__analyse_tree)

        callback_method = getattr(self, node.callback)
        if not any(self.error_object_in_args_of(node)):
            callback_method(node)
        return node

    def error_object_in_args_of(self, node) -> List[bool]:
        args = []
        for arg in node.args:
            if isinstance(arg, Error):
                self.contains_error = True
                self.error_from_ast_node = node # Not really nessesary 
                self.display_error_str = arg.as_string()
                args.append(True)
            else:
                args.append(False)
        return args

    # Cache Node handlers
    def definition(self, node: AstNode):
        node.data_type = 'none'

    # def definition(self, node: AstNode):
    #     rhs = node.args[1].data_type

    #     if ID_definitionsLookup[rhs] == 'undef':
    #         self.assignment_error = IllegalAssignmentError(node.args[0].position, node.args[1].position, "The LHS and RHS of the definiton expression you've entered is not allowed, or had not yet been implemented.", self.raw_code)
    #         self.display_error_str = self.assignment_error.as_string()
    #         self.contains_error = True

    #     node.data_type = 'none'

    def function_def(self, node: Function):
        self.__analyse_tree(node.args[0].executable)

    def tensor_assignment(self, node: TensorNode):
        node.data_type = 'none'

    def sub(self, node: BinaryNode):
        node.data_type = minusOperatorTypes[node.left_child.data_type][node.right_child.data_type]

    def add(self, node: BinaryNode):
        node.data_type = plusOperatorTypes[node.left_child.data_type][node.right_child.data_type]

    def mul(self, node: BinaryNode):
        node.data_type = mulOperatorTypes[node.left_child.data_type][node.right_child.data_type]

    def div(self, node: BinaryNode):
        if isinstance(nominator := node.left_child, Infinitesimal) and isinstance(denominator := node.right_child, Infinitesimal):
            # This is now no longer a divition node, but user means to defined a derivative.
            if nominator.diff_order_as_int != denominator.diff_order_as_int:
                # Pass Error => User defined derivative orders incorrectly non matching.
                node = IllegalSyntaxError(
                                            node.position,
                                            node.position,
                                            f"Differentiation order error: You have a missmatch in diff order \n d^{nominator.diff_order_as_int}.../d...^{denominator.diff_order_as_int}.",
                                            self.raw_code
                                        )
                self.error_from_ast_node = node
                self.contains_error = True
                self.display_error_str = node.as_string()
            else:
                if denominator.expression.callback == 'div':
                    self.analyse(denominator.expression)
                if nominator.expression.callback == 'div':
                    self.analyse(nominator.expression)
                else:
                    node.callback = 'diff'
                    node.args = [nominator.expression, denominator.expression, nominator.diff_order]
                if nominator.data_type == 'tensor':
                    node.data_type = 'tensor'
                else:
                    node.data_type = 'sym_expr'
        else:
            node.data_type = divOperatorTypes[node.left_child.data_type][node.right_child.data_type]

    def pow(self, node: BinaryNode):
        node.data_type = powOperatorTypes[node.left_child.data_type][node.right_child.data_type]

    def neg(self, node: NegNode):
        node_type = node.args[0].data_type
        node.data_type = negOperatorTypes[node_type]

    def pos(self, node: PosNode):
        node_type = node.args[0].data_type
        node.data_type = posOperatorTypes[node_type]

    def constant(self, node: PosNode):
        node.data_type = 'symbol'

    def coordinate_definition(self, node: Definition):
        pass

    ###### THE CALLBACK OF THE NODE HANDLERS BELLOW ARE CREATION => WE KNOW WHAT THE RETURN TYPE IS AS WE ARE CREATING THE OBJECTS

    def function(self, node: AstNode):
        pass

    def int(self, node: IntNode):
        pass

    def float(self, node: FloatNode):
        pass

    def print_(self, node: PrintNode):
        pass

    def tensor(self, node: TensorNode):
        pass

    def symbol(self, node: SymbolNode):
        pass

    def array(self, node: ArrayNode):
        pass

    def not_(self, node: NotNode):
        pass

    def and_(self, node: BinaryNode):
        pass

    def or_(self, node: BinaryNode):
        pass

    def eqequal_(self, node: BinaryNode):
        pass

    def less(self, node: BinaryNode):
        pass

    def greater(self, node: BinaryNode):
        pass

    def lessequal(self, node: BinaryNode):
        pass

    def greaterequal(self, node: BinaryNode):
        pass

    def infinitesimal(self, node: Infinitesimal):
        if node.expression.data_type == 'tensor':
            node.data_type = 'tensor'
        else:
            node.data_type = 'sym_expr'

    ##### BUILT IN FUNCTIONS TO RELATIVISTICPY (will later be implemented by sympy but we are just checking the grammar and Semantics are correct)
    def simplify(self, node: Function):
        node_type = node.args[0].data_type
        node.data_type = simplifyOperatorTypes[node_type]

    def tsimplify(self, node: Function):
        node.data_type = 'array'

    def lim(self, node: Function):
        node.data_type = 'sym_expr'

    def diag(self, node: Function):
        node.data_type = 'array'

    def expand(self, node: Function):
        node.data_type = 'sym_expr'

    def diff(self, node: Function):
        node_type = node.args[0].data_type
        node.data_type = diffOperatorTypes[node_type]

    def integrate(self, node: Function):
        node_type = node.args[0].data_type
        node.data_type = integrateOperatorTypes[node_type]

    def latex(self, node: Function):
        node.data_type = 'sym_expr'

    def subs(self, node: Function):
        node.data_type = 'sym_expr'

    def solve(self, node: Function):
        node.data_type = 'sym_expr'

    def sum(self, node: Function):
        node.data_type = 'sym_expr'

    def sqrt(self, node: Function):
        node.data_type = 'sym_expr'

    def subs(self, node: Function):
        node.data_type = 'sym_expr'

    def dosum(self, node: Function):
        node.data_type = 'sym_expr'

    def prod(self, node: Function):
        node.data_type = 'sym_expr'

    def doprod(self, node: Function):
        node.data_type = 'sym_expr'

    def numerical(self, node: Function):
        node.data_type = 'sym_expr'

    def func_derivative(self, node: Function):
        node.data_type = 'sym_expr'

    def exp(self, node: Function):
        node.data_type = 'sym_expr'

    def dsolve(self, node: Function):
        node.data_type = 'sym_expr'
    
    def RHS(self, node: Function):
        node.data_type = 'sym_expr'

    def LHS(self, node: Function):
        node.data_type = 'sym_expr'

    def sin(self, node: Function):
        node.data_type = trigFunctionFunctionType[node.args[0].data_type]

    def cos(self, node: Function):
        node.data_type = trigFunctionFunctionType[node.args[0].data_type]

    def tan(self, node: Function):
        node.data_type = trigFunctionFunctionType[node.args[0].data_type]

    def asin(self, node: Function):
        node.data_type = trigFunctionFunctionType[node.args[0].data_type]

    def acos(self, node: Function):
        node.data_type = trigFunctionFunctionType[node.args[0].data_type]

    def atan(self, node: Function):
        node.data_type = trigFunctionFunctionType[node.args[0].data_type]

    def sinh(self, node: Function):
        node.data_type = trigFunctionFunctionType[node.args[0].data_type]

    def cosh(self, node: Function):
        node.data_type = trigFunctionFunctionType[node.args[0].data_type]

    def tanh(self, node: Function):
        node.data_type = trigFunctionFunctionType[node.args[0].data_type]

    def asinh(self, node: Function):
        node.data_type = trigFunctionFunctionType[node.args[0].data_type]

    def acosh(self, node: Function):
        node.data_type = trigFunctionFunctionType[node.args[0].data_type]

    def atanh(self, node: Function):
        node.data_type = trigFunctionFunctionType[node.args[0].data_type]
