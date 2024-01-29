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
)
from relativisticpy.parsers.shared.constants import NodeKeys
from relativisticpy.parsers.shared.errors import Error, IllegalAssignmentError, IllegalSyntaxError
from relativisticpy.parsers.types.base import AstNode


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
                    ast_node.inferenced_type if isinstance(ast_node, AstNode) else None,
                    ast_node.inferenced_type
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
        if hasattr(node, "args") and isinstance(node.args, list) and all([isinstance(arg, AstNode) for arg in node.args]):
            for i, arg in enumerate(node.args):
                # Process each child node only if it hasn't been processed yet
                if not arg.inferenced_type:
                    node.args[i] = self.__analyse_tree(arg)

        # Execute the callback for the current node if it's not an error node
        if not isinstance(node, (Error, IllegalSyntaxError)):
            callback_method = getattr(self, node.callback, None)
            if callback_method:
                # Call the callback method with the current node
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
    def assignment(self, node: AstNode):
        if not any(self.error_object_in_args_of(node)):
            lhs_child_type, rhs_child_type = node.args[0].inferenced_type, node.args[1].inferenced_type
            node.inferenced_type = assigningTypes[lhs_child_type][rhs_child_type]

    def definition(self, node: AstNode):
        if not any(self.error_object_in_args_of(node)):
            rhs = node.args[1].inferenced_type

            if ID_definitionsLookup[rhs] == 'undef':
                self.assignment_error = IllegalAssignmentError(node.args[0].position, node.args[1].position, "The LHS and RHS of the definiton expression you've entered is not allowed, or had not yet been implemented.", self.raw_code)
                self.display_error_str = self.assignment_error.as_string()
                self.contains_error = True

            node.inferenced_type = 'none'

    def tensor_component_assignment(self, node: AstNode):
        if not any(self.error_object_in_args_of(node)):
            lhs_child_type, rhs_child_type = node.args[0].inferenced_type, node.args[1].inferenced_type

    def tensor_expr_assignment(self, node: AstNode):
        if not any(self.error_object_in_args_of(node)):
            lhs_child_type, rhs_child_type = node.args[0].inferenced_type, node.args[1].inferenced_type

    def function_def(self, node: AstNode):
        if not any(self.error_object_in_args_of(node)):
            self.__analyse_tree(node.args[0].executable)

    def tensor_definition(self, node: AstNode):
        if not any(self.error_object_in_args_of(node)):
            lhs_child_type, rhs_child_type = node.args[0].inferenced_type, node.args[1].inferenced_type

    def sub(self, node: AstNode):
        if not any(self.error_object_in_args_of(node)):
            lhs_child_type, rhs_child_type = node.args[0].inferenced_type, node.args[1].inferenced_type
            node.inferenced_type = minusOperatorTypes[lhs_child_type][rhs_child_type]

    def add(self, node: AstNode):
        if not any(self.error_object_in_args_of(node)):
            lhs_child_type, rhs_child_type = node.args[0].inferenced_type, node.args[1].inferenced_type
            node.inferenced_type = plusOperatorTypes[lhs_child_type][rhs_child_type]

    def mul(self, node: AstNode):
        if not any(self.error_object_in_args_of(node)):
            lhs_child_type, rhs_child_type = node.args[0].inferenced_type, node.args[1].inferenced_type
            node.inferenced_type = mulOperatorTypes[lhs_child_type][rhs_child_type]

    def div(self, node: AstNode):
        if not any(self.error_object_in_args_of(node)):
            lhs_child_type, rhs_child_type = node.args[0].inferenced_type, node.args[1].inferenced_type
            node.inferenced_type = divOperatorTypes[lhs_child_type][rhs_child_type]

    def pow(self, node: AstNode):
        if not any(self.error_object_in_args_of(node)):
            lhs_child_type, rhs_child_type = node.args[0].inferenced_type, node.args[1].inferenced_type
            node.inferenced_type = powOperatorTypes[lhs_child_type][rhs_child_type]

    def neg(self, node: AstNode):
        if not any(self.error_object_in_args_of(node)):
            node_type = node.args[0].inferenced_type
            node.inferenced_type = negOperatorTypes[node_type]

    def pos(self, node: AstNode):
        if not any(self.error_object_in_args_of(node)):
            node_type = node.args[0].inferenced_type
            node.inferenced_type = posOperatorTypes[node_type]

    ###### THE CALLBACK OF THE NODE HANDLERS BELLOW ARE CREATION => WE KNOW WHAT THE RETURN TYPE IS AS WE ARE CREATING THE OBJECTS

    def function(self, node: AstNode):
        if not any(self.error_object_in_args_of(node)):
            pass

    def int(self, node: AstNode):
        if not any(self.error_object_in_args_of(node)):
            pass

    def float(self, node: AstNode):
        if not any(self.error_object_in_args_of(node)):
            pass

    def print_(self, node: AstNode):
        if not any(self.error_object_in_args_of(node)):
            pass

    def tensor(self, node: AstNode):
        if not any(self.error_object_in_args_of(node)):
            pass

    def symbol(self, node: AstNode):
        if not any(self.error_object_in_args_of(node)):
            pass

    def sym_expr(self, node: AstNode):
        if not any(self.error_object_in_args_of(node)):
            pass

    def array(self, node: AstNode):
        if not any(self.error_object_in_args_of(node)):
            pass

    def not_(self, node: AstNode):
        if not any(self.error_object_in_args_of(node)):
            pass

    def and_(self, node: AstNode):
        if not any(self.error_object_in_args_of(node)):
            pass

    def or_(self, node: AstNode):
        if not any(self.error_object_in_args_of(node)):
            pass

    def eqequal_(self, node: AstNode):
        if not any(self.error_object_in_args_of(node)):
            pass

    def less(self, node: AstNode):
        if not any(self.error_object_in_args_of(node)):
            pass

    def greater(self, node: AstNode):
        if not any(self.error_object_in_args_of(node)):
            pass

    def lessequal(self, node: AstNode):
        if not any(self.error_object_in_args_of(node)):
            pass

    def greaterequal(self, node: AstNode):
        if not any(self.error_object_in_args_of(node)):
            pass

    ##### BUILT IN FUNCTIONS TO RELATIVISTICPY (will later be implemented by sympy but we are just checking the grammar and Semantics are correct)
    def simplify(self, node: AstNode):
        if not any(self.error_object_in_args_of(node)):
            node_type = node.args[0].inferenced_type
            node.inferenced_type = simplifyOperatorTypes[node_type]

    def limit(self, node: AstNode):
        if not any(self.error_object_in_args_of(node)):
            pass

    def expand(self, node: AstNode):
        if not any(self.error_object_in_args_of(node)):
            pass

    def diff(self, node: AstNode):
        test = not any(self.error_object_in_args_of(node))
        if test:
            node_type = node.args[0].inferenced_type
            node.inferenced_type = diffOperatorTypes[node_type]

    def integrate(self, node: AstNode):
        if not any(self.error_object_in_args_of(node)):
            node_type = node.args[0].inferenced_type
            node.inferenced_type = integrateOperatorTypes[node_type]

    def latex(self, node: AstNode):
        if not any(self.error_object_in_args_of(node)):
            pass

    def solve(self, node: AstNode):
        if not any(self.error_object_in_args_of(node)):
            pass

    def numerical(self, node: AstNode):
        if not any(self.error_object_in_args_of(node)):
            pass

    def exp(self, node: AstNode):
        if not any(self.error_object_in_args_of(node)):
            pass

    def dsolve(self, node: AstNode):
        pass

    def sin(self, node: AstNode):
        if not any(self.error_object_in_args_of(node)):
            pass

    def cos(self, node: AstNode):
        if not any(self.error_object_in_args_of(node)):
            pass

    def tan(self, node: AstNode):
        pass

    def asin(self, node: AstNode):
        if not any(self.error_object_in_args_of(node)):
            pass

    def acos(self, node: AstNode):
        if not any(self.error_object_in_args_of(node)):
            pass

    def atan(self, node: AstNode):
        if not any(self.error_object_in_args_of(node)):
            pass

    def sinh(self, node: AstNode):
        if not any(self.error_object_in_args_of(node)):
            pass

    def cosh(self, node: AstNode):
        if not any(self.error_object_in_args_of(node)):
            pass

    def tanh(self, node: AstNode):
        if not any(self.error_object_in_args_of(node)):
            pass

    def asinh(self, node: AstNode):
        if not any(self.error_object_in_args_of(node)):
            pass

    def acosh(self, node: AstNode):
        if not any(self.error_object_in_args_of(node)):
            pass

    def atanh(self, node: AstNode):
        if not any(self.error_object_in_args_of(node)):
            pass
