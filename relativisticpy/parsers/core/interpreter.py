from dataclasses import dataclass
from relativisticpy.parsers.shared.constants import NodeKeys
from relativisticpy.parsers.shared.interfaces.interpreter import IInterpreter
from relativisticpy.parsers.shared.models.semantic_analyzer_node import SANode

@dataclass
class Node:
    node: str
    handler: str
    args: any


class Interpreter(IInterpreter):
    def __init__(self, node_tree_walker):
        self.node_tree_walker = node_tree_walker

    def interpret(self, mathjson):
        if isinstance(mathjson, SANode):
            node_type = mathjson.node
            handler = mathjson.callback
            arguments = mathjson.args
            node_methods = dir(self.node_tree_walker)
            if (
                (node_type in ["tensor", "symbol", "function"])
                and (handler not in node_methods)
                and (node_type in node_methods)
            ):
                return getattr(self.node_tree_walker, node_type)(
                    Node(
                        node_type,
                        handler,
                        [*[self.interpret(arg) for arg in arguments]],
                    )
                )
            elif handler in node_methods: # "developer defined" implemented in ast_traverser 
                handler_return_value = getattr(self.node_tree_walker, handler)(
                    Node(
                        node_type,
                        handler,
                        [*[self.interpret(arg) for arg in arguments]],
                    )
                )
                return handler_return_value
            else:
                raise Exception(
                    f"Method '{handler}' has not been inplemented by {self.node_tree_walker}. Please implement '{handler}' within {self.node_tree_walker.__class__} and declaire it in NodeConfiguration parameter."
                )
        else:
            return mathjson
