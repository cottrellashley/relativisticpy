from dataclasses import dataclass
from relativisticpy.parsers.shared.constants import NodeKeys
from relativisticpy.parsers.shared.interfaces.interpreter import IInterpreter


@dataclass
class Node:
    node: str
    handler: str
    args: any


class Interpreter(IInterpreter):
    def __init__(self, node_tree_walker):
        self.node_tree_walker = node_tree_walker

    def interpret(self, mathjson):
        if isinstance(mathjson, dict):
            node_type = mathjson[NodeKeys.Node.value]
            handler = mathjson[NodeKeys.Handler.value]
            arguments = mathjson[NodeKeys.Arguments.value]
            node_methods = dir(self.node_tree_walker)
            if (
                (node_type in ["object", "function"])
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
