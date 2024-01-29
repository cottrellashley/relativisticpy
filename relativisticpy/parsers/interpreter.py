from dataclasses import dataclass
from typing import List
from relativisticpy.parsers.analyzers.base import ActionTree, GrScriptTree
from relativisticpy.parsers.types.base import AstNode


@dataclass
class ReturnObject:
    value: any
    _type: str


class Interpreter:

    def __init__(self, gr_script: GrScriptTree):
        self.gr_script = gr_script
        self.return_list = []

    def exe_script(self, exe_object: any):
        if self.gr_script.contains_error:
            return self.gr_script.display_error_str
        for action_tree in self.gr_script.action_trees:
            if action_tree.returns_object:
                self.return_list.append(
                    ReturnObject(
                        value=self.exe(action_tree.ast, exe_object),
                        _type=action_tree.return_type,
                    )
                )
            else:
                self.exe(action_tree.ast, exe_object)
        return self.return_list

    def exe(self, ast_node: AstNode, exe_object: any):

        # Check if the node has arguments and process them if necessary.
        # This step ensures that each child node is processed before the current node is passed to the callback.
        if hasattr(ast_node, "args"):
            for i, arg in enumerate(ast_node.args):
                if hasattr(arg, "args"):  # If the argument is an AstNode, process it recursively.
                    ast_node.args[i] = self.exe(arg, exe_object)
                # Note: If the argument is not an AstNode, it's left as is.

        # Use getattr to retrieve the callback method from the exe_object using the node's callback name.
        # Call this method with the full ast_node, which includes its children within node.args.
        callback_method = getattr(exe_object, ast_node.callback)
        return callback_method(ast_node)
