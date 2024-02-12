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
        self.exe_object = exe_object
        if self.gr_script.contains_error:
            return self.gr_script.display_error_str
        for action_tree in self.gr_script.action_trees:
            if action_tree.returns_object:
                callback_result = self.executor(action_tree.ast)
                self.return_list.append(
                    ReturnObject(
                        value=callback_result,
                        _type=action_tree.return_type,
                    )
                )
            else:
                self.executor(action_tree.ast)
        return self.return_list

    def executor(self, ast_node: AstNode):

        # Check if the node has arguments and process them if necessary.
        # This step ensures that each child node is processed before the current node is passed to the callback.
        if hasattr(ast_node, 'is_leaf'):
            if not ast_node.is_leaf:
                ast_node.execute_node(self.executor)

        # Use getattr to retrieve the callback method from the exe_object using the node's callback name.
        # Call this method with the full ast_node, which includes its children within node.args.
        callback_method = getattr(self.exe_object, ast_node.callback)
        return callback_method(ast_node)
