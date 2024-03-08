from dataclasses import dataclass
from typing import List
from relativisticpy.interpreter.analyzers.base import ActionTree, GrScriptTree
from relativisticpy.interpreter.nodes.base import AstNode
from relativisticpy.interpreter.state.scopes import ScopedState

from relativisticpy.interpreter.protocols import Implementer

@dataclass
class ReturnObject:
    value: any
    _type: str


class Interpreter:

    def __init__(self, implementor: Implementer):
        self.implementor = implementor
        self.return_list = []
        self.state = ScopedState()
        self.implementor.state = self.state

    def exe_script(self, gr_script: GrScriptTree):
        self.gr_script = gr_script
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

    def executor(self, ast_node: AstNode): return ast_node.execute_node(self.implementor)
