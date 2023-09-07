from relativisticpy.relparser.interpreter.service_provider import InterpreterServiceProvider
from relativisticpy.relparser.core.interpreter import Interpreter
from relativisticpy.relparser.shared.interfaces.interpreter_service import IInterpreterService

class InterpreterService(IInterpreterService):
    """
     Implements ISerialiser.
    """
    def __init__(self, node_tree_walker):
        self.node_tree_walker = node_tree_walker
        self.interpreter_service = InterpreterServiceProvider(
                                                                interpreter  = Interpreter,
                                                                node = self.node_tree_walker
                                                             )

    def interpret_ast(self, ast):
        return self.interpreter_service.interpret_ast_service(ast)
