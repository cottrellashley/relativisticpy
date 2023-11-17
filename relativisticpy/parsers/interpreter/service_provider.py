from relativisticpy.parsers.shared.interfaces.interpreter import IInterpreter


class InterpreterServiceProvider:
    def __init__(self, interpreter: IInterpreter, node):
        self.node = node
        self.interpreter = interpreter

    def build(self) -> IInterpreter:
        return self.interpreter(self.node)

    def interpret_ast_service(self, ast):
        interpreter = self.build()
        return interpreter.interpret(ast)
