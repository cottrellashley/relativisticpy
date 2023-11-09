from relativisticpy.parser.parser.parser_ import ParserService
from relativisticpy.parser.interpreter.interpreter import InterpreterService
from relativisticpy.parser.shared.models.basic_nodes import NodeConfigurationModel
from relativisticpy.parser.shared.models.mapper import Mappers
from relativisticpy.parser.shared.models.node_keys import ConfigurationModels
from relativisticpy.parser.shared.models.object_configuration import ObjectConfigurationModel

class RelParser:

    def __init__(self, node_tree_walker,  node_configuration: list, object_configuration : list = []):
        self.node_tree_walker = node_tree_walker
        self.node_configurations = ConfigurationModels(
                                                        Mappers.map_from_list(node_configuration, NodeConfigurationModel), 
                                                        Mappers.map_from_list(object_configuration, ObjectConfigurationModel)
                                                      )
        self.parser = ParserService(self.node_configurations)
        self.interpreter = InterpreterService(self.node_tree_walker)

    def exe(self, expression: str):
        AST = self.parser.parse_string(expression)
        return self.interpreter.interpret_ast(AST)

    def parse(self, string: str):
        return self.parser.parse_string(string)

    def tokenize(self, string: str):
        return self.parser.tokenize_string(string)

    def iterpret(self, ast: dict):
        return self.interpreter.interpret_ast(ast)
