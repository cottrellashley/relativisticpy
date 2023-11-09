from relativisticpy.parser.parser.service_provider import ParserServicesProvider
from relativisticpy.parser.core.iterator import Iterator
from relativisticpy.parser.core.lexer import Lexer
from relativisticpy.parser.core.parser_ import Parser
from relativisticpy.parser.core.token import TokenProvider
from relativisticpy.parser.core.nodes import NodeProvider
from relativisticpy.parser.shared.interfaces.parser_service import IParserService
from relativisticpy.parser.shared.models.node_keys import ConfigurationModels

class ParserService(IParserService):
    """
     Implements interface IParserService:
    """
    def __init__(self, node_configuration: ConfigurationModels):
        self.node_configuration = node_configuration
        self.parser_serice = ParserServicesProvider(
                                                        lexer                   = Lexer,
                                                        parser                  = Parser, 
                                                        token_provider          = TokenProvider,
                                                        node_provider           = NodeProvider,
                                                        iterator                = Iterator,
                                                        configuration_models    = self.node_configuration
                                                    )

    def parse_string(self, string: str):
        return self.parser_serice.parse_string_service(string)

    def tokenize_string(self, string: str):
        return self.parser_serice.tokenize_string_service(string)
