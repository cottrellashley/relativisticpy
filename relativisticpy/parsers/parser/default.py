from relativisticpy.parsers.parser.service_provider import ParserServicesProvider
from relativisticpy.parsers.core.iterator import Iterator
from relativisticpy.parsers.core.lexer import Lexer
from relativisticpy.parsers.core.default_parser import Parser
from relativisticpy.parsers.core.token import TokenProvider
from relativisticpy.parsers.core.nodes import NodeProvider
from relativisticpy.parsers.shared.interfaces.parser_service import IParserService
from relativisticpy.parsers.shared.models.node_keys import ConfigurationModels

class DefaultParserService(IParserService):
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
