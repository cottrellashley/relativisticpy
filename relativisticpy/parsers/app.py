# Description of parser module

############### A General/Normal Compiler Structure: ############### 

# The usual process of a compiler involves both front end and back end processes.

# Front End Process:
# The front end of the compiler takes the source code (like Python code) and converts it into a form 
# that the back end can use. This involves several steps:

# 1. Lexer: Breaks down the code into tokens or basic syntax units.
# 2. Parser: Analyzes the structure of these tokens to understand the code's syntax.
# 3. Semantic Analyzer: Checks the code for semantic errors and ensures it makes logical sense.
# 4. IR Generator: Creates an Intermediate Representation of the code, a standardized format 
#    that the back end can work with.

# Back End Process:
# The back end of the compiler takes this Intermediate Representation and turns it into a binary file, 
# which is the machine code that the CPU can execute. This involves:

# 1. Optimizer: Optionally refines the IR for better performance or smaller size.
# 2. Code Generation: Converts the IR into machine code specific to the target CPU architecture.
# 3. Linker: Combines this machine code with other code libraries and resolves references to 
#    create a final executable. Though not strictly part of the compiler, it's often included in the 
#    compilation process.

############### RelativisticPy's Parser & Interpreter Structure: ############### 

# Differences in an Interpreted Language Like Ours:
# However, since our language is interpreted, we do not follow the typical back end process of a compiler. 
# Instead, we have a modified front end process:

# Modified Front End Process:
# Our front end process remains mostly the same, but with a key difference in the last step:

# 1. Lexer
# 2. Parser
# 3. Semantic Analyzer
# 4. Interpreter: Instead of an IR Generator, we use an Interpreter. This Interpreter directly executes 
#    the code, bypassing the need for a separate back end process.

# Interpreter and Execution Implementation:
# In our interpreter, the execution of code is managed by a class that is injected into the interpreter. 
# This means the actual computation for each node in the Abstract Syntax Tree (AST) is dependent on this 
# external class. The interpreter traverses and visits each node of the AST, calling the corresponding 
# method from the external class to implement that node. This design ensures that while the parser and 
# language grammar are fixed and integral to the interpreter, the actual execution logic can vary 
# depending on the implementation of the injected class. This separation allows for flexibility in 
# how language features are implemented and executed.


from relativisticpy.parsers.parser.default import DefaultParserService
from relativisticpy.parsers.interpreter.interpreter import InterpreterService
from relativisticpy.parsers.shared.models.basic_nodes import NodeConfigurationModel
from relativisticpy.parsers.shared.models.mapper import Mappers
from relativisticpy.parsers.shared.models.node_keys import ConfigurationModels
from relativisticpy.parsers.shared.models.object_configuration import (
    ObjectConfigurationModel,
)


class RelParser:
    def __init__(
        self,
        node_tree_walker,
        node_configuration: list,
        object_configuration: list = [],
    ):
        self.node_tree_walker = node_tree_walker
        self.node_configurations = ConfigurationModels(
            Mappers.map_from_list(node_configuration, NodeConfigurationModel),
            Mappers.map_from_list(object_configuration, ObjectConfigurationModel),
        )
        self.parser = DefaultParserService(self.node_configurations)
        self.interpreter = InterpreterService(self.node_tree_walker)

    def exe(self, expression: str):
        _ast = self.parser.parse_string(expression)
        return self.interpreter.interpret_ast(_ast)

    def parse(self, string: str):
        return self.parser.parse_string(string)

    def tokenize(self, string: str):
        return self.parser.tokenize_string(string)

    def iterpret(self, ast: dict):
        return self.interpreter.interpret_ast(ast)
