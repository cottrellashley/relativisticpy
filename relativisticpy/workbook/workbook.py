import re
import json

from relativisticpy.interpreter import RelParser
from relativisticpy.workbook.ast_visitor import RelPyAstNodeTraverser
from loguru import logger
import sys

class Workbook:

    def __init__(self, file_path: str = None, debug_mode: bool = False):
        logger.remove()
        self.file_path = file_path
        self.interpreter = RelParser(RelPyAstNodeTraverser())
        self.debug_mode = debug_mode
        if self.debug_mode:
            logger.remove()  # Removes all handlers added by default
            logger.add(sys.stderr, level="DEBUG")
        else:
            logger.remove()  # Removes all handlers added by default
            logger.add(sys.stderr, level="INFO")

    def markdown(self, path: str):
        # Step 1: Read the markdown content
        with open(path, 'r', encoding='utf-8') as file:
            content = file.read()

        # Step 2: Find all equations using regular expression
        equations = re.findall(r'\$\$(.*?)\$\$', content, re.DOTALL)

        # Step 3: Iterate over the equations, execute them, and store the results
        results = []
        for equation in equations:
            result = self.expr(equation.strip())  # Assuming self.expr can execute the equation
            results.append(result)

        # Step 4: Replace the equations in the content with their results
        modified_content = content
        for equation, result in zip(equations, results):
            modified_content = modified_content.replace(f"$$ {equation} $$", str(result))

        # Step 5: Return the modified content or write it back to the file
        return modified_content

    def expr(self, string: str):
        if self.debug_mode:
            result = self.interpreter.exe(string)
        else:
            try:
                result = self.interpreter.exe(string)
            except Exception as ex:
                return str(ex)

        if isinstance(result, list):
            if len(result) == 1:
                return result[0].value
            elif len(result) == 0:
                return "No return value."
            else:
                return [i.value for i in result][-1]  # TODO: CAREFUL CONCIDERATION OF HOW WE RETURN VALUES TO USER.
        elif isinstance(result, str):
            return result

    def exe(self, string: str):
        result = self.interpreter.exe(string)
        if isinstance(result, list):
            if len(result) == 1:
                return result[0].value
            else:
                return [i.value for i in result]  # TODO: CAREFUL CONCIDERATION OF HOW WE RETURN VALUES TO USER.
        elif isinstance(result, str):
            return result

    def reset(self):
        self.interpreter.node_tree_walker.state.reset()

    def parse(self, string: str):
        return self.interpreter.abstracy_syntax_tree(string)

    def tokens(self, string: str):
        return self.interpreter.tokens(string)
