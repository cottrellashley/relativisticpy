import re

from relativisticpy.relparser import RelParser
from relativisticpy.workbook.ast_traverser import RelPyAstNodeTraverser
from relativisticpy.workbook.cache import RelPyCache


class Workbook:

    _cache = RelPyCache()
    parser = RelParser(RelPyAstNodeTraverser(_cache), RelPyAstNodeTraverser.node_configuration, RelPyAstNodeTraverser.variable_matchers)

    def __init__(self, file_path: str):
        self.file_path = file_path

    def expr(self, string: str):
        if re.search(r'\n|\r\n?', string):
            tasks = []
            lines = string.splitlines()
            for line in lines:
                if line.strip():
                    # If line starts with "#", ignore it
                    if line.startswith("#"):
                        continue
                    
                    # If line contains "#", split and take the part before "#"
                    if "#" in line:
                        line = line.split("#")[0].strip()

                    if ";" in line:
                        lines = line.split(";")
                        for l in lines:
                            tasks.append(l.strip())
                        continue
                    
                    tasks.append(line.strip())
            res = [Workbook.parser.exe(task) for task in tasks if task.strip()]
            return [r for r in res if r != None ]

        return Workbook.parser.exe(string)

    def parse(self, string: str):
        return Workbook.parser.parse(string)

    def tokens(self, string: str):
        return Workbook.parser.tokenize(string)

    def exe(self):
        with open(self.file_path, 'r') as file:
            tasks = []
            for line in file:
                if line.strip():
                    # If line starts with "#", ignore it
                    if line.startswith("#"):
                        continue
                    
                    # If line contains "#", split and take the part before "#"
                    if "#" in line:
                        line = line.split("#")[0].strip()

                    if ";" in line:
                        lines = line.split(";")
                        for l in lines:
                            tasks.append(l.strip())
                        continue
                    
                    tasks.append(line.strip())

            res = [Workbook.parser.exe(task) for task in tasks if task.strip()]

            return [r for r in res if r != None ]
