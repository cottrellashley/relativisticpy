import re
import json

from relativisticpy.parsers import RelParser

from relativisticpy.workbook.ast_visitor import RelPyAstNodeTraverser
from relativisticpy.workbook.state import WorkbookState


class Workbook:

    _cache = WorkbookState()
    parser = RelParser(
        RelPyAstNodeTraverser(_cache)
    )

    @classmethod
    def reset(cls):
        del cls._cache
        del cls.parser
        cls._cache = WorkbookState()
        cls.parser = RelParser( RelPyAstNodeTraverser(cls._cache) )

    def __init__(self, file_path: str = None):
        self.file_path = file_path

    def expr(self, string: str):
        return Workbook.parser.exe(string)

    def parse(self, string: str):
        return Workbook.parser.parse(string)

    def tokens(self, string: str):
        return Workbook.parser.tokenize(string)
