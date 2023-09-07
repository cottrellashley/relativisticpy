import unittest
from parser_ import Parser
from lexer import Lexer
from interpreter import Interpreter

class End2End(unittest.TestCase):

    def test_exp(self):
        exp1 = 'exp(x**2)' # simplify(expr, ratio=1.0, measure=None)
        ans = {'operation': 'EXP',
                'arguments': [{'operation': 'POWER',
                'arguments': [{'operation': 'BUILD_VARIABLE', 'arguments': 'x'},
                {'operation': 'BUILD_INT', 'arguments': '2'}]}]}
        lex = Lexer(exp1).generate_tokens()
        ast = Parser(lex).parse()
        result = Interpreter().visit(ast).dict
        self.assertAlmostEqual(result, ans)