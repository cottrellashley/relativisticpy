import unittest
from relativisticpy.relparser.core.lexer import Lexer
from relativisticpy.relparser.core.parser_ import Parser

class End2End(unittest.TestCase):

    def test_exp(self):
        exp1 = 'exp(x**2)' # simplify(expr, ratio=1.0, measure=None)
        ans = {'operation': 'EXP',
                'arguments': [{'operation': 'POWER',
                'arguments': [{'operation': 'BUILD_VARIABLE', 'arguments': 'x'},
                {'operation': 'BUILD_INT', 'arguments': '2'}]}]}
        lex = Lexer(exp1).generate_tokens()
        ast = Parser(lex).parse()
        self.assertAlmostEqual(ast, ans)