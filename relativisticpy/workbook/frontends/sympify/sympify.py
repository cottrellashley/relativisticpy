from jsonmathpy import JsonMathPy
from relativisticpy.workbook.frontends.sympify.sympy_node import SympyNode
from relativisticpy.workbook.frontends.sympify.node_configuration import node_configuration

class Sympify:

    @classmethod
    def prs(self, expression: str):
        parser = JsonMathPy(SympyNode(), node_configuration)
        return parser.exe(expression)