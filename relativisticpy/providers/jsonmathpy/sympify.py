from jsonmathpy import JsonMathPy
from relativisticpy.providers.jsonmathpy.sympy_node import SympyNode
from relativisticpy.providers.jsonmathpy.node_configuration import node_configuration


def Sympify(expression: str): 
    """Sympify: string => parser => sympy objects""" 
    return JsonMathPy(SympyNode(), node_configuration).exe(expression)