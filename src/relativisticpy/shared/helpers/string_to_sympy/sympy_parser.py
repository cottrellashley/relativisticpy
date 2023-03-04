import re
import sympy as smp
import numpy as np
from sympy import sin, cos, tan, sinh, cosh, tanh, exp, asin, acos, atan
from src.relativisticpy.shared.helpers.string_to_sympy.string_tokenizer import _StringParser

class SympyParser:
    def __init__(self, equationString):
        self.equationString = equationString

    # This will convert a list of variables and convert them into one string, in the same order of the List, 
    # so that sympy can parse the string to sympy objects.
    # Input: List = ['x', 'y', 'z']
    # Output: String = 'x y z '
    def convertListToSympyVariableString(self, stringList):
        varableList = []
        for string in stringList:
            varableList.append(string.replace(' ',''))
        variable = ''
        for i in range(len(varableList)):
            variable += varableList[i] + ' '
        return variable
   
    def convertToSympyObject(self):
        VariableSymbolsList = _StringParser(self.equationString).returnLists()[0]
        FunctionSymbolList = _StringParser(self.equationString).returnLists()[1]
        FunctionSymbols = self.convertListToSympyVariableString(FunctionSymbolList)
        VariableSymbols = self.convertListToSympyVariableString(VariableSymbolsList)
        
        if int(len(_StringParser(self.equationString).returnLists()[0])) == 0:
            return smp.symbols(_StringParser(self.equationString).returnSympyString())
            #return eval(StringParser(self.equationString, self.functionList).returnSympyString())
        elif int(len(_StringParser(self.equationString).returnLists()[1])) == 0:
            W = smp.symbols(VariableSymbols)
            return eval(_StringParser(self.equationString).returnSympyString())
        elif int(len(_StringParser(self.equationString).returnLists()[1])) >= 1:
            W = smp.symbols(VariableSymbols)
            Q = smp.symbols(FunctionSymbols, cls = smp.Function)
            return eval(_StringParser(self.equationString).returnSympyString())