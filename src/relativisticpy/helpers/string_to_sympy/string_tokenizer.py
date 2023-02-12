import re
import sympy as smp
import numpy as np
from sympy import sin, cos, tan, sinh, cosh, tanh, exp, asin, acos, atan

class _StringParser:
    def __init__(self, StringExpression):
        self.StringExpression = StringExpression
    
    def Functions(self):
        Function = "(?<![a-zA-Z])[a-zA-Z](?=\()"
        RemoveSpaces = self.StringExpression.replace(' ', '')
        FunctionList = []
        List = []
        FunctionList = [x for x in re.finditer(Function, RemoveSpaces)]
        for i in FunctionList:
            List.append(i.group())
        return list(dict.fromkeys(List))
            
    def symbolRecognizer(self, Input):
        GreekSymbols = "(theta|phi|omega|sigma|alpha|beta|gamma|epsilon|zeta|eta|kappa|lambda|mu|nu|pi|Theta|Phi|Omega|Sigma|Alpha|Beta|Gamma|Epsilon|Zeta|Eta|Kappa|Lambda|Mu|Nu|Pi)"
        RemoveSpaces = Input.replace(' ', '')
        ListOne = []
        ListOne = [x for x in re.finditer("(?<![a-zA-Z])[a-zA-Z](?![a-zA-Z])", RemoveSpaces)]
        ListTwo = [x for x in re.finditer(GreekSymbols, RemoveSpaces)]

        for i in ListTwo:
            ListOne.append(i)
        return ListOne

    def preDefinedFunctionRecognizer(self):
        SpecialFunctions = ['(sin)', '(cos)', '(tan)', '(sinh)', '(cosh)', '(tanh)', '(asin)', '(atan)', '(acos)', '(asinh)', '(acosh)', '(atanh)', '(exp)', '(pi)']
        RemoveSpaces = self.StringExpression.replace(' ', '')
        ListOfFunctions = []
        for i in SpecialFunctions:
            if [x for x in re.finditer(i, RemoveSpaces)] != []:
                ListOfFunctions.append([x for x in re.finditer(i, RemoveSpaces)])
        return ListOfFunctions

    #symbolRecognizer(Input)[1].group() = "F"
    #symbolRecognizer(Input)[1].span() = (17,18)

    def replacer(self, Input, Replace, Location):
        NewString = Input.replace(' ', '')
        return NewString[:Location[0]] + Replace + NewString[Location[1]:]

    #replacer(Input, 'Q[1]', symbolRecognizer(Input)[1].span())

    def returnLists(self):
        AllSymbolsList = []
        VariableList = []
        FunctionList = []
        
        for i in range(len(self.symbolRecognizer(self.StringExpression))):
            AllSymbolsList.append(self.symbolRecognizer(self.StringExpression)[i].group())
        NoDuplicatesAllSymbolsList = list(dict.fromkeys(AllSymbolsList))
        
        for i in NoDuplicatesAllSymbolsList:
            if i not in self.Functions():
                VariableList.append(i)
            else:
                FunctionList.append(i)

        return [VariableList,FunctionList,AllSymbolsList]

    def returnDictionary(self):
        VariableList = self.returnLists()[0]
        FunctionList = self.returnLists()[1]
        Dic = {}
        if len(VariableList) == 1 and len(FunctionList) == 1:
            for i in range(len(VariableList)):
                Dic.update({VariableList[i] : 'W'.format(i)})
            for i in range(len(FunctionList)):
                Dic.update({FunctionList[i] : 'Q'.format(i)})
        elif len(VariableList) == 1 and len(FunctionList) == 0:
            for i in range(len(VariableList)):
                Dic.update({VariableList[i] : 'W'.format(i)})
        elif len(VariableList) > 1 and len(FunctionList) == 0:
            for i in range(len(VariableList)):
                Dic.update({VariableList[i] : 'W[{}]'.format(i)})
        elif len(VariableList) > 1 and len(FunctionList) > 1:
            for i in range(len(VariableList)):
                Dic.update({VariableList[i] : 'W[{}]'.format(i)})
            for i in range(len(FunctionList)):
                Dic.update({FunctionList[i] : 'Q[{}]'.format(i)})
        elif len(VariableList) == 1 and len(FunctionList) > 1:
            for i in range(len(VariableList)):
                Dic.update({VariableList[i] : 'W'.format(i)})
            for i in range(len(FunctionList)):
                Dic.update({FunctionList[i] : 'Q[{}]'.format(i)})
        elif len(FunctionList) > 1 and len(VariableList) == 1:
            for i in range(len(FunctionList)):
                Dic.update({FunctionList[i] : 'Q[{}]'.format(i)})
            for i in range(len(FunctionList)):
                Dic.update({FunctionList[i] : 'Q'.format(i)})
        elif len(FunctionList) == 1 and len(VariableList) > 1:
            for i in range(len(FunctionList)):
                Dic.update({FunctionList[i] : 'Q'.format(i)})
            for i in range(len(VariableList)):
                Dic.update({VariableList[i] : 'W[{}]'.format(i)})
        return Dic

    def returnSympyString(self):
        New = self.StringExpression
        for i in range(len(self.symbolRecognizer(self.StringExpression))):
            New = self.replacer(New,self.returnDictionary()[self.symbolRecognizer(New)[i].group()], self.symbolRecognizer(New)[i].span())
        return New
