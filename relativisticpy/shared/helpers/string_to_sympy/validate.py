import re
with open('./safety.json', 'r') as _: SafeWords = _.read()

class Stack:
    def __init__(self):
        self.__index = []

    def __len__(self):
        return len(self.__index)

    def push(self,item):
        self.__index.insert(0,item)

    def peek(self):
        if len(self) == 0:
            raise Exception("peek() called on empty stack.")
        return self.__index[0]

    def pop(self):
        if len(self) == 0:
            raise Exception("pop() called on empty stack.")
        return self.__index.pop(0)

    def __str__(self):
        return str(self.__index)

class BasisValidator:
    def __init__(self, basis_string: str = ""):
        self.Basis = self._is_valid_basis(basis_string)
        self.Dimention = len(self.Basis.replace(']','').replace('[','').split(','))

    def _is_valid_basis_values(self, basis):
        valid_basis_values = SafeWords
        pattern0 = "[a-zA-Z][a-zA-Z]+"
        lis = [x for x in re.finditer(pattern0, basis)]
        unrecognizedValues = []
        for i in lis:
            if i.group() not in valid_basis_values:
                unrecognizedValues.append(i.group())
        if basis == "":
            return basis
        elif len(unrecognizedValues) != 0:
            raise ValueError("Some of the values you have entered in the Basis are not valid or recognized.")
        else:
            return basis

    def _is_valid_basis(self, basis):
        basis_after_value_validation = self._is_valid_basis_values(basis)
        B = bool(re.match("^\[(\s*[a-zA-Z]+\s*\,\s*)+(\s*[a-zA-Z]+\s*)\]$", basis_after_value_validation))
        if basis == "":
            return basis
        elif not B:
            raise ValueError("The format of the basis you have entered is not correct.")
        else:
            return basis


class MetricValidator(BasisValidator):
    def __init__(self, metric: str = "", basis: str = ""):
        BasisValidator.__init__(self, basis_string = basis)
        self.Metric = self._is_valid_metric(metric)

    def _metric_brackets_are_valid(self, metric):
        match_all_but_brackets = "[^\[|^\(|^\{|^\]|^\}|^\)]"
        bracketsString = re.sub(match_all_but_brackets, "", metric)
        O = ['{','(','[']
        C = ['}',')',']']
        D = {')':'(','}':'{',']':'['}
        s = Stack()
        for i in bracketsString:
            if i in O:
                s.push(i)
            elif i in C and len(s) != 0:
                if D[i] == s.peek():
                    s.pop()
                else:
                    s.push(i)
            elif i in C and len(s) == 0:
                s.push(i)
        if metric == "":
            return metric
        elif metric and not bool(len(s) == 0):
            raise ValueError("The brackets you have entered for your metric are not balanced.")
        else:
            return metric
    
    def _remove_unwanted_commas(self, metric):
        anythinginoutermost = "\([^\(\)]*\)"
        match_all_but_brackets_and_commas = "[^\[|^\]|^\,]"
        string = metric
        while len([x for x in re.finditer(anythinginoutermost, string)]):
            string = re.sub(anythinginoutermost,'', string)
        return re.sub(match_all_but_brackets_and_commas,'', string)
    
    def _words_in_metric_are_valid(self, metric):
        ListOfSpecialWords = SafeWords
        pattern0 = "[a-zA-Z][a-zA-Z]+"
        lis = [x for x in re.finditer(pattern0, metric)]
        unrecognizedValues = []
        for i in lis:
            if i.group() not in ListOfSpecialWords:
                unrecognizedValues.append(i.group())
        if len(unrecognizedValues) != 0:
            strin = ''
            for i in unrecognizedValues:
                strin = strin + " " + i + ","
            raise ValueError("Value(s)" + strin + " not recognized.")
        else:
            return metric
    
    def _metric_format_is_valid(self, metric):
        if self.Basis:
            metric_without_unwanted_commas = self._remove_unwanted_commas(metric)
            dim = self.Dimention
            pattern0 = "\[(\[(\,){" + str(dim-1) +  "}\]\,){" + str(dim-1) +  "}(\[(\,){" + str(dim-1) +  "}\])\]"
            if not bool(re.search(pattern0, metric_without_unwanted_commas)):
                raise ValueError("The format you have entered for your metric is not valid w.r.t the basis you entered.")
            else:
                return metric
        else:
            return metric

    def _is_valid_metric(self, metric):
        if metric and self.Basis:
            Validation1 = self._metric_brackets_are_valid(metric)
            Validation2 = self._metric_format_is_valid(Validation1)
            Validation3 = self._words_in_metric_are_valid(Validation2)
            return Validation3
        else:
            return metric