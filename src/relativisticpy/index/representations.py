import re
from src.relativisticpy.index.index import Index

class IndexRepresentationA(Index):
    integers = {'0': 0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9}
    def __init__(self, index_representation, order, coordinate_basis):
        integers = {'0': 0, '1':1, '2':2, '3':3, '4':4, '5':5, '6':6, '7':7, '8':8, '9':9}
        self.index_string_representation = self.__str_rep_is_valid(index_representation)
        Index.__init__(self, 
                       symbol       = re.search('[a-zA-Z]+', self.index_string_representation).group(),
                       order        = self.__is_valid_order(order),
                       running      = not bool(re.search('^[^=]*(=)([0-9]+)[^=]*$', self.index_string_representation)),
                       comp_type    = 'covariant' if (self.index_string_representation[0] == "_") else 'contravariant',
                       basis        = self.__is_valid_basis(coordinate_basis),
                       values       = integers[re.split('[=]', self.index_string_representation.replace('{','').replace('}','').replace('_','').replace('^',''))[1].replace(" ","")] if bool(re.match('^[^=]*(=)([0-9]+)[^=]*$', self.index_string_representation)) else None
                       )

    def __is_valid_basis(self, basis):
        return basis
    
    def __str_rep_is_valid(self, index):
        index_ = index.replace(" ", "")
        condition1 = index_[0] == "_" or index_[0] == "^"
        condition2 = index_[1] == "{"
        condition3 = index_[-1] == "}"
        condition4 = bool(re.search('^[^=]*(=)([0-9]+)[^=]*$', index_)) or bool(re.search('^[^=]*[a-zA-Z]+[^=]*$', index_))
        if not condition1:
            raise ValueError("The index {} you have entered, does not contain the necessary characters _ OR ^ to indicate covariace or contravariance.".format(index_))
        elif not condition2 or not condition3:
            raise ValueError("The index {} you have entered, does not contain closing OR opening curly brakets.".format(index_))
        elif not condition4:
            raise ValueError("The following characters you have entered in the index are not recognized: {}".format(index_.replace('{','').replace('}','').replace('_','').replace('^','')))
        else:
            return index_

    def __is_valid_order(self, order):
        valid = type(order) == int
        if valid:
            return order
        else:
            raise ValueError("Index order must be of type: int")

    def __neg__(self):
        if self.index_string_representation[0] == '_':
            return IndexRepresentationA(self.index_string_representation.replace('_', '^'), self.order, self.basis)
        else:
            return IndexRepresentationA(self.index_string_representation.replace('^', '_'), self.order, self.basis)

    def __repr__(self):
        """
        Explain method.
        """
        return f"""IndexRepresentationA( {self.index_string_representation}, {self.order}, {self.basis})"""