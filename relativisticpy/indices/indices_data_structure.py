import itertools as it
from dataclasses import dataclass
from sympy import MutableDenseNDimArray

from relativisticpy.indices.index.index_context_object import IndexContext
from relativisticpy.indices.index.index_data_structure import IndexDataStructure

from relativisticpy.indices.indices_base_object import BaseTensorIndices
from relativisticpy.indices.indices_context_object import IndicesProductContext



@dataclass
class TensorIndicesObject(BaseTensorIndices):

    def __index__(self):
        return tuple([int(i.values) if not i.running else slice(None) for i in self.indices])

# ---- TRUE example ---
# _{a}_{b} == _{b}_{a}        -> true (We can letter use this as a property to define commutator of indices)
# _{a}_{b} == _{a}_{b}        -> true
# ^{a}_{b} == _{b}^{a}        -> true

# ---- FALSE example ---
# _{a}^{b} == _{b}^{a}        -> false
# _{a}_{b} == ^{b}_{a}        -> false
# _{a}_{b} == _{a}_{b}_{c}    -> false
# _{a}_{b} == _{a}_{c}        -> false
    def __eq__(self, other_indices):
        """
        When both indices have equal symbols and equal number of covarient and/or contravariant indices in any order, 
        then they are defined to be equal.
        """
        boolean_combinatorial_list = [i==j for (i, j) in list(it.product(self.indices, other_indices.indices))]
        return boolean_combinatorial_list.count(True) == len(self)


    def __len__(self):
        """
        Number of individual index objects the indices objects holds.
        """
        return len(self.indices)

    # @staticmethod
    # def from_string(string : str, basis : Union[MutableDenseNDimArray, str]):
    #     lis = []
    #     individual_indices = [item for item in re.split('(?=[_^])', string) if item]
    #     for i, index in enumerate(individual_indices):
    #         lis.append(IndexRepresentationA(index, i, basis))
    #     return tuple(lis)

###############################################################################
# INDEX RULES FOR ITERATION OF MULTIPLE INDICES
###############################################################################

    def indices_iterator(self):
        return list(it.product(*[x for x in self.indices]))

    def __iter__(self):
        self.__length = int(len(self.indices_iterator()))
        self.__i = 1
        return self

    def __next__(self):
        """
        Method called when object is passed thorugh an iteration (for or while loops)
        For our case, we simply iterate through the combinations of all the combinatorials.
        These combinatorials are reflective of 
        """
        if self.__i <= self.__length:
            x = self.__i
            self.__i += 1
            return self.indices_iterator()[x-1]
        else:
            raise StopIteration

###############################################################################
# INDEX RULES FROM TENSOR SUMMING IT'S OWN COMPONENTS
###############################################################################

    def __post_init__(self):
        """
        First thing we do when object is initiated, is we check whether the current indices are being summed.
        If they are, we replace it by summed indices, and also return a context object so that other classes
        have the data to do their own responsibilities accordingly.
        """
        if len([0 for i,j in it.product(self.indices,self.indices) if i*j]) > 0:
            self.self_summed = True

    def get_self_summed_contex(self):

        Result  = self.resulting_indices_from_self_sum()
        Old     = self

        return IndicesProductContext(
                                        result      = Result,
                                        old_indices = Old
                                    )


    def resulting_indices_from_self_sum(self):
        A = list(self.indices)

        for (i, j) in it.product(self.indices, self.indices):
            if i*j:
                if i in A:
                    A.remove(i)
                elif j in A:
                    A.remove(j)

        Result : list[IndexDataStructure] = []
        for i in range(len(A)):
            Result.append(
                            IndexDataStructure(
                                                    symbol      = A[i].symbol,
                                                    order       = int(i),
                                                    running     = A[i].running,
                                                    basis       = A[i].basis,
                                                    dimention   = A[i].dimention,
                                                    values      = A[i].values,
                                                    comp_type   = A[i].comp_type,
                                                    child       = True,
                                                    parent      = A[i].parent,
                                                    context     = IndexContext(child_index = A[i])
                                                )
                        )

        # Now we have a new list of indices, representing the resulting tensors indices, we return an Index Object.
        return TensorIndicesObject(
                            indices         = tuple(Result),
                            basis           = self.basis, 
                            dimention       = self.dimention, 
                            rank            = (int([x.comp_type for x in Result].count('contravariant')), int([x.comp_type for x in Result].count('covariant'))),
                            scalar          = len(Result) == 0,
                            shape           = tuple([i.dimention for i in Result]),
                            valid           = True,
                            parent_tensor   = 'Result'
                            )



###############################################################################
# INDEX RULES FROM TENSOR ADDITION
###############################################################################


    def __add__(self, other_indices):
        
        A = self.indices
        B = other_indices.indices

        Result = self.resulting_indices_from_tensor_sum(A, B, 'Result')
        Parent_a = self.add_context_to_indices_from_tensor_sum(A, B, "Parent_A")
        Parent_b = self.add_context_to_indices_from_tensor_sum(B, A, "Parent_B")

        return IndicesProductContext(
                                result  = Result,
                                parentA = Parent_a,
                                parentB = Parent_b
                                    )
    def __sub__(self, other):
        return self + other

    def resulting_indices_from_tensor_sum(self, A : tuple[IndexDataStructure], B : tuple[IndexDataStructure], parent : str):

        An = list(A)
        Bn = list(B)

        # Remove both sumed indices from result if indices are summed over.
        # Remove only last index if they are repeated.
        for (i, j) in list(it.product(A, B)):
            if i*j:
                raise Exception("Tensor sum or subtraction must not have indices summed over: Not a covariant expression.")
    
        for i in An:
            i.parent = parent

        for i in Bn:
            i.parent = parent

        # For new indices, representing resulting combination, we re-enter their orders / locations w.r.t new tensor indices.
        Result : list[IndexDataStructure] = []
        for i in range(len(An)):
            Result.append(
                    IndexDataStructure(
                        symbol      = An[i].symbol,
                        order       = int(i),
                        running     = An[i].running,
                        basis       = An[i].basis,
                        dimention   = An[i].dimention,
                        values      = An[i].values,
                        comp_type   = An[i].comp_type,
                        child       = True,
                        parent      = An[i].parent,
                        context     = IndexContext(child_index = An[i])
                            )
                        )
        
        # Now we have a new list of indices, representing the resulting tensors indices, we return an Index Object.
        return TensorIndicesObject(
                            indices         = tuple(Result),
                            basis           = self.basis, 
                            dimention       = self.dimention, 
                            rank            = (int([x.comp_type for x in Result].count('contravariant')), int([x.comp_type for x in Result].count('covariant'))),
                            scalar          = len(Result) == 0,
                            shape           = tuple([i.dimention for i in Result]),
                            valid           = True,
                            parent_tensor   = 'Result'
                            )

    def add_context_to_indices_from_tensor_sum(self, A : tuple[IndexDataStructure], B : tuple[IndexDataStructure], parent : str):

        A = list(A)
        B = list(B)

        # If index within a tensor product is a summed one, replace it with a new index object which has that contexed information in it: Indx * Indx
        for (i, j) in list(it.product(A, B)):
            if i+j:
                k = A.index(i)
                A = A[:k] + [i+j] + A[k+1:]
        
        # For each index within the tensor indices, inject the parent string to identify the parent tensor later on if needed.
        for i in A:
            i.parent = parent

        return TensorIndicesObject(
                                indices         = tuple(A),
                                basis           = self.basis, 
                                dimention       = self.dimention, 
                                rank            = (int([x.comp_type for x in A].count('contravariant')), int([x.comp_type for x in A].count('covariant'))),
                                scalar          = len(A) == 0,
                                shape           = tuple([i.dimention for i in A]),
                                valid           = True,
                                parent_tensor   = parent
                                )


###############################################################################
# INDEX RULES FROM TENSOR MULTIPLICATION
###############################################################################

    def __mul__(self, other_indices):
        
        A = self.indices
        B = other_indices.indices

        Result   = self.resulting_indices_from_tensor_product(A, B, "Result")
        Parent_a = self.add_context_to_indices_from_tensor_product(A, B, "Parent_A")
        Parent_b = self.add_context_to_indices_from_tensor_product(B, A, "Parent_B")

        return IndicesProductContext(
                                result  = Result,
                                parentA = Parent_a,
                                parentB = Parent_b
                            )

    def resulting_indices_from_tensor_product(self, A : tuple[IndexDataStructure], B : tuple[IndexDataStructure], parent):

        A = list(A)
        B = list(B)

        # Remove both sumed indices from result if indices are summed over.
        # Remove only last index if they are repeated.
        # g^{c}^{a} * R_{a}_{b}_{c}_{d} --->  R^{c}_{b}_{c}_{d}
        for (i, j) in list(it.product(A, B)):
            if i*j:
                if i.metric_parent:
                    A.remove(i)
                    index = B.index(j)
                    B[index] = A[0]
                    A.remove(A[0])
                    break
                elif j.metric_parent:
                    B.remove(j)
                    index = A.index(i)
                    A[index] = B[0]
                    B.remove(B[0])
                    break
                else:
                    A.remove(i)
                    B.remove(j)
            elif i+j:
                raise Exception("Cannot have repeated indices in tensor product, this is not a valid resulting Tesnor.")
        R = A + B

        # For new indices, representing resulting combination, we re-enter their orders / locations w.r.t new tensor indices.
        Result = []
        for i in range(len(R)):
            Result.append(
                    IndexDataStructure(
                        symbol      = R[i].symbol,
                        order       = int(i),
                        running     = R[i].running,
                        basis       = R[i].basis,
                        dimention   = R[i].dimention,
                        values      = R[i].values,
                        comp_type   = R[i].comp_type,
                        child       = True,
                        parent      = parent,
                        context     = IndexContext(child_index = R)
                            )
                        )
        
        # Now we have a new list of indices, representing the resulting tensors indices, we return an Index Object.
        return TensorIndicesObject(
                            indices         = tuple(Result),
                            basis           = self.basis, 
                            dimention       = self.dimention, 
                            rank            = (int([x.comp_type for x in Result].count('contravariant')), int([x.comp_type for x in Result].count('covariant'))),
                            scalar          = len(Result) == 0,
                            shape           = tuple([i.dimention for i in Result]),
                            valid           = True,
                            parent_tensor   = 'Result'
                            )

    def add_context_to_indices_from_tensor_product(self, A : tuple[IndexDataStructure], B : tuple[IndexDataStructure], parent : str):

        A = list(A)
        B = list(B)

        # If index within a tensor product is a summed one, replace it with a new index object which has that contexed information in it: Indx * Indx
        for (i, j) in list(it.product(A, B)):
            if i*j:
                k = A.index(i)
                A = A[:k] + [i*j] + A[k+1:]
        
        # For each index within the tensor indices, inject the parent string to identify the parent tensor later on if needed.
        for i in A:
            i.parent = parent

        return TensorIndicesObject(
                                indices         = tuple(A),
                                basis           = self.basis, 
                                dimention       = self.dimention, 
                                rank            = (int([x.comp_type for x in A].count('contravariant')), int([x.comp_type for x in A].count('covariant'))),
                                scalar          = len(A) == 0,
                                shape           = tuple([i.dimention for i in A]),
                                valid           = True,
                                parent_tensor   = parent
                                )

    def parent_tensor_as_zeros(self):
        return MutableDenseNDimArray.zeros(*self.shape)
