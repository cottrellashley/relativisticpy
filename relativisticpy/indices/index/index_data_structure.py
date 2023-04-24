from dataclasses import dataclass
from relativisticpy.indices.index.index_context_object import IndexContext
from relativisticpy.indices.index.index_object import BaseIndex


@dataclass
class IndexDataStructure(BaseIndex):

    summed : bool = False
    "Is this index being summed with another index within a parent tensor or another tensor in a tensor product."

    repeated : bool = False
    "In a tensor Addition or Subtraction, does this index get repeated against the other tensor."

    child : bool = False
    "Is this index the child index of another index from previous tensor expression."

    context : IndexContext = None
    "Extra information an index object can have. More context we can use for latter in the expressions"

    @staticmethod
    def from_string(str : str):
        return IndexDataStructure()

    def to_string(repr : int):
        pass

    def __repr__(self):
        return f"""IndexDataStructure(
                                        symbol      = {self.symbol},    \n\
                                        order       = {self.order},     \n\
                                        dimention   = {self.dimention}, \n\
                                        running     = {self.running},   \n\
                                        values      = {self.values},    \n\
                                        comp_type   = {self.comp_type}, \n\
                                        basis       = {self.basis},     \n\
                                        parent      = {self.parent},    \n\
                                        summed      = {self.summed},    \n\
                                        repeated    = {self.repeated},  \n\
                                        child       = {self.child},     \n\
                                        context     = {self.context}    \n\
                                    )
                """

    def set_start(self):
        """
        Future note: We could easily implement slices feature for indices. So users can see a slice of a tensor specified via the indices.
        Simply add running indices start of slice and end on end of slice.
        """
        if self.running:
            return 0
        elif not self.running and isinstance(self.values, int):
            return self.values

    def set_end(self):
        """
        Sets the end of the iteration for a index. 
        When iterative methods called, index should run through to its dimention: [0, 1, 2, ... , dim - 1]
        """
        if self.running:
            return self.dimention - 1
        elif not self.running and isinstance(self.values, int):
            return self.values
 
    def __iter__(self):
        self.first_index_value = self.set_start()
        self.last_index_value = self.set_end()
        return self

    def __next__(self):
        if self.first_index_value <= self.last_index_value:
            x = self.first_index_value
            self.first_index_value += 1
            return x
        else:
            raise StopIteration

    def __eq__(self, other):
        if self.symbol == other.symbol and self.comp_type == other.comp_type:
            return True
        else:
            return False
        
    def __str__(self) -> str:
        return self.__repr__()

    def __len__(self) -> int:
        return self.dimention

    def __neg__(self):
        return BaseIndex(
                        symbol          = self.symbol, 
                        order           = self.order,
                        running         = self.running, 
                        basis           = self.basis,
                        dimention       = self.dimention,
                        comp_type       = 'contravariant' if self.comp_type == 'covariant' else 'contravariant',
                        values          = self.values,
                        metric_parent   = self.metric_parent,
                        parent          = self.parent
                        )


    #################################################################
    # ---- Einstein Summation Convention Inplementation ------
    #################################################################
    # This is a personal definition, for simplicity within the rest of the application.
    # When two tensors are multiplied together and the einstein summation convetion is true, this will return the location order of those indices.
    #
    #     Tensor_{a}_{b} * Tensor^{b}_{c} = Tensor_{a}_{c}
    #
    # This operation will return the following when we aply thit logic to the combinatorials of these indices:
    #
    #     [ [1, 0] ] => Which is saying that the index at location/order 1 of the first tensor is being summed with index at location/order 0 from the second tensor.

    def __mul__(self, other : BaseIndex):
        if self.symbol == other.symbol and self.comp_type != other.comp_type and id(self) != id(other):
            return IndexDataStructure(
                                    symbol      = self.symbol,
                                    order       = self.order, 
                                    running     = self.running, 
                                    basis       = self.basis, 
                                    dimention   = self.dimention, 
                                    values      = self.values, 
                                    comp_type   = self.comp_type, 
                                    parent      = self.parent,
                                    summed      = True,
                                    context     = IndexContext(summed_index = other)
                                )
        else:
            return None


    #################################################
    #     ---- Summation of Tesnors Index ------
    #################################################
    #
    #     This is a personal definition, for simplicity within the rest of the application and simliar but different to the multiplication one.
    #     When two tensors are added or subtracted together together, the resulting expression is ONLY a tensor expression if the two tensors being added/subtracted have the same index structure.
    #     However, the indices themselves can sometimes be in different orders:
    #
    #         Tensor_{a}_{b} + Tensor_{b}_{a} = Tensor_{a}_{b}
    #         Tensor_{a}_{b} - Tensor_{b}_{a} = Tensor_{a}_{b}
    #         Tensor_{a}_{b} - Tensor_{a}_{b} = Tensor_{a}_{b}
    #
    #     ---- Example One ----
    #         Tensor_{a}_{b} + Tensor_{b}_{a} = Tensor{_{a}_{b}}:
    #         [ [1, 0],[0,1] ] => Which is saying that the index at location/order 1 of the first tensor is the same as the index at location/order 0 from the second tensor.
    #                             and index at location/order 0 of the first tensor is the same as the index at location/order 1 from the second tensor.
    #                    
    #     ---- Example Two ----
    #
    #         Tensor_{a}_{b} + Tensor_{a}_{b} = Tensor_{a}_{b}:
    #         [ [0, 0],[1,1] ] => Which is saying that the index at location/order 0 of the first tensor is the same as the index at location/order 0 from the second tensor.
    #                             and index at location/order 1 of the first tensor is the same as the index at location/order 1 from the second tensor.
        
    def __add__(self, other):
        if self.symbol == other.symbol and self.comp_type == other.comp_type and id(self) != id(other):
            return IndexDataStructure(
                                    symbol      = self.symbol,
                                    order       = self.order, 
                                    running     = self.running, 
                                    basis       = self.basis, 
                                    dimention   = self.dimention, 
                                    values      = self.values, 
                                    comp_type   = self.comp_type, 
                                    parent      = self.parent, 
                                    repeated    = True,
                                    context     = IndexContext(repeated_index = other)
                                    )
        else:
            return None
            