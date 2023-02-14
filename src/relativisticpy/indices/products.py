from operator import itemgetter
import itertools as it
from src.relativisticpy.indices.data_structure import TensorIndicesObject
from src.relativisticpy.helpers.helpers import transpose_list

class TensorIndicesArithmetic:

    def __init__(self, indicesA : TensorIndicesObject, indicesB : TensorIndicesObject):
        self.indicesA = indicesA
        self.indicesB = indicesB
        self.product_object = self.indicesA + self.indicesB

    def all_component(self):
        ne = [[i.order, i.context.repeated_index.order] for i in self.product_object.parentA.indices]
        repeated_index_locations =  transpose_list(ne)
        return [(IndexA, IndexB) for (IndexA, IndexB) in list(it.product(self.indicesA, self.indicesB)) if itemgetter(*repeated_index_locations[0])(IndexA) == itemgetter(*repeated_index_locations[1])(IndexB)]

    def component(self, components = None):
        res = self.product_object
        if not res.result.scalar and components != None:
            return [(IndicesA, IndicesB) for (IndicesA, IndicesB) in self.all_component() if IndicesA == tuple(components)]
        else:
            return self.all_component()

class TensorIndicesProduct:

    def __init__(self, indicesA : TensorIndicesObject, indicesB : TensorIndicesObject):
        self.indicesA = indicesA
        self.indicesB = indicesB
        self.product_object = self.indicesA * self.indicesB

    def all_component(self):
        """
        Returns combinatorial of indices, where summed index locations are matched.
        """
        ne = [[i.order, i.context.summed_index.order] for i in self.product_object.parentA.indices if i.summed]
        summed_index_locations =  transpose_list(ne)
        return [(IndexA, IndexB) for (IndexA, IndexB) in list(it.product(self.indicesA, self.indicesB)) if itemgetter(*summed_index_locations[0])(IndexA) == itemgetter(*summed_index_locations[1])(IndexB)]

    def return_result_locations(self):
        res = self.product_object
        result_dictionary = {'result_indices_matching_with_A' : [], 'indices_A_matched_with_result' : [], 'result_indices_matching_with_B' : [], 'indices_B_matched_with_result' : []}
        for i in res.result.indices:
            for j in i.context.child_index:
                if i.symbol == j.symbol:
                    if j.parent == 'Parent_A':
                        result_dictionary['result_indices_matching_with_A'].append(i.order) 
                        result_dictionary['indices_A_matched_with_result'].append(j.order) 
                    elif j.parent == 'Parent_B':
                        result_dictionary['result_indices_matching_with_B'].append(i.order) 
                        result_dictionary['indices_B_matched_with_result'].append(j.order)
        return result_dictionary


    def component(self, components = None):
        res = self.product_object
        index_locations = self.return_result_locations()
        result_indices_in_A = index_locations['result_indices_matching_with_A']
        result_indices_in_B = index_locations['result_indices_matching_with_B']
        A_indices_not_summed = index_locations['indices_A_matched_with_result']
        B_indices_not_summed = index_locations['indices_B_matched_with_result']
        if not res.result.scalar and components != None:
            return [(IndicesA, IndicesB) for (IndicesA, IndicesB) in self.all_component() if itemgetter(*A_indices_not_summed)(IndicesA) == itemgetter(*result_indices_in_A)(components) \
                    and itemgetter(*B_indices_not_summed)(IndicesB) == itemgetter(*result_indices_in_B)(components)]
        elif not res.result.scalar and components != None and self.operation in ['+', '-']:
            return [(IndicesA, IndicesB) for (IndicesA, IndicesB) in self.all_component() if IndicesA == tuple(components)]
        else:
            return self.all_component()
