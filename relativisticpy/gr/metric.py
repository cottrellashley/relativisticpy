# Standard Library
from typing import Union
from itertools import product
from operator import itemgetter

# External Modules
from relativisticpy.core import MultiIndexObject, einstein_convention, deserialisable_tensor, Indices, Idx
from relativisticpy.providers import SymbolArray, transpose_list, diff, simplify, tensorproduct, Symbol, IMultiIndexArray
from relativisticpy.providers.helpers import tensor_trace_product

class MetricIndices(Indices):
    # We can allow users to initiate the metric via the __setitem__ method: if user inits the Metric without the comps => they mapp the components

    def __init__(self, *args: Idx): super().__init__(*args)

    def _get_einsum_metric_result(self: 'MetricIndices', other: Union['Indices', 'MetricIndices']) -> 'Indices': # G_{a}_{b} * T^{c}^{d}^{a}^{f} => T^{c}^{d}_{b}^{f} (metric indices) != T_{b}^{c}^{d}^{f} (base indices)
        other_indices = list(other.indices)
        metric_indices = list(self.indices)
        for idx in metric_indices: # iterate on metric indices => if summed with other indices => replace other summed idx with second metric idx
            if idx.is_summed_wrt_indices(other_indices):
                other_indices[other_indices.index(-idx)] = metric_indices[metric_indices.index(idx) - 1]
                break
        return Indices(*other_indices)

    def einsum_product(self, other: 'Indices') -> 'Indices':
        summed_index_locations = transpose_list(self._get_all_summed_locations(other))
        all = [(IndexA, IndexB) for (IndexA, IndexB) in list(product(self, other)) if itemgetter(*summed_index_locations[0])(IndexA) == itemgetter(*summed_index_locations[1])(IndexB)] if len(summed_index_locations) > 0 else [(IndexA, IndexB) for (IndexA, IndexB) in list(product(self, other))]
        res = self._get_einsum_metric_result(other)
        result_indices_in_A = [i[0] for i in res._get_all_repeated_location(self) if len(i) > 0]
        result_indices_in_B = [i[0] for i in res._get_all_repeated_location(other) if len(i) > 0]
        A_indices_not_summed = [i[0] for i in self._get_all_repeated_location(res) if len(i) > 0]
        B_indices_not_summed = [i[0] for i in other._get_all_repeated_location(res) if len(i) > 0]
        
        def generator(idx): # Possible Abstraction => create a method attribute which takes in the function and its arguments as input and structures the if statements in list compr in acordance with what is not an empty array --> apply itemgetter.
            if not res.scalar and idx != None:
                if len(A_indices_not_summed) != 0 and len(B_indices_not_summed) != 0: # e.g. A_{i}_{j}_{s} * B^{i}^{j}_{k}
                    return [(IndicesA, IndicesB) for (IndicesA, IndicesB) in all if itemgetter(*A_indices_not_summed)(IndicesA) == itemgetter(*result_indices_in_A)(idx) and itemgetter(*B_indices_not_summed)(IndicesB) == itemgetter(*result_indices_in_B)(idx)]
                elif len(A_indices_not_summed) == 0 and len(B_indices_not_summed) != 0: # e.g. A_{i}_{j} * B^{i}^{j}_{k}
                    return [(IndicesA, IndicesB) for (IndicesA, IndicesB) in all if itemgetter(*B_indices_not_summed)(IndicesB) == itemgetter(*result_indices_in_B)(idx)]
                elif len(B_indices_not_summed) == 0 and len(A_indices_not_summed) != 0: # e.g. A_{i}_{j}_{k} * B^{i}^{j}
                    return [(IndicesA, IndicesB) for (IndicesA, IndicesB) in all if itemgetter(*A_indices_not_summed)(IndicesA) == itemgetter(*result_indices_in_A)(idx)]
            else:
                return all

        res.generator = generator
        return res

@einstein_convention
@deserialisable_tensor
class Metric(MultiIndexObject):
    # The metric tensor with lower indices, such as g_{ab}, is called the "covariant metric tensor" or simply the "metric tensor".
    # The metric tensor with raised indices, such as g^{ab}, is called the "contravariant metric tensor" or the "inverse metric tensor".
    SYMBOL = "MetricSymbol"
    NAME = "Metric"
    cron_delta = (1,1); contravariant = (0,2); covariant = (2,0)
    _cls_idcs = MetricIndices


    def __init__(self, indices: MetricIndices, components: IMultiIndexArray, basis: IMultiIndexArray):
        super().__init__(indices = indices, components = components, basis = basis)

    @property
    def _(self):
        if self.rank == Metric.contravariant:
            comp = self.components
            ind = self.indices
        else:
            comp = SymbolArray(self.components.tomatrix().inv())
            ind = MetricIndices(*[-j for j in self.indices.indices])
        return Metric(indices = ind, components = comp, basis = self.basis)

    @property     
    def inv(self):
        if self.rank == Metric.covariant:
            comp = self.components
            ind = self.indices
        else:
            comp = SymbolArray(self.components.tomatrix().inv())
            ind = MetricIndices(*[-j for j in self.indices.indices])
        return Metric(indices = ind, components = comp, basis = self.basis)

    def __pow__(self, other):
        if other == -1:
            return self.inv
        else:
            raise ValueError('Cannot raise Tensor to power. Only combatible with taking the inverse by taking the pow of value -1.')

    def rs(self, other, idx): return tensor_trace_product(self.inv.components, other, [[0, idx]])
    def lw(self, other, idx): return tensor_trace_product(self._.components, other, [[0, idx]])

    def get_transformed_metric(self, transformation):

        ############ STEP ONE: #################

        # - Get components from metric
        metric_components = self.components

        # - If component is a sympy object, substitute metric components from given transformation definitions.
        sub = lambda i : i.subs(transformation.transformation.as_dict) if issubclass(type(i), Symbol) else i
        
        # - New components is substituted components
        self.components = SymbolArray([[sub(i) for i in j] for j in metric_components])

        ############# STEP TWO: #################

        # - Get the jacobian matrix from transformation given.
        jacobian = lambda t, b : SymbolArray([[diff(j, i) for j in t] for i in b])
        components = jacobian([i[1] for i in list(transformation.transformation.as_dict.items())], transformation.new_basis.as_symbol)

        # - Acquire indices from metric given.
        a, b = [i.symbol for i in self.indices.indices]

        # - Generate a new index of the derivatives and new matric by na and nb (new a index and new b index)
        indices = '_{n' + a + '}^{' + a + '}_{n' + b + '}^{' + b + '}'

        # - Finally we multiply the metric with new substituted components with a tensor product of the jacobian.
        resulting_GRtensor = self*MultiIndexObject(tensorproduct(components, components), indices, transformation.new_basis.as_string)

        new_components = simplify(resulting_GRtensor.components)
        new_indices = '_{n' + a + '}_{n' + b + '}'

        # Return simplified components
        return Metric(new_components, new_indices, transformation.new_basis.as_string)

    def new_indices(self, indices):
        new_components = self.get_metric() if indices.count('_') == 2 else self.get_inverse()
        return Metric(new_components.components, indices, self.basis, self.signature)

