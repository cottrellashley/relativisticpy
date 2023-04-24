from sympy import MutableDenseNDimArray as Array
from sympy import Symbol, tensorproduct, diff, simplify
from relativisticpy.indices.indices import Indices

from relativisticpy.tensors.core.tensor import GrTensor
from relativisticpy.tensors.core.tensor_context_object import TensorContext
from relativisticpy.tensors.core.transformation_base_objects import CoordinateTransformation




# The metric tensor with lower indices, such as g_{ab}, is called the "covariant metric tensor" or simply the "metric tensor".
# The metric tensor with raised indices, such as g^{ab}, is called the "contravariant metric tensor" or the "inverse metric tensor".
# In general, the covariant and contravariant metric tensors are related by matrix inversion. 
# That is, if g_{ab} is the covariant metric tensor, then g^{ab} is the contravariant metric tensor, and the two are related by:
# g^{ab} = (g_{cd})^{-1}
# where (g_{cd})^{-1} is the inverse of the matrix of g_{cd}.
class Metric(GrTensor):

    def __init__(self, comp, ind, base, signature = None):
        GrTensor.__init__(self, 
                           components = comp,
                           indices = ind,
                           basis = base,
                           context=TensorContext(is_metric_tensor=True))

        # expected_shape = (self.dimention,) * 2
        # if self.shape != expected_shape:
        #     raise ValueError(f"Invalid metric: The shape should be {expected_shape}.")

        # expected_ranks = [(0,2), (2,0)]
        # if self.rank not in expected_ranks:
        #     raise ValueError(f"Invalid metric: The rank should be of (2,0) or (0,2).")

        self.signature = signature

    def get_metric(self):
        if self.rank == (0, 2):
            comp = self.components
            ind = self.indices
        else:
            comp = Array(self.components.tomatrix().inv())
            ind = Indices(tuple([-j for j in self.indices.indices]), self.basis)

        return Metric(comp, ind, self.basis, self.signature)
        
    def get_inverse(self):
        if self.rank == (2, 0):
            comp = self.components
            ind = self.indices
        else:
            comp = Array(self.components.tomatrix().inv())
            ind = Indices(tuple([-j for j in self.indices.indices]), self.basis)

        return Metric(comp, ind, self.basis, self.signature)

    def get_transformed_metric(self, transformation: CoordinateTransformation):

        ############ STEP ONE: #################

        # - Get components from metric
        metric_components = self.components

        # - If component is a sympy object, substitute metric components from given transformation definitions.
        sub = lambda i : i.subs(transformation.transformation.as_dict) if issubclass(type(i), Symbol) else i
        
        # - New components is substituted components
        self.components = Array([[sub(i) for i in j] for j in metric_components])

        ############# STEP TWO: #################

        # - Get the jacobian matrix from transformation given.
        jacobian = lambda t, b : Array([[diff(j, i) for j in t] for i in b])
        components = jacobian([i[1] for i in list(transformation.transformation.as_dict.items())], transformation.new_basis.as_symbol)

        # - Acquire indices from metric given.
        a, b = [i.symbol for i in self.indices.indices]

        # - Generate a new index of the derivatives and new matric by na and nb (new a index and new b index)
        indices = '_{n' + a + '}^{' + a + '}_{n' + b + '}^{' + b + '}'

        # - Finally we multiply the metric with new substituted components with a tensor product of the jacobian.
        resulting_GRtensor = self*GrTensor(tensorproduct(components, components), indices, transformation.new_basis.as_string)

        new_components = simplify(resulting_GRtensor.components)
        new_indices = '_{n' + a + '}_{n' + b + '}'

        # Return simplified components
        return Metric(new_components, new_indices, transformation.new_basis.as_string)

    def new_indices(self, indices):
        new_components = self.get_metric() if indices.count('_') == 2 else self.get_inverse()
        return Metric(new_components.components, indices, self.basis, self.signature)

# class MetricWork(GrTensor):

#     def __init__(self, metric : Metric, indices):
#         self.metric = metric
#         self.comp = self.metric.get_metric() if indices.count('_') == 2 else self.metric.get_inverse()
#         GrTensor.__init__(self ,
#                         components=self.comp.components,
#                         indices= indices,
#                         basis=self.metric.basis
#         )