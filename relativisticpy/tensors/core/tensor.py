
from sympy.tensor.array.dense_ndim_array import MutableDenseNDimArray
from sympy import ones

from relativisticpy.deserializers.index_representations import IndexRepresentationA
from relativisticpy.deserializers.pattern_identifier import TensorRepresentationIdentifier
from relativisticpy.tensors.core.tensor_data_structure import TensorObject

class GrTensor(TensorObject):
    def __init__(self,
                 components,
                 indices,
                 basis,
                 name : str = None,
                 context = None
                 ):
        tensor = TensorRepresentationIdentifier(components, indices, basis)
        TensorObject.__init__(self,
                                components  = tensor.get_components(),
                                basis       = tensor.get_basis(),
                                indices     = tensor.get_indices(),
                                dimention   = tensor.get_dimention(),
                                shape       = tensor.get_shape(),
                                rank        = tensor.get_rank(),
                                scalar      = tensor.get_scalar_bool(),
                                context     = context
        )
        self.name = name

    def gr_tensor_as_dict(self):
        return {
                'name' : self.name,
                'components' : str(self.components),
                'basis' : str(self.basis),
                'dimention' : self.dimention,
                'indices' : IndexRepresentationA(self.indices).indices_as_dict()
                }
    
    def return_tensor_ans_with_numbers(self, number):
        D = self.dimention
        A = int(self.rank)
        Shape = self.return_tensor_ans_shape()
        return MutableDenseNDimArray(ones(D**A)*number, Shape)
