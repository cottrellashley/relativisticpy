from relativisticpy.deserializers.indices import indices_from_string
from relativisticpy.deserializers.mathify import Mathify


def tensor_from_string(
    index_cls,
    indices_cls,
    tensor_cls,
    indices_string_arg,
    components_string,
    basis_string,
):
    return tensor_cls(
        indices=indices_from_string(index_cls, indices_cls, indices_string_arg),
        components=Mathify(components_string),
        basis=Mathify(basis_string),
    )
