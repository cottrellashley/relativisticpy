from dataclasses import dataclass
from relativisticpy.base_tensor.base import BaseTensor

@dataclass
class TensorContext:

    is_metric_tensor : bool = False

    is_result_tensor : bool = False

    is_christoffel : bool = False

    is_covariant_derivative : bool = False

    tensor_name : str = None

    prev_a : BaseTensor = None

    prev_b : BaseTensor = None