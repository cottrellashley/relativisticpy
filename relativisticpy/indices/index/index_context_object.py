from dataclasses import dataclass
from relativisticpy.indices.index.index_object import BaseIndex

@dataclass
class IndexContext:

    summed_index : BaseIndex = None
    "The index to which this index is being summed with."

    repeated_index : BaseIndex = None
    "The index to which this index is the same as (mostly so we know which components to add/subtract)."

    child_index : tuple[BaseIndex] = None
    "The index or indices to which this index is a child to within an expression."
