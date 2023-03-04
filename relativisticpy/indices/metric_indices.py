from dataclasses import dataclass
from relativisticpy.indices.base import BaseTensorIndices

@dataclass
class MetricIndices(BaseTensorIndices):

    def __post_init__(self):
        pass

    def __mul__(self, other : BaseTensorIndices):
        """Perform index raising and lowering."""
        if not isinstance(other, BaseTensorIndices):
            raise TypeError(f"unsupported operand type(s) for *: '{type(self)}' and '{type(other)}'")

        if self.dimention != other.dimention:
            raise ValueError("tensors have different dimensions")

        summed_index = None
        indices = []
        for index in self.indices:
            if index in other.indices:
                if summed_index is not None:
                    raise ValueError("tensors have more than one common index")
                summed_index = index
            else:
                indices.append(index)
        for i, index in enumerate(other.indices):
            if index not in self.indices:
                indices.append(index)
        if summed_index is None:
            raise ValueError("tensors do not have a common index")

        # Determine the factor for index raising or lowering
        if summed_index.contravariant:
            factor = -1
        else:
            factor = 1

        # Compute the new basis and valid flag
        new_basis = self.basis
        valid = self.valid and other.valid
        for index in indices:
            if index.contravariant:
                new_basis = new_basis.contract(index, factor)
            else:
                new_basis = new_basis.contract(index, -factor)
        if summed_index.contravariant:
            new_basis = new_basis.inv()
            self_summed = True
        elif summed_index.covariant:
            self_summed = True

        return MetricIndices(
            indices=indices,
            basis=new_basis,
            dimention=self.dimention,
            rank=self.rank,
            scalar=self.scalar and other.scalar,
            shape=tuple(len(self.indices), len(other.indices)),
            valid=valid,
            self_summed=self_summed,
            parent_tensor=self.parent_tensor
        )