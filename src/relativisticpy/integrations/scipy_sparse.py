r"""Sparse matrices: SciPy's.

A sparse matrix, or a sparse array, is sparse data, which is never made dense: it multiplies
vectors and matrices as a matrix, and its few largest eigenvalues come from an iterative
solver, as with ``wb["L_{ij}"] = matrix``.
"""

from __future__ import annotations

from typing import Any

from relativisticpy.integrations import Data, Example, python_object


def _path() -> dict[str, Any]:
    import scipy.sparse

    rows, columns = [0, 1, 1, 2], [1, 0, 2, 1]
    return {"adjacency": scipy.sparse.csr_matrix(([1.0] * 4, (rows, columns)), shape=(3, 3))}


@python_object(
    "sparse",
    takes=["scipy.sparse.spmatrix", "scipy.sparse.sparray"],
    summary="a SciPy sparse matrix or array: sparse data, never made dense",
    requires="scipy",
    examples=[
        Example(
            r"A_{ij} := \texttt{adjacency}" "\n" r"A_{ij} 1^j",
            [1.0, 2.0, 1.0],
            "the degrees of a path of three vertices",
            given=_path,
        ),
        Example(
            r"A_{ij} := \texttt{adjacency}" "\n" r"A_{ij} A^{ij}",
            4.0,
            "the sum of the squares of the entries: twice the number of edges",
            given=_path,
        ),
    ],
)
class SparseMatrices:
    def becomes(self, value: Any) -> Data:
        return Data(value.tocsr())
