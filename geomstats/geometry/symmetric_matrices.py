"""The vector space of symmetric matrices."""

import geomstats.backend as gs
from geomstats.geometry.embedded_manifold import EmbeddedManifold
from geomstats.geometry.matrices import Matrices

EPSILON = 1e-6
TOLERANCE = 1e-12


class SymmetricMatrices(EmbeddedManifold):
    """Class for the vector space of symmetric matrices of size n."""

    def __init__(self, n):
        assert isinstance(n, int) and n > 0
        super(SymmetricMatrices, self).__init__(
            dimension=int(n * (n + 1) / 2),
            embedding_manifold=Matrices(n, n))
        self.n = n

    def belongs(self, mat, atol=TOLERANCE):
        """Check if mat belongs to the vector space of symmetric matrices."""
        return Matrices(self.n, self.n).is_symmetric(mat=mat, atol=atol)

    def expm(x):
        """Compute the matrix exponential."""
        eigvals, eigvecs = gs.linalg.eigh(x)
        eigvals = gs.exp(eigvals)
        eigvals = gs.from_vector_to_diagonal_matrix(eigvals)
        transp_eigvecs = gs.transpose(eigvecs, axes=(0, 2, 1))
        result = gs.matmul(eigvecs, eigvals)
        result = gs.matmul(result, transp_eigvecs)
        return result