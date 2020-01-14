"""
Grassmannian manifold Gr(n, p),
a set of all p-dimensional subspaces in n-dimensional space,
where p <= n
"""

from geomstats.geometry.embedded_manifold import EmbeddedManifold
from geomstats.geometry.euclidean_space import EuclideanMetric
from geomstats.geometry.matrices_space import MatricesSpace
from geomstats.geometry.riemannian_metric import RiemannianMetric

TOLERANCE = 1e-5
EPSILON = 1e-6


class Grassmannian(EmbeddedManifold):
    """
    Class for Grassmann manifolds Gr(n, k) 
    of k-dimensional subspaces in the n-dimensional euclidean space. 
    """
    def __init__(self, n, k):
        assert isinstance(n, int) and isinstance(k, int)
        assert k <= n

        self.n = n
        self.k = k

        dimension = int(k * (n - k))
        super(Grassmannian, self).__init__(
              dimension=dimension,
              embedding_manifold=MatricesSpace(n, n))

    def belongs(self, point, tolerance=TOLERANCE):
        """
        Check if an (n,n)-matrix is an orthogonal projector 
        onto a subspace of rank p. 
        """
        # check if p* = p and p^2 = p and rank(p) = p 
        return False
        """
        return self.embedding_space.belongs(point, tolerance)\
                and self.embedding_space.is_symmetric(point, tolerance)\
                and is_projector(point, tolerance)\
                and is_of_rank(point, p, tolerance)
        """
    
    """
    def __is_projector(point, tolerance):
        return self.embedding_space.mult(point, point) == point 
    
    def is_of_rank(point, k)
        return True
    """

class GrassmannianCanonicalMetric(RiemannianMetric):

    def __init__(self, n, p):
        assert isinstance(n, int) and isinstance(p, int)
        assert p <= n
        self.n = n
        self.p = p

        dimension = int(p * (n - p))
        super(GrassmannianCanonicalMetric, self).__init__(
                dimension=dimension,
                signature=(dimension, 0, 0))
        self.embedding_metric = EuclideanMetric(n * p)
