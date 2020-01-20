"""
Product of manifolds.
"""

import geomstats.backend as gs
from geomstats.geometry.manifold import Manifold

# TODO(nina): get rid of for loops
# TODO(nina): unit tests


class ProductManifold(Manifold):
    """Class for a product of manifolds M_1 x ... x M_n.

    By default, a point is represented by an array of shape:
        [n_samples, dim_1 + ... + dim_n_manifolds]
    where n_manifolds is the number of manifolds in the product.

    Alternatively, a point can be represented by an array of shape:
        [n_samples, n_manifolds, dim]
    if the n_manifolds have same dimension dim.

    In contrast to the class Landmarks or DiscretizedCruves,
    the manifolds M_1, ..., M_n need not be the same, nor of
    same dimension, but the list of manifolds needs to be provided.
    """
    # TODO(nina): Introduce point_type to decide between the two
    # representations (array shapes) of points in the product.

    def __init__(self, manifolds):
        self.manifolds = manifolds
        self.n_manifolds = len(manifolds)
        dimensions = [manifold.dimension for manifold in manifolds]
        super(ProductManifold, self).__init__(
            dimension=sum(dimensions))

    def belongs(self, point):
        """Check if the point belongs to the manifold."""
        belong = [self.manifold[i].belongs(point[i])
                  for i in range(self.n_manifolds)]
        return gs.all(belong)

    def regularize(self, point):
        """Regularizes the point's coordinates to the canonical representation
        chosen for this manifold.
        """
        regularize_points = [self.manifold[i].regularize(point[i])
                             for i in range(self.n_manifolds)]
        return regularize_points

    def geodesic(self, initial_point,
                 end_point=None, initial_tangent_vec=None,
                 point_type='vector'):
        """Geodesic curve for a product metric seen as the product of the geodesic
        on each space.
        """
        geodesics = gs.asarray([[self.manifold[i].metric.geodesic(
            initial_point,
            end_point=end_point,
            initial_tangent_vec=initial_tangent_vec,
            point_type=point_type)
            for i in range(self.n_manifolds)]])
        return geodesics
