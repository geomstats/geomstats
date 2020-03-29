"""Product of manifolds."""

import geomstats.backend as gs
from geomstats.geometry.manifold import Manifold
from geomstats.geometry.product_riemannian_metric import \
    ProductRiemannianMetric

# TODO(nina): get rid of for loops
# TODO(nina): unit tests


class ProductManifold(Manifold):
    """Class for a product of manifolds M_1 x ... x M_n.

    In contrast to the class Landmarks or DiscretizedCruves,
    the manifolds M_1, ..., M_n need not be the same, nor of
    same dimension, but the list of manifolds needs to be provided.

    By default, a point is represented by an array of shape:
    [n_samples, dim_1 + ... + dim_n_manifolds]
    where n_manifolds is the number of manifolds in the product.
    This type of representation is called 'vector'.

    Alternatively, a point can be represented by an array of shape:
    [n_samples, n_manifolds, dim] if the n_manifolds have same dimension dim.
    This type of representation is called `matrix`.

    Parameters
    ----------
    manifolds : list
        List of manifolds in the product.
    default_point_type : str, {'vector', 'matrix'}
        Default representation of points.
    """

    def __init__(self, manifolds, default_point_type='vector'):
        assert default_point_type in ['vector', 'matrix']
        self.default_point_type = default_point_type

        self.manifolds = manifolds
        self.metric = ProductRiemannianMetric(
            [manifold.metric for manifold in manifolds])

        dimensions = [manifold.dimension for manifold in manifolds]
        super(ProductManifold, self).__init__(
            dimension=sum(dimensions))

    def belongs(self, point, point_type=None):
        """Test if a point belongs to the manifold.

        Parameters
        ----------
        point : array-like, shape=[n_samples, dim]
                           or shape=[n_samples, dim_2, dim_2]
            Point.
        point_type : str, {'vector', 'matrix'}
            Representation of point.

        Returns
        -------
        belongs : array-like, shape=[n_samples, 1]
            Array of booleans evaluating if the corresponding points
            belong to the manifold.
        """
        if point_type is None:
            point_type = self.default_point_type
        assert point_type in ['vector', 'matrix']

        if point_type == 'vector':
            point = gs.to_ndarray(point, to_ndim=2)
        else:
            point = gs.to_ndarray(point, to_ndim=3)

        belongs = gs.empty((point.shape[0], len(self.manifolds)))
        cum_dim = 0
        # FIXME: this only works if the points are in intrinsic representation
        for i, manifold_i in enumerate(self.manifolds):
            cum_dim_next = cum_dim + manifold_i.dimension
            point_i = point[:, cum_dim:cum_dim_next]
            belongs_i = manifold_i.belongs(point_i)
            belongs[:, i] = belongs_i
            cum_dim = cum_dim_next

        belongs = gs.all(belongs, axis=1)
        belongs = gs.to_ndarray(belongs, to_ndim=2)
        return belongs

    def regularize(self, point, point_type=None):
        """Regularize the point into the manifold's canonical representation.

        Parameters
        ----------
        point : array-like, shape=[n_samples, dim]
                           or shape=[n_samples, dim_2, dim_2]
            Point to be regularized.
        point_type : str, {'vector', 'matrix'}
            Representation of point.

        Returns
        -------
        regularized_point : array-like, shape=[n_samples, dim]
                            or shape=[n_samples, dim_2, dim_2]
            Point in the manifold's canonical representation.
        """
        # TODO(nina): Vectorize.
        if point_type is None:
            point_type = self.default_point_type
        assert point_type in ['vector', 'matrix']

        regularized_point = [
            manifold_i.regularize(point_i)
            for manifold_i, point_i in zip(self.manifolds, point)]

        # TODO(nina): Put this in a decorator
        if point_type == 'vector':
            regularized_point = gs.hstack(regularized_point)
        elif point_type == 'matrix':
            regularized_point = gs.vstack(regularized_point)
        return gs.all(regularized_point)
