"""Product of manifolds."""

from joblib import delayed, Parallel

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

    # TODO(nguigs): This only works for 1d points

    def __init__(self, manifolds, default_point_type='vector', n_jobs=1):
        assert default_point_type in ['vector', 'matrix']
        self.default_point_type = default_point_type

        self.manifolds = manifolds
        self.metric = ProductRiemannianMetric(
            [manifold.metric for manifold in manifolds])

        self.dimensions = [manifold.dimension for manifold in manifolds]
        super(ProductManifold, self).__init__(
            dimension=sum(self.dimensions))
        self.n_jobs = n_jobs

    def _detect_intrinsic_extrinsic(self, point, point_type):
        assert point_type in ['vector', 'matrix']
        index = 1 if point_type == 'vector' else 2
        if point.shape[index] == self.dimension:
            intrinsic = True
        elif point.shape[index] == sum(
                [dim + 1 for dim in self.dimensions]):
            intrinsic = False
        else:
            raise ValueError(
                'Input shape does not match the dimension of the manifold,')
        return intrinsic

    @staticmethod
    def _get_method(manifold, method_name, metric_args):
        return getattr(manifold, method_name)(**metric_args)

    def _iterate_over_manifolds(
            self, func, args, intrinsic=False):

        cum_index = gs.cumsum(self.dimensions)[:-1] if intrinsic else \
            gs.cumsum([k + 1 for k in self.dimensions])
        arguments = {key: gs.split(
            args[key], cum_index, axis=1) for key in args.keys()}
        args_list = [{key: arguments[key][j] for key in args.keys()} for j in
                     range(len(self.manifolds))]
        pool = Parallel(n_jobs=self.n_jobs)
        out = pool(
            delayed(self._get_method)(
                self.manifolds[i], func, args_list[i]) for i in range(
                len(self.manifolds)))
        return out

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
        if point_type == 'vector':
            point = gs.to_ndarray(point, to_ndim=2)
            intrinsic = self._detect_intrinsic_extrinsic(point, point_type)
            belongs = self._iterate_over_manifolds(
                'belongs', {'point': point}, intrinsic)
            belongs = gs.hstack(belongs)
            print(belongs)

        elif point_type == 'matrix':
            point = gs.to_ndarray(point, to_ndim=3)
            belongs = gs.stack([
                space.belongs(point[:, i]) for i, space in enumerate(
                    self.manifolds)],
                axis=1)
        print(belongs.shape)
        belongs = gs.all(belongs, axis=1)
        belongs = gs.to_ndarray(belongs, to_ndim=2, axis=1)
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

    def random_uniform(self, n_samples, point_type=None):
        """Sample in the the product space from the uniform distribution.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples.
        point_type : str, {'vector', 'matrix'}
            Representation of point.

        Returns
        -------
        samples : array-like, shape=[n_samples, dimension + 1]
            Points sampled on the hypersphere.
        """
        if point_type is None:
            point_type = self.default_point_type
        assert point_type in ['vector', 'matrix']
        if point_type == 'vector':
            data = self.manifolds[0].random_uniform(n_samples)
            if len(self.manifolds) > 1:
                for i, space in enumerate(self.manifolds[1:]):
                    data = gs.concatenate(
                        [data, space.random_uniform(n_samples)],
                        axis=1)
            return data
        else:
            point = [
                space.random_uniform(n_samples) for space in self.manifolds]
            return gs.stack(point, axis=1)
