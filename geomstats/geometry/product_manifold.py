"""Product of manifolds."""

import joblib

import geomstats.backend as gs
import geomstats.errors
import geomstats.vectorization
from geomstats.geometry.manifold import Manifold
from geomstats.geometry.product_riemannian_metric import \
    ProductRiemannianMetric


class ProductManifold(Manifold):
    """Class for a product of manifolds M_1 x ... x M_n.

    In contrast to the class Landmarks or DiscretizedCruves,
    the manifolds M_1, ..., M_n need not be the same, nor of
    same dimension, but the list of manifolds needs to be provided.

    By default, a point is represented by an array of shape:
    [..., dim_1 + ... + dim_n_manifolds]
    where n_manifolds is the number of manifolds in the product.
    This type of representation is called 'vector'.

    Alternatively, a point can be represented by an array of shape:
    [..., n_manifolds, dim] if the n_manifolds have same dimension dim.
    This type of representation is called `matrix`.

    Parameters
    ----------
    manifolds : list
        List of manifolds in the product.
    default_point_type : str, {'vector', 'matrix'}
        Default representation of points.
        Optional, default: 'vector'.
    n_jobs : int
        Number of jobs for parallel computing.
        Optional, default: 1.
    """

    # FIXME (nguigs): This only works for 1d points

    def __init__(
            self, manifolds, metrics=None,
            default_point_type='vector',
            n_jobs=1, **kwargs):
        geomstats.errors.check_parameter_accepted_values(
            default_point_type, 'default_point_type', ['vector', 'matrix'])

        self.dims = [manifold.dim for manifold in manifolds]
        if metrics is None:
            metrics = [manifold.metric for manifold in manifolds]
        metric = ProductRiemannianMetric(
            metrics, default_point_type=default_point_type, n_jobs=n_jobs)
        super(ProductManifold, self).__init__(
            dim=sum(self.dims), metric=metric,
            default_point_type=default_point_type, **kwargs)
        self.manifolds = manifolds
        self.n_jobs = n_jobs

    @staticmethod
    def _get_method(manifold, method_name, metric_args):
        return getattr(manifold, method_name)(**metric_args)

    def _iterate_over_manifolds(
            self, func, args, intrinsic=False):

        cum_index = gs.cumsum(self.dims)[:-1] if intrinsic else \
            gs.cumsum([k + 1 for k in self.dims])
        arguments = {}
        float_args = {}
        for key, value in args.items():
            if not isinstance(value, float):
                arguments[key] = gs.split(value, cum_index, axis=-1)
            else:
                float_args[key] = value
        args_list = [
            {key: arguments[key][j] for key in arguments}
            for j in range(len(self.manifolds))]
        pool = joblib.Parallel(n_jobs=self.n_jobs)
        out = pool(
            joblib.delayed(self._get_method)(
                self.manifolds[i], func, {**args_list[i], **float_args})
            for i in range(len(self.manifolds)))
        return out

    def belongs(self, point, atol=gs.atol):
        """Test if a point belongs to the manifold.

        Parameters
        ----------
        point : array-like, shape=[..., {dim, [n_manifolds, dim_each]}]
            Point.
        atol : float,
            Tolerance.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if the point belongs to the manifold.
        """
        point_type = self.default_point_type

        if point_type == 'vector':
            intrinsic = self.metric.is_intrinsic(point)
            belongs = self._iterate_over_manifolds(
                'belongs', {'point': point, 'atol': atol}, intrinsic)
            belongs = gs.stack(belongs, axis=-1)

        else:
            belongs = gs.stack([
                space.belongs(point[..., i, :], atol) for i, space in
                enumerate(self.manifolds)],
                axis=-1)

        belongs = gs.all(belongs, axis=-1)
        return belongs

    def regularize(self, point):
        """Regularize the point into the manifold's canonical representation.

        Parameters
        ----------
        point : array-like, shape=[..., {dim, [n_manifolds, dim_each]}]
            Point to be regularized.
        point_type : str, {'vector', 'matrix'}
            Representation of point.
            Optional, default: None.

        Returns
        -------
        regularized_point : array-like,
            shape=[..., {dim, [n_manifolds, dim_each]}]
            Point in the manifold's canonical representation.
        """
        point_type = self.default_point_type

        if point_type == 'vector':
            intrinsic = self.metric.is_intrinsic(point)
            regularized_point = self._iterate_over_manifolds(
                'regularize', {'point': point}, intrinsic)
            regularized_point = gs.concatenate(regularized_point, axis=-1)
        elif point_type == 'matrix':
            regularized_point = [
                manifold_i.regularize(point[..., i, :])
                for i, manifold_i in enumerate(self.manifolds)]
            regularized_point = gs.stack(regularized_point, axis=1)
        return regularized_point

    def random_point(self, n_samples=1, bound=1.):
        """Sample in the product space from the uniform distribution.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples.
        bound : float
            Bound of the interval in which to sample for non compact manifolds.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[..., {dim, [n_manifolds, dim_each]}]
            Points sampled on the hypersphere.
        """
        point_type = self.default_point_type
        geomstats.errors.check_parameter_accepted_values(
            point_type, 'point_type', ['vector', 'matrix'])

        if point_type == 'vector':
            data = self.manifolds[0].random_point(n_samples, bound)
            if len(self.manifolds) > 1:
                for space in self.manifolds[1:]:
                    samples = space.random_point(n_samples, bound)
                    data = gs.concatenate([data, samples], axis=-1)
            return data

        point = [
            space.random_point(n_samples, bound) for space in self.manifolds]
        samples = gs.stack(point, axis=-2)
        return samples

    def projection(self, point):
        """Project a point in product embedding manifold on each manifold.

        Parameters
        ----------
        point : array-like, shape=[..., {dim, [n_manifolds, dim_each]}]
            Point in embedding manifold.

        Returns
        -------
        projected : array-like, shape=[..., {dim, [n_manifolds, dim_each]}]
            Projected point.
        """
        point_type = self.default_point_type
        geomstats.errors.check_parameter_accepted_values(
            point_type, 'point_type', ['vector', 'matrix'])

        if point_type == 'vector':
            intrinsic = self.metric.is_intrinsic(point)
            projected_point = self._iterate_over_manifolds(
                'projection', {'point': point}, intrinsic)
            projected_point = gs.concatenate(projected_point, axis=-1)
        elif point_type == 'matrix':
            projected_point = [
                manifold_i.projection(point[..., i, :])
                for i, manifold_i in enumerate(self.manifolds)]
            projected_point = gs.stack(projected_point, axis=-2)
        return projected_point

    def to_tangent(self, vector, base_point):
        """Project a vector to a tangent space of the manifold.

        The tangent space of the product manifold is the direct sum of
        tangent spaces.

        Parameters
        ----------
        vector : array-like, shape=[..., dim]
            Vector.
        base_point : array-like, shape=[..., dim]
            Point on the manifold.

        Returns
        -------
        tangent_vec : array-like, shape=[..., dim]
            Tangent vector at base point.
        """
        point_type = self.default_point_type
        geomstats.errors.check_parameter_accepted_values(
            point_type, 'point_type', ['vector', 'matrix'])

        if point_type == 'vector':
            intrinsic = self.metric.is_intrinsic(base_point)
            tangent_vec = self._iterate_over_manifolds(
                'to_tangent',
                {'base_point': base_point, 'vector': vector}, intrinsic)
            tangent_vec = gs.concatenate(tangent_vec, axis=-1)
        elif point_type == 'matrix':
            tangent_vec = [
                manifold_i.to_tangent(vector[..., i, :], base_point[..., i, :])
                for i, manifold_i in enumerate(self.manifolds)]
            tangent_vec = gs.stack(tangent_vec, axis=-2)
        return tangent_vec

    def is_tangent(self, vector, base_point, atol=gs.atol):
        """Check whether the vector is tangent at base_point.

        The tangent space of the product manifold is the direct sum of
        tangent spaces.

        Parameters
        ----------
        vector : array-like, shape=[..., dim]
            Vector.
        base_point : array-like, shape=[..., dim]
            Point on the manifold.
        atol : float
            Absolute tolerance.
            Optional, default: backend atol.

        Returns
        -------
        is_tangent : bool
            Boolean denoting if vector is a tangent vector at the base point.
        """
        point_type = self.default_point_type
        geomstats.errors.check_parameter_accepted_values(
            point_type, 'point_type', ['vector', 'matrix'])

        if point_type == 'vector':
            intrinsic = self.metric.is_intrinsic(base_point)
            is_tangent = self._iterate_over_manifolds(
                'is_tangent',
                {'base_point': base_point, 'vector': vector, 'atol': atol},
                intrinsic)
            is_tangent = gs.stack(is_tangent, axis=-1)

        else:
            is_tangent = gs.stack([
                space.is_tangent(
                    vector[..., i, :], base_point[..., i, :], atol=atol)
                for i, space in enumerate(self.manifolds)], axis=-1)

        is_tangent = gs.all(is_tangent, axis=-1)
        return is_tangent
