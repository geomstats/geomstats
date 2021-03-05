"""The n-dimensional hypersphere.

The n-dimensional hypersphere embedded in (n+1)-dimensional
Euclidean space.
"""

import logging
from itertools import product

import geomstats.algebra_utils as utils
import geomstats.backend as gs
import geomstats.vectorization
from geomstats.geometry.embedded_manifold import EmbeddedManifold
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.euclidean import EuclideanMetric
from geomstats.geometry.riemannian_metric import RiemannianMetric

TOLERANCE = 1e-6
EPSILON = 1e-4


class _Hypersphere(EmbeddedManifold):
    """Private class for the n-dimensional hypersphere.

    Class for the n-dimensional hypersphere embedded in the
    (n+1)-dimensional Euclidean space.

    By default, points are parameterized by their extrinsic
    (n+1)-coordinates.

    Parameters
    ----------
    dim : int
        Dimension of the hypersphere.
    """

    def __init__(self, dim):
        super(_Hypersphere, self).__init__(
            dim=dim,
            embedding_manifold=Euclidean(dim + 1))
        self.embedding_metric = self.embedding_manifold.metric

    def belongs(self, point, tolerance=TOLERANCE):
        """Test if a point belongs to the hypersphere.

        This tests whether the point's squared norm in Euclidean space is 1.

        Parameters
        ----------
        point : array-like, shape=[..., dim + 1]
            Point in Euclidean space.
        tolerance : float
            Tolerance at which to evaluate norm == 1.
            Optional, default: 1e-6.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if point belongs to the hypersphere.
        """
        point_dim = gs.shape(point)[-1]
        if point_dim != self.dim + 1:
            if point_dim is self.dim:
                logging.warning(
                    'Use the extrinsic coordinates to '
                    'represent points on the hypersphere.')
            belongs = False
            if gs.ndim(point) == 2:
                belongs = gs.tile([belongs], (point.shape[0],))
            return belongs
        sq_norm = gs.sum(point ** 2, axis=-1)
        diff = gs.abs(sq_norm - 1)
        return gs.less_equal(diff, tolerance)

    def regularize(self, point):
        """Regularize a point to the canonical representation.

        Regularize a point to the canonical representation chosen
        for the hypersphere, to avoid numerical issues.

        Parameters
        ----------
        point : array-like, shape=[..., dim + 1]
            Point on the hypersphere.

        Returns
        -------
        projected_point : array-like, shape=[..., dim + 1]
            Point in canonical representation chosen for the hypersphere.
        """
        if not gs.all(self.belongs(point)):
            raise ValueError('Points do not belong to the manifold.')

        return self.projection(point)

    def projection(self, point):
        """Project a point on the hypersphere.

        Parameters
        ----------
        point : array-like, shape=[..., dim + 1]
            Point in embedding Euclidean space.

        Returns
        -------
        projected_point : array-like, shape=[..., dim + 1]
            Point projected on the hypersphere.
        """
        norm = gs.linalg.norm(point, axis=-1)
        projected_point = gs.einsum('...,...i->...i', 1. / norm, point)

        return projected_point

    def to_tangent(self, vector, base_point):
        """Project a vector to the tangent space.

        Project a vector in Euclidean space
        on the tangent space of the hypersphere at a base point.

        Parameters
        ----------
        vector : array-like, shape=[..., dim + 1]
            Vector in Euclidean space.
        base_point : array-like, shape=[..., dim + 1]
            Point on the hypersphere defining the tangent space,
            where the vector will be projected.

        Returns
        -------
        tangent_vec : array-like, shape=[..., dim + 1]
            Tangent vector in the tangent space of the hypersphere
            at the base point.
        """
        sq_norm = gs.sum(base_point ** 2, axis=-1)
        inner_prod = self.embedding_metric.inner_product(base_point, vector)
        coef = inner_prod / sq_norm
        tangent_vec = vector - gs.einsum('...,...j->...j', coef, base_point)

        return tangent_vec

    def spherical_to_extrinsic(self, point_spherical):
        """Convert point from spherical to extrinsic coordinates.

        Convert from the spherical coordinates in the hypersphere
        to the extrinsic coordinates in Euclidean space.
        Only implemented in dimension 2.

        Parameters
        ----------
        point_spherical : array-like, shape=[..., dim]
            Point on the sphere, in spherical coordinates.

        Returns
        -------
        point_extrinsic : array_like, shape=[..., dim + 1]
            Point on the sphere, in extrinsic coordinates in Euclidean space.
        """
        if self.dim != 2:
            raise NotImplementedError(
                'The conversion from spherical coordinates'
                ' to extrinsic coordinates is implemented'
                ' only in dimension 2.')

        theta = point_spherical[..., 0]
        phi = point_spherical[..., 1]

        point_extrinsic = gs.stack(
            [gs.sin(theta) * gs.cos(phi),
             gs.sin(theta) * gs.sin(phi),
             gs.cos(theta)],
            axis=-1)

        if not gs.all(self.belongs(point_extrinsic)):
            raise ValueError('Points do not belong to the manifold.')

        return point_extrinsic

    @geomstats.vectorization.decorator(['else', 'vector', 'vector'])
    def tangent_spherical_to_extrinsic(self, tangent_vec_spherical,
                                       base_point_spherical):
        """Convert tangent vector from spherical to extrinsic coordinates.

        Convert from the spherical coordinates in the hypersphere
        to the extrinsic coordinates in Euclidean space for a tangent
        vector. Only implemented in dimension 2.

        Parameters
        ----------
        tangent_vec_spherical : array-like, shape=[..., dim]
            Tangent vector to the sphere, in spherical coordinates.
        base_point_spherical : array-like, shape=[..., dim]
            Point on the sphere, in spherical coordinates.

        Returns
        -------
        tangent_vec_extrinsic : array-like, shape=[..., dim + 1]
            Tangent vector to the sphere, at base point,
            in extrinsic coordinates in Euclidean space.
        """
        if self.dim != 2:
            raise NotImplementedError(
                'The conversion from spherical coordinates'
                ' to extrinsic coordinates is implemented'
                ' only in dimension 2.')

        n_samples = base_point_spherical.shape[0]
        theta = base_point_spherical[:, 0]
        phi = base_point_spherical[:, 1]

        zeros = gs.zeros(n_samples)

        jac = gs.concatenate([gs.array([[
            [gs.cos(theta[i]) * gs.cos(phi[i]),
             - gs.sin(theta[i]) * gs.sin(phi[i])],
            [gs.cos(theta[i]) * gs.sin(phi[i]),
             gs.sin(theta[i]) * gs.cos(phi[i])],
            [- gs.sin(theta[i]),
             zeros[i]]]]) for i in range(n_samples)], axis=0)

        tangent_vec_extrinsic = gs.einsum(
            '...ij,...j->...i', jac, tangent_vec_spherical)

        return tangent_vec_extrinsic

    def intrinsic_to_extrinsic_coords(self, point_intrinsic):
        """Convert point from intrinsic to extrinsic coordinates.

        Convert from the intrinsic coordinates in the hypersphere,
        to the extrinsic coordinates in Euclidean space.

        Parameters
        ----------
        point_intrinsic : array-like, shape=[..., dim]
            Point on the hypersphere, in intrinsic coordinates.

        Returns
        -------
        point_extrinsic : array-like, shape=[..., dim + 1]
            Point on the hypersphere, in extrinsic coordinates in
            Euclidean space.
        """
        sq_coord_0 = 1. - gs.sum(point_intrinsic ** 2, axis=-1)
        if gs.any(gs.less(sq_coord_0, 0.)):
            raise ValueError('Square-root of a negative number.')
        coord_0 = gs.sqrt(sq_coord_0)

        point_extrinsic = gs.concatenate([
            coord_0[..., None], point_intrinsic], axis=-1)

        return point_extrinsic

    def extrinsic_to_intrinsic_coords(self, point_extrinsic):
        """Convert point from extrinsic to intrinsic coordinates.

        Convert from the extrinsic coordinates in Euclidean space,
        to some intrinsic coordinates in the hypersphere.

        Parameters
        ----------
        point_extrinsic : array-like, shape=[..., dim + 1]
            Point on the hypersphere, in extrinsic coordinates in
            Euclidean space.

        Returns
        -------
        point_intrinsic : array-like, shape=[..., dim]
            Point on the hypersphere, in intrinsic coordinates.
        """
        point_intrinsic = point_extrinsic[..., 1:]

        return point_intrinsic

    def _replace_values(self, samples, new_samples, indcs):
        replaced_indices = [
            i for i, is_replaced in enumerate(indcs) if is_replaced]
        value_indices = list(product(replaced_indices, range(self.dim + 1)))
        return gs.assignment(samples, gs.flatten(new_samples), value_indices)

    def random_uniform(self, n_samples=1, tol=1e-6):
        """Sample in the hypersphere from the uniform distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        tol : float
            Tolerance.
            Optional, default: 1e-6.

        Returns
        -------
        samples : array-like, shape=[..., dim + 1]
            Points sampled on the hypersphere.
        """
        size = (n_samples, self.dim + 1)

        samples = gs.random.normal(size=size)
        while True:
            norms = gs.linalg.norm(samples, axis=1)
            indcs = gs.isclose(norms, 0.0, atol=tol)
            num_bad_samples = gs.sum(indcs)
            if num_bad_samples == 0:
                break
            new_samples = gs.random.normal(
                size=(num_bad_samples, self.dim + 1))
            samples = self._replace_values(samples, new_samples, indcs)

        samples = gs.einsum('..., ...i->...i', 1 / norms, samples)
        if n_samples == 1:
            samples = gs.squeeze(samples, axis=0)
        return samples

    def random_von_mises_fisher(self, kappa=10, n_samples=1):
        """Sample in the 2-sphere with the von Mises distribution.

        Sample in the 2-sphere with the von Mises distribution centered at the
        north pole.

        References
        ----------
        https://en.wikipedia.org/wiki/Von_Mises_distribution

        Parameters
        ----------
        kappa : int
            Kappa parameter of the von Mises distribution.
            Optional, default: 10.
        n_samples : int
            Number of samples.
            Optional, default: 1.

        Returns
        -------
        point : array-like, shape=[..., 3]
            Points sampled on the sphere in extrinsic coordinates
            in Euclidean space of dimension 3.
        """
        if self.dim != 2:
            raise NotImplementedError(
                'Sampling from the von Mises Fisher distribution'
                'is only implemented in dimension 2.')
        angle = 2. * gs.pi * gs.random.rand(n_samples)
        angle = gs.to_ndarray(angle, to_ndim=2, axis=1)
        unit_vector = gs.hstack((gs.cos(angle), gs.sin(angle)))
        scalar = gs.random.rand(n_samples)

        coord_z = 1. + 1. / kappa * gs.log(
            scalar + (1. - scalar) * gs.exp(gs.array(-2. * kappa)))
        coord_z = gs.to_ndarray(coord_z, to_ndim=2, axis=1)

        coord_xy = gs.sqrt(1. - coord_z**2) * unit_vector

        point = gs.hstack((coord_xy, coord_z))

        if n_samples == 1:
            point = gs.squeeze(point, axis=0)
        return point


class HypersphereMetric(RiemannianMetric):
    """Class for the Hypersphere Metric.

    Parameters
    ----------
    dim : int
        Dimension of the hypersphere.
    """

    def __init__(self, dim):
        super(HypersphereMetric, self).__init__(
            dim=dim,
            signature=(dim, 0, 0))
        self.embedding_metric = EuclideanMetric(dim + 1)
        self._space = _Hypersphere(dim=dim)

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
        """Compute the inner-product of two tangent vectors at a base point.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., dim + 1]
            First tangent vector at base point.
        tangent_vec_b : array-like, shape=[..., dim + 1]
            Second tangent vector at base point.
        base_point : array-like, shape=[..., dim + 1], optional
            Point on the hypersphere.

        Returns
        -------
        inner_prod : array-like, shape=[...,]
            Inner-product of the two tangent vectors.
        """
        inner_prod = self.embedding_metric.inner_product(
            tangent_vec_a, tangent_vec_b, base_point)

        return inner_prod

    def squared_norm(self, vector, base_point=None):
        """Compute the squared norm of a vector.

        Squared norm of a vector associated with the inner-product
        at the tangent space at a base point.

        Parameters
        ----------
        vector : array-like, shape=[..., dim + 1]
            Vector on the tangent space of the hypersphere at base point.
        base_point : array-like, shape=[..., dim + 1], optional
            Point on the hypersphere.

        Returns
        -------
        sq_norm : array-like, shape=[..., 1]
            Squared norm of the vector.
        """
        sq_norm = self.embedding_metric.squared_norm(vector)
        return sq_norm

    def exp(self, tangent_vec, base_point):
        """Compute the Riemannian exponential of a tangent vector.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., dim + 1]
            Tangent vector at a base point.
        base_point : array-like, shape=[..., dim + 1]
            Point on the hypersphere.

        Returns
        -------
        exp : array-like, shape=[..., dim + 1]
            Point on the hypersphere equal to the Riemannian exponential
            of tangent_vec at the base point.
        """
        hypersphere = Hypersphere(dim=self.dim)
        proj_tangent_vec = hypersphere.to_tangent(tangent_vec, base_point)
        norm2 = self.embedding_metric.squared_norm(proj_tangent_vec)

        coef_1 = utils.taylor_exp_even_func(
            norm2, utils.cos_close_0, order=4)
        coef_2 = utils.taylor_exp_even_func(
            norm2, utils.sinc_close_0, order=4)
        exp = (gs.einsum('...,...j->...j', coef_1, base_point)
               + gs.einsum('...,...j->...j', coef_2, proj_tangent_vec))

        return exp

    @geomstats.vectorization.decorator(['else', 'vector', 'vector'])
    def log(self, point, base_point, **kwargs):
        """Compute the Riemannian logarithm of a point.

        Parameters
        ----------
        point : array-like, shape=[..., dim + 1]
            Point on the hypersphere.
        base_point : array-like, shape=[..., dim + 1]
            Point on the hypersphere.

        Returns
        -------
        log : array-like, shape=[..., dim + 1]
            Tangent vector at the base point equal to the Riemannian logarithm
            of point at the base point.
        """
        inner_prod = self.embedding_metric.inner_product(base_point, point)
        cos_angle = gs.clip(inner_prod, -1., 1.)
        squared_angle = gs.arccos(cos_angle) ** 2
        coef_1_ = utils.taylor_exp_even_func(
            squared_angle, utils.inv_sinc_close_0, order=5)
        coef_2_ = utils.taylor_exp_even_func(
            squared_angle, utils.inv_tanc_close_0, order=5)
        log = (gs.einsum('...,...j->...j', coef_1_, point)
               - gs.einsum('...,...j->...j', coef_2_, base_point))

        return log

    def dist(self, point_a, point_b):
        """Compute the geodesic distance between two points.

        Parameters
        ----------
        point_a : array-like, shape=[..., dim + 1]
            First point on the hypersphere.
        point_b : array-like, shape=[..., dim + 1]
            Second point on the hypersphere.

        Returns
        -------
        dist : array-like, shape=[..., 1]
            Geodesic distance between the two points.
        """
        norm_a = self.embedding_metric.norm(point_a)
        norm_b = self.embedding_metric.norm(point_b)
        inner_prod = self.embedding_metric.inner_product(point_a, point_b)

        cos_angle = gs.einsum(
            '...,...->...', inner_prod, 1. / (norm_a * norm_b))
        cos_angle = gs.clip(cos_angle, -1, 1)

        dist = gs.arccos(cos_angle)

        return dist

    def squared_dist(self, point_a, point_b):
        """Squared geodesic distance between two points.

        Parameters
        ----------
        point_a : array-like, shape=[..., dim]
            Point on the hypersphere.
        point_b : array-like, shape=[..., dim]
            Point on the hypersphere.

        Returns
        -------
        sq_dist : array-like, shape=[...,]
        """
        return self.dist(point_a, point_b) ** 2

    @staticmethod
    def parallel_transport(tangent_vec_a, tangent_vec_b, base_point):
        r"""Compute the parallel transport of a tangent vector.

        Closed-form solution for the parallel transport of a tangent vector a
        along the geodesic defined by :math: `t \mapsto exp_(base_point)(t*
        tangent_vec_b)`.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., dim + 1]
            Tangent vector at base point to be transported.
        tangent_vec_b : array-like, shape=[..., dim + 1]
            Tangent vector at base point, along which the parallel transport
            is computed.
        base_point : array-like, shape=[..., dim + 1]
            Point on the hypersphere.

        Returns
        -------
        transported_tangent_vec: array-like, shape=[..., dim + 1]
            Transported tangent vector at `exp_(base_point)(tangent_vec_b)`.
        """
        theta = gs.linalg.norm(tangent_vec_b, axis=-1)
        normalized_b = gs.einsum('..., ...i->...i', 1 / theta, tangent_vec_b)
        pb = gs.einsum('...i,...i->...', tangent_vec_a, normalized_b)
        p_orth = tangent_vec_a - gs.einsum('..., ...i->...i', pb, normalized_b)
        transported = \
            - gs.einsum('..., ...i->...i', gs.sin(theta) * pb, base_point)\
            + gs.einsum('..., ...i->...i', gs.cos(theta) * pb, normalized_b)\
            + p_orth
        return transported

    def christoffels(self, point, point_type='spherical'):
        """Compute the Christoffel symbols at a point.

        Only implemented in dimension 2 and for spherical coordinates.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point on hypersphere where the Christoffel symbols are computed.

        point_type: str, {'spherical', 'intrinsic', 'extrinsic'}
            Coordinates in which to express the Christoffel symbols.
            Optional, default: 'spherical'.

        Returns
        -------
        christoffel : array-like, shape=[..., contravariant index, 1st
                                         covariant index, 2nd covariant index]
            Christoffel symbols at point.
        """
        if self.dim != 2 or point_type != 'spherical':
            raise NotImplementedError(
                'The Christoffel symbols are only implemented'
                ' for spherical coordinates in the 2-sphere')

        point = gs.to_ndarray(point, to_ndim=2)
        christoffel = []
        for sample in point:
            gamma_0 = gs.array(
                [[0, 0], [0, - gs.sin(sample[0]) * gs.cos(sample[0])]])
            gamma_1 = gs.array([[0, gs.cos(sample[0]) / gs.sin(sample[0])],
                                [gs.cos(sample[0]) / gs.sin(sample[0]), 0]])
            christoffel.append(gs.stack([gamma_0, gamma_1]))

        christoffel = gs.stack(christoffel)
        if gs.ndim(christoffel) == 4 and gs.shape(christoffel)[0] == 1:
            christoffel = gs.squeeze(christoffel, axis=0)
        return christoffel


class Hypersphere(_Hypersphere):
    """Class for the n-dimensional hypersphere.

    Class for the n-dimensional hypersphere embedded in the
    (n+1)-dimensional Euclidean space.

    By default, points are parameterized by their extrinsic
    (n+1)-coordinates.

    Parameters
    ----------
    dim : int
        Dimension of the hypersphere.
    """

    def __init__(self, dim):
        super(Hypersphere, self).__init__(dim)
        self.metric = HypersphereMetric(dim)
