"""The n-dimensional hypersphere.

The n-dimensional hypersphere embedded in (n+1)-dimensional
Euclidean space.
"""

import logging
import math

import geomstats.backend as gs
from geomstats.geometry.embedded_manifold import EmbeddedManifold
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.euclidean import EuclideanMetric
from geomstats.geometry.riemannian_metric import RiemannianMetric

TOLERANCE = 1e-6
EPSILON = 1e-8

COS_TAYLOR_COEFFS = [1., 0.,
                     - 1.0 / math.factorial(2), 0.,
                     + 1.0 / math.factorial(4), 0.,
                     - 1.0 / math.factorial(6), 0.,
                     + 1.0 / math.factorial(8), 0.]
INV_SIN_TAYLOR_COEFFS = [0., 1. / 6.,
                         0., 7. / 360.,
                         0., 31. / 15120.,
                         0., 127. / 604800.]
INV_TAN_TAYLOR_COEFFS = [0., - 1. / 3.,
                         0., - 1. / 45.,
                         0., - 2. / 945.,
                         0., -1. / 4725.]


class Hypersphere(EmbeddedManifold):
    """Class for the n-dimensional hypersphere.

    Class for the n-dimensional hypersphere embedded in the
    (n+1)-dimensional Euclidean space.

    By default, points are parameterized by their extrinsic
    (n+1)-coordinates.

    Parameters
    ----------
    dimension: int
        Dimension of the hypersphere.
    """

    def __init__(self, dimension):
        assert isinstance(dimension, int) and dimension > 0
        super(Hypersphere, self).__init__(
            dimension=dimension,
            embedding_manifold=Euclidean(dimension + 1))
        self.embedding_metric = self.embedding_manifold.metric
        self.metric = HypersphereMetric(dimension)

    def belongs(self, point, tolerance=TOLERANCE):
        """Test if a point belongs to the hypersphere.

        This tests whether the point's squared norm in Euclidean space is 1.

        Parameters
        ----------
        point : array-like, shape=[n_samples, dimension + 1]
            Points in Euclidean space.
        tolerance : float, optional
            Tolerance at which to evaluate norm == 1 (default: TOLERANCE).

        Returns
        -------
        belongs : array-like, shape=[n_samples, 1]
            Array of booleans evaluating if each point belongs to
            the hypersphere.
        """
        point = gs.asarray(point)
        point_dim = point.shape[-1]
        if point_dim != self.dimension + 1:
            if point_dim is self.dimension:
                logging.warning(
                    'Use the extrinsic coordinates to '
                    'represent points on the hypersphere.')
            return gs.array([[False]])
        sq_norm = self.embedding_metric.squared_norm(point)
        diff = gs.abs(sq_norm - 1)
        return gs.less_equal(diff, tolerance)

    def regularize(self, point):
        """Regularize a point to the canonical representation.

        Regularize a point to the canonical representation chosen
        for the hypersphere, to avoid numerical issues.

        Parameters
        ----------
        point : array-like, shape=[n_samples, dimension + 1]
            Points on the hypersphere.

        Returns
        -------
        projected_point : array-like, shape=[n_samples, dimension + 1]
            Points in canonical representation chosen for the hypersphere.
        """
        assert gs.all(self.belongs(point))

        return self.projection(point)

    def projection(self, point):
        """Project a point on the hypersphere.

        Parameters
        ----------
        point : array-like, shape=[n_samples, dimension + 1]
            Point in embedding Euclidean space.

        Returns
        -------
        projected_point : array-like, shape=[n_samples, dimension + 1]
            Point projected on the hypersphere.
        """
        point = gs.to_ndarray(point, to_ndim=2)

        norm = self.embedding_metric.norm(point)
        projected_point = point / norm

        return projected_point

    def projection_to_tangent_space(self, vector, base_point):
        """Project a vector to the tangent space.

        Project a vector in Euclidean space
        on the tangent space of the hypersphere at a base point.

        Parameters
        ----------
        vector : array-like, shape=[n_samples, dimension + 1]
            Vector in Euclidean space.
        base_point : array-like, shape=[n_samples, dimension + 1]
            Point on the hypersphere defining the tangent space,
            where the vector will be projected.

        Returns
        -------
        tangent_vec : array-like, shape=[n_samples, dimension + 1]
            Tangent vector in the tangent space of the hypersphere
            at the base point.
        """
        vector = gs.to_ndarray(vector, to_ndim=2)
        base_point = gs.to_ndarray(base_point, to_ndim=2)

        sq_norm = self.embedding_metric.squared_norm(base_point)
        inner_prod = self.embedding_metric.inner_product(base_point, vector)
        coef = inner_prod / sq_norm
        tangent_vec = vector - gs.einsum('ni,nj->nj', coef, base_point)

        return tangent_vec

    def spherical_to_extrinsic(self, point_spherical):
        """Convert point from spherical to extrinsic coordinates.

        Convert from the spherical coordinates in the hypersphere
        to the extrinsic coordinates in Euclidean space.
        Only implemented in dimension 2.

        Parameters
        ----------
        point_spherical : array-like, shape=[n_samples, dimension]
            Point on the sphere, in spherical coordinates.

        Returns
        -------
        point_extrinsic : array_like, shape=[n_samples, dimension + 1]
            Point on the sphere, in extrinsic coordinates in Euclidean space.
        """
        if self.dimension != 2:
            raise NotImplementedError(
                'The conversion from spherical coordinates'
                ' to extrinsic coordinates is implemented'
                ' only in dimension 2.')
        point_spherical = gs.to_ndarray(point_spherical, to_ndim=2)
        theta = point_spherical[:, 0]
        phi = point_spherical[:, 1]
        point_extrinsic = gs.zeros(
            (point_spherical.shape[0], self.dimension + 1))
        point_extrinsic[:, 0] = gs.sin(theta) * gs.cos(phi)
        point_extrinsic[:, 1] = gs.sin(theta) * gs.sin(phi)
        point_extrinsic[:, 2] = gs.cos(theta)
        assert self.belongs(point_extrinsic).all()

        return point_extrinsic

    def tangent_spherical_to_extrinsic(self, tangent_vec_spherical,
                                       base_point_spherical):
        """Convert tangent vector from spherical to extrinsic coordinates.

        Convert from the spherical coordinates in the hypersphere
        to the extrinsic coordinates in Euclidean space for a tangent
        vector. Only implemented in dimension 2.

        Parameters
        ----------
        tangent_vec_spherical : array-like, shape=[n_samples, dimension]
            Tangent vector to the sphere, in spherical coordinates.
        base_point_spherical : array-like, shape=[n_samples, dimension]
            Point on the sphere, in spherical coordinates.

        Returns
        -------
        tangent_vec_extrinsic : array-like, shape=[n_samples, dimension + 1]
            Tangent vector to the sphere, at base point,
            in extrinsic coordinates in Euclidean space.
        """
        if self.dimension != 2:
            raise NotImplementedError(
                'The conversion from spherical coordinates'
                ' to extrinsic coordinates is implemented'
                ' only in dimension 2.')
        base_point_spherical = gs.to_ndarray(base_point_spherical, to_ndim=2)
        tangent_vec_spherical = gs.to_ndarray(tangent_vec_spherical, to_ndim=2)
        n_samples = base_point_spherical.shape[0]
        theta = base_point_spherical[:, 0]
        phi = base_point_spherical[:, 1]
        jac = gs.zeros((n_samples, self.dimension + 1, self.dimension))
        jac[:, 0, 0] = gs.cos(theta) * gs.cos(phi)
        jac[:, 0, 1] = - gs.sin(theta) * gs.sin(phi)
        jac[:, 1, 0] = gs.cos(theta) * gs.sin(phi)
        jac[:, 1, 1] = gs.sin(theta) * gs.cos(phi)
        jac[:, 2, 0] = - gs.sin(theta)
        tangent_vec_extrinsic = gs.einsum('nij,nj->ni', jac,
                                          tangent_vec_spherical)

        return tangent_vec_extrinsic

    def intrinsic_to_extrinsic_coords(self, point_intrinsic):
        """Convert point from intrinsic to extrinsic coordinates.

        Convert from the intrinsic coordinates in the hypersphere,
        to the extrinsic coordinates in Euclidean space.

        Parameters
        ----------
        point_intrinsic : array-like, shape=[n_samples, dimension]
            Point on the hypersphere, in intrinsic coordinates.

        Returns
        -------
        point_extrinsic : array-like, shape=[n_samples, dimension + 1]
            Point on the hypersphere, in extrinsic coordinates in
            Euclidean space.
        """
        point_intrinsic = gs.to_ndarray(point_intrinsic, to_ndim=2)

        # FIXME: The next line needs to be guarded against taking the sqrt of
        #        negative numbers.
        coord_0 = gs.sqrt(1. - gs.linalg.norm(point_intrinsic, axis=-1) ** 2)
        coord_0 = gs.to_ndarray(coord_0, to_ndim=2, axis=-1)

        point_extrinsic = gs.concatenate([coord_0, point_intrinsic], axis=-1)

        return point_extrinsic

    def extrinsic_to_intrinsic_coords(self, point_extrinsic):
        """Convert point from extrinsic to intrinsic coordinates.

        Convert from the extrinsic coordinates in Euclidean space,
        to some intrinsic coordinates in the hypersphere.

        Parameters
        ----------
        point_extrinsic : array-like, shape=[n_samples, dimension + 1]
            Point on the hypersphere, in extrinsic coordinates in
            Euclidean space.

        Returns
        -------
        point_intrinsic : array-like, shape=[n_samples, dimension]
            Point on the hypersphere, in intrinsic coordinates.
        """
        point_extrinsic = gs.to_ndarray(point_extrinsic, to_ndim=2)

        point_intrinsic = point_extrinsic[:, 1:]

        return point_intrinsic

    def random_uniform(self, n_samples=1):
        """Sample in the hypersphere from the uniform distribution.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples.

        Returns
        -------
        samples : array-like, shape=[n_samples, dimension + 1]
            Points sampled on the hypersphere.
        """
        size = (n_samples, self.dimension + 1)

        samples = gs.random.normal(size=size)
        while True:
            norms = gs.linalg.norm(samples, axis=1)
            indcs = gs.isclose(norms, 0.0)
            num_bad_samples = gs.sum(indcs)
            if num_bad_samples == 0:
                break
            samples[indcs, :] = gs.random.normal(
                size=(num_bad_samples, self.dimension + 1))

        return gs.einsum('n, ni->ni', 1 / norms, samples)

    def random_von_mises_fisher(self, kappa=10, n_samples=1):
        """Sample in the 2-sphere with the von Mises distribution.

        Sample in the 2-sphere with the von Mises distribution centered at the
        north pole.

        References
        ----------
        https://en.wikipedia.org/wiki/Von_Mises_distribution

        Parameters
        ----------
        kappa : int, optional
            Kappa parameter of the von Mises distribution.
        n_samples : int, optional
            Number of samples.

        Returns
        -------
        point : array-like, shape=[n_samples, 3]
            Points sampled on the sphere in extrinsic coordinates
            in Euclidean space of dimension 3.
        """
        if self.dimension != 2:
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

        return point


class HypersphereMetric(RiemannianMetric):
    """Class for the Hypersphere Metric.

    Parameters
    ----------
    dimension : int
        Dimension of the hypersphere.
    """

    def __init__(self, dimension):
        super(HypersphereMetric, self).__init__(
            dimension=dimension,
            signature=(dimension, 0, 0))
        self.embedding_metric = EuclideanMetric(dimension + 1)

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
        """Compute the inner-product of two tangent vectors at a base point.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[n_samples, dimension + 1]
            First tangent vector at base point.
        tangent_vec_b : array-like, shape=[n_samples, dimension + 1]
            Second tangent vector at base point.
        base_point : array-like, shape=[n_samples, dimension + 1], optional
            Point on the hypersphere.

        Returns
        -------
        inner_prod : array-like, shape=[n_samples, 1]
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
        vector : array-like, shape=[n_samples, dimension + 1]
            Vector on the tangent space of the hypersphere at base point.
        base_point : array-like, shape=[n_samples, dimension + 1], optional
            Point on the hypersphere.

        Returns
        -------
        sq_norm : array-like, shape=[n_samples, 1]
            Squared norm of the vector.
        """
        sq_norm = self.embedding_metric.squared_norm(vector)
        return sq_norm

    def exp(self, tangent_vec, base_point):
        """Compute the Riemannian exponential of a tangent vector.

        Parameters
        ----------
        tangent_vec : array-like, shape=[n_samples, dimension + 1]
            Tangent vector at a base point.
        base_point : array-like, shape=[n_samples, dimension + 1]
            Point on the hypersphere.

        Returns
        -------
        exp : array-like, shape=[n_samples, dimension + 1]
            Point on the hypersphere equal to the Riemannian exponential
            of tangent_vec at the base point.
        """
        tangent_vec = gs.to_ndarray(tangent_vec, to_ndim=2)
        base_point = gs.to_ndarray(base_point, to_ndim=2)

        # TODO(nina): Decide on metric.space or space.metric
        #  for the hypersphere
        # TODO(nina): Raise error when vector is not tangent
        n_base_points, extrinsic_dim = base_point.shape
        n_tangent_vecs, _ = tangent_vec.shape

        hypersphere = Hypersphere(dimension=extrinsic_dim - 1)
        proj_tangent_vec = hypersphere.projection_to_tangent_space(
            tangent_vec, base_point)
        norm_tangent_vec = self.embedding_metric.norm(proj_tangent_vec)

        mask_0 = gs.isclose(norm_tangent_vec, 0.)
        mask_non0 = ~mask_0

        coef_1 = gs.zeros((n_tangent_vecs, 1))
        coef_2 = gs.zeros((n_tangent_vecs, 1))
        norm2 = norm_tangent_vec[mask_0]**2
        norm4 = norm2**2
        norm6 = norm2**3
        coef_1[mask_0] = 1. - norm2 / 2. + norm4 / 24. - norm6 / 720.
        coef_2[mask_0] = 1. - norm2 / 6. + norm4 / 120. - norm6 / 5040.

        coef_1[mask_non0] = gs.cos(norm_tangent_vec[mask_non0])
        coef_2[mask_non0] = gs.sin(norm_tangent_vec[mask_non0]) / \
            norm_tangent_vec[mask_non0]

        n_coef_1 = n_tangent_vecs
        if n_coef_1 != n_base_points:
            if n_coef_1 == 1:
                coef_1 = gs.squeeze(coef_1, axis=0)
                einsum_str = 'i,nj->nj'
            elif n_base_points == 1:
                base_point = gs.squeeze(base_point, axis=0)
                einsum_str = 'ni,j->nj'
            else:
                raise ValueError('Shape mismatch in einsum.')
        else:
            einsum_str = 'ni,nj->nj'

        exp = (gs.einsum(einsum_str, coef_1, base_point)
               + gs.einsum('ni,nj->nj', coef_2, proj_tangent_vec))

        return exp

    def log(self, point, base_point):
        """Compute the Riemannian logarithm of a point.

        Parameters
        ----------
        point : array-like, shape=[n_samples, dimension + 1]
            Point on the hypersphere.
        base_point : array-like, shape=[n_samples, dimension + 1]
            Point on the hypersphere.

        Returns
        -------
        log : array-like, shape=[n_samples, dimension + 1]
            Tangent vector at the base point equal to the Riemannian logarithm
            of point at the base point.
        """
        point = gs.to_ndarray(point, to_ndim=2)
        base_point = gs.to_ndarray(base_point, to_ndim=2)

        norm_base_point = self.embedding_metric.norm(base_point)
        norm_point = self.embedding_metric.norm(point)
        inner_prod = self.embedding_metric.inner_product(base_point, point)
        cos_angle = inner_prod / (norm_base_point * norm_point)
        cos_angle = gs.clip(cos_angle, -1., 1.)

        angle = gs.arccos(cos_angle)
        angle = gs.to_ndarray(angle, to_ndim=1)
        angle = gs.to_ndarray(angle, to_ndim=2, axis=1)

        mask_0 = gs.isclose(angle, 0.)
        mask_else = gs.equal(mask_0, gs.array(False))

        mask_0_float = gs.cast(mask_0, gs.float32)
        mask_else_float = gs.cast(mask_else, gs.float32)

        coef_1 = gs.zeros_like(angle)
        coef_2 = gs.zeros_like(angle)

        coef_1 += mask_0_float * (
            1. + INV_SIN_TAYLOR_COEFFS[1] * angle ** 2
            + INV_SIN_TAYLOR_COEFFS[3] * angle ** 4
            + INV_SIN_TAYLOR_COEFFS[5] * angle ** 6
            + INV_SIN_TAYLOR_COEFFS[7] * angle ** 8)
        coef_2 += mask_0_float * (
            1. + INV_TAN_TAYLOR_COEFFS[1] * angle ** 2
            + INV_TAN_TAYLOR_COEFFS[3] * angle ** 4
            + INV_TAN_TAYLOR_COEFFS[5] * angle ** 6
            + INV_TAN_TAYLOR_COEFFS[7] * angle ** 8)

        # This avoids division by 0.
        angle += mask_0_float * 1.

        coef_1 += mask_else_float * angle / gs.sin(angle)
        coef_2 += mask_else_float * angle / gs.tan(angle)

        log = (gs.einsum('ni,nj->nj', coef_1, point)
               - gs.einsum('ni,nj->nj', coef_2, base_point))

        mask_same_values = gs.isclose(point, base_point)

        mask_else = gs.equal(mask_same_values, gs.array(False))
        mask_else_float = gs.cast(mask_else, gs.float32)
        mask_else_float = gs.to_ndarray(mask_else_float, to_ndim=1)
        mask_else_float = gs.to_ndarray(mask_else_float, to_ndim=2)
        mask_not_same_points = gs.sum(mask_else_float, axis=1)
        mask_same_points = gs.isclose(mask_not_same_points, 0.)
        mask_same_points = gs.cast(mask_same_points, gs.float32)
        mask_same_points = gs.to_ndarray(mask_same_points, to_ndim=2, axis=1)

        mask_same_points_float = gs.cast(mask_same_points, gs.float32)

        log -= mask_same_points_float * log

        return log

    def dist(self, point_a, point_b):
        """Compute the geodesic distance between two points.

        Parameters
        ----------
        point_a : array-like, shape=[n_samples, dimension + 1]
            First point on the hypersphere.
        point_b : array-like, shape=[n_samples, dimension + 1]
            Second point on the hypersphere.

        Returns
        -------
        dist : array-like, shape=[n_samples, 1]
            Geodesic distance between the two points.
        """
        norm_a = self.embedding_metric.norm(point_a)
        norm_b = self.embedding_metric.norm(point_b)
        inner_prod = self.embedding_metric.inner_product(point_a, point_b)

        cos_angle = inner_prod / (norm_a * norm_b)
        cos_angle = gs.clip(cos_angle, -1, 1)

        dist = gs.arccos(cos_angle)

        return dist

    def parallel_transport(self, tangent_vec_a, tangent_vec_b, base_point):
        """Compute the parallel transport of a tangent vector.

        Closed-form solution for the parallel transport of a tangent vector a
        along the geodesic defined by exp_(base_point)(tangent_vec_b).

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[n_samples, dimension + 1]
            Tangent vector at base point to be transported.
        tangent_vec_b : array-like, shape=[n_samples, dimension + 1]
            Tangent vector at base point, along which the parallel transport
            is computed.
        base_point : array-like, shape=[n_samples, dimension + 1]
            Point on the hypersphere.

        Returns
        -------
        transported_tangent_vec: array-like, shape=[n_samples, dimension + 1]
            Transported tangent vector at exp_(base_point)(tangent_vec_b).
        """
        tangent_vec_a = gs.to_ndarray(tangent_vec_a, to_ndim=2)
        tangent_vec_b = gs.to_ndarray(tangent_vec_b, to_ndim=2)
        base_point = gs.to_ndarray(base_point, to_ndim=2)
        # TODO(nguigs): work around this condition
        assert len(base_point) == len(tangent_vec_a) == len(tangent_vec_b)
        theta = gs.linalg.norm(tangent_vec_b, axis=1)
        normalized_b = gs.einsum('n, ni->ni', 1 / theta, tangent_vec_b)
        pb = gs.einsum('ni,ni->n', tangent_vec_a, normalized_b)
        p_orth = tangent_vec_a - gs.einsum('n,ni->ni', pb, normalized_b)
        transported = - gs.einsum('n,ni->ni', gs.sin(theta) * pb, base_point)\
            + gs.einsum('n,ni->ni', gs.cos(theta) * pb, normalized_b)\
            + p_orth
        return transported

    def christoffels(self, point, point_type='spherical'):
        """Compute the Christoffel symbols at a point.

        Only implemented in dimension 2 and for spherical coordinates.

        Parameters
        ----------
        point : array-like, shape=[n_samples, dimension]
            Point on hypersphere where the Christoffel symbols are computed.

        point_type: str, {'spherical', 'intrinsic', 'extrinsic'}
            Coordinates in which to express the Christoffel symbols.

        Returns
        -------
        christoffel : array-like, shape=[n_samples, contravariant index, 1st
                                         covariant index, 2nd covariant index]
            Christoffel symbols at point.
        """
        if self.dimension != 2 or point_type != 'spherical':
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

        return gs.stack(christoffel)
