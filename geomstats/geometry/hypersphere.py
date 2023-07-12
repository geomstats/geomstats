"""The n-dimensional hypersphere.

The n-dimensional hypersphere embedded in (n+1)-dimensional
Euclidean space.

Lead author: Nina Miolane.
"""

import logging
import math
from itertools import product

from scipy.stats import beta

import geomstats.algebra_utils as utils
import geomstats.backend as gs
from geomstats.geometry.base import LevelSet
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.riemannian_metric import RiemannianMetric


class _Hypersphere(LevelSet):
    """Private class for the n-dimensional hypersphere.

    Class for the n-dimensional hypersphere embedded in the
    (n+1)-dimensional Euclidean space.

    By default, points are parameterized by their extrinsic
    (n+1)-coordinates. For dimensions 1 and 2, this can be changed with the
    `default_coords_type` parameter. For dimensions 1 (the circle),
    the intrinsic coordinates correspond angles in radians, with 0. mapping
    to point [1., 0.]. For dimension 2, the intrinsic coordinates are the
    spherical coordinates from the north pole, i.e. where angles [0., 0.]
    correspond to point [0., 0., 1.].

    Parameters
    ----------
    dim : int
        Dimension of the hypersphere.

    default_coords_type : str, {'extrinsic', 'intrinsic'}
        Type of representation for dimensions 1 and 2.
    """

    def __init__(self, dim, default_coords_type="extrinsic", equip=True):
        self.dim = dim
        super().__init__(
            dim=dim,
            default_coords_type=default_coords_type,
            equip=equip,
        )

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return HypersphereMetric

    def _define_embedding_space(self):
        return Euclidean(self.dim + 1)

    def submersion(self, point):
        """Submersion that defines the manifold.

        Parameters
        ----------
        point : array-like, shape=[..., dim + 1]

        Returns
        -------
        submersed_point : array-like, shape=[...]
        """
        return gs.sum(point**2, axis=-1) - 1.0

    def tangent_submersion(self, vector, point):
        """Tangent submersion.

        Parameters
        ----------
        vector : array-like, shape=[..., dim+1]
        point : array-like, shape=[..., dim+1]

        Returns
        -------
        submersed_vector : array-like, shape=[...]
        """
        return 2 * gs.sum(point * vector, axis=-1)

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
        return gs.einsum("...,...i->...i", 1.0 / norm, point)

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
        sq_norm = gs.sum(base_point**2, axis=-1)
        inner_prod = self.embedding_space.metric.inner_product(base_point, vector)
        coef = inner_prod / sq_norm
        return vector - gs.einsum("...,...j->...j", coef, base_point)

    @staticmethod
    def angle_to_extrinsic(point_angle):
        """Convert point from angle to extrinsic coordinates.

        Convert from the angle in radians to the extrinsic coordinates in
        2d plane. Angle 0 corresponds to point [1., 0.] and is expected in
        [-Pi, Pi). This method is only implemented in dimension 1.

        Parameters
        ----------
        point_angle : array-like, shape=[..., 1]
            Point on the circle, i.e. an angle in radians in [-Pi, Pi].

        Returns
        -------
        point_extrinsic : array_like, shape=[..., 2]
            Point on the sphere, in extrinsic coordinates in Euclidean space.
        """
        cos = gs.cos(point_angle)
        sin = gs.sin(point_angle)
        return gs.hstack([cos, sin])

    @staticmethod
    def extrinsic_to_angle(point_extrinsic):
        """Compute the angle of a point in the plane.

        Convert from the extrinsic coordinates in the 2d plane to angle in
        radians. Angle 0 corresponds to point [1., 0.]. This method is only
        implemented in dimension 1.

        Parameters
        ----------
        point_extrinsic : array-like, shape=[..., 2]
            Point on the circle, in extrinsic coordinates in Euclidean space.

        Returns
        -------
        point_angle : array_like, shape=[..., 1]
            Point on the circle, i.e. an angle in radians in [-Pi, Pi].
        """
        return gs.expand_dims(
            gs.arctan2(point_extrinsic[..., 1], point_extrinsic[..., 0]),
            axis=-1,
        )

    def spherical_to_extrinsic(self, point_spherical):
        """Convert point from spherical to extrinsic coordinates.

        Convert from the spherical coordinates in the hypersphere
        to the extrinsic coordinates in Euclidean space.
        Spherical coordinates are defined from the north pole, i.e. that
        angles [0., 0.] correspond to point [0., 0., 1.].
        Only implemented in dimension 2.

        Parameters
        ----------
        point_spherical : array-like, shape=[..., 2]
            Point on the sphere, in spherical coordinates.

        Returns
        -------
        point_extrinsic : array_like, shape=[..., 3]
            Point on the sphere, in extrinsic coordinates in Euclidean space.
        """
        theta = point_spherical[..., 0]
        phi = point_spherical[..., 1]

        point_extrinsic = gs.stack(
            [gs.sin(theta) * gs.cos(phi), gs.sin(theta) * gs.sin(phi), gs.cos(theta)],
            axis=-1,
        )

        return point_extrinsic

    def tangent_spherical_to_extrinsic(
        self, tangent_vec_spherical, base_point_spherical
    ):
        """Convert tangent vector from spherical to extrinsic coordinates.

        Convert from the spherical coordinates in the hypersphere
        to the extrinsic coordinates in Euclidean space for a tangent
        vector. Only implemented in dimension 2.

        Parameters
        ----------
        tangent_vec_spherical : array-like, shape=[..., 2]
            Tangent vector to the sphere, in spherical coordinates.
        base_point_spherical : array-like, shape=[..., 2]
            Point on the sphere, in spherical coordinates.

        Returns
        -------
        tangent_vec_extrinsic : array-like, shape=[..., 3]
            Tangent vector to the sphere, at base point,
            in extrinsic coordinates in Euclidean space.
        """
        if self.dim != 2:
            raise NotImplementedError(
                "The conversion from spherical coordinates"
                " to extrinsic coordinates is implemented"
                " only in dimension 2."
            )

        axes = (2, 0, 1) if base_point_spherical.ndim == 2 else (0, 1)
        theta = base_point_spherical[..., 0]
        phi = base_point_spherical[..., 1]
        phi = gs.where(theta == 0.0, 0.0, phi)

        zeros = gs.zeros_like(theta)

        jac = gs.array(
            [
                [gs.cos(theta) * gs.cos(phi), -gs.sin(theta) * gs.sin(phi)],
                [gs.cos(theta) * gs.sin(phi), gs.sin(theta) * gs.cos(phi)],
                [-gs.sin(theta), zeros],
            ]
        )
        jac = gs.transpose(jac, axes)

        return gs.einsum("...ij,...j->...i", jac, tangent_vec_spherical)

    def extrinsic_to_spherical(self, point_extrinsic):
        """Convert point from extrinsic to spherical coordinates.

        Convert from the extrinsic coordinates, i.e. embedded in Euclidean
        space of dim 3 to spherical coordinates in the hypersphere.
        Spherical coordinates are defined from the north pole, i.e.
        angles [0., 0.] correspond to point [0., 0., 1.].
        Only implemented in dimension 2.

        Parameters
        ----------
        point_extrinsic : array-like, shape=[..., 3]
            Point on the sphere, in extrinsic coordinates.

        Returns
        -------
        point_spherical : array_like, shape=[..., 2]
            Point on the sphere, in spherical coordinates relative to the
            north pole.
        """
        theta = gs.arccos(point_extrinsic[..., -1])
        x = point_extrinsic[..., 0]
        y = point_extrinsic[..., 1]
        phi = gs.arctan2(y, x)
        phi = gs.where(phi < 0, phi + 2 * gs.pi, phi)
        return gs.stack([theta, phi], axis=-1)

    def tangent_extrinsic_to_spherical(
        self, tangent_vec, base_point=None, base_point_spherical=None
    ):
        """Convert tangent vector from extrinsic to spherical coordinates.

        Convert a tangent vector from the extrinsic coordinates in Euclidean
        space to the spherical coordinates in the hypersphere for.
        Spherical coordinates are considered from the north pole [0., 0.,
        1.]. This method is only implemented in dimension 2.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., 3]
            Tangent vector to the sphere, in spherical coordinates.
        base_point : array-like, shape=[..., 3]
            Point on the sphere. Unused if `base_point_spherical` is given.
            Optional, default : None.
        base_point_spherical : array-like, shape=[..., 2]
            Point on the sphere, in spherical coordinates. Either
            `base_point` or `base_point_spherical` must be given.
            Optional, default : None.

        Returns
        -------
        tangent_vec_spherical : array-like, shape=[..., 2]
            Tangent vector to the sphere, at base point,
            in spherical coordinates relative to the north pole [0., 0., 1.].
        """
        if self.dim != 2:
            raise NotImplementedError(
                "The conversion from to extrinsic coordinates "
                "spherical coordinates is implemented"
                " only in dimension 2."
            )
        if base_point is None and base_point_spherical is None:
            raise ValueError(
                "A base point must be given, either in "
                "extrinsic or in spherical coordinates."
            )
        if base_point_spherical is None and base_point is not None:
            base_point_spherical = self.extrinsic_to_spherical(base_point)

        axes = (2, 0, 1) if base_point_spherical.ndim == 2 else (0, 1)
        theta = base_point_spherical[..., 0]
        phi = base_point_spherical[..., 1]

        theta_safe = gs.where(gs.abs(theta) < gs.atol, gs.atol, theta)
        zeros = gs.zeros_like(theta)
        jac_close_0 = gs.array(
            [[gs.ones_like(theta), zeros, zeros], [zeros, gs.ones_like(theta), zeros]]
        )

        jac = gs.array(
            [
                [
                    gs.cos(theta) * gs.cos(phi),
                    gs.cos(theta) * gs.sin(phi),
                    -gs.sin(theta),
                ],
                [
                    -gs.sin(phi) / gs.sin(theta_safe),
                    gs.cos(phi) / gs.sin(theta_safe),
                    zeros,
                ],
            ]
        )

        jac = gs.transpose(jac, axes)
        jac_close_0 = gs.transpose(jac_close_0, axes)
        theta_criterion = gs.einsum("...,...ij->...ij", theta, gs.ones_like(jac))
        jac = gs.where(gs.abs(theta_criterion) < gs.atol, jac_close_0, jac)

        return gs.einsum("...ij,...j->...i", jac, tangent_vec)

    def intrinsic_to_extrinsic_coords(self, point_intrinsic):
        """Convert point from intrinsic to extrinsic coordinates.

        Convert from the intrinsic coordinates in the hypersphere,
        to the extrinsic coordinates in Euclidean space.
        For dimensions 1 (the circle), the intrinsic coordinates correspond
        angles in radians, with 0. mapping to point [1., 0.]. For dimension
        2, the intrinsic coordinates are the spherical coordinates from the
        north pole, i.e. that angles [0., 0.] correspond to point [0., 0., 1.].

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
        if self.dim == 2:
            return self.spherical_to_extrinsic(point_intrinsic)
        if self.dim == 1:
            return self.angle_to_extrinsic(point_intrinsic)

        raise NotImplementedError(
            "Intrinsic coordinates are only implemented in dimension 1 and 2."
        )

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
        if self.dim == 2:
            return self.extrinsic_to_spherical(point_extrinsic)
        if self.dim == 1:
            return self.extrinsic_to_angle(point_extrinsic)

        raise NotImplementedError(
            "Intrinsic coordinates are only implemented in dimension 1 and 2."
        )

    def _replace_values(self, samples, new_samples, indcs):
        replaced_indices = [i for i, is_replaced in enumerate(indcs) if is_replaced]
        value_indices = list(product(replaced_indices, range(self.dim + 1)))
        return gs.assignment(samples, gs.flatten(new_samples), value_indices)

    def random_point(self, n_samples=1, bound=1.0):
        """Sample in the hypersphere from the uniform distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : unused

        Returns
        -------
        samples : array-like, shape=[..., dim + 1]
            Points sampled on the hypersphere.
        """
        return self.random_uniform(n_samples)

    def random_uniform(self, n_samples=1):
        """Sample in the hypersphere from the uniform distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[..., dim + 1]
            Points sampled on the hypersphere.
        """
        size = (n_samples, self.dim + 1)

        samples = gs.random.normal(size=size)
        while True:
            norms = gs.linalg.norm(samples, axis=1)
            indcs = gs.isclose(norms, 0.0, atol=gs.atol)
            num_bad_samples = gs.sum(indcs)
            if num_bad_samples == 0:
                break
            new_samples = gs.random.normal(size=(num_bad_samples, self.dim + 1))
            samples = self._replace_values(samples, new_samples, indcs)

        samples = gs.einsum("..., ...i->...i", 1 / norms, samples)
        if n_samples == 1:
            samples = gs.squeeze(samples, axis=0)

        if self.dim in [1, 2] and self.default_coords_type == "intrinsic":
            return self.extrinsic_to_intrinsic_coords(samples)
        return samples

    def random_von_mises_fisher(self, mu=None, kappa=10, n_samples=1, max_iter=100):
        """Sample with the von Mises-Fisher distribution.

        This distribution corresponds to the maximum entropy distribution
        given a mean. In dimension 2, a closed form expression is available.
        In larger dimension, rejection sampling is used according to [Wood94]_

        References
        ----------
        https://en.wikipedia.org/wiki/Von_Mises-Fisher_distribution

        .. [Wood94]   Wood, Andrew T. A. “Simulation of the von Mises Fisher
                      Distribution.” Communications in Statistics - Simulation
                      and Computation, June 27, 2007.
                      https://doi.org/10.1080/03610919408813161.

        Parameters
        ----------
        mu : array-like, shape=[dim]
            Mean parameter of the distribution.
        kappa : float
            Kappa parameter of the von Mises distribution.
            Optional, default: 10.
        n_samples : int
            Number of samples.
            Optional, default: 1.
        max_iter : int
            Maximum number of trials in the rejection algorithm. In case it
            is reached, the current number of samples < n_samples is returned.
            Optional, default: 100.

        Returns
        -------
        point : array-like, shape=[n_samples, dim + 1]
            Points sampled on the sphere in extrinsic coordinates
            in Euclidean space of dimension dim + 1.
        """
        dim = self.dim
        if dim == 1:
            raise NotImplementedError("Not implemented for dim == 1")

        if dim == 2:
            angle = 2.0 * gs.pi * gs.random.rand(n_samples)
            angle = gs.to_ndarray(angle, to_ndim=2, axis=1)
            unit_vector = gs.hstack((gs.cos(angle), gs.sin(angle)))
            scalar = gs.random.rand(n_samples)

            coord_x = 1.0 + 1.0 / kappa * gs.log(
                scalar + (1.0 - scalar) * gs.exp(gs.array(-2.0 * kappa))
            )
            coord_x = gs.to_ndarray(coord_x, to_ndim=2, axis=1)
            coord_yz = gs.sqrt(1.0 - coord_x**2) * unit_vector
            sample = gs.hstack((coord_x, coord_yz))

        else:
            # rejection sampling in the general case
            sqrt = gs.sqrt(4 * kappa**2.0 + dim**2)
            envelop_param = (-2 * kappa + sqrt) / dim
            node = (1.0 - envelop_param) / (1.0 + envelop_param)
            correction = kappa * node + dim * gs.log(1.0 - node**2)

            n_accepted, n_iter = 0, 0
            result = []
            while (n_accepted < n_samples) and (n_iter < max_iter):
                sym_beta = beta.rvs(dim / 2, dim / 2, size=n_samples - n_accepted)
                sym_beta = gs.cast(sym_beta, node.dtype)
                coord_x = (1 - (1 + envelop_param) * sym_beta) / (
                    1 - (1 - envelop_param) * sym_beta
                )
                accept_tol = gs.random.rand(n_samples - n_accepted)
                criterion = (
                    kappa * coord_x + dim * gs.log(1 - node * coord_x) - correction
                ) > gs.log(accept_tol)
                result.append(coord_x[criterion])
                n_accepted += gs.sum(criterion)
                n_iter += 1
            if n_accepted < n_samples:
                logging.warning(
                    "Maximum number of iteration reached in rejection "
                    "sampling before n_samples were accepted."
                )
            coord_x = gs.concatenate(result)
            coord_rest = _Hypersphere(dim - 1).random_uniform(n_accepted)
            coord_rest = gs.einsum(
                "...,...i->...i", gs.sqrt(1 - coord_x**2), coord_rest
            )
            sample = gs.concatenate([coord_x[..., None], coord_rest], axis=1)

        if mu is not None:
            sample = utils.rotate_points(sample, mu)

        return sample if (n_samples > 1) else sample[0]

    def random_riemannian_normal(
        self, mean=None, precision=None, n_samples=1, max_iter=100
    ):
        r"""Sample from the Riemannian normal distribution.

        The Riemannian normal distribution, or spherical normal in this case,
        is defined by the probability density function (with respect to the
        Riemannian volume measure) proportional to:

        .. math::
                \exp \Big \left(- \frac{\lambda}{2} \mathtm{arccos}^2(x^T\mu)
                \Big \right)

        where :math:`\mu` is the mean and :math:`\lambda` is the isotropic
        precision. For the anisotropic case,
        :math:`\log_{\mu}(x)^T \Lambda \log_{\mu}(x)` is used instead.

        A rejection algorithm is used to sample from this distribution [Hau18]_

        Parameters
        ----------
        mean : array-like, shape=[dim]
            Mean parameter of the distribution.
            Optional, default: (0,...,0,1) (the north pole).
        precision : float or array-like, shape=[dim, dim]
            Inverse of the covariance parameter of the normal distribution.
            If a float is passed, the covariance matrix is precision times
            identity.
            Optional, default: identity.
        n_samples : int
            Number of samples.
            Optional, default: 1.
        max_iter : int
            Maximum number of trials in the rejection algorithm. In case it
            is reached, the current number of samples < n_samples is returned.
            Optional, default: 100.

        Returns
        -------
        point : array-like, shape=[n_samples, dim + 1]
            Points sampled on the sphere.

        References
        ----------
        .. [Hau18]  Hauberg, Soren. “Directional Statistics with the
                    Spherical Normal Distribution.”
                    In 2018 21st International Conference on Information
                    Fusion (FUSION), 704–11, 2018.
                    https://doi.org/10.23919/ICIF.2018.8455242.
        """
        dim = self.dim
        n_accepted, n_iter = 0, 0
        result = []
        if precision is None:
            precision_ = gs.eye(self.dim)
        elif isinstance(precision, (float, int)):
            precision_ = precision * gs.eye(self.dim)
        else:
            precision_ = precision
        precision_2 = precision_ + (dim - 1) / gs.pi * gs.eye(dim)
        tangent_cov = gs.linalg.inv(precision_2)

        def threshold(random_v):
            """Compute the acceptance threshold."""
            squared_norm = gs.sum(random_v**2, axis=-1)
            sinc = utils.taylor_exp_even_func(squared_norm, utils.sinc_close_0) ** (
                dim - 1
            )
            threshold_val = sinc * gs.exp(squared_norm * (dim - 1) / 2 / gs.pi)
            return threshold_val, squared_norm**0.5

        while (n_accepted < n_samples) and (n_iter < max_iter):
            envelope = gs.random.multivariate_normal(
                gs.zeros(dim), tangent_cov, size=(n_samples - n_accepted,)
            )
            thresh, norm = threshold(envelope)
            proposal = gs.random.rand(n_samples - n_accepted)
            criterion = gs.logical_and(norm <= gs.pi, proposal <= thresh)
            result.append(envelope[criterion])
            n_accepted += gs.sum(criterion)
            n_iter += 1
        if n_accepted < n_samples:
            logging.warning(
                "Maximum number of iteration reached in rejection "
                "sampling before n_samples were accepted."
            )
        tangent_sample_intr = gs.concatenate(result)
        tangent_sample = gs.concatenate(
            [tangent_sample_intr, gs.zeros(n_accepted)[:, None]], axis=1
        )

        north_pole = gs.array([0.0] * dim + [1.0])
        if mean is not None:
            mean_from_north = self.metric.log(mean, north_pole)
            tangent_sample_at_pt = self.metric.parallel_transport(
                tangent_sample, north_pole, mean_from_north
            )
        else:
            tangent_sample_at_pt = tangent_sample
            mean = north_pole
        sample = self.metric.exp(tangent_sample_at_pt, mean)
        return sample[0] if (n_samples == 1) else sample


class HypersphereMetric(RiemannianMetric):
    """Class for the Hypersphere Metric."""

    def metric_matrix(self, base_point=None):
        """Metric matrix at the tangent space at a base point.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim + 1]
            Base point.
            Optional, default: None.

        Returns
        -------
        mat : array-like, shape=[..., dim + 1, dim + 1]
            Inner-product matrix.
        """
        return self._space.embedding_space.metric.metric_matrix(base_point)

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
        return self._space.embedding_space.metric.inner_product(
            tangent_vec_a, tangent_vec_b, base_point
        )

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
        return self._space.embedding_space.metric.squared_norm(vector)

    def exp(self, tangent_vec, base_point, **kwargs):
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
        proj_tangent_vec = self._space.to_tangent(tangent_vec, base_point)
        norm2 = self._space.embedding_space.metric.squared_norm(proj_tangent_vec)

        coef_1 = utils.taylor_exp_even_func(norm2, utils.cos_close_0, order=4)
        coef_2 = utils.taylor_exp_even_func(norm2, utils.sinc_close_0, order=4)
        exp = gs.einsum("...,...j->...j", coef_1, base_point) + gs.einsum(
            "...,...j->...j", coef_2, proj_tangent_vec
        )

        return exp

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
        inner_prod = self._space.embedding_space.metric.inner_product(base_point, point)
        cos_angle = gs.clip(inner_prod, -1.0, 1.0)
        squared_angle = gs.arccos(cos_angle) ** 2
        coef_1_ = utils.taylor_exp_even_func(
            squared_angle, utils.inv_sinc_close_0, order=5
        )
        coef_2_ = utils.taylor_exp_even_func(
            squared_angle, utils.inv_tanc_close_0, order=5
        )
        log = gs.einsum("...,...j->...j", coef_1_, point) - gs.einsum(
            "...,...j->...j", coef_2_, base_point
        )

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
        embedding_metric = self._space.embedding_space.metric
        norm_a = embedding_metric.norm(point_a)
        norm_b = embedding_metric.norm(point_b)
        inner_prod = embedding_metric.inner_product(point_a, point_b)

        cos_angle = inner_prod / (norm_a * norm_b)
        cos_angle = gs.clip(cos_angle, -1, 1)

        return gs.arccos(cos_angle)

    def squared_dist(self, point_a, point_b, **kwargs):
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

    def parallel_transport(
        self, tangent_vec, base_point, direction=None, end_point=None
    ):
        r"""Compute the parallel transport of a tangent vector.

        Closed-form solution for the parallel transport of a tangent vector
        along the geodesic between two points `base_point` and `end_point`
        or alternatively defined by :math:`t \mapsto exp_{(base\_point)}(
        t*direction)`.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., dim + 1]
            Tangent vector at base point to be transported.
        base_point : array-like, shape=[..., dim + 1]
            Point on the hypersphere.
        direction : array-like, shape=[..., dim + 1]
            Tangent vector at base point, along which the parallel transport
            is computed.
            Optional, default : None.
        end_point : array-like, shape=[..., dim + 1]
            Point on the hypersphere. Point to transport to. Unused if
            `tangent_vec_b` is given.
            Optional, default : None.

        Returns
        -------
        transported_tangent_vec: array-like, shape=[..., dim + 1]
            Transported tangent vector at
            `end_point=exp_(base_point)(tangent_vec_b)`.
        """
        if direction is None:
            if end_point is not None:
                direction = self.log(end_point, base_point)
            else:
                raise ValueError(
                    "Either an end_point or a tangent_vec_b must be given to define the"
                    " geodesic along which to transport."
                )
        theta = gs.linalg.norm(direction, axis=-1)
        eps = gs.where(theta == 0.0, 1.0, theta)
        normalized_b = gs.einsum("...,...i->...i", 1 / eps, direction)
        pb = gs.dot(tangent_vec, normalized_b)
        p_orth = tangent_vec - gs.einsum("...,...i->...i", pb, normalized_b)
        transported = (
            -gs.einsum("...,...i->...i", gs.sin(theta) * pb, base_point)
            + gs.einsum("...,...i->...i", gs.cos(theta) * pb, normalized_b)
            + p_orth
        )
        return transported

    def christoffels(self, point, coords_type="spherical"):
        """Compute the Christoffel symbols at a point.

        Only implemented in dimension 2 and for spherical coordinates.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point on hypersphere where the Christoffel symbols are computed.

        coords_type: str, {'spherical', 'intrinsic', 'extrinsic'}
            Coordinates in which to express the Christoffel symbols.
            Optional, default: 'spherical'.

        Returns
        -------
        christoffel : array-like, shape=[..., contravariant index, 1st
                                         covariant index, 2nd covariant index]
            Christoffel symbols at point.
        """
        if self._space.dim != 2 or coords_type != "spherical":
            raise NotImplementedError(
                "The Christoffel symbols are only implemented"
                " for spherical coordinates in the 2-sphere"
            )

        point = gs.to_ndarray(point, to_ndim=2)
        christoffel = []
        for sample in point:
            gamma_0 = gs.array([[0, 0], [0, -gs.sin(sample[0]) * gs.cos(sample[0])]])
            gamma_1 = gs.array(
                [
                    [0, gs.cos(sample[0]) / gs.sin(sample[0])],
                    [gs.cos(sample[0]) / gs.sin(sample[0]), 0],
                ]
            )
            christoffel.append(gs.stack([gamma_0, gamma_1]))

        christoffel = gs.stack(christoffel)
        if gs.ndim(christoffel) == 4 and gs.shape(christoffel)[0] == 1:
            christoffel = gs.squeeze(christoffel, axis=0)
        return christoffel

    def curvature(self, tangent_vec_a, tangent_vec_b, tangent_vec_c, base_point):
        r"""Compute the curvature.

        For three tangent vectors at a base point :math:`x,y,z`,
        the curvature is defined by
        :math:`R(x, y)z = \nabla_{[x,y]}z
        - \nabla_z\nabla_y z + \nabla_y\nabla_x z`, where :math:`\nabla`
        is the Levi-Civita connection. In the case of the hypersphere,
        we have the closed formula
        :math:`R(x,y)z = \langle x, z \rangle y - \langle y,z \rangle x`.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., dim]
            Tangent vector at `base_point`.
        tangent_vec_b : array-like, shape=[..., dim]
            Tangent vector at `base_point`.
        tangent_vec_c : array-like, shape=[..., dim]
            Tangent vector at `base_point`.
        base_point :  array-like, shape=[..., dim]
            Point on the hypersphere.

        Returns
        -------
        curvature : array-like, shape=[..., dim]
            Tangent vector at `base_point`.
        """
        inner_ac = self.inner_product(tangent_vec_a, tangent_vec_c)
        inner_bc = self.inner_product(tangent_vec_b, tangent_vec_c)
        first_term = gs.einsum("...,...i->...i", inner_bc, tangent_vec_a)
        second_term = gs.einsum("...,...i->...i", inner_ac, tangent_vec_b)
        return -first_term + second_term

    def _normalization_factor_odd_dim(self, variances):
        """Compute the normalization factor - odd dimension."""
        dim = self._space.dim
        half_dim = int((dim + 1) / 2)
        area = 2 * gs.pi**half_dim / math.factorial(half_dim - 1)
        comb = gs.comb(dim - 1, half_dim - 1)

        erf_arg = gs.sqrt(variances / 2) * gs.pi
        first_term = (
            area
            / (2**dim - 1)
            * comb
            * gs.sqrt(gs.pi / (2 * variances))
            * gs.erf(erf_arg)
        )

        def summand(k):
            exp_arg = -((dim - 1 - 2 * k) ** 2) / 2 / variances
            erf_arg_2 = (gs.pi * variances - (dim - 1 - 2 * k) * 1j) / gs.sqrt(
                2 * variances
            )
            sign = (-1.0) ** k
            comb_2 = gs.comb(k, dim - 1)
            return sign * comb_2 * gs.exp(exp_arg) * gs.real(gs.erf(erf_arg_2))

        if half_dim > 2:
            sum_term = gs.sum(gs.stack([summand(k)] for k in range(half_dim - 2)))
        else:
            sum_term = summand(0)
        coef = area / 2 / erf_arg * gs.pi**0.5 * (-1.0) ** (half_dim - 1)

        return first_term + coef / 2 ** (dim - 2) * sum_term

    def _normalization_factor_even_dim(self, variances):
        """Compute the normalization factor - even dimension."""
        dim = self._space.dim
        half_dim = (dim + 1) / 2
        area = 2 * gs.pi**half_dim / math.gamma(half_dim)

        def summand(k):
            exp_arg = -((dim - 1 - 2 * k) ** 2) / 2 / variances
            erf_arg_1 = (dim - 1 - 2 * k) * 1j / gs.sqrt(2 * variances)
            erf_arg_2 = (gs.pi * variances - (dim - 1 - 2 * k) * 1j) / gs.sqrt(
                2 * variances
            )
            sign = (-1.0) ** k
            comb = gs.comb(dim - 1, k)
            erf_terms = gs.imag(gs.erf(erf_arg_2) + gs.erf(erf_arg_1))
            return sign * comb * gs.exp(exp_arg) * erf_terms

        half_dim_2 = int((dim - 2) / 2)
        if half_dim_2 > 0:
            sum_term = gs.sum(gs.stack([summand(k)] for k in range(half_dim_2)))
        else:
            sum_term = summand(0)
        coef = (
            area
            * (-1.0) ** half_dim_2
            / 2 ** (dim - 2)
            * gs.sqrt(gs.pi / 2 / variances)
        )

        return coef * sum_term

    def normalization_factor(self, variances):
        """Return normalization factor of the Gaussian distribution.

        Parameters
        ----------
        variances : array-like, shape=[n,]
            Variance of the distribution.

        Returns
        -------
        norm_func : array-like, shape=[n,]
            Normalisation factor for all given variances.
        """
        if self._space.dim % 2 == 0:
            return self._normalization_factor_even_dim(variances)
        return self._normalization_factor_odd_dim(variances)

    def norm_factor_gradient(self, variances):
        """Compute the gradient of the normalization factor.

        Parameters
        ----------
        variances : array-like, shape=[n,]
            Variance of the distribution.

        Returns
        -------
        norm_func : array-like, shape=[n,]
            Normalisation factor for all given variances.
        """

        def func(var):
            return gs.sum(self.normalization_factor(var))

        _, grad = gs.autodiff.value_and_grad(func)(variances)
        return _, grad

    def curvature_derivative(
        self,
        tangent_vec_a,
        tangent_vec_b=None,
        tangent_vec_c=None,
        tangent_vec_d=None,
        base_point=None,
    ):
        r"""Compute the covariant derivative of the curvature.

        The derivative of the curvature vanishes since the hypersphere is a
        constant curvature space.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., dim]
            Tangent vector at `base_point` along which the curvature is
            derived.
        tangent_vec_b : array-like, shape=[..., dim]
            Unused tangent vector at `base_point` (since curvature derivative
            vanishes).
        tangent_vec_c : array-like, shape=[..., dim]
            Unused tangent vector at `base_point` (since curvature derivative
            vanishes).
        tangent_vec_d : array-like, shape=[..., dim]
            Unused tangent vector at `base_point` (since curvature derivative
            vanishes).
        base_point : array-like, shape=[..., dim]
            Unused point on the hypersphere.

        Returns
        -------
        curvature_derivative : array-like, shape=[..., dim]
            Tangent vector at base point.
        """
        return gs.zeros_like(tangent_vec_a)

    def injectivity_radius(self, base_point):
        """Compute the radius of the injectivity domain.

        This is is the supremum of radii r for which the exponential map is a
        diffeomorphism from the open ball of radius r centered at the base
        point onto its image.
        In the case of the sphere, it does not depend on the base point and is
        Pi everywhere.

        Parameters
        ----------
        base_point : array-like, shape=[..., dim+1]
            Point on the manifold.

        Returns
        -------
        radius : float
            Injectivity radius.
        """
        return gs.pi


class Hypersphere(_Hypersphere):
    """Class for the n-dimensional hypersphere.

    Class for the n-dimensional hypersphere embedded in the
    (n+1)-dimensional Euclidean space.

    By default, points are parameterized by their extrinsic
    (n+1)-coordinates. For dimensions 1 and 2, this can be changed with the
    `default_coords_type` parameter. For dimensions 1 (the circle),
    the intrinsic coordinates correspond angles in radians, with 0. mapping
    to point [1., 0.]. For dimension 2, the intrinsic coordinates are the
    spherical coordinates from the north pole, i.e. where angles [0.,
    0.] correspond to point [0., 0., 1.].

    Parameters
    ----------
    dim : int
        Dimension of the hypersphere.

    default_coords_type : str, {'extrinsic', 'intrinsic'}
        Type of representation for dimensions 1 and 2.
    """

    def __init__(self, dim, default_coords_type="extrinsic", equip=True):
        super().__init__(dim, default_coords_type, equip=equip)
