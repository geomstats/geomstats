"""The n-dimensional hypersphere.

The n-dimensional hypersphere embedded in (n+1)-dimensional
Euclidean space.

Lead author: Nina Miolane.
"""

import abc
import logging
import math

from scipy.stats import beta

import geomstats.algebra_utils as utils
import geomstats.backend as gs
from geomstats._einsum import scalarvecmul
from geomstats.geometry.base import DiffeomorphicManifold, LevelSet
from geomstats.geometry.diffeo import Diffeo
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.manifold import Manifold
from geomstats.geometry.pullback_metric import PullbackDiffeoMetric
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.vectorization import get_batch_shape, repeat_out

logger = logging.getLogger(__name__)


class AngleToExtrinsicDiffeo(Diffeo):
    r"""Diffeomorphism from angles to the circle in extrinsic coordinates.

    .. math::

        \theta \mapsto (\cos \theta, \sin \theta)

    Angle 0 corresponds to point [1., 0.]. The inverse returns angles in
    :math:`(-\pi, \pi]`.
    """

    def __call__(self, base_point):
        """Convert angles to extrinsic coordinates.

        Parameters
        ----------
        base_point : array-like, shape=[..., 1]
            Angle in radians.

        Returns
        -------
        image_point : array-like, shape=[..., 2]
            Point on the circle, in extrinsic coordinates.
        """
        return gs.concatenate([gs.cos(base_point), gs.sin(base_point)], axis=-1)

    def inverse(self, image_point):
        """Convert extrinsic coordinates to angles.

        Parameters
        ----------
        image_point : array-like, shape=[..., 2]
            Point on the circle, in extrinsic coordinates.

        Returns
        -------
        base_point : array-like, shape=[..., 1]
            Angle in radians.
        """
        return gs.expand_dims(
            gs.arctan2(image_point[..., 1], image_point[..., 0]), axis=-1
        )

    def tangent(self, tangent_vec, base_point=None, image_point=None):
        r"""Convert a tangent vector from angle to extrinsic coordinates.

        .. math::

            d\theta \mapsto d\theta \, (-\sin \theta, \cos \theta)

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., 1]
            Tangent vector at base point.
        base_point : array-like, shape=[..., 1]
            Angle in radians.
        image_point : array-like, shape=[..., 2]
            Point on the circle, in extrinsic coordinates.

        Returns
        -------
        image_tangent_vec : array-like, shape=[..., 2]
            Tangent vector at image point.
        """
        if base_point is None:
            base_point = self.inverse(image_point)

        direction = gs.concatenate([-gs.sin(base_point), gs.cos(base_point)], axis=-1)
        return tangent_vec * direction

    def inverse_tangent(self, image_tangent_vec, image_point=None, base_point=None):
        r"""Convert a tangent vector from extrinsic to angle coordinates.

        .. math::

            (dx, dy) \mapsto -y \, dx + x \, dy

        Parameters
        ----------
        image_tangent_vec : array-like, shape=[..., 2]
            Tangent vector at image point.
        image_point : array-like, shape=[..., 2]
            Point on the circle, in extrinsic coordinates.
        base_point : array-like, shape=[..., 1]
            Angle in radians.

        Returns
        -------
        tangent_vec : array-like, shape=[..., 1]
            Tangent vector at base point.
        """
        if image_point is None:
            image_point = self(base_point)

        return gs.expand_dims(
            -image_point[..., 1] * image_tangent_vec[..., 0]
            + image_point[..., 0] * image_tangent_vec[..., 1],
            axis=-1,
        )


class SphericalToExtrinsicDiffeo(Diffeo):
    r"""Diffeomorphism from spherical to extrinsic coordinates on the sphere.

    .. math::

        (\theta, \phi) \mapsto
        (\sin \theta \cos \phi, \sin \theta \sin \phi, \cos \theta)

    Spherical coordinates are defined from the north pole, i.e. angles
    [0., 0.] correspond to point [0., 0., 1.]. The inverse returns
    :math:`\theta \in [0, \pi]` and :math:`\phi \in [0, 2\pi)`.
    """

    def __call__(self, base_point):
        """Convert spherical to extrinsic coordinates.

        Parameters
        ----------
        base_point : array-like, shape=[..., 2]
            Point on the sphere, in spherical coordinates.

        Returns
        -------
        image_point : array-like, shape=[..., 3]
            Point on the sphere, in extrinsic coordinates.
        """
        theta = base_point[..., 0]
        phi = base_point[..., 1]

        return gs.stack(
            [gs.sin(theta) * gs.cos(phi), gs.sin(theta) * gs.sin(phi), gs.cos(theta)],
            axis=-1,
        )

    def inverse(self, image_point):
        """Convert extrinsic to spherical coordinates.

        Parameters
        ----------
        image_point : array-like, shape=[..., 3]
            Point on the sphere, in extrinsic coordinates.

        Returns
        -------
        base_point : array-like, shape=[..., 2]
            Point on the sphere, in spherical coordinates.
        """
        theta = gs.arccos(image_point[..., -1])
        phi = gs.arctan2(image_point[..., 1], image_point[..., 0])
        phi = gs.where(phi < 0, phi + 2 * gs.pi, phi)
        return gs.stack([theta, phi], axis=-1)

    def tangent(self, tangent_vec, base_point=None, image_point=None):
        """Convert a tangent vector from spherical to extrinsic coordinates.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., 2]
            Tangent vector at base point.
        base_point : array-like, shape=[..., 2]
            Point on the sphere, in spherical coordinates.
        image_point : array-like, shape=[..., 3]
            Point on the sphere, in extrinsic coordinates.

        Returns
        -------
        image_tangent_vec : array-like, shape=[..., 3]
            Tangent vector at image point.
        """
        if base_point is None:
            base_point = self.inverse(image_point)

        theta = base_point[..., 0]
        phi = gs.where(theta == 0.0, 0.0, base_point[..., 1])
        d_theta = tangent_vec[..., 0]
        d_phi = tangent_vec[..., 1]

        return gs.stack(
            [
                gs.cos(theta) * gs.cos(phi) * d_theta
                - gs.sin(theta) * gs.sin(phi) * d_phi,
                gs.cos(theta) * gs.sin(phi) * d_theta
                + gs.sin(theta) * gs.cos(phi) * d_phi,
                -gs.sin(theta) * d_theta,
            ],
            axis=-1,
        )

    def inverse_tangent(self, image_tangent_vec, image_point=None, base_point=None):
        """Convert a tangent vector from extrinsic to spherical coordinates.

        At the north pole, where spherical coordinates are singular, the
        first two extrinsic components are returned.

        Parameters
        ----------
        image_tangent_vec : array-like, shape=[..., 3]
            Tangent vector at image point.
        image_point : array-like, shape=[..., 3]
            Point on the sphere, in extrinsic coordinates.
        base_point : array-like, shape=[..., 2]
            Point on the sphere, in spherical coordinates.

        Returns
        -------
        tangent_vec : array-like, shape=[..., 2]
            Tangent vector at base point.
        """
        if base_point is None:
            base_point = self.inverse(image_point)

        theta = base_point[..., 0]
        phi = base_point[..., 1]
        d_x = image_tangent_vec[..., 0]
        d_y = image_tangent_vec[..., 1]
        d_z = image_tangent_vec[..., 2]

        close_to_pole = gs.abs(theta) < gs.atol
        sin_theta_safe = gs.where(close_to_pole, 1.0, gs.sin(theta))

        d_theta = (
            gs.cos(theta) * gs.cos(phi) * d_x
            + gs.cos(theta) * gs.sin(phi) * d_y
            - gs.sin(theta) * d_z
        )
        d_phi = (-gs.sin(phi) * d_x + gs.cos(phi) * d_y) / sin_theta_safe

        d_theta = gs.where(close_to_pole, d_x, d_theta)
        d_phi = gs.where(close_to_pole, d_y, d_phi)
        return gs.stack([d_theta, d_phi], axis=-1)


class Hypersphere(LevelSet):
    """Class for the n-dimensional hypersphere.

    Class for the n-dimensional hypersphere embedded in the
    (n+1)-dimensional Euclidean space.

    Points are parameterized by their extrinsic (n+1)-coordinates. For
    intrinsic coordinates, see `CircleAngles` and
    `SphereSphericalCoordinates`.

    Parameters
    ----------
    dim : int
        Dimension of the hypersphere.
    equip : bool
        If True, equip space with default metric.
    """

    def __init__(self, dim, equip=True):
        self.dim = dim
        super().__init__(
            dim=dim,
            shape=(dim + 1,),
            intrinsic=False,
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
        return scalarvecmul(1.0 / norm, point)

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
        return vector - scalarvecmul(coef, base_point)

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

        See `HypersphereUniform`.

        Parameters
        ----------
        n_samples : int
            Number of samples.

        Returns
        -------
        samples : array-like, shape=[..., dim + 1]
            Points sampled on the hypersphere.
        """
        return HypersphereUniform(self).sample(n_samples)

    def random_von_mises_fisher(self, mu=None, kappa=10, n_samples=1, max_iter=100):
        """Sample with the von Mises-Fisher distribution.

        See `VonMisesFisher`.

        Parameters
        ----------
        mu : array-like, shape=[dim + 1]
            Mean parameter of the distribution.
        kappa : float
            Kappa parameter of the von Mises distribution.
        n_samples : int
            Number of samples.
        max_iter : int
            Maximum number of trials in the rejection algorithm.
            Only used if dim > 2.

        Returns
        -------
        point : array-like, shape=[n_samples, dim + 1]
            Points sampled on the sphere.
        """
        return VonMisesFisher(self, mu=mu, kappa=kappa, max_iter=max_iter).sample(
            n_samples=n_samples
        )

    def random_riemannian_normal(
        self, mean=None, precision=None, n_samples=1, max_iter=100
    ):
        """Sample from the Riemannian normal distribution.

        See `HypersphereRiemannianNormal`.

        Parameters
        ----------
        mean : array-like, shape=[dim + 1]
            Mean parameter of the distribution.
        precision : float or array-like, shape=[dim, dim]
            Inverse of the covariance parameter of the normal distribution.
        n_samples : int
            Number of samples.
        max_iter : int
            Maximum number of trials in the rejection algorithm.

        Returns
        -------
        point : array-like, shape=[n_samples, dim + 1]
            Points sampled on the sphere.
        """
        return HypersphereRiemannianNormal(
            self, mean=mean, precision=precision, max_iter=max_iter
        ).sample(n_samples=n_samples)


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
        proj_tangent_vec = self._space.to_tangent(tangent_vec, base_point)
        norm2 = self._space.embedding_space.metric.squared_norm(proj_tangent_vec)

        coef_1 = utils.taylor_exp_even_func(norm2, utils.cos_close_0, order=4)
        coef_2 = utils.taylor_exp_even_func(norm2, utils.sinc_close_0, order=4)
        return scalarvecmul(coef_1, base_point) + scalarvecmul(
            coef_2, proj_tangent_vec
        )

    def log(self, point, base_point):
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
        if gs.any(gs.isclose(inner_prod, -1.0)):
            raise ValueError(
                "The Logarithm is not well-defined for antipodal points: "
                f"{point} and {base_point}."
            )
        cos_angle = gs.clip(inner_prod, -1.0, 1.0)
        squared_angle = gs.arccos(cos_angle) ** 2
        coef_1_ = utils.taylor_exp_even_func(
            squared_angle, utils.inv_sinc_close_0, order=5
        )
        coef_2_ = utils.taylor_exp_even_func(
            squared_angle, utils.inv_tanc_close_0, order=5
        )
        return scalarvecmul(coef_1_, point) - scalarvecmul(coef_2_, base_point)

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

    def squared_dist(self, point_a, point_b):
        """Squared geodesic distance between two points.

        Parameters
        ----------
        point_a : array-like, shape=[..., dim + 1]
            Point on the hypersphere.
        point_b : array-like, shape=[..., dim + 1]
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
        along the geodesic between two points ``base_point`` and ``end_point``
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
            ``tangent_vec_b`` is given.
            Optional, default : None.

        Returns
        -------
        transported_tangent_vec: array-like, shape=[..., dim + 1]
            Transported tangent vector at
            ``end_point=exp_(base_point)(tangent_vec_b)``.
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
        normalized_b = scalarvecmul(1 / eps, direction)
        pb = gs.dot(tangent_vec, normalized_b)
        p_orth = tangent_vec - scalarvecmul(pb, normalized_b)
        return (
            -scalarvecmul(gs.sin(theta) * pb, base_point)
            + scalarvecmul(gs.cos(theta) * pb, normalized_b)
            + p_orth
        )

    def curvature(self, tangent_vec_a, tangent_vec_b, tangent_vec_c, base_point):
        r"""Compute the curvature.

        For three tangent vectors at a base point :math:`x,y,z`,
        the curvature is defined by
        :math:`R(x, y)z = \nabla_{[x,y]}z
        - \nabla_z\nabla_y z + \nabla_y\nabla_x z`, where :math:`\nabla`
        is the Levi-Civita connection. In the case of the hypersphere,
        we have the closed formula
        :math:`R(x,y)z = \langle y,z \rangle x - \langle x, z \rangle y`.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., dim + 1]
            Tangent vector at ``base_point``.
        tangent_vec_b : array-like, shape=[..., dim + 1]
            Tangent vector at ``base_point``.
        tangent_vec_c : array-like, shape=[..., dim + 1]
            Tangent vector at ``base_point``.
        base_point :  array-like, shape=[..., dim + 1]
            Point on the hypersphere.

        Returns
        -------
        curvature : array-like, shape=[..., dim + 1]
            Tangent vector at ``base_point``.
        """
        if self._space.dim == 1:
            raise NotImplementedError("Curvature is not implemented for the circle.")

        inner_ac = self.inner_product(tangent_vec_a, tangent_vec_c)
        inner_bc = self.inner_product(tangent_vec_b, tangent_vec_c)
        first_term = scalarvecmul(inner_bc, tangent_vec_a)
        second_term = scalarvecmul(inner_ac, tangent_vec_b)
        return first_term - second_term

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
        tangent_vec_a : array-like, shape=[..., dim + 1]
            Tangent vector at `base_point` along which the curvature is
            derived.
        tangent_vec_b : array-like, shape=[..., dim + 1]
            Unused tangent vector at `base_point` (since curvature derivative
            vanishes).
        tangent_vec_c : array-like, shape=[..., dim + 1]
            Unused tangent vector at `base_point` (since curvature derivative
            vanishes).
        tangent_vec_d : array-like, shape=[..., dim + 1]
            Unused tangent vector at `base_point` (since curvature derivative
            vanishes).
        base_point : array-like, shape=[..., dim + 1]
            Unused point on the hypersphere.

        Returns
        -------
        curvature_derivative : array-like, shape=[..., dim + 1]
            Tangent vector at base point.
        """
        curvature_derivative = gs.zeros_like(tangent_vec_a)
        return repeat_out(
            self._space.point_ndim,
            curvature_derivative,
            tangent_vec_a,
            tangent_vec_b,
            tangent_vec_c,
            tangent_vec_d,
            base_point,
            out_shape=self._space.shape,
        )

    def injectivity_radius(self, base_point=None):
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
        radius : array-like, shape=[...,]
            Injectivity radius.
        """
        radius = gs.array(gs.pi)
        return repeat_out(self._space.point_ndim, radius, base_point)


class CircleAngles(Manifold):
    r"""Circle parametrized by angles.

    The circle :math:`\mathbb{R} / 2\pi\mathbb{Z}`, with points represented
    by angles in radians. Angle 0 corresponds to point [1., 0.] of the circle
    in extrinsic coordinates (see `AngleToExtrinsicDiffeo`).

    Parameters
    ----------
    equip : bool
        If True, equip space with default metric.
    """

    def __init__(self, equip=True):
        super().__init__(dim=1, shape=(1,), intrinsic=True, equip=equip)

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return CircleAnglesMetric

    def belongs(self, point, atol=gs.atol):
        """Evaluate if a point belongs to the circle.

        Any angle is a representative of a point of the circle, so this only
        checks the shape of the input.

        Parameters
        ----------
        point : array-like, shape=[..., 1]
            Point to evaluate.
        atol : float
            Unused.

        Returns
        -------
        belongs : array-like, shape=[...,]
            Boolean evaluating if point belongs to the circle.
        """
        batch_shape = get_batch_shape(self.point_ndim, point)
        if point.shape[-self.point_ndim :] == self.shape:
            return gs.ones(batch_shape, dtype=bool)
        return gs.zeros(batch_shape, dtype=bool)

    def is_tangent(self, vector, base_point=None, atol=gs.atol):
        """Check whether the vector is tangent at base_point.

        Parameters
        ----------
        vector : array-like, shape=[..., 1]
            Vector.
        base_point : array-like, shape=[..., 1]
            Point on the circle.
        atol : float
            Unused.

        Returns
        -------
        is_tangent : array-like, shape=[...,]
            Boolean denoting if vector is a tangent vector at the base point.
        """
        batch_shape = get_batch_shape(self.point_ndim, vector, base_point)
        if vector.shape[-self.point_ndim :] == self.shape:
            return gs.ones(batch_shape, dtype=bool)
        return gs.zeros(batch_shape, dtype=bool)

    def to_tangent(self, vector, base_point=None):
        """Project a vector to a tangent space of the circle.

        Any vector is tangent, so the vector itself is returned.

        Parameters
        ----------
        vector : array-like, shape=[..., 1]
            Vector.
        base_point : array-like, shape=[..., 1]
            Point on the circle.

        Returns
        -------
        tangent_vec : array-like, shape=[..., 1]
            Tangent vector at base point.
        """
        return repeat_out(
            self.point_ndim, gs.copy(vector), vector, base_point, out_shape=self.shape
        )

    def regularize(self, point):
        r"""Regularize angles to :math:`(-\pi, \pi]`.

        Parameters
        ----------
        point : array-like, shape=[..., 1]
            Point on the circle.

        Returns
        -------
        regularized_point : array-like, shape=[..., 1]
            Angle in :math:`(-\pi, \pi]`.
        """
        return gs.arctan2(gs.sin(point), gs.cos(point))

    def random_point(self, n_samples=1, bound=1.0):
        r"""Sample angles uniformly in :math:`[-\pi, \pi)`.

        Parameters
        ----------
        n_samples : int
            Number of samples.
        bound : float
            Unused.

        Returns
        -------
        samples : array-like, shape=[..., 1]
            Angles in radians.
        """
        size = (n_samples, 1) if n_samples != 1 else (1,)
        return gs.random.uniform(low=-gs.pi, high=gs.pi, size=size)


class CircleAnglesMetric(RiemannianMetric):
    r"""Flat metric on the circle parametrized by angles.

    Exponential and logarithm are computed in closed form, modulo
    :math:`2\pi`.
    """

    def metric_matrix(self, base_point=None):
        """Metric matrix at the tangent space at a base point.

        Parameters
        ----------
        base_point : array-like, shape=[..., 1]
            Base point.

        Returns
        -------
        mat : array-like, shape=[..., 1, 1]
            Inner-product matrix.
        """
        return repeat_out(
            self._space.point_ndim, gs.ones((1, 1)), base_point, out_shape=(1, 1)
        )

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
        """Inner product between two tangent vectors at a base point.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., 1]
            Tangent vector at base point.
        tangent_vec_b : array-like, shape=[..., 1]
            Tangent vector at base point.
        base_point : array-like, shape=[..., 1]
            Base point.

        Returns
        -------
        inner_product : array-like, shape=[...,]
            Inner-product.
        """
        inner_product = gs.sum(tangent_vec_a * tangent_vec_b, axis=-1)
        return repeat_out(
            self._space.point_ndim,
            inner_product,
            tangent_vec_a,
            tangent_vec_b,
            base_point,
        )

    def exp(self, tangent_vec, base_point):
        r"""Compute the exponential map.

        .. math::

            \exp_\theta(v) = \theta + v \mod 2\pi

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., 1]
            Tangent vector at base point.
        base_point : array-like, shape=[..., 1]
            Point on the circle.

        Returns
        -------
        exp : array-like, shape=[..., 1]
            Point on the circle, in :math:`(-\pi, \pi]`.
        """
        return self._space.regularize(base_point + tangent_vec)

    def log(self, point, base_point):
        r"""Compute the logarithm map.

        The difference of angles, taken in :math:`(-\pi, \pi]`.

        Parameters
        ----------
        point : array-like, shape=[..., 1]
            Point on the circle.
        base_point : array-like, shape=[..., 1]
            Point on the circle.

        Returns
        -------
        log : array-like, shape=[..., 1]
            Tangent vector at base point.
        """
        return self._space.regularize(point - base_point)

    def dist(self, point_a, point_b):
        """Compute the distance between two points.

        Parameters
        ----------
        point_a : array-like, shape=[..., 1]
            Point on the circle.
        point_b : array-like, shape=[..., 1]
            Point on the circle.

        Returns
        -------
        dist : array-like, shape=[...,]
            Distance.
        """
        return gs.abs(self.log(point_b, point_a))[..., 0]

    def parallel_transport(
        self, tangent_vec, base_point, direction=None, end_point=None
    ):
        """Compute the parallel transport of a tangent vector.

        The circle is flat, so the parallel transport is the identity.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., 1]
            Tangent vector at base point to be transported.
        base_point : array-like, shape=[..., 1]
            Point on the circle.
        direction : array-like, shape=[..., 1]
            Tangent vector at base point, along which the parallel transport
            is computed.
        end_point : array-like, shape=[..., 1]
            Point on the circle to transport to.

        Returns
        -------
        transported_tangent_vec : array-like, shape=[..., 1]
            Transported tangent vector.
        """
        return repeat_out(
            self._space.point_ndim,
            gs.copy(tangent_vec),
            tangent_vec,
            base_point,
            direction,
            end_point,
            out_shape=self._space.shape,
        )

    def injectivity_radius(self, base_point=None):
        r"""Compute the radius of the injectivity domain.

        On the circle, it is :math:`\pi` everywhere.

        Parameters
        ----------
        base_point : array-like, shape=[..., 1]
            Point on the circle.

        Returns
        -------
        radius : array-like, shape=[...,]
            Injectivity radius.
        """
        radius = gs.array(gs.pi)
        return repeat_out(self._space.point_ndim, radius, base_point)


class SphereSphericalCoordinates(DiffeomorphicManifold):
    """Sphere parametrized by spherical coordinates.

    Spherical coordinates are defined from the north pole, i.e. angles
    [0., 0.] correspond to point [0., 0., 1.] of the sphere in extrinsic
    coordinates (see `SphericalToExtrinsicDiffeo`).

    Parameters
    ----------
    equip : bool
        If True, equip space with default metric.
    """

    def __init__(self, equip=True):
        super().__init__(
            dim=2,
            shape=(2,),
            diffeo=SphericalToExtrinsicDiffeo(),
            image_space=Hypersphere(2),
            intrinsic=True,
            equip=equip,
        )

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return SphereSphericalCoordinatesMetric

    def random_point(self, n_samples=1, bound=1.0):
        """Sample spherical coordinates uniformly on the sphere.

        Parameters
        ----------
        n_samples : int
            Number of samples.
        bound : float
            Unused.

        Returns
        -------
        samples : array-like, shape=[..., 2]
            Points in spherical coordinates.
        """
        return self.diffeo.inverse(self.image_space.random_point(n_samples))


class SphereSphericalCoordinatesMetric(PullbackDiffeoMetric):
    """Metric on the sphere in spherical coordinates.

    Pulled back from the sphere in extrinsic coordinates.
    """

    def __init__(self, space):
        super().__init__(
            space=space, diffeo=space.diffeo, image_space=space.image_space
        )

    def metric_matrix(self, base_point=None):
        r"""Metric matrix at the tangent space at a base point.

        .. math::

            G(\theta, \phi) = \begin{pmatrix} 1 & 0 \\
            0 & \sin^2 \theta \end{pmatrix}

        Parameters
        ----------
        base_point : array-like, shape=[..., 2]
            Base point.

        Returns
        -------
        mat : array-like, shape=[..., 2, 2]
            Inner-product matrix.
        """
        theta = base_point[..., 0]
        diagonal = gs.stack([gs.ones_like(theta), gs.sin(theta) ** 2], axis=-1)
        return gs.vec_to_diag(diagonal)

    def christoffels(self, base_point):
        """Compute the Christoffel symbols at a point.

        Parameters
        ----------
        base_point : array-like, shape=[..., 2]
            Point in spherical coordinates.

        Returns
        -------
        christoffel : array-like, shape=[..., 2, 2, 2]
            Christoffel symbols at point, with the contravariant index first.
        """
        batch_shape = get_batch_shape(self._space.point_ndim, base_point)

        theta = base_point[..., 0]

        gamma_0 = gs.stack(
            [
                gs.zeros(batch_shape + (2,)),
                gs.stack(
                    [gs.zeros(batch_shape), -gs.sin(theta) * gs.cos(theta)], axis=-1
                ),
            ],
            axis=-1,
        )
        gamma_1 = gs.stack(
            [
                gs.stack(
                    [gs.zeros(batch_shape), gs.cos(theta) / gs.sin(theta)], axis=-1
                ),
                gs.stack(
                    [gs.cos(theta) / gs.sin(theta), gs.zeros(batch_shape)], axis=-1
                ),
            ],
            axis=-1,
        )

        return gs.stack([gamma_0, gamma_1], axis=-3)


class Distribution(abc.ABC):
    """Probability distribution on a manifold.

    Parameters
    ----------
    space : Manifold
        Space the distribution is defined on.
    """

    def __init__(self, space):
        self.space = space

    @abc.abstractmethod
    def sample(self, n_samples=1):
        """Sample from the distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples.

        Returns
        -------
        samples : array-like, shape=[..., *space.shape]
            Points sampled on the space.
        """


class HypersphereUniform(Distribution):
    """Uniform distribution on the hypersphere.

    Samples are normalized standard Gaussian vectors.

    Parameters
    ----------
    space : Hypersphere
        Hypersphere.
    """

    def sample(self, n_samples=1):
        """Sample from the distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples.

        Returns
        -------
        samples : array-like, shape=[..., dim + 1]
            Points sampled on the hypersphere.
        """
        size = (n_samples, self.space.dim + 1)

        samples = gs.random.normal(size=size)
        while True:
            norms = gs.linalg.norm(samples, axis=1)
            mask = gs.isclose(norms, 0.0, atol=gs.atol)
            num_bad_samples = gs.sum(mask)
            if num_bad_samples == 0:
                break

            samples[mask] = gs.random.normal(size=(num_bad_samples, self.space.dim + 1))

        samples = scalarvecmul(1 / norms, samples)
        if n_samples == 1:
            samples = gs.squeeze(samples, axis=0)

        return samples


class VonMisesFisher:
    r"""Von Mises-Fisher distribution on the hypersphere.

    Its density with respect to the uniform measure on the hypersphere is

    .. math::

        f(x; \mu, \kappa) \propto \exp(\kappa \, \mu^T x)

    where :math:`\mu` is the mean direction and :math:`\kappa > 0` the
    concentration: samples concentrate around :math:`\mu` as :math:`\kappa`
    grows, and approach the uniform distribution as it goes to 0. It is the
    maximum entropy distribution on the hypersphere for a given mean.

    Instantiating this class returns the sampler that corresponds to the
    dimension of the space: a closed form on the 2-sphere, and rejection
    sampling [W1994]_ in larger dimensions.

    Parameters
    ----------
    space : Hypersphere
        Hypersphere of dimension at least 2.
    mu : array-like, shape=[dim + 1]
        Mean direction, a unit vector. If None, the first vector of the
        canonical basis is used.
    kappa : float
        Concentration parameter.
    max_iter : int
        Maximum number of trials in the rejection algorithm. In case it is
        reached, fewer samples than requested are returned. Only used if
        dim > 2.

    Raises
    ------
    NotImplementedError
        If the space is the circle.

    References
    ----------
    https://en.wikipedia.org/wiki/Von_Mises-Fisher_distribution

    .. [W1994]   Wood, Andrew T. A. “Simulation of the von Mises Fisher
                  Distribution.” Communications in Statistics - Simulation
                  and Computation, June 27, 2007.
                  https://doi.org/10.1080/03610919408813161.
    """

    def __new__(cls, space, mu=None, kappa=10, max_iter=100):
        """Return the sampler that corresponds to the dimension of the space."""
        if space.dim == 1:
            raise NotImplementedError("Not implemented for dim == 1")

        if space.dim == 2:
            return _SphereVonMisesFisher(space, mu=mu, kappa=kappa)

        return _HypersphereVonMisesFisher(space, mu=mu, kappa=kappa, max_iter=max_iter)


class _VonMisesFisherBase(Distribution):
    """Base class for the von Mises-Fisher distribution.

    Subclasses sample around the first vector of the canonical basis, and
    samples are then rotated to the mean direction.

    Parameters
    ----------
    space : Hypersphere
        Hypersphere.
    mu : array-like, shape=[dim + 1]
        Mean direction, a unit vector. If None, the first vector of the
        canonical basis is used.
    kappa : float
        Concentration parameter.
    """

    def __init__(self, space, mu=None, kappa=10):
        super().__init__(space)
        self.mu = mu
        self.kappa = kappa

    @abc.abstractmethod
    def _sample_at_first_axis(self, n_samples):
        """Sample with mean direction the first vector of the canonical basis.

        Parameters
        ----------
        n_samples : int
            Number of samples.

        Returns
        -------
        sample : array-like, shape=[n_samples, dim + 1]
            Points sampled on the hypersphere.
        """

    def sample(self, n_samples=1):
        """Sample from the distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples.

        Returns
        -------
        point : array-like, shape=[n_samples, dim + 1]
            Points sampled on the sphere.
        """
        sample = self._sample_at_first_axis(n_samples)

        if self.mu is not None:
            sample = utils.rotate_points(sample, self.mu)

        return sample if (n_samples > 1) else sample[0]


class _SphereVonMisesFisher(_VonMisesFisherBase):
    r"""Von Mises-Fisher distribution on the 2-sphere, sampled in closed form.

    On the 2-sphere, the coordinate :math:`w = x^T e_1` along the mean
    direction has density proportional to :math:`e^{\kappa w}` on
    :math:`[-1, 1]`, so it is sampled by inverting its cumulative
    distribution function:

    .. math::

        w = 1 + \frac{1}{\kappa} \log\left(u + (1 - u) e^{-2 \kappa}\right),
        \quad u \sim \mathcal{U}(0, 1)

    The remaining coordinates are :math:`\sqrt{1 - w^2} (\cos \phi,
    \sin \phi)`, with :math:`\phi` uniform in :math:`[0, 2\pi)`.

    Parameters
    ----------
    space : Hypersphere
        Hypersphere of dimension 2.
    mu : array-like, shape=[3]
        Mean direction, a unit vector. If None, the first vector of the
        canonical basis is used.
    kappa : float
        Concentration parameter.
    """

    def _sample_at_first_axis(self, n_samples):
        """Sample with mean direction the first vector of the canonical basis.

        Parameters
        ----------
        n_samples : int
            Number of samples.

        Returns
        -------
        sample : array-like, shape=[n_samples, 3]
            Points sampled on the 2-sphere.
        """
        kappa = self.kappa
        angle = 2.0 * gs.pi * gs.random.rand(n_samples)
        angle = gs.to_ndarray(angle, to_ndim=2, axis=1)

        unit_vector = gs.hstack((gs.cos(angle), gs.sin(angle)))
        scalar = gs.random.rand(n_samples)

        coord_x = 1.0 + 1.0 / kappa * gs.log(
            scalar + (1.0 - scalar) * gs.exp(gs.array(-2.0 * kappa))
        )
        coord_x = gs.to_ndarray(coord_x, to_ndim=2, axis=1)
        coord_yz = gs.sqrt(1.0 - coord_x**2) * unit_vector
        return gs.hstack((coord_x, coord_yz))


class _HypersphereVonMisesFisher(_VonMisesFisherBase):
    r"""Von Mises-Fisher distribution on hyperspheres of dimension larger than 2.

    The coordinate along the mean direction is sampled by rejection sampling,
    with an envelope built from a symmetric Beta distribution [W1994]_. The
    remaining coordinates are a uniform point on the hypersphere of one
    dimension less, scaled by :math:`\sqrt{1 - w^2}`, where :math:`w` is the
    coordinate along the mean direction.

    Parameters
    ----------
    space : Hypersphere
        Hypersphere of dimension larger than 2.
    mu : array-like, shape=[dim + 1]
        Mean direction, a unit vector. If None, the first vector of the
        canonical basis is used.
    kappa : float
        Concentration parameter.
    max_iter : int
        Maximum number of trials in the rejection algorithm. In case it is
        reached, fewer samples than requested are returned.
    """

    def __init__(self, space, mu=None, kappa=10, max_iter=100):
        super().__init__(space, mu=mu, kappa=kappa)
        self.max_iter = max_iter

    def _sample_at_first_axis(self, n_samples):
        """Sample with mean direction the first vector of the canonical basis.

        Parameters
        ----------
        n_samples : int
            Number of samples.

        Returns
        -------
        sample : array-like, shape=[n_accepted, dim + 1]
            Points sampled on the hypersphere. Fewer than `n_samples` if
            the maximum number of iterations is reached.
        """
        dim = self.space.dim
        kappa = self.kappa
        max_iter = self.max_iter

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
            logger.warning(
                "Maximum number of iteration reached in rejection "
                "sampling before n_samples were accepted."
            )

        coord_x = gs.concatenate(result)
        coord_rest = HypersphereUniform(Hypersphere(dim - 1, equip=False)).sample(
            n_accepted
        )
        coord_rest = scalarvecmul(gs.sqrt(1 - coord_x**2), coord_rest)
        sample = gs.concatenate([coord_x[..., None], coord_rest], axis=1)
        return sample


class HypersphereRiemannianNormal(Distribution):
    r"""Riemannian normal distribution on the hypersphere.

    The Riemannian normal distribution, or spherical normal in this case,
    is defined by the probability density function (with respect to the
    Riemannian volume measure) proportional to:

    .. math::
            \exp \Big \left(- \frac{\lambda}{2} \mathtm{arccos}^2(x^T\mu)
            \Big \right)

    where :math:`\mu` is the mean and :math:`\lambda` is the isotropic
    precision. For the anisotropic case,
    :math:`\log_{\mu}(x)^T \Lambda \log_{\mu}(x)` is used instead.

    A rejection algorithm is used to sample from this distribution [H2018]_.

    Parameters
    ----------
    space : Hypersphere
        Hypersphere, equipped with its metric.
    mean : array-like, shape=[dim + 1]
        Mean parameter of the distribution. If None, the north pole
        (0, ..., 0, 1) is used.
    precision : float or array-like, shape=[dim, dim]
        Inverse of the covariance parameter of the normal distribution.
        If a float is passed, the covariance matrix is precision times
        identity. If None, the identity is used.
    max_iter : int
        Maximum number of trials in the rejection algorithm. In case it is
        reached, fewer samples than requested are returned.

    References
    ----------
    .. [H2018]  Hauberg, Soren. “Directional Statistics with the
                Spherical Normal Distribution.”
                In 2018 21st International Conference on Information
                Fusion (FUSION), 704–11, 2018.
                https://doi.org/10.23919/ICIF.2018.8455242.
    """

    def __init__(self, space, mean=None, precision=None, max_iter=100):
        super().__init__(space)
        self.mean = mean
        self.precision = precision
        self.max_iter = max_iter

    def sample(self, n_samples=1):
        """Sample from the distribution.

        Parameters
        ----------
        n_samples : int
            Number of samples.

        Returns
        -------
        point : array-like, shape=[n_samples, dim + 1]
            Points sampled on the sphere.
        """
        dim = self.space.dim
        mean, precision = self.mean, self.precision
        max_iter = self.max_iter

        n_accepted, n_iter = 0, 0
        result = []
        if precision is None:
            precision_ = gs.eye(dim)
        elif isinstance(precision, (float, int)):
            precision_ = precision * gs.eye(dim)
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
            logger.warning(
                "Maximum number of iteration reached in rejection "
                "sampling before n_samples were accepted."
            )
        tangent_sample_intr = gs.concatenate(result)
        tangent_sample = gs.concatenate(
            [tangent_sample_intr, gs.zeros(n_accepted)[:, None]], axis=1
        )

        north_pole = gs.array([0.0] * dim + [1.0])
        if mean is not None:
            mean_from_north = self.space.metric.log(mean, north_pole)
            tangent_sample_at_pt = self.space.metric.parallel_transport(
                tangent_sample, north_pole, mean_from_north
            )
        else:
            tangent_sample_at_pt = tangent_sample
            mean = north_pole

        sample = self.space.metric.exp(tangent_sample_at_pt, mean)
        return sample[0] if (n_samples == 1) else sample
