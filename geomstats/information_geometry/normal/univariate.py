"""Information manifold of univariate normal distributions.

Lead authors: Antoine Collas, Alice Le Brigant.
"""

import math

from scipy.stats import norm

import geomstats.backend as gs
from geomstats._einsum import scalarmatmul
from geomstats.geometry.euclidean import CoordinateScalingDiffeo
from geomstats.geometry.poincare_half_space import PoincareHalfSpace
from geomstats.geometry.pullback_metric import PullbackDiffeoMetric
from geomstats.geometry.scalar_product_metric import ScalarProductMetric
from geomstats.information_geometry.base import (
    InformationManifoldMixin,
    ScipyUnivariateRandomVariable,
)
from geomstats.vectorization import repeat_out


class UnivariateNormalDistributions(InformationManifoldMixin, PoincareHalfSpace):
    """Class for the manifold of univariate normal distributions.

    This is upper half-plane.
    """

    def __init__(self, equip=True):
        super().__init__(dim=2, support_shape=(), equip=equip)
        self._scp_rv = UnivariateNormalDistributionsRandomVariable(self)

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return UnivariateNormalMetric

    @staticmethod
    def random_point(n_samples=1, bound=1.0):
        """Sample parameters of normal distributions.

        The uniform distribution on [-bound/2, bound/2]x[0, bound] is used.

        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        bound : float
            Side of the square where the normal parameters are sampled.
            Optional, default: 5.

        Returns
        -------
        samples : array-like, shape=[..., 2]
            Sample of points representing normal distributions.
        """
        means = -bound + 2 * bound * gs.random.rand(n_samples)
        stds = bound * gs.random.rand(n_samples)
        if n_samples == 1:
            return gs.array((means[0], stds[0]))
        return gs.transpose(gs.stack((means, stds)))

    def sample(self, point, n_samples=1):
        """Sample from the normal distribution.

        Sample from the normal distribution with parameters provided
        by point.

        Parameters
        ----------
        point : array-like, shape=[..., 2]
            Point representing a normal distribution (mean and scale).
        n_samples : int
            Number of points to sample with each pair of parameters in point.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[..., n_samples]
            Sample from normal distributions.
        """
        return self._scp_rv.rvs(point, n_samples)

    def point_to_pdf(self, point):
        """Compute pdf associated to point.

        Compute the probability density function of the normal
        distribution with parameters provided by point.

        Parameters
        ----------
        point : array-like, shape=[..., 2]
            Point representing a normal distribution (mean and scale).

        Returns
        -------
        pdf : function
            Probability density function of the normal distribution with
            parameters provided by point.
        """
        means = gs.expand_dims(point[..., 0], axis=-1)
        stds = gs.expand_dims(point[..., 1], axis=-1)
        pdf_normalization = 1.0 / gs.sqrt(2 * gs.pi * stds**2)

        def pdf(x):
            """Generate parameterized function for normal pdf.

            Parameters
            ----------
            x : array-like, shape=[n_samples,]
                Points at which to compute the probability density function.

            Returns
            -------
            pdf_at_x : array-like, shape=[..., n_samples]
                Values of pdf at x for each value of the parameters provided
                by point.
            """
            x = gs.reshape(gs.array(x), (-1,))
            return pdf_normalization * gs.exp(-((x - means) ** 2) / (2 * stds**2))

        return pdf


class UnivariateNormalMetric(PullbackDiffeoMetric):
    r"""Class for the Fisher information metric on univariate normal distributions.

    This is the pullback of twice the metric of the Poincare upper half-plane
    by the diffeomorphism :math:`(mean, std) -> (mean / \sqrt{2}, std)`.
    """

    def __init__(self, space):
        super().__init__(
            space,
            diffeo=CoordinateScalingDiffeo(gs.array([1 / math.sqrt(2.0), 1.0])),
            image_space=PoincareHalfSpace(dim=2).equip_with_metric(
                ScalarProductMetric, scale=2.0
            ),
        )

    @staticmethod
    def metric_matrix(base_point):
        """Compute the metric matrix at the tangent space at base_point.

        Parameters
        ----------
        base_point : array-like, shape=[..., 2]
            Point representing a normal distribution (mean and scale).

        Returns
        -------
        mat : array-like, shape=[..., 2, 2]
            Metric matrix.
        """
        stds = base_point[..., 1]
        const = 1 / stds**2
        mat = gs.array([[1.0, 0], [0, 2]])
        return scalarmatmul(const, mat)

    def sectional_curvature(self, tangent_vec_a, tangent_vec_b, base_point=None):
        r"""Compute the sectional curvature.

        In the literature sectional curvature is noted K.

        For two orthonormal tangent vectors :math:`x,y` at a base point,
        the sectional curvature is defined by :math:`K(x,y) = <R(x, y)x, y>`.

        For non-orthonormal vectors, it is
        :math:`K(x,y) = <R(x, y)y, x> / (<x, x><y, y> - <x, y>^2)`.

        sectional_curvature(X, Y, P) = K(X,Y) where X, Y are tangent vectors
        at base point P.

        The information manifold of univariate normal distributions has constant
        sectional curvature given by :math:`K = - 1/2`.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., 2]
            Tangent vector at `base_point`.
        tangent_vec_b : array-like, shape=[..., 2]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., 2]
            Point in the manifold.

        Returns
        -------
        sectional_curvature : array-like, shape=[...,]
            Sectional curvature at `base_point`.
        """
        sectional_curv = gs.array(-0.5)
        return repeat_out(
            self._space.point_ndim,
            sectional_curv,
            tangent_vec_a,
            tangent_vec_b,
            base_point,
        )


class UnivariateNormalDistributionsRandomVariable(ScipyUnivariateRandomVariable):
    """A univariate normal random variable."""

    def __init__(self, space):
        super().__init__(space, norm.rvs, norm.pdf)

    @staticmethod
    def _flatten_params(point, pre_flat_shape):
        loc = gs.expand_dims(point[..., 0], axis=-1)
        scale = gs.expand_dims(point[..., 1], axis=-1)

        flat_loc = gs.reshape(gs.broadcast_to(loc, pre_flat_shape), (-1,))
        flat_scale = gs.reshape(gs.broadcast_to(scale, pre_flat_shape), (-1,))
        return {"loc": flat_loc, "scale": flat_scale}
