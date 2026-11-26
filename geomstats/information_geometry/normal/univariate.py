"""Information manifold of univariate normal distributions.

Lead authors: Antoine Collas, Alice Le Brigant.
"""

import math

from scipy.stats import norm

import geomstats.backend as gs
from geomstats._einsum import scalarmatmul
from geomstats.geometry.euclidean import CoordinateScalingDiffeo
from geomstats.geometry.poincare_half_space import PoincareHalfSpace
from geomstats.geometry.pullback_metric import CoordinatePullbackDiffeoMetric
from geomstats.geometry.scalar_product_metric import ScalarProductMetric
from geomstats.information_geometry.base import (
    InformationManifoldMixin,
    ScipyUnivariateRandomVariable,
)
from geomstats.vectorization import repeat_out


class UnivariateNormalDistributions(InformationManifoldMixin, PoincareHalfSpace):
    """Class for the manifold of univariate normal distributions.

    This is the upper half-plane.
    """

    def __init__(self, equip=True):
        super().__init__(dim=2, support_shape=(), equip=equip)
        self._scp_rv = UnivariateNormalDistributionsRandomVariable(self)

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return UnivariateNormalMetric

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


class UnivariateNormalMetric(CoordinatePullbackDiffeoMetric):
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

    @staticmethod
    def inner_product_derivative_matrix(base_point):
        r"""Compute derivative of the inner prod matrix at base point.

        The only nonzero entries are :math:`\partial_{\sigma} g_{\mu\mu}
        = -2 / \sigma^3` and :math:`\partial_{\sigma} g_{\sigma\sigma}
        = -4 / \sigma^3`.

        Parameters
        ----------
        base_point : array-like, shape=[..., 2]
            Point representing a normal distribution (mean and scale).

        Returns
        -------
        metric_derivative : array-like, shape=[..., 2, 2, 2]
            Derivative of the inner-product matrix, where the index
            k of the derivation is last: :math:`mat_{ijk} = \partial_k g_{ij}`.
        """
        mat = gs.array([[1.0, 0.0], [0.0, 2.0]])
        return gs.einsum(
            "...,ij,k->...ijk",
            -2 / base_point[..., 1] ** 3,
            mat,
            gs.array([0.0, 1.0]),
        )

    @staticmethod
    def _christoffels_pattern():
        r"""Compute the Christoffel symbols at :math:`\sigma = 1`.

        They are :math:`\Gamma^{\mu}_{\mu\sigma} = -1`,
        :math:`\Gamma^{\sigma}_{\mu\mu} = 1 / 2` and
        :math:`\Gamma^{\sigma}_{\sigma\sigma} = -1`.

        Returns
        -------
        pattern : array-like, shape=[2, 2, 2]
            Pattern, with the contravariant index on the first dimension.
        """
        return gs.array([[[0.0, -1.0], [-1.0, 0.0]], [[0.5, 0.0], [0.0, -1.0]]])

    def christoffels(self, base_point):
        r"""Compute the Christoffel symbols.

        The nonzero symbols are :math:`\Gamma^{\mu}_{\mu\sigma} = -1 / \sigma`,
        :math:`\Gamma^{\sigma}_{\mu\mu} = 1 / (2 \sigma)` and
        :math:`\Gamma^{\sigma}_{\sigma\sigma} = -1 / \sigma`.

        Parameters
        ----------
        base_point : array-like, shape=[..., 2]
            Point representing a normal distribution (mean and scale).

        Returns
        -------
        christoffels : array-like, shape=[..., 2, 2, 2]
            Christoffel symbols, with the contravariant index on
            the first dimension.
        """
        return gs.einsum(
            "...,kij->...kij", 1 / base_point[..., 1], self._christoffels_pattern()
        )

    def jacobian_christoffels(self, base_point):
        r"""Compute the Jacobian of the Christoffel symbols.

        The symbols only depend on :math:`\sigma`, as :math:`1 / \sigma`.

        Parameters
        ----------
        base_point : array-like, shape=[..., 2]
            Point representing a normal distribution (mean and scale).

        Returns
        -------
        jac : array-like, shape=[..., 2, 2, 2, 2]
            Jacobian of the Christoffel symbols.
            :math:`jac[..., i, j, k, l] = dGamma^i_{jk} / dx_l`
        """
        return gs.einsum(
            "...,kij,l->...kijl",
            -1 / base_point[..., 1] ** 2,
            self._christoffels_pattern(),
            gs.array([0.0, 1.0]),
        )

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

    def scalar_curvature(self, base_point=None):
        """Compute the scalar curvature.

        It is twice the constant sectional curvature, i.e. :math:`-1`.

        Parameters
        ----------
        base_point : array-like, shape=[..., 2]
            Point representing a normal distribution (mean and scale).

        Returns
        -------
        scalar_curvature : array-like, shape=[...,]
            Scalar curvature at `base_point`.
        """
        return repeat_out(self._space.point_ndim, gs.array(-1.0), base_point)


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
