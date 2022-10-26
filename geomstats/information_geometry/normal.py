"""Statistical Manifold of normal distributions with the Fisher metric.

Lead author: Alice Le Brigant.
"""

from scipy.stats import norm

import geomstats.backend as gs
import geomstats.errors
from geomstats.geometry.poincare_half_space import (
    PoincareHalfSpace,
    PoincareHalfSpaceMetric,
)
from geomstats.geometry.pullback_metric import PullbackDiffeoMetric
from geomstats.information_geometry.base import InformationManifoldMixin


class NormalDistributions(InformationManifoldMixin, PoincareHalfSpace):
    """Class for the manifold of univariate normal distributions.

    This is upper half-plane.
    """

    def __init__(self):
        super().__init__(dim=2)
        self.metric = NormalMetric()

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
            Point representing a normal distribution (location and scale).
        n_samples : int
            Number of points to sample with each pair of parameters in point.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[..., n_samples]
            Sample from normal distributions.
        """
        geomstats.errors.check_belongs(point, self)
        point = gs.to_ndarray(point, to_ndim=2)
        samples = []
        for loc, scale in point:
            samples.append(gs.array(norm.rvs(loc, scale, size=n_samples)))
        return samples[0] if len(point) == 1 else gs.stack(samples)

    def point_to_pdf(self, point):
        """Compute pdf associated to point.

        Compute the probability density function of the normal
        distribution with parameters provided by point.

        Parameters
        ----------
        point : array-like, shape=[..., 2]
            Point representing a normal distribution (location and scale).

        Returns
        -------
        pdf : function
            Probability density function of the normal distribution with
            parameters provided by point.
        """
        geomstats.errors.check_belongs(point, self)
        means = point[..., 0]
        stds = point[..., 1]
        means = gs.to_ndarray(means, to_ndim=2)
        stds = gs.to_ndarray(stds, to_ndim=2)

        def pdf(x):
            """Generate parameterized function for normal pdf.

            Parameters
            ----------
            x : array-like, shape=[n_points,]
                Points at which to compute the probability density function.
            """
            x = gs.to_ndarray(x, to_ndim=2, axis=-1)
            return (1.0 / gs.sqrt(2 * gs.pi * stds**2)) * gs.exp(
                -((x - means) ** 2) / (2 * stds**2)
            )

        return pdf


class NormalMetric(PullbackDiffeoMetric):
    """Class for the Fisher information metric on normal distributions.

    This is the pullback of the metric of the Poincare upper half-plane
    by the diffeomorphism :math:`(mean, std) -> (mean, sqrt{2} std)`.
    """

    def __init__(self):
        super().__init__(dim=2)

    def define_embedding_metric(self):
        r"""Define the metric to pull back.

        This is the metric of the Poincare upper half-plane
        with a scaling factor of 2.

        Returns
        -------
        embedding_metric : RiemannianMetric object
            The metric of the Poincare upper half-plane.
        """
        return PoincareHalfSpaceMetric(dim=2, scale=2)

    def diffeomorphism(self, base_point):
        r"""Image of base point in the Poincare upper half-plane.

        This is the image by the diffeomorphism
        :math:`(mean, std) -> (mean, sqrt{2} std)`.

        Parameters
        ----------
        base_point : array-like, shape=[..., 2]
            Point representing a normal distribution. Coordinates
            are mean and standard deviation.

        Returns
        -------
        image_point : array-like, shape=[..., 2]
            Image of base_point in the Poincare upper half-plane.
        """
        image_point = gs.copy(base_point)
        image_point[..., 0] /= gs.sqrt(2.0)
        return image_point

    def inverse_diffeomorphism(self, image_point):
        r"""Inverse image of a point in the Poincare upper half-plane.

        This is the inverse image by the diffeomorphism
        :math:`(mean, std) -> (mean, sqrt{2} std)`.

        Parameters
        ----------
        image_point : array-like, shape=[..., 2]
            Point in the upper half-plane.

        Returns
        -------
        base_point : array-like, shape=[..., 2]
            Inverse image of the image point, representing a normal
            distribution. Coordinates are mean and standard deviation.
        """
        base_point = gs.copy(image_point)
        base_point[..., 0] *= gs.sqrt(2.0)
        return base_point

    def tangent_diffeomorphism(self, tangent_vec, base_point):
        r"""Image of tangent vector.

        This is the image by the tangent map of the diffeomorphism
        :math:`(mean, std) -> (mean, sqrt{2} std)`.

        Parameters
        ----------
        tangent_vec : array-like, shape=[..., 2]
            Tangent vector at base point.

        base_point : array-like, shape=[..., 2]
            Base point representing a normal distribution.

        Returns
        -------
        image_tangent_vec : array-like, shape=[..., 2]
            Image tangent vector at image of the base point.
        """
        return self.diffeomorphism(tangent_vec)

    def inverse_tangent_diffeomorphism(self, image_tangent_vec, image_point):
        r"""Inverse image of tangent vector.

        This is the inverse image by the tangent map of the diffeomorphism
        :math:`(mean, std) -> (mean, sqrt{2} std)`.

        Parameters
        ----------
        image_tangent_vec : array-like, shape=[..., 2]
            Image of a tangent vector at image_point.

        image_point : array-like, shape=[..., 2]
            Image of a point representing a normal distribution.

        Returns
        -------
        tangent_vec : array-like, shape=[..., 2]
            Inverse image of image_tangent_vec.
        """
        return self.inverse_diffeomorphism(image_tangent_vec)

    @staticmethod
    def metric_matrix(base_point=None):
        """Compute the metric matrix at the tangent space at base_point.

        Parameters
        ----------
        base_point : array-like, shape=[..., 2]
            Point representing a normal distribution (location and scale).

        Returns
        -------
        mat : array-like, shape=[..., 2, 2]
            Metric matrix.
        """
        stds = base_point[..., 1]
        stds = gs.to_ndarray(stds, to_ndim=1)
        metric_mat = gs.stack(
            [gs.array([[1.0 / std**2, 0.0], [0.0, 2.0 / std**2]]) for std in stds],
            axis=0,
        )

        if metric_mat.ndim == 3 and metric_mat.shape[0] == 1:
            return metric_mat[0]
        return metric_mat

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
        sectional_curv = -0.5
        if (
            tangent_vec_a.ndim == 1
            and tangent_vec_b.ndim == 1
            and (base_point is None or base_point.ndim == 1)
        ):
            return gs.array(sectional_curv)

        n_sec_curv = []
        if base_point is not None and base_point.ndim == 2:
            n_sec_curv.append(base_point.shape[0])
        if tangent_vec_a.ndim == 2:
            n_sec_curv.append(tangent_vec_a.shape[0])
        if tangent_vec_b.ndim == 2:
            n_sec_curv.append(tangent_vec_b.shape[0])
        n_sec_curv = max(n_sec_curv)

        return gs.tile(sectional_curv, (n_sec_curv,))
