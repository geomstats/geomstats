"""Statistical Manifold of beta distributions with the Fisher metric.

Lead author: Alice Le Brigant.
"""

from scipy.stats import beta

import geomstats.backend as gs
from geomstats.geometry.symmetric_matrices import SymmetricMatrices
from geomstats.information_geometry.base import ScipyUnivariateRandomVariable
from geomstats.information_geometry.dirichlet import (
    DirichletDistributions,
    DirichletMetric,
)


class BetaDistributions(DirichletDistributions):
    r"""Class for the manifold of beta distributions.

    This is Beta = :math:`R_+^* \times R_+^*`, the upper-right
    quadrant of the Euclidean plane.

    Attributes
    ----------
    dim : int
        Dimension of the manifold of beta distributions, equal to 2.
    embedding_space : Manifold
        Embedding manifold.
    """

    def __init__(self, equip=True):
        super().__init__(dim=2, equip=equip)
        self.support_shape = ()
        self._scp_rv = BetaDistributionsRandomVariable(self)

    @staticmethod
    def default_metric():
        """Metric to equip the space with if equip is True."""
        return BetaMetric

    def sample(self, point, n_samples=1):
        """Sample from the beta distribution.

        Sample from the beta distribution with parameters provided by point.
        This gives samples in the segment [0, 1].

        Parameters
        ----------
        point : array-like, shape=[..., 2]
            Point representing a beta distribution.
        n_samples : int
            Number of points to sample with each pair of parameters in point.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[..., n_samples]
            Sample from beta distributions.)
        """
        return self._scp_rv.rvs(point, n_samples)

    def point_to_pdf(self, point):
        """Compute pdf associated to point.

        Compute the probability density function of the beta
        distribution with parameters provided by point.

        Parameters
        ----------
        point : array-like, shape=[..., 2]
            Point representing a beta distribution.

        Returns
        -------
        pdf : function
            Probability density function of the beta distribution with
            parameters provided by point.
        """
        alpha = gs.expand_dims(point[..., 0], axis=-1)
        beta = gs.expand_dims(point[..., 1], axis=-1)

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
            return (
                x ** (alpha - 1)
                * (1 - x) ** (beta - 1)
                / (gs.gamma(alpha) * gs.gamma(beta) / gs.gamma(alpha + beta))
            )

        return pdf

    @staticmethod
    def maximum_likelihood_fit(data, loc=0, scale=1, epsilon=1e-6):
        """Estimate parameters from samples.

        This a wrapper around scipy's maximum likelihood estimator to
        estimate the parameters of a beta distribution from samples.

        Parameters
        ----------
        data : array-like, shape=[..., n_samples]
            Data to estimate parameters from. Arrays of
            different length may be passed.
        loc : float
            Location parameter of the distribution to estimate parameters
            from. It is kept fixed during optimization.
            Optional, default: 0.
        scale : float
            Scale parameter of the distribution to estimate parameters
            from. It is kept fixed during optimization.
            Optional, default: 1.

        Returns
        -------
        parameter : array-like, shape=[..., 2]
            Estimate of parameter obtained by maximum likelihood.
        """
        data = gs.where(data == 1.0, 1.0 - epsilon, data)
        data = gs.where(data == 0.0, epsilon, data)
        data = gs.to_ndarray(data, to_ndim=2)
        parameters = []
        for sample in data:
            param_a, param_b, _, _ = beta.fit(sample, floc=loc, fscale=scale)
            parameters.append(gs.array([param_a, param_b]))
        return parameters[0] if len(data) == 1 else gs.stack(parameters)


class BetaMetric(DirichletMetric):
    """Class for the Fisher information metric on beta distributions."""

    @staticmethod
    def metric_det(point):
        """Compute the determinant of the metric.

        Parameters
        ----------
        point : array-like, shape=[..., 2]
            Point representing a beta distribution.


        Returns
        -------
        metric_det : array-like, shape=[...,]
            Determinant of the metric.
        """
        param_a = point[..., 0]
        param_b = point[..., 1]
        metric_det = gs.polygamma(1, param_a) * gs.polygamma(1, param_b) - gs.polygamma(
            1, param_a + param_b
        ) * (gs.polygamma(1, param_a) + gs.polygamma(1, param_b))
        return metric_det

    def metric_matrix(self, base_point):
        """Compute the inner-product matrix.

        Compute the inner-product matrix of the Fisher information metric
        at the tangent space at base point.

        Parameters
        ----------
        base_point : array-like, shape=[..., 2]
            Base point.

        Returns
        -------
        mat : array-like, shape=[..., 2, 2]
            Inner-product matrix.
        """
        param_a = base_point[..., 0]
        param_b = base_point[..., 1]
        trigamma_sum = gs.polygamma(1, param_a + param_b)
        vector = gs.stack(
            [
                gs.polygamma(1, param_a) - trigamma_sum,
                -trigamma_sum,
                gs.polygamma(1, param_b) - trigamma_sum,
            ],
            axis=-1,
        )
        return SymmetricMatrices.matrix_representation(vector)

    def christoffels(self, base_point):
        """Compute the Christoffel symbols.

        Compute the Christoffel symbols of the Fisher information metric.

        Parameters
        ----------
        base_point : array-like, shape=[..., 2]
            Base point.

        Returns
        -------
        christoffels : array-like, shape=[..., 2, 2, 2]
            Christoffel symbols, with the contravariant index on
            the first dimension.
            :math:`christoffels[..., i, j, k] = Gamma^i_{jk}`
        """

        def _coefficients(param_a, param_b):
            trigamma_a = gs.polygamma(1, param_a)
            tetragamma_a = gs.polygamma(2, param_a)
            trigamma_b = gs.polygamma(1, param_b)
            tetragamma_b = gs.polygamma(2, param_b)
            trigamma_ab = gs.polygamma(1, param_a + param_b)
            tetragamma_ab = gs.polygamma(2, param_a + param_b)
            denominator = 2 * (
                trigamma_a * trigamma_b - trigamma_ab * (trigamma_a + trigamma_b)
            )

            coef_1 = (
                tetragamma_a * (trigamma_b - trigamma_ab) - trigamma_b * tetragamma_ab
            ) / denominator
            coef_2 = -trigamma_b * tetragamma_ab / denominator
            coef_3 = (
                tetragamma_b * trigamma_ab - trigamma_b * tetragamma_ab
            ) / denominator
            return coef_1, coef_2, coef_3

        param_a, param_b = base_point[..., 0], base_point[..., 1]
        coef_1, coef_2, coef_3 = _coefficients(param_a, param_b)
        coef_4, coef_5, coef_6 = _coefficients(param_b, param_a)
        christoffels_a = SymmetricMatrices.matrix_representation(
            gs.stack([coef_1, coef_2, coef_3], axis=-1)
        )
        christoffels_b = SymmetricMatrices.matrix_representation(
            gs.stack([coef_6, coef_5, coef_4], axis=-1)
        )
        return gs.stack([christoffels_a, christoffels_b], axis=-3)

    def sectional_curvature(self, tangent_vec_a, tangent_vec_b, base_point=None):
        """Compute the sectional curvature.

        The space is two-dimensional, so the sectional curvature only
        depends on the base point.

        Parameters
        ----------
        tangent_vec_a : array-like, shape=[..., 2]
            Tangent vector at `base_point`.
        tangent_vec_b : array-like, shape=[..., 2]
            Tangent vector at `base_point`.
        base_point : array-like, shape=[..., 2]
            Base point.

        Returns
        -------
        sectional_curvature : array-like, shape=[...,]
            Sectional curvature at `base_point`.
        """
        param_a, param_b = base_point[..., 0], base_point[..., 1]
        trigamma_a = gs.polygamma(1, param_a)
        trigamma_b = gs.polygamma(1, param_b)
        trigamma_ab = gs.polygamma(1, param_a + param_b)
        tetragamma_a = gs.polygamma(2, param_a)
        tetragamma_b = gs.polygamma(2, param_b)
        tetragamma_ab = gs.polygamma(2, param_a + param_b)

        curvature = (
            tetragamma_a
            * tetragamma_b
            * tetragamma_ab
            * (
                trigamma_a / tetragamma_a
                + trigamma_b / tetragamma_b
                - trigamma_ab / tetragamma_ab
            )
            / (4 * self.metric_det(base_point) ** 2)
        )
        batch_zeros = gs.zeros_like(tangent_vec_a[..., 0] + tangent_vec_b[..., 0])
        return curvature + batch_zeros


class BetaDistributionsRandomVariable(ScipyUnivariateRandomVariable):
    """A beta random variable."""

    def __init__(self, space):
        super().__init__(space, beta.rvs, beta.pdf)

    @staticmethod
    def _flatten_params(point, pre_flat_shape):
        param_a = gs.expand_dims(point[..., 0], axis=-1)
        param_b = gs.expand_dims(point[..., 1], axis=-1)

        flat_param_a = gs.reshape(gs.broadcast_to(param_a, pre_flat_shape), (-1,))
        flat_param_b = gs.reshape(gs.broadcast_to(param_b, pre_flat_shape), (-1,))
        return {"a": flat_param_a, "b": flat_param_b}
