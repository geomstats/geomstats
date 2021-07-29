"""Statistical Manifold of beta distributions with the Fisher metric."""

from scipy.stats import beta

import geomstats.backend as gs
import geomstats.errors
from geomstats.information_geometry.dirichlet import DirichletDistributions
from geomstats.information_geometry.dirichlet import DirichletMetric

EPSILON = 1e-6


class BetaDistributions(DirichletDistributions):
    r"""Class for the manifold of beta distributions.

    This is :math: Beta = `R_+^* \times R_+^*`, the upper-right
    quadrant of the Euclidean plane.

    Attributes
    ----------
    dim : int
        Dimension of the manifold of beta distributions, equal to 2.
    embedding_space : Manifold
        Embedding manifold.
    """

    def __init__(self):
        super(BetaDistributions, self).__init__(
            dim=2)
        self.metric = BetaMetric()

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
            Sample from beta distributions.
        """
        geomstats.errors.check_belongs(point, self)
        point = gs.to_ndarray(point, to_ndim=2)
        samples = []
        for param_a, param_b in point:
            samples.append(gs.array(
                beta.rvs(param_a, param_b, size=n_samples)))
        return samples[0] if len(point) == 1 else gs.stack(samples)

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
        geomstats.errors.check_belongs(point, self)
        a_params = point[..., 0]
        b_params = point[..., 1]

        def pdf(x):
            """Generate parameterized function for normal pdf.

            Parameters
            ----------
            x : array-like, shape=[n_points,]
                Points at which to compute the probability density function.
            """
            x = gs.array(x, gs.float32)
            x = gs.to_ndarray(x, to_ndim=1)

            pdf_at_x = [
                gs.array(beta.pdf(x, a=a, b=b)) for a, b
                in zip(a_params, b_params)]
            pdf_at_x = gs.stack(pdf_at_x, axis=-1)

            return pdf_at_x
        return pdf

    @staticmethod
    def maximum_likelihood_fit(data, loc=0, scale=1):
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
        data = gs.cast(data, gs.float32)
        data = gs.where(data == 1., 1. - EPSILON, data)
        data = gs.where(data == 0., EPSILON, data)
        data = gs.to_ndarray(data, to_ndim=2)
        parameters = []
        for sample in data:
            param_a, param_b, _, _ = beta.fit(sample, floc=loc, fscale=scale)
            parameters.append(gs.array([param_a, param_b]))
        return parameters[0] if len(data) == 1 else gs.stack(parameters)


class BetaMetric(DirichletMetric):
    """Class for the Fisher information metric on beta distributions."""

    def __init__(self):
        super(BetaMetric, self).__init__(dim=2)

    @staticmethod
    def metric_det(param_a, param_b):
        """Compute the determinant of the metric.

        Parameters
        ----------
        param_a : array-like, shape=[...,]
            First parameter of the beta distribution.
        param_b : array-like, shape=[...,]
            Second parameter of the beta distribution.

        Returns
        -------
        metric_det : array-like, shape=[...,]
            Determinant of the metric.
        """
        metric_det = gs.polygamma(1, param_a) * gs.polygamma(1, param_b) - \
            gs.polygamma(1, param_a + param_b) * (gs.polygamma(1, param_a) +
                                                  gs.polygamma(1, param_b))
        return metric_det
