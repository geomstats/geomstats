"""Mixin for manifolds of probability distributions."""

from scipy.integrate import quad

import geomstats.backend as gs
import geomstats.errors
from geomstats.geometry.riemannian_metric import RiemannianMetric


class InformationManifold:
    """Mixin for manifolds of probability distributions."""

    def sample(self, point, n_samples=1):
        """Sample from the probability distribution.

        Sample from the probability distribution with parameters provided
        by point. This gives n_samples points.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point representing a probability distribution.
        n_samples : int
            Number of points to sample for each set of parameters in point.
            Optional, default: 1.

        Returns
        -------
        samples : array-like, shape=[..., n_samples]
            Sample from the probability distributions.
        """
        raise NotImplementedError("The sample method is not yet implemented.")

    def _hard_pdf(self, point, x):
        """Hard-coded probability density function.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point representing a probability distribution.
        x : array-like, shape=[...,]
            Point where to evaluate the pdf
        """
        raise NotImplementedError("The pdf has not been hardcoded yet.")

    def point_to_pdf(self, point):
        """Compute pdf associated to point.

        Compute the probability density function of the probability
        distribution with parameters provided by point.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point representing a probability distribution.

        Returns
        -------
        pdf : function
            Probability density function of the probability distribution with
            parameters provided by point.
        """
        geomstats.errors.check_belongs(point, self)
        point = gs.to_ndarray(point, to_ndim=2)

        def pdf(x):
            """Generate parameterized function for pdf.

            Parameters
            ----------
            x : array-like, shape=[n_points, dim]
                Points at which to compute the probability
                density function.

            Returns
            -------
            pdf_at_x : array-like, shape=[..., n_points]
                Values of pdf at x for each value of the parameters provided
                by point.
            """
            raise NotImplementedError("The pdf method has not yet been implemented.")

        return pdf

    def point_to_cdf(self, point):
        """Compute cdf associated to point.

        Compute the cumulative density function of the probability
        distribution with parameters provided by point.

        Parameters
        ----------
        point : array-like, shape=[..., dim]
            Point representing a probability distribution.

        Returns
        -------
        cdf : function
            Cumulative density function of the probability distribution with
            parameters provided by point.
        """
        geomstats.errors.check_belongs(point, self)
        point = gs.to_ndarray(point, to_ndim=2)

        def cdf(x):
            """Generate parameterized function for cdf.

            Parameters
            ----------
            x : array-like, shape=[n_points, dim]
                Points at which to compute the probability
                density function.

            Returns
            -------
            cdf_at_x : array-like, shape=[..., n_points]
                Values of cdf at x for each value of the parameters provided
                by point.
            """
            raise NotImplementedError("The cdf method has not yet been implemented.")

        return cdf


class FisherRaoMetric(RiemannianMetric):
    """Class to derive the information metric from the pdf in InformationManifold."""

    def __init__(self, pdf, support, dim):
        self.pdf = pdf
        self.support = support
        self.dim = dim

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

        def log_pdf_at_x(x):
            return lambda point: gs.log(self.pdf(point, x))

        def sqrtscore(x):
            return gs.autodiff.jacobian(log_pdf_at_x(x))(base_point)

        def score(x):
            return gs.einsum("...i, ...j -> ...ij", sqrtscore(x), sqrtscore(x))

        return gs.array(
            [
                [
                    quad(
                        lambda x: score(x)[i, j] * self.pdf(base_point, x),
                        *self.support
                    )[0]
                    for i in range(self.dim)
                ]
                for j in range(self.dim)
            ]
        )
