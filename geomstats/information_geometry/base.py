"""Mixin for manifolds of probability distributions."""


class InformationManifoldMixin:
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
        raise NotImplementedError("`point_to_pdf` has not yet been implemented.")

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
        raise NotImplementedError("`point_to_cdf` has not yet been implemented.")
