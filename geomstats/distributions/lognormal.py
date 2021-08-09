"""LogNormal Distribution"""

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean, EuclideanMetric
from geomstats.geometry.matrices import Matrices, MatricesMetric
from geomstats.geometry.spd_matrices import (
    SPDMatrices,
    SPDMetricAffine,
    SPDMetricLogEuclidean,
)


class _LogNormalSPD:
    """LogNormal Distribution on manifold of SPD Matrices"""

    def __init__(self, manifold, mean, cov):
        n = mean.shape[-1]
        metric = manifold.metric
        if metric is None:
            manifold.metric = SPDMetricLogEuclidean(n)
        else:
            if (
                not isinstance(metric, SPDMetricLogEuclidean) and
                not isinstance(metric, SPDMetricAffine)
            ):
                raise ValueError(
                    "Invalid Metric, "
                    "Should be of type SPDMetricLogEuclidean"
                    "or SPDMetricAffine")

        self.manifold = manifold
        self.mean = mean
        self.cov = cov

    def samples_sym(self, mean_vec, cov, n_samples):
        """Generate symmetric matrices."""
        n = self.mean.shape[-1]
        samples_euclidean = gs.random.multivariate_normal(
            mean_vec, cov, (n_samples,))
        diag = samples_euclidean[:, :n]
        off_diag = samples_euclidean[:, n:] / gs.sqrt(2.0)
        samples_sym = gs.mat_from_diag_triu_tril(
            diag=diag, tri_upp=off_diag, tri_low=off_diag)
        return samples_sym

    def sample(self, n_samples):
        """Generate samples for SPD manifold"""
        if isinstance(self.manifold.metric, SPDMetricLogEuclidean):
            sym_matrix = self.manifold.logm(self.mean)
            mean_euclidean = gs.hstack(
                (gs.diagonal(sym_matrix)[None, :],
                 gs.sqrt(2.0) * gs.triu_to_vec(sym_matrix, k=1)[None, :]))[0]
            _samples = self.samples_sym(mean_euclidean, self.cov, n_samples)

        else:
            samples_sym = self.samples_sym(
                gs.zeros(self.manifold.dim), self.cov, n_samples)
            mean_half = self.manifold.powerm(self.mean, 0.5)
            _samples = Matrices.mul(mean_half, samples_sym, mean_half)

        return self.manifold.expm(_samples)


class _LogNormalEuclidean:
    """LogNormal Distribution on Euclidean Space"""

    def __init__(self, manifold, mean, cov):
        n = mean.shape[-1]
        metric = manifold.metric
        if metric is None:
            manifold.metric = EuclideanMetric(n)
        else:
            if type(metric) not in (EuclideanMetric, MatricesMetric):
                raise ValueError(
                    "Invalid Metric, "
                    "Should be of type EuclideanMetric or MatricesMetric")

        self.manifold = manifold
        self.mean = mean
        self.cov = cov

    def sample(self, n_samples):
        """Generate samples for Euclidean Manifold"""
        _samples = gs.random.multivariate_normal(
            self.mean, self.cov, (n_samples,))
        return gs.exp(_samples)


def _sampler_factory(manifold, mean, cov):
    """Factory method for creating sampler"""
    if isinstance(manifold, Euclidean):
        return _LogNormalEuclidean(manifold, mean, cov)
    return _LogNormalSPD(manifold, mean, cov)


class LogNormal:
    """LogNormal Distribution on manifold of SPD Matrices and Euclidean Spaces.

    (1) For Euclidean Spaces, if X is distributed as Normal(mean, cov), then
    exp(X) is distributed as LogNormal(mean, cov).

    (2) For SPDMatrices, there are different distributions based on metric

        a)LogEuclidean Metric : With this metric, LogNormal distribution
        is defined by transforming the mean -- an SPD matrix -- into
        a symmetric matrix through the Matrix logarithm, which gives
        the element "log-mean" that now belongs to a vector space.
        This log-mean and given cov are used to parametrize a Normal
        Distribution.

        b)AffineInvariant Metric : X is distributed as LogNormal(mean, cov)
        if exp(mean^{1/2}.X.mean^{1/2}) is distributed as Normal(0, cov)

    Parameters
    ----------
    manifold : Manifold obj, {Euclidean(n), SPDMatrices(n)}
        Manifold to sample over. Manifold should
        be instance of Euclidean or SPDMatrices.
    mean : array-like,
            shape=[dim] if manifold is Euclidean Space
            shape=[n, n] if manifold is SPD Manifold
        Mean of the distribution.
    cov : array-like,
            shape=[dim, dim] if manifold is Euclidean Space
            shape=[n*(n+1)/2, n*(n+1)/2] if manifold is SPD Manifold
        Covariance of the distribution.


    Example
    --------
    >>> import geomstats.backend as gs
    >>> from geomstats.geometry.spd_matrices import SPDMatrices
    >>> from geomstats.distributions.lognormal import LogNormal
    >>> mean = 2 * gs.eye(3)
    >>> cov  = gs.eye(6)
    >>> SPDManifold = SPDMatrices(3, metric=SPDMetricAffine(3))
    >>> LogNormalSampler = LogNormal(SPDManifold, mean, cov)
    >>> data = LogNormalSampler.sample(5)

    References
    ----------
    .. [LNGASPD2016] A. Schwartzman,
    "LogNormal distributions and"
    "Geometric Averages of Symmetric Positive Definite Matrices.",
    International Statistical Review 84.3 (2016): 456-486.
    """
    def __init__(self, manifold, mean, cov=None):

        if (
            not isinstance(manifold, SPDMatrices) and
            not isinstance(manifold, Euclidean)
        ):
            raise ValueError(
                "Invalid Manifold object, "
                "Should be of type SPDMatrices or Euclidean")

        if not manifold.belongs(mean):
            raise ValueError(
                "Invalid Value in mean, doesn't belong to ",
                type(manifold).__name__)

        if cov is not None:
            valid_cov_shape = (manifold.dim, manifold.dim)
            if (
                cov.ndim != 2 or
                (cov.shape[0], cov.shape[1]) != valid_cov_shape
            ):
                raise ValueError("Invalid Shape, "
                                 "cov should have shape", valid_cov_shape)

        else:
            cov = gs.eye(self.manifold.dim)

        self.__sampler = _sampler_factory(manifold, mean, cov)

    @property
    def manifold(self):
        return self.__sampler.manifold

    @property
    def mean(self):
        return self.__sampler.mean

    @property
    def cov(self):
        return self.__sampler.cov

    def sample(self, n_samples=1):
        """Generate samples

        Parameters
        ----------
        n_samples : int
                Number of samples to be generated

        Returns
        -------
        samples : array-like, shape=[n_samples, dim]
                              if manifold is Euclidean
                              shape=[n_samples, n, n]
                              if manifold is SPDManifold
            Samples from LogNormal distribution.
        """
        return self.__sampler.sample(n_samples)

    def pdf(self, points):
        """Evaluate probability density function for given points"""
        raise NotImplementedError
