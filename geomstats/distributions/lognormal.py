"""LogNormal Sampler."""

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.matrices import Matrices
from geomstats.geometry.spd_matrices import (
    SPDMatrices, 
    SPDMetricLogEuclidean,
    SPDMetricAffine,
) 


class LogNormal:
    """LogNormal Sampler on manifold of SPD Matrices and Euclidean Spaces.

    (1) For Euclidean Spaces, if X is distributed as Normal(mean, cov), then
    exp(X) is distributed as LogNormal(mean, cov).

    (2) For SPDMatrices, the LogNormal distribution is defined by transforming
    the mean -- an SPD matrix -- into a symmetric matrix through
    the Matrix logarithm, which gives the element "log-mean" that now belongs
    to a vector space. This log-mean and given cov are used to parametrize
    a Normal Distribution.

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

    Returns
    --------
    samples : array-like, shape=[..., dim]
        Samples from LogNormal distribution.

    Example
    --------
    >>> import geomstats.backend as gs
    >>> from geomstats.geometry.spd_matrices import SPDMatrices
    >>> from geomstats.sampling.lognormal import LogNormal
    >>> mean = 2 * gs.eye(3)
    >>> cov  = gs.eye(6)
    >>> SPDManifold = SPDMatrices(3)
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

        n = mean.shape[-1]
        metric = manifold.metric
        if isinstance(manifold, Euclidean):
            cov_n = n
        else:
            cov_n = (n * (n + 1)) // 2
            if metric is None:
                metric = SPDMetricLogEuclidean(n)
            else:
                if (
                    not isinstance(metric, SPDMetricLogEuclidean) and
                    not isinstance(metric, SPDMetricAffine)
                ):
                    raise ValueError(
                        "Invalid Metric, "
                        "Should be of type SPDMetricLogEuclidean or SPDMetricAffine")


        if cov is not None:
            if (
                cov.ndim != 2 or
                (cov.shape[0], cov.shape[1]) != (cov_n, cov_n)
            ):
                valid_shape = (cov_n, cov_n)
                raise ValueError("Invalid Shape, "
                                 "cov should have shape", valid_shape)

        else:
            cov = gs.eye(cov_n)

        self.__manifold = manifold
        self.__mean = mean
        self.__cov = cov
        self.__mean_dim = n
        self.__cov_dim = cov_n 
        
    @property
    def manifold(self):
        return self.__manifold

    @property
    def mean(self):
        return self.__mean

    @property
    def cov(self):
        return self.__cov
    
    @property
    def mean_dim(self):
        return self.__mean_dim
    
    @property
    def cov_dim(self):
        return self.__cov_dim
    
    def __sample_spd(self, n_samples):
        def samples_sym(mean, cov, n_samples):
            samples_euclidean = gs.random.multivariate_normal(
            mean, cov, (n_samples,))
            diag = samples_euclidean[:, :self.mean_dim]
            off_diag = samples_euclidean[:, self.mean_dim:]/gs.sqrt(2.0)
            samples_sym = gs.mat_from_diag_triu_tril(
            diag=diag, tri_upp=off_diag, tri_low=off_diag)
            return samples_sym
        
        if isinstance(self.manifold.metric, SPDMetricLogEuclidean):
            sym_matrix = self.manifold.logm(self.mean)
            mean_euclidean = gs.hstack(
                (gs.diagonal(sym_matrix)[None, :],
                gs.sqrt(2.0) * gs.triu_to_vec(sym_matrix, k=1)[None, :]))[0]
            _samples = samples_sym(mean_euclidean, self.cov, n_samples)

        else:
            samples_sym = samples_sym(gs.zero(self.cov_n), self.cov, n_samples)
            mean_half = self.manifold.powerm(self.mean, 0.5)
            _samples = Matrices.mul(mean_half, samples_sym, mean_half)

        return self.manifold.expm(_samples)
        
    def __sample_euclidean(self, samples):
        _samples = gs.random.multivariate_normal(
            self.mean, self.cov, (samples,))
        return gs.exp(_samples)

    def sample(self, samples=1):
        if isinstance(self.manifold, Euclidean):
            return self.__sample_euclidean(samples)
        return self.__sample_spd(samples)
    