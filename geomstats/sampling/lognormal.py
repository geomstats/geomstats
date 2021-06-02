"""LogNormal Sampler on the manifold of SPD Matrices and Euclidean Spaces"""

import geomstats.backend as gs
from geomstats.geometry.euclidean import Euclidean
from geomstats.geometry.spd_matrices import SPDMatrices


class LogNormal:
    """LogNormal Sampler

    Parameters
    ----------
    manifold: Manifold Object
        Manifold over which to sample
    mean: array-like, shape=[..., dim]
        Mean of the Distribution.
    cov: array-like, shape=[..., dim]
        Covariance of the Distribution.
    Returns
    --------
    samples

    Examples
    --------
    import geomstats.backend as gs
    from geomstats.geometry.spd_matrices import SPDMatrices
    from geomstats.sampling.lognormal import LogNormal

    mean = 2*gs.eye(3)
    cov  = gs.eye(6)
    SPDManifold = SPDMatrices(3)
    LogNormalSampler = LogNormal(SPDManifold, mean, cov)
    data = LogNormalSampler.sample(5)

    References
    ----------
    Lognormal Distributions and Geometric Averages of
    Symmetric Positive Definite Matrices
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5222531/
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
        if isinstance(manifold, Euclidean):
            cov_n = n
        else:
            cov_n = (n * (n + 1)) // 2
            
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

        self.manifold = manifold
        self.mean = mean
        self.cov = cov

    #TODO (sait): simplify hstack after pytorch version upgrade
    def _sample_spd(self, samples):
        n = self.mean.shape[-1]
        sym_matrix = self.manifold.logm(self.mean)
        mean_euclidean = gs.hstack(
            (gs.reshape(gs.diagonal(sym_matrix), [1, -1]),
             gs.sqrt(2) * gs.reshape(gs.triu_to_vec(sym_matrix, k = 1), [1, -1])))[0]
        samples_euclidean = gs.random.multivariate_normal(
            mean_euclidean, self.cov, (samples,))
        diag = samples_euclidean[:, :n]
        off_diag = samples_euclidean[:, n:] / gs.sqrt(2)
        samples_sym = gs.mat_from_diag_triu_tril(
            diag=diag, triu=off_diag, tril=off_diag)
        samples_spd = self.manifold.expm(samples_sym)
        return samples_spd

    def _sample_euclidean(self, samples):
        _samples = gs.random.multivariate_normal(
            self.mean, self.cov, (samples,))
        return gs.exp(_samples)

    def sample(self, samples=1):

        if isinstance(self.manifold, Euclidean):
            return self._sample_euclidean(samples)
        return self._sample_spd(samples)
