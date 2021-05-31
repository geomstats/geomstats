"""LogNormal Sampler on the manifold of SPD Matrices and Euclidean Spaces"""

import geomstats.backend as gs
from geomstats.geometry.spd_matrices import SPDMatrices
from geomstats.geometry.euclidean import  Euclidean

class LogNormal:
    """LogNormal Sampler

    Parameters:
    ----------
    manifold: str, {\'Euclidean\', \'SPDmanifold\'}
        Name of the Manifold
    mean: array-like, shape=[..., dim]
        Mean of the Distribution.
    cov: array-like, shape=[..., dim]
        Covariance of the Distribution.   
    Returns:
    --------
    samples

    Examples:
    --------
    import geomstats.backend as gs
    from geomstats.geometry.spd_matrices import SPDMatrices
    from geomstats.sampling.lognormal import LogNormal

    mean = 2*gs.eye(3)
    cov  = gs.eye(6)
    SPDManifold = SPDMatrices(3)
    LogNormalSampler = LogNormal(SPDManifold, mean, cov)
    data = LogNormalSampler.sample(5)

    References:
    ----------
    Lognormal Distributions and Geometric Averages of Symmetric Positive Definite Matrices
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5222531/
    """
    def __init__(self, manifold, mean, cov=None):

        if (not isinstance(manifold, SPDMatrices) and 
            not isinstance(manifold, Euclidean)):
            raise ValueError(
                "Invalid Manifold object, Should be of type SPDMatrices or Euclidean")

        if not manifold.belongs(mean):
            raise ValueError(
                "Invalid Value in mean, doesn't belong to ", type(manifold).__name__)
                
        n = mean.shape[-1]
        cov_n = (n * (n + 1)) // 2
        if cov is not None:
            if (cov.ndim != 2 and 
                (cov.shape[0] != cov_n or cov.shape[1] != cov_n)):
                valid_shape = (self.cov_n, self.cov_n)
                raise ValueError("Invalid Shape, cov should have shape" , valid_shape) 
        if cov is None:
            cov = gs.eye(self.cov_n)    

        self.manifold = manifold  
        self.mean = mean
        self.cov  = cov


    def _sample_spd(self, samples):
        n = self.mean.shape[-1]
        i, = gs.diag_indices(n, ndim=1)
        j, k = gs.triu_indices(n, k=1)
        sym_matrix = self.manifold.logm(self.mean)
        mean_euclidean = gs.hstack((sym_matrix[i, i], gs.sqrt(2) * sym_matrix[j, k]))
        _samples = gs.zeros((samples, n, n))
        samples_euclidean = gs.random.multivariate_normal(mean_euclidean, self.cov, samples)
        off_diag = samples_euclidean[:, n:]/gs.sqrt(2)
        _samples[:, i, i] = samples_euclidean[:, :n]
        _samples[:, j, k] = off_diag
        _samples[:, k, j] = off_diag
        samples_spd = self.manifold.expm(_samples)
        return samples_spd

    def _sample_euclidean(self, samples):
        _samples = gs.random.multivariate_normal(self.mean, self.cov, samples)
        return gs.exp(_samples)

    def sample(self, samples=1):

        if self.manifold == 'Euclidean':
            return self._sample_euclidean(samples)

        if self.manifold == 'SPDmanifold':
            return self._sample_spd(samples)
