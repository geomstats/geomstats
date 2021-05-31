import geomstats.geometry.euclidean as euclidean
import geomstats.geometry.spd_matrices as spd
import geomstats.backend as gs

class LogNormal:
    """ LogNormal Sampler 

    Parameters:
    ----------
    manifold: str, {\'Euclidean\', \'SPDmanifold\'}
        Name of the Manifold
    mean: array-like, shape=[..., dim]
        Mean of the Distribution
    cov: array-like, shape=[..., dim]
        Covariance of the Distribution. Should be Positive Semi Definite
    validate_args: boolean
        When True Checks for validity of mean,cov.
        checks if mean is Symmetric Positive Definite Matrix
        checks if cov  is Symmetric Positive Semi Definite Matrix    
    Returns:
    --------
    samples

    Examples:
    --------
    import geomstats.geometry.spd_matrices as spd
    from geomstats.sampling.lognormal import LogNormal

    SPDManifold = spd.SPDmatrices(3)
    mean = 2*gs.eye(3)
    cov  = gs.eye(6)
    LogNormalSampler = LogNormal(SPDManifold, mean, cov)
    data = LogNormalSampler.sample(5)

    References:
    ----------
    Lognormal Distributions and Geometric Averages of Symmetric Positive Definite Matrices
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5222531/
    """
    def __init__(self, manifold, mean, cov=None, validate= True):

        if (not isinstance(manifold, spd.SPDMatrices) and 
            not isinstance(manifold, euclidean.Euclidean)):
            raise ValueError(
                "Invalid Manifold object, Should be of type SPDMatrices or Euclidean")

        if validate:    
            if not manifold.belongs(mean):
                raise ValueError(
                    "Invalid Value in mean, doesn't belong to ", type(manifold).__name__)
        
        n = mean.shape[-1]
        cov_n = (self.n*(self.n+1))//2
        if cov is not None:
            if (cov.ndim != 2 and 
                (cov.shape[0] != cov_n or cov.shape[1] != cov_n)):
                valid_shape = (self.cov_n,self.cov_n)
                raise ValueError("Invalid Shape, cov should have shape" , valid_shape) 
        if self.cov is None:
            cov = gs.eye(self.cov_n)    

        self.manifold = manifold  
        self.mean = mean
        self.cov  = cov


    def _sample_spd(self, samples):
        i,  = gs.diag_indices(self.n, ndim=1)
        j,k = gs.triu_indices(self.n, k=1)
        sym_matrix = self.manifold.logm(self.mean)
        mean_euclidean = gs.hstack((sym_matrix[i,i], gs.sqrt(2)*sym_matrix[j,k]))
        _samples = gs.zeros((samples, self.n, self.n))
        samples_euclidean = gs.random.multivariate_normal(mean_euclidean, self.cov, samples)
        off_diag = samples_euclidean[:, self.n:]/gs.sqrt(2)
        _samples[:, i, i] = samples_euclidean[:, :self.n]
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
