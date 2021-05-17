class LogNormal:
    """ LogNormal Sampler (Currently only done for SPDmanifold,EuclideanSpace)

    Parameters:
    ----------
    manifold: Name of Manifold

    """
    def __init__(self,manifold,mean,cov=None):
        self.manifold = manifold
        self.mean = mean
        self.n  = self.mean.shape[-1]

    
    def _sample_spd(self,samples):
        pass

    def _sample_euclidean(self,samples):
        pass    

    def sample(self,samples=1):

        if self.manifold == 'Euclidean':
            return self._sample_euclidean(samples)

        if self.manifold == 'SPDmanifold':
            return self._sample_spd(samples)    
