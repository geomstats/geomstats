import geomstats.geometry.spd_matrices as spd

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
        self.cov_n = (self.n*(self.n+1))//2
        self.cov = cov
        if self.cov is None:
            self.cov = gs.eye(self.cov_n)
        if self.manifold == 'SPDmanifold':
            mean_SPDmanifold = spd.SPDMatrices(self.n)
            cov_SPDmanifold  = spd.SPDMatrices(self.cov_n)

            if (not mean_SPDmanifold.belongs(self.mean)):
                 raise ValueError("Invalid Value in mean" , mean)




    
    def _sample_spd(self,samples):
        pass

    def _sample_euclidean(self,samples):
        pass    

    def sample(self,samples=1):

        if self.manifold == 'Euclidean':
            return self._sample_euclidean(samples)

        if self.manifold == 'SPDmanifold':
            return self._sample_spd(samples)    
