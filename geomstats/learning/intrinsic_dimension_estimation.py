"""Intrinsic Dimension Estimation"""

import geomstats.backend as gs

class LevinaBickelEstimator:
    """Levina Bickel Estimator for intrinsic dimension

   

    References
    ----------
    .. [2004] . "Maximum Likelihood Estimation of Intrinsic Dimension" 
    Advances in Neural Information Processing Systems 17 (NIPS 2004)
    https://arxiv.org/abs/1908.09326
    """

    def __init__(self, min_neighbors=5, max_neighbors=20):
        self.min_neighbors = min_neighbors
        self.max_neighbors = max_neighbors

    def fit(self, X):
        

        pairwise_sub = gs.expand_dims(X, axis = 0) - gs.expand_dims(X, axis=1)
        pairwise_dist = gs.linalg.norm(pairwise_sub, axis = -1)
        sorted_dist = gs.sort(pairwise_dist , axis = -1)
        self.log_sorted_dist = gs.log(sorted_dist)


    def T(self, j,x):
        return gs.exp(self.log_sorted_dist[x-1, j-1])

    def predict(self):
        num = self.sorted_dist[:, self.min_neighbors-1: self.max_neighbors]
        int_dim_for_each_k = []
        for k in range(self.min_neighbors , self.max_neighbors+1):
            t1 = self.log_sorted_dist[:, k-1]
            t2 = gs.mean(self.log_sorted_dist[:,:k-1], axis=1)
            int_dim_k = gs.mean(1.0/(t1-t2))
            int_dim_for_each_k.append(int_dim_k)

        intrinsic_dim = sum(int_dim_for_each_k)/len(int_dim_for_each_k)  
        return intrinsic_dim

