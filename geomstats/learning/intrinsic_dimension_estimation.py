"""Intrinsic Dimension Estimation."""

import geomstats.backend as gs


class LevinaBickelEstimator:
    r"""Levina Bickel Estimator for intrinsic dimension estimation.

    Levian Bickel Estimator calculates intrinsic dimension of
    manifolds embedded in euclidean space through maximum likelihood
    estimation.

    .. math::
        T_j(x_i)=\text{j th Nearest Neighbor to }  x_i \text{ in the data}

        m_k(x_i)=[\frac{1}{k-1} \sum_{j=1}^{k}
        \frac{\log T_k(x)}{\log T_{j}(x)}]^{-1}

        m_k=\frac{1}{N} \sum_{i=1}^{N} m_{k}(x_i)

        m=\frac{1}{k_2 - k_1 + 1} \sum_{k=k_1}^{k_2} m_k

    Parameters
    ----------
    min_neighbors : int
        Minimum Neighbors used for computing estimator
        Optional, default: 5.
    max_neighbors : int
        Maximum Neighbors used for computing estimator
        Optional, default: 20.

    Example
    -------
    >>> from sklearn import  datasets
    >>> import geomstats.backend as gs
    >>> import geomstats.learning.intrinsic_dimension_estimation as ide
    >>> from ide import LevinaBickelEstimator
    >>> X, _ = datasets.make_swiss_roll(n_samples=1500)
    >>> X = gs.array(X)
    >>> lbe = LevinaBickelEstimator()
    >>> predicted_intrinsic_dim =lbe.fit(X).predict()

    Reference
    ----------
    .. [2004] . "Maximum Likelihood Estimation of Intrinsic Dimension"
    Advances in Neural Information Processing Systems 17 (NIPS 2004)
    https://arxiv.org/abs/1908.09326
    """

    def __init__(self, min_neighbors=5, max_neighbors=20):
        self.min_neighbors = min_neighbors
        self.max_neighbors = max_neighbors
        self.log_sorted_dist = None

    def fit(self, X):
        """Calculate distance matrix

        Parameters
        ----------
        X : array-like
            Data sampled from manifold

        Returns
        -------
        self : object
            Returns self.
        """
        pairwise_sub = gs.expand_dims(X, axis=0) - gs.expand_dims(X, axis=1)
        pairwise_dist = gs.linalg.norm(pairwise_sub, axis=-1)
        sorted_dist = gs.sort(pairwise_dist, axis=-1)[:, 1:]
        self.log_sorted_dist = gs.log(sorted_dist)
        return self

    def T(self, j, x):
        """Return jth nearest neighbor to x in data X.

        Parameters
        ---------
        j : int
            j th nearest neighbor
        x : int, ranges from 0 to N-1
            index of the point in the data 

        Returns
        -------
        X : array like, shape=[N, dim]
        """
        return gs.exp(self.log_sorted_dist[x - 1, j - 1])

    def predict(self):
        """Predict intrinsic dimension."""
        int_dim_for_each_k = []
        for k in range(self.min_neighbors, self.max_neighbors + 1):
            t1 = self.log_sorted_dist[:, k - 1]
            t2 = gs.mean(self.log_sorted_dist[:, :k - 1], axis=1)
            int_dim_k = gs.mean(1.0 / (t1 - t2))
            int_dim_for_each_k.append(int_dim_k)

        intrinsic_dim = sum(int_dim_for_each_k) / len(int_dim_for_each_k)
        return intrinsic_dim
