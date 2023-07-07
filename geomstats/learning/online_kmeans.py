"""Online kmeans algorithm on Manifolds.

Lead author: Alice Le Brigant.
"""

import logging

from sklearn.base import BaseEstimator, ClusterMixin

import geomstats.backend as gs


class OnlineKMeans(BaseEstimator, ClusterMixin):
    """Online k-means clustering.

    Online k-means clustering seeks to divide a set of data points into
    a specified number of classes, while minimizing intra-class variance.
    It is closely linked to discrete quantization, which computes the closest
    approximation of the empirical distribution of the dataset by a discrete
    distribution supported by a smaller number of points with respect to the
    Wasserstein distance. The algorithm used can either be seen as an online
    version of the k-means algorithm or as Competitive Learning Riemannian
    Quantization (see [LBP2019]_).

    Parameters
    ----------
    space : Manifold
        Equipped manifold. At each iteration,
        one of the cluster centers is moved in the direction of the new
        datum, according the exponential map of the underlying space, which
        is a method of metric.
    n_clusters : int
        Number of clusters of the k-means clustering, or number of desired
        atoms of the quantized distribution.
    n_repetitions : int, default=20
        The cluster centers are updated using decreasing step sizes, each
        of which stays constant for n_repetitions iterations to allow a better
        exploration of the data points.
    max_iter : int, default=5e4
        Maximum number of iterations. If it is reached, the
        quantization may be inacurate.

    Attributes
    ----------
    cluster_centers_ : array, [n_clusters, n_features]
        Coordinates of cluster centers.
    labels_ :
        Labels of each point.

    Notes
    -----
    * Required metric methods: `exp`, `log`, `dist`, `closest_neighbor_index`.

    Example
    -------
    >>> from geomstats.geometry.hypersphere import Hypersphere
    >>> from geomstats.learning.onlinekmeans import OnlineKmeans
    >>> sphere = Hypersphere(dim=2)
    >>> metric = sphere.metric
    >>> X = sphere.random_von_mises_fisher(kappa=10, n_samples=50)
    >>> clustering = OnlineKmeans(metric=metric,n_clusters=4).fit(X)
    >>> clustering.cluster_centers_
    >>> clustering.labels_


    References
    ----------
    .. [LBP2019] A. Le Brigant and S. Puechmorel, Optimal Riemannian
       quantization with an application to air traffic analysis. J. Multivar.
       Anal. 173 (2019), 685 - 703.
    """

    def __init__(
        self,
        space,
        n_clusters,
        n_repetitions=20,
        atol=1e-5,
        max_iter=500,
    ):
        self.space = space
        self.n_clusters = n_clusters
        self.n_repetitions = n_repetitions
        self.atol = atol
        self.max_iter = max_iter

        self.cluster_centers_ = None
        self.labels_ = None

    def fit(self, X, y=None):
        """Perform clustering.

        Perform online version of k-means algorithm on data contained in X.
        The data points are treated sequentially and the cluster centers are
        updated one at a time. This version of k-means avoids computing the
        mean of each cluster at each iteration and is therefore less
        computationally intensive than the offline version.

        In the setting of quantization of probability distributions, this
        algorithm is also known as Competitive Learning Riemannian Quantization.
        It computes the closest approximation of the empirical distribution of
        data by a discrete distribution supported by a smaller number of points
        with respect to the Wasserstein distance. This smaller number of points
        is n_clusters.

        Parameters
        ----------
        X : array-like, shape=[n_samples, n_features]
            Input data. It is treated sequentially by the algorithm, i.e.
            one datum is chosen randomly at each iteration.
        y : None
            Target values. Ignored.

        Returns
        -------
        self : object
            Returns self.
        """
        n_samples = X.shape[0]

        random_indices = gs.random.randint(
            low=0, high=n_samples, size=(self.n_clusters,)
        )
        cluster_centers = gs.get_slice(X, random_indices)

        gap = 1.0

        for iteration in range(self.max_iter):
            step_size = gs.floor(gs.array((iteration + 1) / self.n_repetitions)) + 1

            random_index = gs.random.randint(low=0, high=n_samples, size=(1,))
            point = gs.get_slice(X, random_index)[0]

            index_to_update = self.space.metric.closest_neighbor_index(
                point, cluster_centers
            )
            center_to_update = gs.copy(gs.get_slice(cluster_centers, index_to_update))

            tangent_vec_update = self.space.metric.log(
                point=point, base_point=center_to_update
            ) / (step_size + 1)
            new_center = self.space.metric.exp(
                tangent_vec=tangent_vec_update, base_point=center_to_update
            )
            gap = self.space.metric.dist(center_to_update, new_center)
            if gap == 0 and iteration == 0:
                gap = gs.array(1.0)

            cluster_centers[index_to_update, :] = new_center

            if gs.isclose(gap, 0, atol=self.atol):
                break
        else:
            logging.warning(
                "Maximum number of iterations {} reached. The"
                "clustering may be inaccurate".format(self.max_iter)
            )

        labels = self.space.metric.closest_neighbor_index(X, cluster_centers)

        self.cluster_centers_ = cluster_centers
        self.labels_ = labels

        return self

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : array-like, shape=[n_samples, n_features]
            New data to predict.

        Returns
        -------
        labels : int
            Index of the cluster each sample belongs to.
        """
        if self.cluster_centers_ is None:
            raise Exception("Not fitted")

        return self.space.metric.closest_neighbor_index(X, self.cluster_centers_)
