"""Online kmeans algorithm on Manifolds."""

import logging

from sklearn.base import BaseEstimator, ClusterMixin

import geomstats.backend as gs


# TODO (nkoep): Move this into the OnlineKMeans class.

def online_kmeans(X, metric, n_clusters, n_repetitions=20,
                  tolerance=1e-5, max_iter=5e4):
    """Perform online K-means clustering.

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
    X : array-like, shape=[..., n_features]
        Input data. It is treated sequentially by the algorithm, i.e.
        one datum is chosen randomly at each iteration.
    metric : object
        Metric of the space in which the data lives. At each iteration,
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

    Returns
    -------
    cluster_centers : array, shape=[n_clusters, n_features]
        Coordinates of cluster centers.
    labels : array, shape=[n_samples]
        Cluster labels for each point.
    """
    n_samples = X.shape[0]

    random_indices = gs.random.randint(low=0, high=n_samples,
                                       size=(n_clusters,))
    cluster_centers = gs.get_slice(X, gs.cast(random_indices, gs.int32))

    gap = 1.0
    iteration = 0

    while iteration < max_iter:
        iteration += 1
        step_size = gs.floor(gs.array(iteration / n_repetitions)) + 1

        random_index = gs.random.randint(low=0, high=n_samples, size=(1,))
        point = gs.get_slice(X, gs.cast(random_index, gs.int32))

        index_to_update = metric.closest_neighbor_index(point, cluster_centers)
        center_to_update = gs.copy(
            gs.get_slice(cluster_centers, index_to_update))

        tangent_vec_update = metric.log(
            point=point, base_point=center_to_update
        ) / (step_size + 1)
        new_center = metric.exp(
            tangent_vec=tangent_vec_update,
            base_point=center_to_update
        )
        gap = metric.dist(center_to_update, new_center)
        if gap == 0 and iteration == 1:
            gap = gs.array(1.0)

        cluster_centers[index_to_update, :] = new_center

        if gs.isclose(gap, 0, atol=tolerance):
            break

    if iteration == max_iter - 1:
        logging.warning(
            'Maximum number of iterations {} reached. The'
            'clustering may be inaccurate'.format(max_iter))

    labels = gs.zeros(n_samples)
    for i in range(n_samples):
        labels[i] = int(metric.closest_neighbor_index(X[i], cluster_centers))

    return cluster_centers, labels


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
    metric : object
        Metric of the space in which the data lives. At each iteration,
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

    def __init__(self, metric, n_clusters, n_repetitions=20,
                 tolerance=1e-5, max_iter=5e4, point_type='vector'):
        self.metric = metric
        self.n_clusters = n_clusters
        self.n_repetitions = n_repetitions
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.point_type = point_type

    def fit(self, X):
        """Perform clustering.

        Parameters
        ----------
        X : array-like, shape=[n_features, n_samples]
            Samples to cluster.
        """
        self.cluster_centers_, self.labels_ = \
            online_kmeans(X=X, metric=self.metric,
                          n_clusters=self.n_clusters,
                          n_repetitions=self.n_repetitions,
                          tolerance=self.tolerance,
                          max_iter=self.max_iter)

        return self

    def predict(self, point):
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : array-like, shape=[n_features]
            New data to predict.

        Returns
        -------
        labels : int
            Index of the cluster each sample belongs to.
        """
        return self.metric.closest_neighbor_index(point, self.cluster_centers_)
