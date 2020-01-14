"""Optimal quantization algorithm on Manifolds.
"""

import geomstats.backend as gs

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted



def quantization(X, metric, n_clusters, n_repetitions=20,
                 tolerance=1e-5, n_max_iterations=5e4):
    """
    Perform quantization of data using the Competitive Learning
    Riemannian Quantization algorithmm. It computes the closest
    approximation of the empirical distribution of data by a
    discrete distribution supported by a smaller number of points
    with respect to the Wasserstein distance. It is an online
    version of the k-means clustering algorithm.

    Parameters
    ----------

    X : array-like, shape=[n_samples, n_features]
        Input data. It is treated sequentially by the algorithm, i.e.
        one datum is chosen randomly at each iteration.

    metric : object
        Metric of the space in which the data lives. At each iteration,
        one of the centers of the quantization is moved in the direction
        of the new datum, according the exponential map of the underlying
        space, which is a method of metric.

    n_clusters : int
        Number of desired atoms of the approximation distribution, or
        equivalently number of centers of the clusters of the k-means
        clustering.

    n_repetitions : int, default=20
        The centers of the quantization are updated using decreasing
        step sizes, each of which stays constant for n_repetitions
        iterations to allow a better exploration of the data points.

    n_max_iterations : int, default=5e4
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
    n_features = X.shape[-1]

    random_indices = gs.random.randint(low=0, high=n_samples,
                                           size=(n_clusters,))
    cluster_centers = X[gs.cast(random_indices, gs.int32), :]

    gap = 1.0
    iteration = 0

    while iteration < n_max_iterations:
        iteration += 1
        step_size = gs.floor(gs.array(iteration / n_repetitions)) + 1

        random_index = gs.random.randint(low=0, high=n_samples, size=(1,))
        point = X[gs.cast(random_index, gs.int32), :]

        index_to_update = metric.closest_neighbor_index(point, cluster_centers)
        center_to_update = gs.copy(cluster_centers[index_to_update, :])

        tangent_vec_update = metric.log(
                point=point, base_point=center_to_update
                ) / (step_size+1)
        new_center = metric.exp(
                tangent_vec=tangent_vec_update, base_point=center_to_update
                )
        gap = metric.dist(center_to_update, new_center)
        if gap == 0 and iteration == 1:
            gap = gs.array(1.0)

        cluster_centers[index_to_update, :] = new_center

        if gs.isclose(gap, 0, atol=tolerance):
            break

    if iteration == n_max_iterations-1:
        print('Maximum number of iterations {} reached. The'
                'quantization may be inaccurate'.format(n_max_iterations))

    labels = gs.zeros(n_samples)
    for i in range(n_samples):
        labels[i] = int(metric.closest_neighbor_index(X[i], cluster_centers))

    return cluster_centers, labels


class Quantization(BaseEstimator, ClusterMixin):
    """Optimal Riemannian quantization

    Quantization computes the closest approximation of the empirical
    distribution of a dataset by a discrete distribution supported by a
    smaller number of points with respect to the Wasserstein distance. It
    yields a k-means clustering of the data. The algorithm used is
    Competitive Learning Riemannian Quantization, and corresponds to an
    online version of the k-means clustering algorithm.

    Parameters
    ----------

    metric : object
        Metric of the space in which the data lives. At each iteration,
        one of the centers of the quantization is moved in the direction
        of the new datum, according the exponential map of the underlying
        space, which is a method of metric.

    n_clusters : int
        Number of desired atoms of the approximation distribution, or
        equivalently number of centers of the clusters of the k-means
        clustering.

    n_repetitions : int, default=20
        The centers of the quantization are updated using decreasing
        step sizes, each of which stays constant for n_repetitions
        iterations to allow a better exploration of the data points.

    n_max_iterations : int, default=5e4
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
    >>> from geomstats.learning.quantization import Quantization
    >>> sphere = Hypersphere(dimension=2)
    >>> metric = sphere.metric
    >>> X = sphere.random_von_mises_fisher(kappa=10, n_samples=50)
    >>> clustering = Quantization(metric=metric,n_clusters=4).fit(X)
    >>> clustering.cluster_centers_
    >>> clustering.labels_

    Reference
    ---------
    A. Le Brigant and S. Puechmorel, Optimal Riemannian quantization
    with an application to air traffic analysis. J. Multivar. Anal.
    173 (2019), 685 - 703.
    """
    def __init__(self, metric, n_clusters, n_repetitions=20,
            tolerance=1e-5, n_max_iterations=5e4):
        self.metric = metric
        self.n_clusters = n_clusters
        self.n_repetitions = n_repetitions
        self.tolerance = tolerance
        self.n_max_iterations = n_max_iterations

    def fit(self, X):
        """Perform clustering.

        Parameters
        ----------
        X : array-like, shape=[n_features, n_samples]
            Samples to cluster.
        """
        self.cluster_centers_, self.labels_ = \
            quantization(X=X, metric=self.metric,
                    n_clusters=self.n_clusters,
                    n_repetitions=self.n_repetitions,
                    tolerance=self.tolerance,
                    n_max_iterations=self.n_max_iterations)

        return self

    def predict(self, point):
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : {array-like}, shape=[n_features]
            New data to predict.

        Returns
        -------
        labels : Index of the cluster each sample belongs to.
        """

        return self.metric.closest_neighbor_index(point, self.cluster_centers_)
