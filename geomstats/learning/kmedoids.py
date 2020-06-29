"""K-medoids clustering."""

import logging

from sklearn.base import BaseEstimator, ClusterMixin

import geomstats.backend as gs
from geomstats.learning._template import TransformerMixin


class RiemannianKMedoids(TransformerMixin, ClusterMixin, BaseEstimator):
    """Class for K-medoids clustering on manifolds.

    K-medoids algorithm using Riemannian manifolds.

    Parameters
    ----------
    metric : object of class RiemannianMetric
        The geomstats Riemmanian metric associate to the space used.
    n_clusters : int
        Number of clusters (k value of k-medoids).
        Optional, default: 8.
    init : str
        How to initialize centroids at the beginning of the algorithm. The
        choice 'random' will select training points as initial centroids
        uniformly at random.
        Optional, default: 'random'.
    cluster_centers_ : array-like, shape=[n_clusters, dim]
        Array of cluster centers.
    labels_ : array-like, shape=[n_clusters, dim]
        Labels predicted for each data sample.
    medoid_indices_ : array-like, shape=[n_clusters]
        Indices of the cluster centers from the data array.

    Example
    -------
    Available example on the Poincaré Ball and Hypersphere manifolds
    :mod:`examples.plot_kmedoids_manifolds`
    """

    def __init__(
            self, metric, n_clusters=8, init='random'):
        self.metric = metric
        self.n_clusters = n_clusters
        self.init = init
        self.cluster_centers_ = None
        self.labels_ = None
        self.medoid_indices_ = None

    def _initialize_medoids(self, distances):
        """Select initial medoids when beginning clustering."""
        if self.init == "random":
            medoids = gs.random.choice(
                gs.arange(len(distances)), self.n_clusters)
        else:
            logging.error('Unknown initialization method.')

        return medoids

    def fit(self, data, max_iter=100):
        """Provide clusters centroids and data labels.

        Labels data by minimizing the distance between data points
        and cluster centroids chosen from the data points.
        Minimization is performed by swapping the centroids and data points.

        Parameters
        ----------
        data : array-like, shape=[n_samples, dim]
            Training data, where n_samples is the number of samples and
            dim is the number of dimensions.
        max_iter : int
            Maximum number of iterations.
            Optional, default: 100.

        Returns
        -------
        self : array-like, shape=[n_clusters,]
            Centroids.
        """
        distances = self.metric.dist_pairwise(data)

        medoids_indices = self._initialize_medoids(distances)

        for iteration in range(max_iter):

            old_medoids_indices = gs.copy(medoids_indices)

            labels = gs.argmin(distances[medoids_indices, :], axis=0)

            self._update_medoid_indexes(distances, labels, medoids_indices)

            if gs.all(old_medoids_indices == medoids_indices):
                break
            if iteration == max_iter - 1:
                logging.warning(
                    'Maximum number of iteration reached before '
                    'convergence. Consider increasing max_iter to '
                    'improve the fit.'
                )

        self.cluster_centers_ = data[medoids_indices]
        self.labels_ = labels
        self.medoid_indices_ = medoids_indices

        return self.cluster_centers_

    def _update_medoid_indexes(self, distances, labels, medoid_indices):

        for cluster in range(self.n_clusters):

            cluster_index = gs.where(labels == cluster)[0]

            if len(cluster_index) == 0:
                logging.warning('One cluster is empty.')
                continue

            in_cluster_distances = distances[
                cluster_index, gs.expand_dims(cluster_index, axis=-1)]

            in_cluster_all_costs = gs.sum(in_cluster_distances, axis=1)

            min_cost_index = gs.argmin(in_cluster_all_costs)

            min_cost = in_cluster_all_costs[min_cost_index]

            current_cost = in_cluster_all_costs[
                gs.argmax(cluster_index == medoid_indices[cluster])]

            if min_cost < current_cost:
                medoid_indices[cluster] = cluster_index[min_cost_index]

    def predict(self, data):
        """Predict the closest cluster for each sample in X.

        Parameters
        ----------
        data : array-like, shape=[n_samples, dim,]
            Training data, where n_samples is the number of samples and
            dim is the number of dimensions.

        Returns
        -------
        labels : array-like, shape=[n_samples,]
            Index of the cluster each sample belongs to.
        """
        labels = gs.zeros(len(data))

        for point_index, point_value in enumerate(data):
            distances = gs.zeros(len(self.cluster_centers_))
            for cluster_index, cluster_value in enumerate(
                    self.cluster_centers_):
                distances[cluster_index] = self.metric.dist(
                    point_value, cluster_value)

            labels[point_index] = gs.argmin(distances)

        return labels
