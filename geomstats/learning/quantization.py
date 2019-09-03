"""Mean shift clustering algorithm on Manifolds.
"""


import geomstats.backend as gs

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted


N_CLUSTERS = 10
TOLERANCE = 1e-5
N_REPETITIONS = 20
N_MAX_ITERATIONS = 50000


def quantization(points, metric, n_clusters=N_CLUSTERS,
            n_repetitions=N_REPETITIONS, tolerance=TOLERANCE,
            n_max_iterations=N_MAX_ITERATIONS):
    """Perform mean shift clustering of data.

    Parameters
    ----------

    X : array-like, shape=[n_samples, n_features]
        Input data.

    bandwidth : float, optional
        Kernel bandwidth.

        If bandwidth is not given, it is determined using a heuristic based on
        the median of all pairwise distances. This will take quadratic time in
        the number of samples. The sklearn.cluster.estimate_bandwidth function
        can be used to do this more efficiently.

    seeds : array-like, shape=[n_seeds, n_features] or None
        Point used as initial kernel locations. If None and bin_seeding=False,
        each data point is used as a seed. If None and bin_seeding=True,
        see bin_seeding.

    bin_seeding : boolean, default=False
        If true, initial kernel locations are not locations of all
        points, but rather the location of the discretized version of
        points, where points are binned onto a grid whose coarseness
        corresponds to the bandwidth. Setting this option to True will speed
        up the algorithm because fewer seeds will be initialized.
        Ignored if seeds argument is not None.

    min_bin_freq : int, default=1
       To speed up the algorithm, accept only those bins with at least
       min_bin_freq points as seeds.

    cluster_all : boolean, default True
        If true, then all points are clustered, even those orphans that are
        not within any kernel. Orphans are assigned to the nearest kernel.
        If false, then orphans are given cluster label -1.

    max_iter : int, default 300
        Maximum number of iterations, per seed point before the clustering
        operation terminates (for that seed point), if has not converged yet.

    n_jobs : int or None, optional (default=None)
        The number of jobs to use for the computation. This works by computing
        each of the n_init runs in parallel.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

        .. versionadded:: 0.17
           Parallel Execution using *n_jobs*.

    Returns
    -------

    cluster_centers : array, shape=[n_clusters, n_features]
        Coordinates of cluster centers.

    labels : array, shape=[n_samples]
        Cluster labels for each point.

    """
    n_points = points.shape[0]
    dimension = points.shape[-1]

    random_indices = gs.random.randint(low=0, high=n_points,
                                           size=(n_clusters,))
    cluster_centers = points[gs.cast(random_indices, gs.int32), :]

    gap = 1.0
    iteration = 0

    while iteration < n_max_iterations:
        iteration += 1
        step_size = gs.floor(iteration / n_repetitions) + 1

        random_index = gs.random.randint(low=0, high=n_points, size=(1,))
        point = points[gs.ix_(random_index, gs.arange(dimension))]

        index_to_update = self.closest_neighbor_index(point, centers)
        center_to_update = centers[index_to_update, :]

        tangent_vec_update = self.log(
                point=point, base_point=center_to_update
                ) / (step_size+1)
        new_center = self.exp(
                tangent_vec=tangent_vec_update, base_point=center_to_update
                )
        gap = self.dist(center_to_update, new_center)
        gap = (gap != 0) * gap + (gap == 0)

        cluster_centers[index_to_update, :] = new_center

        if gs.isclose(gap, 0, atol=tolerance):
                break

    if iteration == n_max_iterations-1:
        print('Maximum number of iterations {} reached. The'
                'quantization may be inaccurate'.format(n_max_iterations))

    labels = gs.zeros(n_points)
    for i in range(n_points):
        labels[i] = metric.closest_neighbor_index(points[i], cluster_centers)

    return cluster_centers, labels


class Quantization(BaseEstimator, ClusterMixin):
    """Mean shift clustering using a flat kernel.

    Mean shift clustering aims to discover "blobs" in a smooth density of
    samples. It is a centroid-based algorithm, which works by updating
    candidates for centroids to be the mean of the points within a given
    region. These candidates are then filtered in a post-processing stage to
    eliminate near-duplicates to form the final set of centroids.

    Seeding is performed using a binning technique for scalability.

    Parameters
    ----------
    bandwidth : float, optional
        Bandwidth used in the RBF kernel.

        If not given, the bandwidth is estimated using
        sklearn.cluster.estimate_bandwidth; see the documentation for that
        function for hints on scalability (see also the Notes, below).

    seeds : array, shape=[n_samples, n_features], optional
        Seeds used to initialize kernels. If not set,
        the seeds are calculated by clustering.get_bin_seeds
        with bandwidth as the grid size and default values for
        other parameters.

    bin_seeding : boolean, optional
        If true, initial kernel locations are not locations of all
        points, but rather the location of the discretized version of
        points, where points are binned onto a grid whose coarseness
        corresponds to the bandwidth. Setting this option to True will speed
        up the algorithm because fewer seeds will be initialized.
        default value: False
        Ignored if seeds argument is not None.

    min_bin_freq : int, optional
       To speed up the algorithm, accept only those bins with at least
       min_bin_freq points as seeds. If not defined, set to 1.

    cluster_all : boolean, default True
        If true, then all points are clustered, even those orphans that are
        not within any kernel. Orphans are assigned to the nearest kernel.
        If false, then orphans are given cluster label -1.

    n_jobs : int or None, optional (default=None)
        The number of jobs to use for the computation. This works by computing
        each of the n_init runs in parallel.

        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Attributes
    ----------
    cluster_centers_ : array, [n_clusters, n_features]
        Coordinates of cluster centers.

    labels_ :
        Labels of each point.
    """
    def __init__(self, metric, n_clusters=N_CLUSTERS,
            n_repetitions=N_REPETITIONS, tolerance=TOLERANCE,
            n_max_iterations=N_MAX_ITERATIONS):
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
            optimal_quantization(X, n_clusters=self.n_clusters,
                    n_repetitions=self.n_repetitions,
                    tolerance=self.tolerance,
                    n_max_repetitions=self.n_max_repetitions)

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
