"""Mean shift clustering algorithm on Manifolds.
"""


from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted


def mean_shift(X, bandwidth=None, seeds=None, bin_seeding=False,
               min_bin_freq=1, cluster_all=True, max_iter=300,
               n_jobs=None):
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
    raise NotImplementedError()


class MeanShift(BaseEstimator, ClusterMixin):
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
    def __init__(self, bandwidth=None, seeds=None, bin_seeding=False,
                 min_bin_freq=1, cluster_all=True, n_jobs=None):
        self.bandwidth = bandwidth
        self.seeds = seeds
        self.bin_seeding = bin_seeding
        self.cluster_all = cluster_all
        self.min_bin_freq = min_bin_freq
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Perform clustering.

        Parameters
        ----------
        X : array-like, shape=[n_samples, n_features]
            Samples to cluster.

        y : Ignored

        """
        X = check_array(X)
        self.cluster_centers_, self.labels_ = \
            mean_shift(X, bandwidth=self.bandwidth, seeds=self.seeds,
                       min_bin_freq=self.min_bin_freq,
                       bin_seeding=self.bin_seeding,
                       cluster_all=self.cluster_all, n_jobs=self.n_jobs)
        return self

    def predict(self, X):
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape=[n_samples, n_features]
            New data to predict.

        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        check_is_fitted(self, "cluster_centers_")

        return pairwise_distances_argmin(X, self.cluster_centers_)
