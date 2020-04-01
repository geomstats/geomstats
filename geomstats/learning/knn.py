"""The KNN classifier on manifolds."""

from sklearn.neighbors import KNeighborsClassifier

from geomstats.geometry.euclidean import Euclidean

EUCLIDEAN = Euclidean(dimension=1)
EUCLIDEAN_DISTANCE = EUCLIDEAN.metric.dist


class KNearestNeighborsClassifier(KNeighborsClassifier):
    """Classifier implementing the k-nearest neighbors vote on manifolds.

    Parameters
    ----------
    n_neighbors : int, optional (default = 5)
        Number of neighbors to use by default.
    weights : str or callable, optional (default = 'uniform')
        Weight function used in prediction.  Possible values:
        - 'uniform' : uniform weights.  All points in each neighborhood
          are weighted equally.
        - 'distance' : weight points by the inverse of their distance.
          in this case, closer neighbors of a query point will have a
          greater influence than neighbors which are further away.
        - [callable] : a user-defined function which accepts an
          array of distances, and returns an array of the same shape
          containing the weights.
    metric : callable or string, default EuclideanMetric
        The distance metric to use.
        The 'minkowski' string metric with p=2 is equivalent to the standard
        Euclidean metric.
        See the documentation of the DistanceMetric class in the scikit-learn
        library for a list of available metrics.
        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square during fit.
    dimension : int, optional (default = 1)
        Dimension of the manifold used if the metric parameter is callable.
    metric_params : dict, optional (default = None)
        Additional keyword arguments for the metric function.
    p : integer, optional (default = 2)
        Power parameter for the 'minkowski' string metric.
        When p = 1, this is equivalent to using manhattan_distance (l1),
        and euclidean_distance (l2) for p = 2.
        For arbitrary p, minkowski_distance (l_p) is used.
    n_jobs : int or None, optional (default=None)
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1; ``-1`` means using all processors.

    Attributes
    ----------
    classes_ : array, shape=[n_classes,]
        Class labels known to the classifier
    effective_metric_ : callable or string
        The distance metric used. It will be same as the `metric` parameter
        or a synonym of it, e.g. 'euclidean' if the `metric` parameter set to
        'minkowski' and `p` parameter set to 2.
    effective_metric_params_ : dict
        Additional keyword arguments for the metric function. For most metrics
        will be same with `metric_params` parameter, but may also contain the
        `p` parameter value if the `effective_metric_` attribute is set to
        'minkowski'.
    outputs_2d_ : bool
        False when `y`'s shape is (n_samples, ) or (n_samples, 1) during fit
        otherwise True.

    References
    ----------
    This algorithm uses the scikit-learn library:
    https://github.com/scikit-learn/scikit-learn/blob/95d4f0841/sklearn/
    neighbors/_classification.py#L25
    """

    def __init__(self, n_neighbors=5,
                 weights='uniform',
                 metric=EUCLIDEAN_DISTANCE,
                 metric_params=None,
                 p=2,
                 n_jobs=None,
                 **kwargs):

        super().__init__(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm='brute',
            metric=metric,
            metric_params=metric_params,
            p=p,
            n_jobs=n_jobs,
            **kwargs)
