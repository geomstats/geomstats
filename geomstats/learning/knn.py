"""The KNN classifier on manifolds."""

from sklearn.neighbors import KNeighborsClassifier


class KNearestNeighborsClassifier(KNeighborsClassifier):
    """Classifier implementing the k-nearest neighbors vote on manifolds.

    Parameters
    ----------
    n_neighbors : int, optional (default = 5)
        Number of neighbors to use by default.
    weights : string or callable, optional (default = 'uniform')
        Weight function used in prediction. Possible values:
        - 'uniform' : uniform weights.  All points in each neighborhood
          are weighted equally.
        - 'distance' : weight points by the inverse of their distance.
          in this case, closer neighbors of a query point will have a
          greater influence than neighbors which are further away.
        - [callable] : a user-defined function which accepts an
          array of distances, and returns an array of the same shape
          containing the weights.
    p : integer, optional (default = 2)
        Power parameter for the 'minkowski' string distance.
        When p = 1, this is equivalent to using manhattan_distance (l1),
        and euclidean_distance (l2) for p = 2.
        For arbitrary p, minkowski_distance (l_p) is used.
    distance : string or callable, optional (default = 'minkowski')
        The distance metric to use.
        The default distance is minkowski, and with p=2 is equivalent to the
        standard Euclidean distance.
        See the documentation of the DistanceMetric class in the scikit-learn
        library for a list of available distances.
        If distance is "precomputed", X is assumed to be a distance matrix and
        must be square during fit.
    distance_params : dict, optional (default = None)
        Additional keyword arguments for the distance function.
    n_jobs : int or None, optional (default = None)
        The number of parallel jobs to run for neighbors search.
        ``None`` means 1; ``-1`` means using all processors.

    Attributes
    ----------
    classes_ : array, shape=[n_classes,]
        Class labels known to the classifier
    effective_metric_ : string or callable
        The distance metric used. It will be same as the `distance` parameter
        or a synonym of it, e.g. 'euclidean' if the `distance` parameter set to
        'minkowski' and `p` parameter set to 2.
    effective_metric_params_ : dict
        Additional keyword arguments for the distance function.
        For most distances will be same with `distance_params` parameter,
        but may also contain the `p` parameter value if the
        `effective_metric_` attribute is set to 'minkowski'.
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
                 p=2,
                 distance='minkowski',
                 distance_params=None,
                 n_jobs=None,
                 **kwargs):

        super().__init__(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm='brute',
            p=p,
            metric=distance,
            metric_params=distance_params,
            n_jobs=n_jobs,
            **kwargs)
