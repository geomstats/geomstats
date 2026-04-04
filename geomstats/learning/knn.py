"""The KNN classifier on manifolds.

Lead author: Yann Cabanes.
"""

import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.neighbors import KNeighborsClassifier

import geomstats.backend as gs


def wrap(function):
    """Wrap a function to first convert args to arrays."""

    def wrapped_function(*args, **kwargs):
        new_args = map(gs.from_numpy, args)
        return function(*new_args, **kwargs)

    return wrapped_function


class SimpleGeodesicKNearestNeighborsClassifier:
    """Classifier implementing the k-nearest neighbors vote on geodesic metric spaces.

    Parameters
    ----------
    space : PointSet
        Space equipped with a metric.
    n_neighbors : int, optional (default = 5)
        Number of neighbors to use by default.
    """

    def __init__(self, space, n_neighbors=5):
        self.space = space
        self.dist = space.metric.dist
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        """Fit the k-nearest neighbors classifier from the training dataset.

        Parameters
        ----------
        X : array-like of shape (n_samples, object)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : SimpleGeodesicKNearestNeighborsClassifier
            The fitted k-nearest neighbors classifier.
        """
        self._fit_X = gs.array(X)
        self._fit_y = gs.array(y)
        self._is_fitted = True
        return self

    def predict_one(self, x):
        """
        Predict the class labels for one provided data point.

        Parameters
        ----------
        X : object
            Test sample.

        Returns
        -------
        y : general label
            Class labels for one data sample.
        """
        if not hasattr(self, "_is_fitted"):
            raise NotFittedError("Must fit model before predicting.")

        dists = [self.dist(x, xx) for xx in self._fit_X]
        indices_by_dist = sorted(range(len(dists)), key=lambda k: dists[k])

        labels = self._fit_y[indices_by_dist[: self.n_neighbors]]
        unique_values, counts = np.unique(labels, return_counts=True)

        return unique_values[np.argmax(counts)]

    def predict(self, X):
        """
        Predict the class labels for provided data.

        Parameters
        ----------
        X : array-like of shape (n_queries, object)
            Test samples.

        Returns
        -------
        y : ndarray of shape (n_queries,)
            Class labels for each data sample.
        """
        return gs.array([self.predict_one(x) for x in X])


class KNearestNeighborsClassifier(KNeighborsClassifier):
    """Classifier implementing the k-nearest neighbors vote on manifolds.

    Parameters
    ----------
    space : Manifold
        Equipped manifold.
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

    def __init__(
        self,
        space,
        n_neighbors=5,
        weights="uniform",
        n_jobs=None,
    ):
        self.space = space

        distance = wrap(space.metric.dist)
        super().__init__(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm="brute",
            metric=distance,
            n_jobs=n_jobs,
        )
