"""The MDM classifier on manifolds."""

import geomstats.backend as gs
from geomstats.learning.frechet_mean import FrechetMean


class RiemannianMinimumDistanceToMeanClassifier:
    r"""Classifier implementing the MDM scheme on manifolds.

    Parameters
    ----------
    riemannian_metric : RiemannianMetric
        Riemannian metric to be Used.
    n_classes: int
        Number of classes.
    point_type : str, {\'vector\', \'matrix\'}
        Point type.
        Optional, default: \'matrix\'.
    """

    def __init__(
            self,
            riemannian_metric,
            n_classes,
            point_type='matrix'):
        self.riemannian_metric = riemannian_metric
        self.n_classes = n_classes
        self.point_type = point_type
        self.mean_estimates_ = None

    def fit(self, X, y):
        """Compute Frechet mean of each class.

        Parameters
        ----------
        X : array-like, shape=[n_samples, dim]
                              if point_type='vector'
                              shape=[n_samples, n, n]
                              if point_type='matrix'
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape=[n_samples, n_classes]
            Training labels, where n_classes is the number of classes.
        """
        mean_estimator = FrechetMean(
            metric=self.riemannian_metric,
            point_type=self.point_type)
        frechet_means = []
        for c in range(self.n_classes):
            data_class = self.split_data_in_classes(X, y, c)
            frechet_means.append(mean_estimator.fit(data_class).estimate_)
        self.mean_estimates_ = gs.array(frechet_means)

    def predict(self, X):
        """
        Compute closest neighbor according to riemannian_metric.

        Parameters
        ----------
        X : array-like, shape=[n_samples, dim]
                              if point_type='vector'
                              shape=[n_samples, n , n]
                              if point_type='matrix'
            Test data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        y : array-like, shape=[n_samples, n_classes]
            Predicted labels, where n_classes is the number of classes.
        """
        n_samples = X.shape[0]
        _labels = []
        for i in range(n_samples):
            label = self.riemannian_metric.closest_neighbor_index(
                X[i], self.mean_estimates_)
            _labels.append(label)
        y = gs.one_hot(_labels, self.n_classes)
        return y

    @staticmethod
    def split_data_in_classes(X, y, c):
        """Split a labelled dataset in sub-datasets of each label.

        Parameters
        ----------
        X : array-like, shape=[n_samples, dim]
                              if point_type='vector'
                              shape=[n_samples, n, n]
                              if point_type='matrix'
            Labelled dataset, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape=[n_samples, n_classes]
            Labels, where n_classes is the number of classes.
        c : int
            Class index

        Returns
        -------
        X_split : array-like, shape=[n_samples_in_class, dim]
                             if point_type='vector'
                             shape=[n_samples_in_class, n, n]
                             if point_type='matrix'
            Labelled dataset,
            Where n_samples_in_class is the number of samples in class c
        """
        return X[gs.argmax(y, axis=1) == c]
