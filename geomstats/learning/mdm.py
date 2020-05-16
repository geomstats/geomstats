"""The MDM classifier on manifolds."""

import geomstats.backend as gs
from geomstats.learning.frechet_mean import FrechetMean


class RiemannianMinimumDistanceToMeanClassifier:
    """
    Classifier implementing the MDM scheme on manifolds.

    Attributes
    ----------
    riemannian_metric : string or callable.
        The distance metric to use.
    n_clusters: int, number of clusters.
    mean_estimate_: array-like, shape=[n_classes, n_features] if
    point_type='vector'
                   shape=[n_classes, n_features, n_features]
                    if point_type='matrix'
       Frechet means of each class.
    """

    def __init__(
            self,
            riemannian_metric,
            n_clusters,
            point_type='matrix'):
        self.riemannian_metric = riemannian_metric
        self.n_clusters = n_clusters
        self.point_type = point_type
        self.mean_estimate_ = None

    def fit(self, X, y):
        """
        Compute Frechet mean of each class.

        :param X: array-like, shape=[n_samples, n_features]
                              if point_type='vector'
                              shape=[n_samples, n_features, n_features]
                              if point_type='matrix'
                  Training data, where n_samples is the number of samples
                  and n_features is the number of features.
        :param y: array-like, shape=[n_samples, n_classes]
                  Training labels, where n_classes is the number of classes.
        """
        mean_estimator = FrechetMean(
            metric=self.riemannian_metric,
            point_type=self.point_type)
        frechet_means = []
        for c in range(self.n_clusters):
            data_class = self.split_data_in_classes(X, y, c)
            # frechet_means.append(mean_estimator.fit(data_class).estimate_[0])
            frechet_means.append(mean_estimator.fit(data_class).estimate_)
        self.mean_estimate_ = gs.array(frechet_means)

    def predict(self, X):
        """
        Compute closest neighbor according to riemannian_metric.

        :param X: array-like, shape=[n_samples, n_features]
                              if point_type='vector'
                              shape=[n_samples, n_features, n_features]
                              if point_type='matrix'
                  Test data, where n_samples is the number of samples
                  and n_features is the number of features.
        :return: y: array-like, shape=[n_samples, n_classes]
                    Predicted labels, where n_classes is the number of classes.
        """
        n_samples = X.shape[0]
        y = gs.zeros((n_samples, self.n_clusters))
        for i in range(n_samples):
            label = self.riemannian_metric.closest_neighbor_index(
                X[i], self.mean_estimate_)
            y[i, label] = 1
        return y

    @staticmethod
    def split_data_in_classes(X, y, c):
        """
        Split a labelled dataset in sub-datasets of each label.

        :param X: array-like, shape=[n_samples, n_features]
                              if point_type='vector'
                              shape=[n_samples, n_features, n_features]
                              if point_type='matrix'
                  Labelled dataset, where n_samples is the number of samples
                  and n_features is the number of features.
        :param y: array-like, shape=[n_samples, n_classes]
                  Labels, where n_classes is the number of classes.
        :param c: int
                  Class index
        :return: array-like, shape=[n_samples_in_class, n_features]
                             if point_type='vector'
                             shape=[n_samples_in_class, n_features, n_features]
                             if point_type='matrix'
                  Labelled dataset,
                  where n_samples_in_class is the number of samples in class c
        """
        return X[gs.where(gs.where(y)[1] == c)]
