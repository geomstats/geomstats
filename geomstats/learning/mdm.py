"""The MDM classifier on manifolds."""

import geomstats.backend as gs
from geomstats.learning.frechet_mean import FrechetMean


class RiemannianMinimumDistanceToMeanClassifier():
    """
    Classifier implementing the MDM scheme on manifolds.

    Parameters
    ----------
    riemannian_metric : string or callable.
        The distance metric to use.

    Attributes
    ----------
    G: array-like, shape=[n_classes, n_features] if point_type='vector'
                   shape=[n_classes, n_features, n_features]
                    if point_type='matrix'
       Frechet means of each class.
    """

    def __init__(
            self,
            riemannian_metric,
            mean_method='default',
            verbose=0,
            point_type='matrix'):
        self.riemannian_metric = riemannian_metric
        self.point_type = point_type

    def fit(self, X, Y):
        """
        Compute Frechet mean of each class.

        :param X: array-like, shape=[n_samples, n_features]
                              if point_type='vector'
                              shape=[n_samples, n_features, n_features]
                              if point_type='matrix'
                  Training data, where n_samples is the number of samples
                  and n_features is the number of features.
        :param Y: array-like, shape=[n_samples, n_classes]
                  Training labels, where n_classes is the number of classes.
        """
        n_classes = Y.shape[-1]
        mean_estimator = FrechetMean(
            metric=self.riemannian_metric,
            point_type=self.point_type)
        frechet_means = []
        for c in range(n_classes):
            data_class = self.split_data_in_classes(X, Y, c)
            # frechet_means.append(mean_estimator.fit(data_class).estimate_[0])
            frechet_means.append(mean_estimator.fit(data_class).estimate_)
        self.G = gs.array(frechet_means)
        return

    def predict(self, X):
        """
        Compute closest neighbor according to riemannian_metric.

        :param X: array-like, shape=[n_samples, n_features]
                              if point_type='vector'
                              shape=[n_samples, n_features, n_features]
                              if point_type='matrix'
                  Test data, where n_samples is the number of samples
                  and n_features is the number of features.
        :return: Y: array-like, shape=[n_samples, n_classes]
                    Predicted labels, where n_classes is the number of classes.
        """
        n_samples = X.shape[0]
        n_classes = self.G.shape[0]
        Y = gs.zeros((n_samples, n_classes))
        for i in range(n_samples):
            c = self.riemannian_metric.closest_neighbor_index(X[i], self.G)
            Y[i, c] = 1
        return Y

    def split_data_in_classes(self, X, Y, c):
        """
        Split a labelled dataset in sub-datasets of each label.

        :param X: array-like, shape=[n_samples, n_features]
                              if point_type='vector'
                              shape=[n_samples, n_features, n_features]
                              if point_type='matrix'
                  Labelled dataset, where n_samples is the number of samples
                  and n_features is the number of features.
        :param Y: array-like, shape=[n_samples, n_classes]
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
        return X[gs.where(gs.where(Y)[1] == c)]
