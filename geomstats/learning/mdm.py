"""The MDM classifier on manifolds.

Lead authors: Daniel Brooks and Quentin Barthelemy.
"""

from scipy.special import softmax
from sklearn.metrics import accuracy_score

import geomstats.backend as gs
from geomstats.learning.frechet_mean import FrechetMean


class RiemannianMinimumDistanceToMeanClassifier:
    r"""Minimum Distance to Mean (MDM) classifier on manifolds.

    Classification by nearest centroid. For each of the given classes, a
    centroid is estimated according to the chosen metric. Then, for each new
    point, the class is affected according to the nearest centroid (see
    [BBCJ2012]_).

    Parameters
    ----------
    riemannian_metric : RiemannianMetric
        Riemannian metric to be used.
    n_classes : int
        Number of classes.

    Attributes
    ----------
    mean_estimates_ : list
        If fit, centroids computed on training set.
    classes_ : list
        If fit, classes of training set.

    References
    ----------
    .. [BBCJ2012] A. Barachant, S. Bonnet, M. Congedo and C. Jutten, Multiclass
        Brain-Computer Interface Classification by Riemannian Geometry. IEEE
        Trans. Biomed. Eng., vol. 59, pp. 920-928, 2012.
    """

    def __init__(self, riemannian_metric, n_classes):
        self.riemannian_metric = riemannian_metric
        self.n_classes = n_classes
        self.mean_estimates_ = None
        self.classes_ = None

    def fit(self, X, y):
        """Compute Frechet mean of each class.

        Parameters
        ----------
        X : array-like, shape=[n_samples, *metric.shape]
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape=[n_samples,]
            Training labels.
        """
        self.classes_ = gs.unique(y)
        mean_estimator = FrechetMean(metric=self.riemannian_metric)
        frechet_means = []
        for c in self.classes_:
            X_c = X[gs.where(y == c, True, False)]
            frechet_means.append(mean_estimator.fit(X_c).estimate_)
        self.mean_estimates_ = gs.array(frechet_means)

    def predict(self, X):
        """Compute closest neighbor according to riemannian_metric.

        Parameters
        ----------
        X : array-like, shape=[n_samples, *metric.shape]
            Test data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        y : array-like, shape=[n_samples,]
            Predicted labels.
        """
        indices = self.riemannian_metric.closest_neighbor_index(X, self.mean_estimates_)
        if gs.ndim(indices) == 0:
            indices = gs.expand_dims(indices, 0)

        return gs.take(self.classes_, indices)

    def predict_proba(self, X):
        """Compute probabilities.

        Compute probabilities to belong to classes according to
        riemannian_metric.

        Parameters
        ----------
        X : array-like, shape=[n_samples, *metric.shape]
            Test data, where n_samples is the number of samples
            and n_features is the number of features.

        Returns
        -------
        probas : array-like, shape=[n_samples, n_classes]
            Probability of the sample for each class in the model.
        """
        n_samples = X.shape[0]
        probas = []
        for i in range(n_samples):
            dist2 = self.riemannian_metric.squared_dist(X[i], self.mean_estimates_)
            probas.append(softmax(-dist2))
        return gs.array(probas)

    def score(self, X, y, weights=None):
        """Compute score on the given test data and labels.

        Compute the score defined as accuracy.

        Parameters
        ----------
        X : array-like, shape=[n_samples, *metric.shape]
            Test data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape=[n_samples,]
            True labels for `X`.
        weights : array-like, shape=[n_samples,]
            Weights associated to the samples.
            Optional, default: None.

        Returns
        -------
        score : float
            Mean accuracy of ``self.predict(X)`` wrt. `y`.
        """
        return accuracy_score(y, self.predict(X), sample_weight=weights)
