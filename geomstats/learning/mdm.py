"""The MDM classifier on manifolds.

Lead authors: Daniel Brooks and Quentin Barthelemy.
"""

from scipy.special import softmax
from sklearn.metrics import accuracy_score

import geomstats.backend as gs
from geomstats.learning.frechet_mean import FrechetMean


class RiemannianMinimumDistanceToMeanClassifier:
    """Minimum Distance to Mean (MDM) classifier on manifolds.

    Classification by nearest centroid. For each of the given classes, a
    centroid is estimated according to the chosen metric. Then, for each new
    point, the class is affected according to the nearest centroid [BBCJ2012]_.

    Parameters
    ----------
    riemannian_metric : RiemannianMetric
        Riemannian metric to be used.

    Attributes
    ----------
    n_classes_ : int
        If fit, number of classes.
    classes_ : list
        If fit, n_classes labels of training set.
    mean_estimates_ : list of arrays-like of shape=[*metric.shape]
        If fit, n_classes centroids computed on training set.

    References
    ----------
    .. [BBCJ2012] A. Barachant, S. Bonnet, M. Congedo and C. Jutten, Multiclass
        Brain-Computer Interface Classification by Riemannian Geometry. IEEE
        Trans. Biomed. Eng., vol. 59, pp. 920-928, 2012.
    """

    def __init__(self, riemannian_metric):
        self.riemannian_metric = riemannian_metric
        self.n_classes_ = None
        self.classes_ = None
        self.mean_estimates_ = None

    def fit(self, X, y):
        """Compute Frechet mean of each class.

        Parameters
        ----------
        X : array-like, shape=[n_samples, *metric.shape]
            Training input samples.
        y : array-like, shape=[n_samples,]
            Training labels.
        """
        self.classes_ = gs.unique(y)
        self.n_classes_ = len(self.classes_)
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
            Test samples.

        Returns
        -------
        y : array-like, shape=[n_samples,]
            Predicted labels.
        """
        indices = self.riemannian_metric.closest_neighbor_index(
            X,
            self.mean_estimates_,
        )
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
            Test samples.

        Returns
        -------
        probas : array-like, shape=[n_samples, n_classes]
            Probability of the sample for each class in the model.
        """
        n_samples = X.shape[0]
        probas = []
        for i in range(n_samples):
            dist2 = self.riemannian_metric.squared_dist(
                X[i],
                self.mean_estimates_,
            )
            probas.append(softmax(-dist2))
        return gs.array(probas)

    def score(self, X, y, weights=None):
        """Compute score on the given test data and labels.

        Compute the score defined as accuracy.

        Parameters
        ----------
        X : array-like, shape=[n_samples, *metric.shape]
            Test samples.
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
