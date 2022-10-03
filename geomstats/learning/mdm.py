"""The MDM classifier on manifolds.

Lead authors: Daniel Brooks and Quentin Barthelemy.
"""

from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin

import geomstats.backend as gs
from geomstats.learning._template import TransformerMixin
from geomstats.learning.frechet_mean import FrechetMean


class RiemannianMinimumDistanceToMean(
    BaseEstimator,
    ClassifierMixin,
    TransformerMixin,
):
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
    classes_ : array-like, shape=[n_classes,]
        If fit, labels of training set.
    mean_estimates_ : array-like, shape=[n_classes, *metric.shape]
        If fit, centroids computed on training set.

    References
    ----------
    .. [BBCJ2012] A. Barachant, S. Bonnet, M. Congedo and C. Jutten, Multiclass
        Brain-Computer Interface Classification by Riemannian Geometry. IEEE
        Trans. Biomed. Eng., vol. 59, pp. 920-928, 2012.
    """

    def __init__(self, riemannian_metric):
        self.riemannian_metric = riemannian_metric
        self.classes_ = None
        self.mean_estimates_ = None

    def fit(self, X, y, weights=None):
        """Compute Frechet mean of each class.

        Parameters
        ----------
        X : array-like, shape=[n_samples, *metric.shape]
            Training input samples.
        y : array-like, shape=[n_samples,]
            Training labels.
        weights : array-like, shape=[n_samples,]
            Weights associated to the samples.
            Optional, default: None, in which case it is equally weighted.

        Returns
        -------
        self : object
            Returns self.
        """
        self.classes_ = gs.unique(y)
        self.n_classes_ = len(self.classes_)
        if weights is None:
            weights = gs.ones(X.shape[0])
        weights /= gs.sum(weights)

        mean_estimator = FrechetMean(metric=self.riemannian_metric)
        frechet_means = []
        for c in self.classes_:
            X_c = X[gs.where(y == c, True, False)]
            weights_c = weights[gs.where(y == c, True, False)]
            mean_c = mean_estimator.fit(X_c, None, weights_c).estimate_
            frechet_means.append(mean_c)
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

    def transform(self, X):
        """Compute distances to each centroid.

        Compute distances to each centroid according to riemannian_metric.

        Parameters
        ----------
        X : array-like, shape=[n_samples, *metric.shape]
            Test samples.

        Returns
        -------
        dist : ndarray, shape=[n_samples, n_classes]
            Distances to each centroid.
        """
        n_samples = X.shape[0]
        dists = []
        for i in range(n_samples):
            dist = self.riemannian_metric.dist(
                X[i],
                self.mean_estimates_,
            )
            dists.append(dist)
        return gs.array(dists)
