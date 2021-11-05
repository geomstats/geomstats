"""The MDM classifier on manifolds."""

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
        Riemannian metric to be Used.
    n_classes: int
        Number of classes.
    point_type : str, {\'vector\', \'matrix\'}
        Point type.
        Optional, default: \'matrix\'.

    Attributes
    ----------
    mean_estimates_ : list
        Centroids.
    classes_ : list
        Classes.

    References
    ----------
    .. [BBCJ2012] A. Barachant, S. Bonnet, M. Congedo and C. Jutten, Multiclass
        Brain-Computer Interface Classification by Riemannian Geometry. IEEE
        Trans. Biomed. Eng., vol. 59, pp. 920-928, 2012.
    """

    def __init__(self, riemannian_metric, n_classes, point_type="matrix"):
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
        y : array-like, shape=[n_samples,]
            Training labels.
        """
        self.classes_ = gs.unique(y)
        mean_estimator = FrechetMean(
            metric=self.riemannian_metric, point_type=self.point_type
        )
        frechet_means = []
        for c in self.classes_:
            data_class = X[y == c]
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
        y : array-like, shape=[n_samples,]
            Predicted labels.
        """
        y = []
        for i in range(X.shape[0]):
            index = self.riemannian_metric.closest_neighbor_index(
                X[i], self.mean_estimates_
            )
            y.append(self.classes_[index])
        return y

    def score(self, X, y, weights=None):
        """Compute score on the given test data and labels.

        Compute the score defined as accuracy.

        Parameters
        ----------
        X : array-like, shape=[n_samples, dim]
                              if point_type='vector'
                              shape=[n_samples, n , n]
                              if point_type='matrix'
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
