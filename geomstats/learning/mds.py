"""Multidimensional scaling (MDS)."""

from sklearn.manifold import MDS
from sklearn.manifold._mds import smacof

import geomstats.backend as gs


class MetricMDS(MDS):
    r"""Metric multidimensional scaling (MDS).

    MDS is used to translate pairwise distances of N objects into
    n points mapped into abstract Cartesian space, usually two-dimensional.

    In general, MDS doesn't need a metric, though this implementation will use one.

    Parameters
    ----------
    space : PointSet
        Space equipped with a distance metric.
    n_components : int
        Number of dimensions in which to immerse the dissimilarities.

    Attributes
    ----------
    embedding_ : array-like, shape=[n_samples, n_components]
        Data transformed in the new space.

    Notes
    -----
    * Required metric methods for general case:
        * `dist`

    References
    ----------
    This algorithm uses the scikit-learn library:
    https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html
    """

    def __init__(self, space, n_components=2):
        self.space = space
        self.n_components = 2

        super().__init__(n_components=self.n_components, dissimilarity="precomputed")

    def fit_transform(self, X, y=None):
        """Fit the data and return the embedded coordinates.

        Parameters
        ----------
        X : array-like, shape=[n_samples, *metric.shape]
            Training input samples.
        y: ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        X_new : array-like, shape=[n_samples, n_components]
            X transformed in the new space.
        """
        n_samples = len(X)

        self.dissimilarity_matrix_ = gs.zeros((n_samples, n_samples))
        max_dist = 0
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dist = self.space.metric.dist(X[i], X[j])

                self.dissimilarity_matrix_[i, j] = dist
                self.dissimilarity_matrix_[j, i] = dist

                if dist > max_dist:
                    max_dist = dist

        self.dissimilarity_matrix_ /= max_dist

        self.embedding_, self.stress_, self.n_iter_ = smacof(
            self.dissimilarity_matrix_,
            # metric=self._metric_mds,
            n_components=self.n_components,
            n_init=4,
            # n_jobs=self.n_jobs,
            # max_iter=self.max_iter,
            # verbose=self.verbose,
            # eps=self.eps,
            random_state=self.random_state,
            return_n_iter=True,
            # normalized_stress=self.normalized_stress,
        )

        return self.embedding_
