"""Multidimensional scaling (MDS)."""

from sklearn.manifold import MDS as _MDS

import geomstats.backend as gs


def pairwise_dists(points, space):
    """Compute the pairwise distance between points.

    Parameters
    ----------
    points : array-like, shape=[n_samples, dim]
        Set of points in the manifold.
    space : Manifold or PointSet

    Returns
    -------
    pairwise_dists : array-like, shape=[n_samples, n_samples]
        Pairwise distance matrix between all the points.
    """
    n_samples = len(points)

    pairwise_dists = gs.zeros((n_samples, n_samples))
    for i in range(n_samples):
        dists = space.metric.dist(points[i], points[i + 1 :])
        pairwise_dists[i, i + 1 :] = dists
        pairwise_dists[i + 1 :, i] = dists
    return pairwise_dists


class MDS(_MDS):
    r"""Multidimensional scaling (MDS).

    MDS is used to translate pairwise distances of N objects into
    n points mapped into abstract Cartesian space, usually two-dimensional.

    In general, MDS doesn't need a metric, though this implementation will use one.

    Parameters
    ----------
    space : Manifold or PointSet
        Space equipped with a distance metric.
    n_components : int
        Number of dimensions in which to immerse the dissimilarities.

    Attributes
    ----------
    embedding_ : array-like, shape=[n_samples, n_components]
        Data transformed in the new space.

    Notes
    -----
    * For all other parameters see documentation in scikit-learn.
    * Required metric methods for general case:
        * `dist`

    References
    ----------
    This algorithm uses the scikit-learn library:
    https://scikit-learn.org/stable/modules/generated/sklearn.manifold.MDS.html
    """

    def __init__(
        self,
        space,
        n_components=2,
        metric_mds=True,
        n_init=1,
        init="classical_mds",
        max_iter=300,
        verbose=0,
        eps=1e-06,
        n_jobs=None,
        random_state=None,
        metric_params=None,
        normalized_stress="auto",
    ):
        self.space = space

        super().__init__(
            n_components=n_components,
            metric_mds=metric_mds,
            n_init=n_init,
            init=init,
            max_iter=max_iter,
            verbose=verbose,
            eps=eps,
            n_jobs=n_jobs,
            random_state=random_state,
            metric="precomputed",
            metric_params=metric_params,
            normalized_stress=normalized_stress,
        )

    def fit_transform(self, X, y=None, init=None):
        """Fit the data and return the embedded coordinates.

        Parameters
        ----------
        X : array-like, shape=[n_samples, *metric.shape]
            Training input samples.
        y: ignored
            Not used, present for API consistency by convention.
        init : ndarray of shape (n_samples, n_components), default=None
            Starting configuration of the embedding to initialize the SMACOF
            algorithm. By default, the algorithm is initialized with a randomly
            chosen array. DON'T USE

        Returns
        -------
        X_new : array-like, shape=[n_samples, n_components]
            X transformed in the new space.
        """
        dissimilarity_matrix = pairwise_dists(X, self.space)
        self.embedding_ = gs.from_numpy(super().fit_transform(dissimilarity_matrix))
        self.dissimilarity_matrix_ = gs.from_numpy(dissimilarity_matrix)
        return self.embedding_
