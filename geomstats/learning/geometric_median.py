"""Geometric median."""

import logging

from sklearn.base import BaseEstimator

import geomstats.backend as gs


class GeometricMedian(BaseEstimator):
    """Geometric median.

    Parameters
    ----------
    space : Manifold
        Equipped manifold.
    max_iter : int
        Maximum number of iterations for the algorithm.
        Optional, default : 100
    lr : float
        Learning rate to be used for the algorithm.
        Optional, default : 1.0
    init_point : array-like, shape=[*space.shape]
        Initialization to be used in the start.
        Optional, default : None, in which case it uses last sample.
    print_every : int
        Print updated median after print_every iterations.
        Optional, default : None
    epsilon : float
        Tolerance for stopping the algorithm (distance between two successive
        estimates).
        Optional, default : gs.atol

    Attributes
    ----------
    estimate_ : array-like, shape=[*space.shape]
        If fit, geometric median.

    References
    ----------
    .. [FVJ2009] Fletcher PT, Venkatasubramanian S and Joshi S.
        "The geometric median on Riemannian manifolds with application to
        robust atlas estimation", NeuroImage, 2009
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2735114/
    """

    def __init__(
        self,
        space,
        max_iter=100,
        lr=1.0,
        init_point=None,
        print_every=None,
        epsilon=gs.atol,
    ):
        self.space = space
        self.max_iter = max_iter
        self.lr = lr
        self.init_point = init_point
        self.print_every = print_every
        self.epsilon = epsilon

        self.estimate_ = None

    def _iterate_once(self, current_median, X, weights, lr):
        """Compute a single iteration of Weiszfeld algorithm.

        Parameters
        ----------
        current_median : array-like, shape=[*space.shape]
            Current median.
        X : array-like, shape=[n_samples, *space.shape]
            Training input samples.
        weights : array-like, shape=[n_samples,]
            Weights for weighted sum.
        lr : float
            Learning rate for the current iteration.

        Returns
        -------
        updated_median : array-like, shape=[*metric.shape]
            Updated median after single iteration.
        """
        dists = self.space.metric.dist(current_median, X)
        is_non_zero = dists > gs.atol
        if not gs.any(is_non_zero):
            return current_median

        w = weights[is_non_zero] / dists[is_non_zero]
        logs = self.space.metric.log(X[is_non_zero], current_median)
        v_k = gs.einsum("n,n...->...", w / gs.sum(w), logs)
        updated_median = self.space.metric.exp(lr * v_k, current_median)
        return updated_median

    def fit(self, X, y=None, weights=None):
        """Compute the weighted geometric median.

        Compute the geometric median on manifold using Weiszfeld algorithm.

        Parameters
        ----------
        X : array-like, shape=[n_samples, *metric.shape]
            Training input samples.
        y : None
            Target values. Ignored.
        weights : array-like, shape=[n_samples,]
            Weights associated to the samples.
            Optional, default: None, in which case it is equally weighted.

        Returns
        -------
        self : object
            Returns self.
        """
        median = X[-1] if self.init_point is None else self.init_point
        if weights is None:
            weights = gs.ones(X.shape[0])
        weights /= gs.sum(weights)

        for iteration in range(self.max_iter):
            new_median = self._iterate_once(median, X, weights, self.lr)
            shift = self.space.metric.dist(new_median, median)

            if shift < self.epsilon:
                break
            median = new_median

            if self.print_every and (iteration + 1) % self.print_every == 0:
                logging.info(f"median at iteration {iteration+1}:\n{median}")
        else:
            logging.warning(
                "Maximum number of iterations %s reached. "
                "The median may be inaccurate",
                self.max_iter,
            )

        self.estimate_ = median
        return self
