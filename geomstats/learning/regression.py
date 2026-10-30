"""Regression on Manifolds/Geodesic metric spaces."""

import random

from sklearn.base import BaseEstimator

import geomstats.backend as gs
from geomstats.learning.euclidean import LinearRegression

from ._utils import _is_graph_space, _warn_max_iterations


class AACRegression(BaseEstimator):
    r"""Generalized Geodesic Regression (GGR) on Graph Space.

    The Align All and Compute (AAC) algorithm for GGR estimation is
    introduced in [CFV2022]_ and it estimates the GGR for
    :math:`\{(s_i, X_i)\in \mathbb{R}^p\times X/T\}` a set of labeled or unlabeled
    graphs as output and a set of scalar or vector as input:
    :math:`f: \mathbb{R}^p \rightarrow X/T`. The idea is to iteratively estimate a OLS
    regression model between a set of regressors and a set of flattened adjacency
    matrices and align the graphs to the current GGR estimator using the optimal
    alignment for regression. The optimal alignment for regression consists in aligning
    the graph with the corresponding predicted graph along the regression model to
    decrease the prediction error. The algorithm stops as soon as the loss in two
    consecutive estimations is lower then math:`\epsilon` or the maximum number of
    iteration is reached. The initialization step consists in aligning all the data with
    respect to a initial point.

    Parameters
    ----------
    space : GraphSpace
        Graph space total space with a quotient structure.
    epsilon: float, default=1e-6
        Stopping criterion for the estimation step, i.e., the distance between loss
        function in two consecutive estimation steps.
    max_iter: int, default = 20
        Stopping criterion on the maximum number of iterations.
    init_point: array-like, shape=[n_nodes, n_nodes] or GraphPoint, default random.
        Algorithm initialization.
    save_last_y: bool, default=True
        Flag to save the data as aligned in the last algorithm iteration.
    total_space_estimator_kwargs : dict
        Total space estimator keyword arguments.

    Attributes
    ----------
    total_space_estimator: BaseEstimator
        Method for the estimation of the OLS Regression for a set of flattened adjacency
        matrices in the total space.
        Check geomstats.learning._sklearn_wrapper for details.
        Default: ``sklearn.linear_model.LinearRegression``.
    aligned_y_: array-like, shape=[n_samples, n_nodes, n_nodes] or set of GraphPoint.
        Set of aligned data as after the last call of fit.
        Saved if ``self.save_last_y is True``.

    References
    ----------
    .. [CFV2022]  Calissano, A., Feragen, A., Vantini, S.
        “Graph-valued regression: prediction of unlabelled networks in a non-Euclidean
        Graph Space.”Journal of Multivariate Analysis 190 - 104950, (2022).
        https://doi.org/10.1016/j.jmva.2022.104950.
    """

    def __init__(
        self,
        space,
        *,
        epsilon=1e-3,
        max_iter=20,
        init_point=None,
        total_space_estimator_kwargs=None,
        save_last_y=True,
    ):
        self.space = space
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.init_point = init_point
        self.save_last_y = save_last_y

        self.total_space_estimator_kwargs = total_space_estimator_kwargs or {}
        self.total_space_estimator = LinearRegression(
            image_space=self.space,
            **self.total_space_estimator_kwargs,
        )
        self.n_iter_ = None
        self.aligned_y_ = None

    def fit(self, X, y):
        """Fit the Generalized Geodesic Regression.

        Parameters
        ----------
        X : array-like, shape=[n_samples, p].
            Dataset of regressors to estimate the GGR.
        y : array-like, shape=[n_samples, n_nodes, n_nodes] or set of GraphPoint.
            Dataset to estimate the GGR.

        Returns
        -------
        self : object
            Returns self.
        """
        y_ = random.choice(y) if self.init_point is None else self.init_point
        aligned_y = self.space.aligner.align(y, y_)

        previous_pred_dist = 1e6
        for iteration in range(self.max_iter):
            self.total_space_estimator.fit(X, aligned_y)
            y_pred = self.total_space_estimator.predict(X)

            aligned_y = self.space.aligner.align(aligned_y, y_pred)
            pred_dist = gs.sum(self.space.metric.dist(y_pred, aligned_y))

            dist_diff = gs.abs(previous_pred_dist - pred_dist)
            if dist_diff < self.epsilon:
                break

            previous_pred_dist = pred_dist
        else:
            _warn_max_iterations(self.max_iter)

        if self.save_last_y:
            self.aligned_y_ = aligned_y
        self.n_iter_ = iteration

        return self

    def predict(self, X):
        """Predict using the generalized geodesic regression.

        Predict a graph or a set of graphs corresponding to the given regressors. It
        uses the total space prediction.

        Parameters
        ----------
        X : array-like, shape=[n_samples, n_nodes, n_nodes] or set of GraphPoint
            Dataset to estimate the GGR.

        Returns
        -------
        prediction : array-like, shape=[n_samples, n_nodes, n_nodes] or set of
            GraphPoint
            Predicted unlabeled graphs.
        """
        return self.total_space_estimator.predict(X)


def GeneralizedGeodesicRegression(space, **kwargs):
    r"""Generalized Geodesic Regression.

    Parameters
    ----------
    space : Manifold
        Equipped manifold.
    """
    if _is_graph_space(space):
        return AACRegression(space, **kwargs)

    raise NotImplementedError("GGPCA is only implemented for graphspace.")
