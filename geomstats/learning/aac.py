"""Align All and Compute for Graph Space.

Lead author: Anna Calissano.
"""

import logging
import random

from sklearn.base import BaseEstimator

import geomstats.backend as gs
from geomstats.errors import check_parameter_accepted_values
from geomstats.learning._sklearn_wrapper import WrappedLinearRegression, WrappedPCA
from geomstats.learning.frechet_mean import FrechetMean


def _warn_max_iterations(iteration, max_iter):
    if iteration == max_iter:
        logging.warning(
            f"Maximum number of iterations {max_iter} reached. "
            "The estimate may be inaccurate"
        )


class _AACFrechetMean(BaseEstimator):
    r"""Class AAC for Frechet Mean on Graph Space.

    The Align All and Compute (AAC) algorithm for Frechet Mean (FM)estimation is
    introduced in [Calissano2020] and it estimates the Frechet Mean for a set of
    labeled or unlabeled graphs. The idea is to optimally aligned the graphs to the
    current mean estimator using the optimal alignment between the graphs and the mean
    (graph matching between two graphs) and compute the mean estimation between the
    aligned adjacency matrices (the arithmetic mean in the euclidean space of dimension
    :math:`nodes \times nodes`). The algorithm stops as soon as the distance between two
    consecutive estimations is lower then :math:`\epsilon` or the maximum number of
    iterations is reached. The initialization step consists in aligning the data
    with respect to an initial point.

    Parameters
    ----------
    metric : GraphSpaceMetric
        Metric Class on Graph Space.
    epsilon: float, default=1e-6
        Stopping criterion for the estimation step, i.e., the distance between two
        consecutive estimators.
    max_iter: int, default = 20
        Stopping criterion on the maximum number of iterations.
    init_point: array-like, shape=[n_nodes, n_nodes] or GraphPoint, default random.
        Algorithm initialization.
    total_space_estimator_kwargs : dict
        Total space estimator keyword arguments.

    Attributes
    ----------
    total_space_estimator : BaseEstimator
        Frechet mean estimator in total space.
    estimate_ : array-like, mean=[n_nodes, n_nodes]
        Mean.
    n_iter_ : int
        Number of performed iterations.

    References
    ----------
    .. [Calissano2020]  Calissano, A., Feragen, A., Vantini, S.
        “Graph Space: Geodesic Principal Components for a Population of
        Network-valued Data.” Mox report 14, 2020.
        https://mox.polimi.it/reports-and-theses/publication-results/?id=855.
    """

    def __init__(
        self,
        metric,
        *,
        epsilon=1e-6,
        max_iter=20,
        init_point=None,
        total_space_estimator_kwargs=None,
    ):
        self.metric = metric
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.init_point = init_point

        self.total_space_estimator_kwargs = total_space_estimator_kwargs or {}
        self.total_space_estimator = FrechetMean(
            self.metric.total_space_metric, **self.total_space_estimator_kwargs
        )

        self.estimate_ = None
        self.n_iter_ = None

    def fit(self, X, y=None):
        r"""Fit the Frechet Mean.

        Parameters
        ----------
        X : array-like, shape=[n_samples, n_nodes, n_nodes] or set of GraphPoint.
            Dataset to estimate the FM.
        y : Ignored
            Ignored.

        Returns
        -------
        self : object
            Returns self.
        """
        previous_estimate = (
            random.choice(X) if self.init_point is None else self.init_point
        )
        aligned_X = X
        error = self.epsilon + 1
        iteration = 0
        while error > self.epsilon and iteration < self.max_iter:
            iteration += 1

            aligned_X = self.metric.align_point_to_point(previous_estimate, aligned_X)
            new_estimate = self.total_space_estimator.fit(aligned_X).estimate_
            error = self.metric.total_space_metric.dist(previous_estimate, new_estimate)

            previous_estimate = new_estimate

        _warn_max_iterations(iteration, self.max_iter)

        self.estimate_ = new_estimate
        self.n_iter_ = iteration

        return self


class _AACGGPCA(BaseEstimator):
    r"""Class AAC for Generalized Geodesic Principal Components (GGPCA) on Graph Space.

    The Align All and Compute (AAC) algorithm for GGPCA estimation is
    introduced in [Calissano2020] and it estimates the GGPCA for a set of
    labeled or unlabeled graphs. The idea is to optimally aligned the graphs to the
    current GGPCA estimator using the optimal alignment between the graphs and the
    geodesics and then compute the GGPCA estimation between the aligned adjacency
    matrices (the PCA in the euclidean space of dimension :math:`nodes \times nodes`).
    The algorithm stops as soon as the percentage of variance explained by PCA in two
    consecutive estimations is lower then :math:`\epsilon` or the maximum number of
    iteration is reached. The initialization step consists in aligning all the data with
    respect to an initial point.

    Parameters
    ----------
    metric : GraphSpaceMetric
        Metric Class on Graph Space.
    epsilon: float, default=1e-6
        Stopping criterion for the estimation step, i.e., the distance between two
        consecutive estimators.
    max_iter: int, default = 20
        Stopping criterion on the maximum number of iterations.
    init_point: array-like, shape=[n_nodes, n_nodes] or GraphPoint, default random.
        Algorithm initialization.
    n_components: int
        Number of principal components to be estimated. Notice that the convergence is
        ensured only for the first principal component.

    Attributes
    ----------
    total_space_estimator: BaseEstimator
        Method for the estimation of the PCA for a set of flattened adjacency matrices
        in the total space. Check geomstats.learning._sklearn_wrapper for details.
        Default: ``sklearn.decomposition.PCA``.
    n_iter_ : int
        Number of performed iterations.

    References
    ----------
    .. [Calissano2020]  Calissano, A., Feragen, A., Vantini, S.
        “Graph Space: Geodesic Principal Components for a Population of
        Network-valued Data.” Mox report 14, 2020.
        https://mox.polimi.it/reports-and-theses/publication-results/?id=855.
    """

    def __init__(
        self, metric, *, n_components=2, epsilon=1e-6, max_iter=20, init_point=None
    ):
        self.metric = metric
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.init_point = init_point
        self.n_components = n_components
        self.total_space_estimator = WrappedPCA(n_components=self.n_components)
        self.n_iter_ = None

    @property
    def components_(self):
        r"""Principal Components in the total space.

        GGPCA expressed as vectors in the total space.
        """
        return self.total_space_estimator.reshaped_components_

    @property
    def explained_variance_(self):
        r"""Variance Explained along the GGPCA."""
        return self.total_space_estimator.explained_variance_

    @property
    def explained_variance_ratio_(self):
        r"""Percentage of Variance Explained along the GGPCA."""
        return self.total_space_estimator.explained_variance_ratio_

    @property
    def mean_(self):
        r"""Mean at the last iteration."""
        return self.total_space_estimator.reshaped_mean_

    def fit(self, X, y=None):
        r"""Fit the GGPCA.

        Parameters
        ----------
        X : array-like, shape=[n_samples, n_nodes, n_nodes] or set of GraphPoint.
            Dataset to estimate the GGPCA.
        y : Ignored
            Ignored.

        Returns
        -------
        self : object
            Returns self.

        Note: Default method in the total space is sklearn.decomposition.PCA where
        the input data are centered but not scaled for each feature.
        """
        x = random.choice(X) if self.init_point is None else self.init_point
        aligned_X = self.metric.align_point_to_point(x, X)

        self.total_space_estimator.fit(aligned_X)
        previous_expl = self.total_space_estimator.explained_variance_ratio_[0]

        error = self.epsilon + 1
        iteration = 0
        while error > self.epsilon and iteration < self.max_iter:
            iteration += 1
            mean = self.total_space_estimator.reshaped_mean_
            direc = self.total_space_estimator.reshaped_components_[0]

            geodesic = self.metric.total_space_metric.geodesic(
                initial_point=mean, initial_tangent_vec=direc
            )

            aligned_X = self.metric.align_point_to_geodesic(geodesic, aligned_X)
            self.total_space_estimator.fit(aligned_X)
            expl_ = self.total_space_estimator.explained_variance_ratio_[0]

            error = expl_ - previous_expl
            previous_expl = expl_

        self.n_iter_ = iteration
        _warn_max_iterations(iteration, self.max_iter)

        return self


class _AACRegression(BaseEstimator):
    r"""Class AAC for Generalized Geodesic Regression (GGR) on Graph Space.

    The Align All and Compute (AAC) algorithm for GGR estimation is
    introduced in [Calissano2022] and it estimates the GGR for
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
    metric : GraphSpaceMetric
        Metric Class on Graph Space.
    epsilon: float, default=1e-6
        Stopping criterion for the estimation step, i.e., the distance between loss
        function in two consecutive estimation steps.
    max_iter: int, default = 20
        Stopping criterion on the maximum number of iterations.
    init_point: array-like, shape=[n_nodes, n_nodes] or GraphPoint, default random.
        Algorithm initialization.
    total_space_estimator_kwargs : dict
        Total space estimator keyword arguments.

    Attributes
    ----------
    total_space_estimator: BaseEstimator
        Method for the estimation of the OLS Regression for a set of flattened adjacency
        matrices in the total space.
        Check geomstats.learning._sklearn_wrapper for details.
        Default: ``sklearn.linear_model.LinearRegression``.

    References
    ----------
    .. [Calissano2022]  Calissano, A., Feragen, A., Vantini, S.
        “Graph-valued regression: prediction of unlabelled networks in a non-Euclidean
        Graph Space.”Journal of Multivariate Analysis 190 - 104950, (2022).
        https://doi.org/10.1016/j.jmva.2022.104950.
    """

    def __init__(
        self,
        metric,
        *,
        epsilon=1e-6,
        max_iter=20,
        init_point=None,
        total_space_estimator_kwargs=None,
    ):
        self.metric = metric
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.init_point = init_point

        self.total_space_estimator_kwargs = total_space_estimator_kwargs or {}
        self.total_space_estimator = WrappedLinearRegression(
            **self.total_space_estimator_kwargs
        )
        self.n_iter_ = None

    def _compute_pred_error(self, y_pred, y):
        r"""Compute the prediction error."""
        # order matters for reuse of perm_
        return gs.sum(self.metric.dist(y_pred, y))

    def fit(self, X, y):
        r"""Fit the Generalized Geodesic Regression.

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
        aligned_y = self.metric.align_point_to_point(y_, y)

        self.total_space_estimator.fit(X, aligned_y)
        previous_y_pred = self.total_space_estimator.predict(X)
        previous_pred_dist = self._compute_pred_error(previous_y_pred, aligned_y)

        error = self.epsilon + 1
        iteration = 0
        while (error < 0.0 or error > self.epsilon) and iteration < self.max_iter:
            iteration += 1

            # align point to point using previous computed alignment
            aligned_y = self.metric.space.permute(aligned_y, self.metric.perm_)

            self.total_space_estimator.fit(X, aligned_y)
            y_pred = self.total_space_estimator.predict(X)

            pred_dist = self._compute_pred_error(y_pred, aligned_y)
            error = previous_pred_dist - pred_dist

            previous_y_pred = y_pred
            previous_pred_dist = pred_dist

        self.n_iter_ = iteration
        _warn_max_iterations(iteration, self.max_iter)

        return self

    def predict(self, X):
        r"""Predict using the generalized geodesic regression.

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


class AAC:
    r"""Class for Align all and Compute algorithm on Graph Space.

    The Align All and Compute (AAC) algorithm is introduced in [Calissano2020] and it
    allows to compute different statistical estimators: the Frechet Mean, the
    Generalized Geodesic Principal components and the Regression for a set of labeled or
    unlabeled graphs.
    The idea is to optimally aligned the graphs to the current
    estimator using the correct alignment technique and compute the current estimation
    using the geometrical property of the total space, i.e., the Euclidean space of
    adjacency matrices.

    Parameters
    ----------
    metric : GraphSpaceMetric
        Metric Class on Graph Space.
    estimate : str
        Desired estimator. One of the following:
        - "frechet_mean": Frechet Mean estimation [Calissano2020]
        - "ggpca": Generalized Geodesic Principal Components [Calissano2020]
        - "regression": Graph-on-vector regression model [Calissano2022]

    Examples
    --------
    Available example on Graph Space:
    :mod:`notebooks.19_practical_methods__aac`
    Available example on Graph Space with real world data:
    :mod:`notebooks.20_real_world_application__graph_space`

    References
    ----------
    .. [Calissano2020]  Calissano, A., Feragen, A., Vantini, S.
        “Graph Space: Geodesic Principal Components for a Population of
        Network-valued Data.” Mox report 14, 2020.
        https://mox.polimi.it/reports-and-theses/publication-results/?id=855.
    .. [Calissano2022]  Calissano, A., Feragen, A., Vantini, S.
        “Graph-valued regression: prediction of unlabelled networks in a non-Euclidean
        Graph Space.”Journal of Multivariate Analysis 190 - 104950, (2022).
        https://doi.org/10.1016/j.jmva.2022.104950.
    """

    MAP_ESTIMATE = {
        "frechet_mean": _AACFrechetMean,
        "ggpca": _AACGGPCA,
        "regression": _AACRegression,
    }

    def __new__(cls, metric, *args, estimate="frechet", **kwargs):
        r"""Class for Align all and Compute algorithm on Graph Space."""
        check_parameter_accepted_values(
            estimate, "estimate", list(cls.MAP_ESTIMATE.keys())
        )

        return cls.MAP_ESTIMATE[estimate](metric, *args, **kwargs)
