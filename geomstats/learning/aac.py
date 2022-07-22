import logging
import random

import scipy

import geomstats.backend as gs
from geomstats.learning._sklearn_wrapper import WrappedLinearRegression, WrappedPCA
from geomstats.learning.frechet_mean import FrechetMean

# TODO: create AAC and control flow with __new__


def _warn_max_iterations(iteration, max_iter):
    if iteration == max_iter:
        logging.warning(
            f"Maximum number of iterations {max_iter} reached. "
            "The estimate may be inaccurate"
        )


class AACFrechet:
    def __init__(
        self,
        metric,
        *,
        epsilon=1e-6,
        max_iter=20,
        init_point=None,
        mean_estimator_kwargs=None,
    ):
        self.metric = metric
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.init_point = init_point

        mean_estimator_kwargs = mean_estimator_kwargs or {}
        self.mean_estimator = FrechetMean(
            self.metric.total_space_metric, **mean_estimator_kwargs
        )

        self.estimate_ = None
        self.n_iter_ = None

    def fit(self, X):
        previous_estimate = (
            random.choice(X) if self.init_point is None else self.init_point
        )
        aligned_X = X
        error = self.epsilon + 1
        iteration = 0
        while error > self.epsilon and iteration < self.max_iter:
            iteration += 1

            aligned_X = self.metric.align_point_to_point(previous_estimate, aligned_X)
            new_estimate = self.mean_estimator.fit(aligned_X).estimate_
            error = self.metric.total_space_metric.dist(previous_estimate, new_estimate)

            previous_estimate = new_estimate

        _warn_max_iterations(iteration, self.max_iter)

        self.estimate_ = new_estimate
        self.n_iter_ = iteration

        return self


class AACGPC:
    def __init__(
        self, metric, *, n_components=2, epsilon=1e-6, max_iter=20, init_point=None
    ):
        self.metric = metric
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.init_point = init_point

        self.pca_solver = WrappedPCA(n_components=n_components)
        self.aligner = None

    @property
    def components_(self):
        return self.pca_solver.reshaped_components_

    @property
    def explained_variance_(self):
        return self.pca_solver.explained_variance_

    @property
    def explained_variance_ratio_(self):
        return self.pca_solver.explained_variance_ratio_

    @property
    def singular_values_(self):
        return self.pca_solver.singular_values_

    @property
    def mean_(self):
        return self.pca_solver.reshaped_mean_

    def set_default_aligner(self, s_min, s_max, n_sample_points=10):
        self.set_aligner("default", s_min, s_max, n_sample_points=n_sample_points)

    def set_aligner(self, aligner, *args, **kwargs):
        if aligner == "default":
            self.aligner = PointToGeodesicAligner(self.metric, *args, **kwargs)
        else:
            self.aligner = aligner

    def fit(self, X, y=None):
        x = random.choice(X) if self.init_point is None else self.init_point
        aligned_X = self.metric.align_point_to_point(x, X)

        self.pca_solver.fit(aligned_X)
        previous_expl = self.pca_solver.explained_variance_ratio_[0]

        error = self.epsilon + 1
        iteration = 0
        while error > self.epsilon and iteration < self.max_iter:
            iteration += 1
            mean = self.pca_solver.reshaped_mean_
            direc = self.pca_solver.reshaped_components_[0]

            geodesic = self.metric.total_space_metric.geodesic(
                initial_point=mean, initial_tangent_vec=direc
            )

            aligned_X = self.aligner.align(geodesic, aligned_X)
            self.pca_solver.fit(aligned_X)
            expl_ = self.pca_solver.explained_variance_ratio_[0]

            error = expl_ - previous_expl
            previous_expl = expl_

        _warn_max_iterations(iteration, self.max_iter)

        return self


class PointToGeodesicAligner:
    # TODO: move to metric
    # TODO: create base class

    def __init__(self, metric, s_min, s_max, n_sample_points=10):
        self.metric = metric
        self.s_min = s_min
        self.s_max = s_max
        self.n_sample_points = n_sample_points

        self.min_dists_ = None

    @property
    def _s(self):
        return gs.linspace(self.s_min, self.s_max, num=self.n_sample_points)

    def _get_gamma_s(self, geodesic):
        return geodesic(self._s)

    def align(self, geodesic, x):
        gamma_s = self._get_gamma_s(geodesic)

        n_points = 1 if gs.ndim(x) == 2 else gs.shape(x)[0]
        if n_points > 1:
            gamma_s = gs.repeat(gamma_s, n_points, axis=0)
            rep_x = gs.concatenate([x for _ in range(self.n_sample_points)])
        else:
            rep_x = x

        dists = gs.reshape(
            self.metric.dist(gamma_s, rep_x), (self.n_sample_points, n_points)
        )

        min_dists_idx = gs.argmin(dists, axis=0)
        perm_indices = min_dists_idx * n_points + gs.arange(n_points)
        if n_points == 1:
            perm_indices = perm_indices[0]

        perms = gs.take(self.metric.perm_, perm_indices, axis=0)

        # TODO: delete?
        self.min_dists_ = gs.take(
            gs.transpose(dists),
            min_dists_idx + gs.arange(n_points) * self.n_sample_points,
        )

        return self.metric.space.permute(x, perms)


class GeodesicToPointAligner:
    def __init__(self, metric, method="BFGS"):
        self.metric = metric

        self.method = method

        self.opt_results_ = None

    def _objective(self, s, x, geodesic):
        point = geodesic(s)
        dist = self.metric.dist(point, x)

        return dist

    def align(self, geodesic, x):
        n_points = 1 if gs.ndim(x) == 2 else gs.shape(x)[0]

        if n_points == 1:
            x = gs.expand_dims(x, axis=0)

        perms = []
        min_dists = []
        opt_results = []
        for xx in x:
            s0 = 0.0
            res = scipy.optimize.minimize(
                self._objective, x0=s0, args=(xx, geodesic), method=self.method
            )
            perms.append(self.metric.perm_[0])
            min_dists.append(res.fun)

            opt_results.append(res)

        self.min_dists_ = gs.array(min_dists)
        self.opt_results_ = opt_results

        new_x = self.metric.space.permute(x, gs.array(perms))
        return new_x[0] if n_points == 1 else new_x


class AACRegression:
    def __init__(
        self, metric, *, epsilon=1e-6, max_iter=20, init_point=None, model_kwargs=None
    ):
        self.metric = metric
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.init_point = init_point

        # TODO: set regressor?
        model_kwargs = model_kwargs or {}
        self.regressor = WrappedLinearRegression(**model_kwargs)

    def fit(self, X, y):
        y_ = random.choice(y) if self.init_point is None else self.init_point
        aligned_y = self.metric.align_point_to_point(y_, y)

        self.regressor.fit(X, aligned_y)
        previous_y_pred = self.regressor.predict(X)

        error = self.epsilon + 1
        iteration = 0
        while error > self.epsilon and iteration < self.max_iter:
            iteration += 1
            aligned_y = self.metric.align_point_to_point(previous_y_pred, aligned_y)

            self.regressor.fit(X, aligned_y)
            y_pred = self.regressor.predict(X)

            # TODO: squared distances?
            error = gs.sum(self.metric.dist(previous_y_pred, y_pred))

            previous_y_pred = y_pred

        _warn_max_iterations(iteration, self.max_iter)

        return self

    def predict(self, X):
        return self.regressor.predict(X)
