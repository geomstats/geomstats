import logging
import random

import scipy
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

import geomstats.backend as gs
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


class _WrappedPCA(PCA):
    # TODO: wrap by manipulating __new__?

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._init_shape = None

    def __repr__(self):
        # to use *args and **kwargs
        return object.__repr__(self)

    @property
    def reshaped_components_(self):
        if self.components_ is None:
            return None
        return gs.reshape(self.components_, (self.n_components, *self._init_shape[1:]))

    @property
    def reshaped_mean_(self):
        if self.mean_ is None:
            return None

        return gs.reshape(self.mean_, self._init_shape[1:])

    def _reshape(self, x):
        return gs.reshape(x, (x.shape[0], -1))

    def _reshape_X(self, X):
        self._init_shape = X.shape
        return self._reshape(X)

    def fit(self, X, y=None):
        return super().fit(self._reshape_X(X))

    def fit_transform(self, X, y=None):
        return super().fit_transform(self._reshape_X(X))

    def score_samples(self, X, y=None):
        return super().score_samples(self._reshape(X))

    def score(self, X, y=None):
        return super().score(self._reshape(X))


class AACGPC:
    def __init__(
        self, metric, *, n_components=2, epsilon=1e-6, max_iter=20, init_point=None
    ):
        self.metric = metric
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.init_point = init_point

        self.pca_solver = _WrappedPCA(n_components=n_components)
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


class _WrappedLinearRegression(LinearRegression):
    # TODO: wrap by manipulating __new__?

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._init_shape_X = None
        self._init_shape_y = None

    def __repr__(self):
        # to use *args and **kwargs
        return object.__repr__(self)

    def _reshape(self, x):
        return gs.reshape(x, (x.shape[0], -1))

    def _reshape_X(self, X):
        self._init_shape_X = X.shape
        return self._reshape(X)

    def _reshape_y(self, y):
        self._init_shape_y = y.shape
        return self._reshape(y)

    def _reshape_out(self, out):
        return gs.reshape(out, (out.shape[0], *self._init_shape_y[1:]))

    def fit(self, X, y):
        return super().fit(self._reshape_X(X), y=self._reshape_y(y))

    def predict(self, X):
        return self._reshape_out(super().predict(self._reshape(X)))


class AACRegression:
    def __init__(
        self, metric, *, epsilon=1e-6, max_iter=20, init_point=None, model_kwargs=None
    ):
        self.metric = metric
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.init_point = init_point

        model_kwargs = model_kwargs or {}
        self.model = _WrappedLinearRegression(**model_kwargs)

    def fit(self, X, y):
        y_ = random.choice(y) if self.init_point is None else self.init_point
        aligned_y = self.metric.align_point_to_point(y_, y)

        self.model.fit(X, aligned_y)
        previous_y_pred = self.model.predict(X)

        error = self.epsilon + 1
        iteration = 0
        while error > self.epsilon and iteration < self.max_iter:
            iteration += 1
            aligned_y = self.metric.align_point_to_point(previous_y_pred, aligned_y)

            self.model.fit(X, aligned_y)
            y_pred = self.model.predict(X)

            # TODO: squared distances?
            error = gs.sum(self.metric.dist(previous_y_pred, y_pred))
            print(error)

            previous_y_pred = y_pred

        _warn_max_iterations(iteration, self.max_iter)

        return self

    def predict(self, X):
        return self.model.predict(X)
