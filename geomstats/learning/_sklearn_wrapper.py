from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

import geomstats.backend as gs


class WrappedPCA(PCA):
    # TODO: wrap by manipulating __new__?

    def __init__(
        self,
        n_components=None,
        *,
        copy=True,
        whiten=False,
        svd_solver="auto",
        tol=0.0,
        iterated_power="auto",
        n_oversamples=10,
        power_iteration_normalizer="auto",
        random_state=None,
    ):
        super().__init__(
            n_components,
            copy=copy,
            whiten=whiten,
            svd_solver=svd_solver,
            tol=tol,
            iterated_power=iterated_power,
            n_oversamples=n_oversamples,
            power_iteration_normalizer=power_iteration_normalizer,
            random_state=random_state,
        )
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


class WrappedLinearRegression(LinearRegression):
    # TODO: wrap by manipulating __new__?

    def __init__(
        self,
        *,
        fit_intercept=True,
        copy_X=True,
        n_jobs=None,
        positive=False,
    ):
        super().__init__(
            fit_intercept=fit_intercept, copy_X=copy_X, n_jobs=n_jobs, positive=positive
        )

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
